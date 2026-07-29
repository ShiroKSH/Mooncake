#include <mooncake_communicator.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <numeric>
#include <stdexcept>
#include <utility>

#include <glog/logging.h>

#include "memory_location.h"
#include "pg_utils.h"

namespace mooncake {
namespace {

constexpr const char* kRegisterBufferError =
    "failed to register local collective memory";
// A non-zero operation size is required to ensure that the worker creates a
// task for the barrier.
constexpr size_t kBarrierDummySize = 1;

AgentInterface& requireAgent(
    const std::shared_ptr<MooncakePGContext>& context) {
    PG_CHECK(context, "Mooncake process context is null");
    PG_CHECK(context->agent_host, "Mooncake process context is not connected");
    return *context->agent_host;
}

void checkCuda(cudaError_t error, const char* operation) {
    PG_CHECK(error == cudaSuccess, operation, ": ", cudaGetErrorString(error));
}

void copyToDevice(void* dst, const void* src, size_t bytes,
                  cudaStream_t stream) {
    checkCuda(
        cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream),
        "cudaMemcpyAsync");
}

}  // namespace

MooncakePGContext::~MooncakePGContext() {
    // Shutdown AgentHost first so its process-level unregisterAgent RPC can
    // reach the local Coordinator before that Coordinator is torn down.
    if (agent_host) agent_host->shutdown();
    // Shutdown CoordinatorHost second so rank 0 fails pending proposals.
    if (coordinator_host) coordinator_host->shutdown();
}

void MooncakePGContext::initializeDataPlane(int rank, int world_size) {
    std::lock_guard<std::mutex> lock(initialize_mutex_);
    PG_CHECK(rank >= 0 && rank < world_size,
             "global rank is outside the process world");
    PG_CHECK(world_size > 0 && world_size <= kMaxNumRanks,
             "max_world_size is outside the supported range");
    if (max_world_size != 0) {
        PG_CHECK(global_rank == rank && max_world_size == world_size,
                 "Mooncake process context was initialized with a "
                 "different rank or world size");
        return;
    }

    global_rank = rank;
    max_world_size = world_size;

    // Ordering constraint: AgentHost::start() sends registerAgent immediately,
    // which includes LinkManager's localServerName() and getWarmupRecvAddr().
    // These must be non-empty, so the engine and LinkManager must be
    // initialized before connectCoordinator starts the AgentHost.
    if (!engine_initialized) {
#ifdef USE_MACA
        PG_CHECK(std::getenv("MC_MACA_HOST_TRANSPORT") != nullptr,
                 "MACA PG requires MC_MACA_HOST_TRANSPORT=1");
#endif
        engine->init(P2PHANDSHAKE, host_ip);
        engine_initialized = true;
    }
    if (!link_manager.isInitialized()) {
        link_manager.init(rank, world_size, engine);
    }
}

std::string MooncakePGContext::launchCoordinator() {
    std::lock_guard<std::mutex> lock(initialize_mutex_);
    PG_CHECK(max_world_size > 0,
             "initializeDataPlane must run before launchCoordinator");
    PG_CHECK(global_rank == 0, "only global rank 0 may start the coordinator");
    if (!coordinator_host) {
        coordinator_host = std::make_unique<CoordinatorHost>(
            host_ip, max_world_size, fault_reconciliation_window_us);
        coordinator_host->start();
    }
    return coordinator_host->getListenAddr();
}

AgentHost& MooncakePGContext::connectCoordinator(
    const std::string& coordinator_address) {
    std::lock_guard<std::mutex> lock(initialize_mutex_);
    PG_CHECK(max_world_size > 0,
             "initializeDataPlane must run before connectCoordinator");
    if (!agent_host) {
        agent_host = std::make_unique<AgentHost>(
            coordinator_address, host_ip, global_rank, max_world_size,
            link_manager, fault_reconciliation_window_us);
        agent_host->start();
    }
    return *agent_host;
}

void MooncakePGContext::setHostIp(std::string value) {
    std::lock_guard<std::mutex> lock(initialize_mutex_);
    host_ip = std::move(value);
}

void MooncakePGContext::setExternalEngine(TransferEngine* transfer_engine) {
    std::lock_guard<std::mutex> lock(initialize_mutex_);
    PG_CHECK(max_world_size == 0,
             "transfer engine must be configured before PG init");
    if (transfer_engine) {
        engine = transfer_engine;
        engine_initialized = true;
    } else {
        engine = owned_engine.get();
        engine_initialized = false;
    }
}

void MooncakePGContext::setDeviceFilter(std::vector<std::string> filters) {
    std::lock_guard<std::mutex> lock(initialize_mutex_);
    engine->setWhitelistFilters(std::move(filters));
}

void MooncakePGContext::setFaultReconciliationWindow(int64_t timeout_us) {
    std::lock_guard<std::mutex> lock(initialize_mutex_);
    fault_reconciliation_window_us = timeout_us;
}

/**
 * @brief Initialize Mooncake communicator state from the framework-neutral
 * communicator configuration.
 */
MooncakeCommunicator::MooncakeCommunicator(
    std::shared_ptr<MooncakePGContext> context,
    MooncakeCommunicatorConfig config)
    : context_(std::move(context)),
      agent_(requireAgent(context_)),
      rank_(config.rank),
      size_(config.size),
      max_group_size_(config.max_group_size > 0 ? config.max_group_size
                                                : config.size),
      device_index_(config.device_index),
      is_cpu_(config.is_cpu),
      active_ranks_mirror_(config.active_ranks_mirror),
      active_ranks_mirror_count_(config.active_ranks_mirror_count),
      active_ranks_mirror_is_device_(config.active_ranks_mirror_is_device) {
    PG_CHECK(size_ > 0 && size_ <= max_group_size_,
             "group size exceeds max_group_size");
    PG_CHECK(max_group_size_ > 0 && max_group_size_ <= kMaxNumRanks,
             "max_group_size is outside the supported range");
    PG_CHECK(rank_ >= 0 && rank_ < size_, "rank is outside the initial group");
    PG_CHECK(!config.group_bootstrap_id.empty(),
             "group bootstrap id must not be empty");
    PG_CHECK(!config.auto_sync_on_failure || config.auto_deactivate_on_failure,
             "auto_sync_on_failure requires "
             "auto_deactivate_on_failure");
    if (active_ranks_mirror_) {
        PG_CHECK(
            active_ranks_mirror_count_ >= static_cast<size_t>(max_group_size_),
            "active-ranks mirror is too small");
    }

    // Build rank order from global_ranks. rank order is a mapping from in-group
    // rank to global rank.
    std::vector<GlobalRank> initial_rank_order;
    if (config.global_ranks.size() == static_cast<size_t>(size_)) {
        initial_rank_order = config.global_ranks;
    } else {
        // Fallback with identical mapping.
        initial_rank_order.resize(size_);
        std::iota(initial_rank_order.begin(), initial_rank_order.end(), 0);
    }

    // Memory location for device-specific buffers. Always kWildcardLocation for
    // a CPU communicator.
    std::string location = kWildcardLocation;
    if (!is_cpu_) {
        if (device_index_ >= 0) {
            checkCuda(cudaSetDevice(device_index_), "cudaSetDevice");
        } else {
            checkCuda(cudaGetDevice(&device_index_), "cudaGetDevice");
        }
        location = GPU_PREFIX + std::to_string(device_index_);
    }

    // Register collective buffers.
    for (size_t index = 0; index < 2; ++index) {
        if (is_cpu_) {
            send_buffer_[index] = std::malloc(kBufferSize);
            recv_buffer_[index] = std::malloc(kBufferSize);
            PG_CHECK(send_buffer_[index] && recv_buffer_[index],
                     "failed to allocate CPU collective buffers");
        } else {
            checkCuda(cudaMalloc(&send_buffer_[index], kBufferSize),
                      "cudaMalloc send buffer");
            checkCuda(cudaMalloc(&recv_buffer_[index], kBufferSize),
                      "cudaMalloc recv buffer");
        }
        PG_CHECK(context_->engine->registerLocalMemory(
                     send_buffer_[index], kBufferSize, location) == 0,
                 kRegisterBufferError);
        PG_CHECK(context_->engine->registerLocalMemory(
                     recv_buffer_[index], kBufferSize, location) == 0,
                 kRegisterBufferError);

        // Register CPU synchronization regions.
        cpu_sync_send_region_[index] = new int32_t[kMaxNumRanks]{};
        cpu_sync_recv_region_[index] = new int32_t[kMaxNumRanks]{};
        PG_CHECK(context_->engine->registerLocalMemory(
                     cpu_sync_send_region_[index],
                     kMaxNumRanks * sizeof(int32_t), kWildcardLocation) == 0,
                 kRegisterBufferError);
        PG_CHECK(context_->engine->registerLocalMemory(
                     cpu_sync_recv_region_[index],
                     kMaxNumRanks * sizeof(int32_t), kWildcardLocation) == 0,
                 kRegisterBufferError);
    }

    if (is_cpu_) {
        p2p_device_worker_ =
            context_->p2p_device_worker_manager.getCPUWorker(context_->engine);
        worker_ = context_->worker_manager.GetCPUWorker();
    } else {
        p2p_device_worker_ = context_->p2p_device_worker_manager.getCUDAWorker(
            device_index_, context_->engine);
        worker_ = context_->worker_manager.GetCUDAWorker(device_index_);
        preloadReduceKernels();
    }
    worker_->Start();

    p2p_proxy_ = std::make_shared<P2PProxy>(
        context_->engine,
        P2PProxy::Options{.is_cpu = is_cpu_,
                          .rank = rank_,
                          .size = max_group_size_,
                          .cuda_device_index = device_index_,
                          .p2p_timeout_us = &context_->p2p_timeout_us});
    p2p_device_worker_->registerProxy(p2p_proxy_);

    meta_ = std::make_shared<TransferGroupMeta>();
    for (int index = 0; index < kMaxNumRanks; ++index) {
        meta_->segmentIDs[index] = static_cast<TransferMetadata::SegmentID>(-1);
        meta_->rankEpochs[index] = 0;
        meta_->rankStates[index] = RankState::Offline;
    }
    meta_->rank = rank_;
    meta_->globalRank = initial_rank_order[rank_];
    for (int index = 0; index < max_group_size_; ++index) {
        // initial_rank_order only has `size_` entries; for remaining extension
        // slots default to identity so that in-group rank i maps to global rank
        // i until applyViewUpdate overwrites it.
        meta_->rank_order[index] =
            index < size_ ? initial_rank_order[index] : index;
    }
    meta_->maxGroupSize = max_group_size_;  // slot capacity
    meta_->activeSize.store(size_, std::memory_order_relaxed);
    meta_->taskCount = 0;
    meta_->collectiveTimeoutUs = &context_->collective_timeout_us;
    meta_->engine = context_->engine;
    meta_->communicator = this;
    meta_->autoSyncOnFailure = config.auto_sync_on_failure;
    p2p_proxy_->bindMeta(meta_);

    // Active ranks will be filled by applyViewUpdate, so only allocate their
    // storage here.
    meta_->maybeActivatable = new bool[max_group_size_]{};
    if (is_cpu_) {
        meta_->activeRanks = new bool[max_group_size_]{};
        meta_->activeRanksDevice = meta_->activeRanks;
    } else {
        checkCuda(
            cudaHostAlloc(&meta_->activeRanks, max_group_size_ * sizeof(bool),
                          cudaHostAllocMapped),
            "cudaHostAlloc active ranks");
        checkCuda(cudaHostGetDevicePointer(&meta_->activeRanksDevice,
                                           meta_->activeRanks, 0),
                  "cudaHostGetDevicePointer active ranks");
        std::fill_n(meta_->activeRanks, max_group_size_, false);
    }

    // Initial local endpoint info.
    meta_->segmentInfos[rank_] = GroupEndpointInfo{
        .send_buffer = {reinterpret_cast<uint64_t>(send_buffer_[0]),
                        reinterpret_cast<uint64_t>(send_buffer_[1])},
        .recv_buffer = {reinterpret_cast<uint64_t>(recv_buffer_[0]),
                        reinterpret_cast<uint64_t>(recv_buffer_[1])},
        .send_sync = {reinterpret_cast<uint64_t>(cpu_sync_send_region_[0]),
                      reinterpret_cast<uint64_t>(cpu_sync_send_region_[1])},
        .recv_sync = {reinterpret_cast<uint64_t>(cpu_sync_recv_region_[0]),
                      reinterpret_cast<uint64_t>(cpu_sync_recv_region_[1])},
        .p2p_credit_region =
            reinterpret_cast<uint64_t>(p2p_proxy_->credit_region()),
        .p2p_ack_region = reinterpret_cast<uint64_t>(p2p_proxy_->ack_region()),
    };

    // Control Plane Initialization

    // Wait for Agent registration.
    PG_CHECK(agent_.waitUntilRegistered(std::chrono::seconds(30)),
             "Agent registration timed out");

    // The framework-provided group id is only a bootstrap id. The Coordinator
    // resolves it together with rank order into a process-lifetime GroupId. CPU
    // and device communicators use independent namespaces.
    auto bootstrap_id = std::string(is_cpu_ ? "cpu:" : "device:") +
                        std::move(config.group_bootstrap_id);

    // Register this group with the Agent, publish the local endpoint, and block
    // until the Coordinator says it is ready. Group registration is
    // synchronous.
    meta_->group_id = agent_.registerGroup(
        std::move(bootstrap_id), max_group_size_, std::move(initial_rank_order),
        config.group_resolve_policy, config.auto_deactivate_on_failure, this);

    if (!isValidGroup()) {
        // Registration rejection is scoped to this communicator. Keep the Agent
        // and every other group untouched, and use the pre-join local-only
        // collective behavior with an effective {self} membership.
        std::fill_n(meta_->activeRanks, max_group_size_, false);
        meta_->activeRanks[rank_] = true;
        meta_->autoSyncOnFailure = false;
        syncActiveRanksMirror();
        refreshSegmentID(rank_);
        LOG(WARNING) << "Mooncake communicator rank=" << meta_->globalRank
                     << " is using local-only execution because group "
                        "registration was rejected";
        return;
    }

    agent_.publishLocalEndpoint(buildEndpointMetadata());
    (void)agent_.waitUntilGroupReady(meta_->group_id,
                                     std::chrono::seconds(300));

    // Initialize all peer segment IDs from the LinkManager. Subsequent updates
    // (endpoint changes, disconnects) are handled by NotifyLinkRefreshed.
    for (int local = 0; local < max_group_size_; ++local) {
        refreshSegmentID(local);
    }
}

MooncakeCommunicator::~MooncakeCommunicator() {
    try {
        shutdown();
    } catch (const std::exception& error) {
        LOG(ERROR) << "Mooncake communicator shutdown failed: " << error.what();
    }
}

int MooncakeCommunicator::getSize() const {
    if (!meta_ || meta_->extensionMode.load(std::memory_order_acquire) !=
                      CollectiveExtensionState::Normal) {
        return size_;
    }
    return meta_->activeSize.load(std::memory_order_acquire);
}

void MooncakeCommunicator::prepareOp(OpType op) const {
    const auto mode = meta_->extensionMode.load(std::memory_order_acquire);
    if (isValidGroup()) {
        PG_CHECK(meta_->rankStates[meta_->globalRank] != RankState::Offline,
                 "rank ", meta_->globalRank,
                 " is offline and cannot perform operations");
    }
    // P2P operations don't require the rank to be active in the group.
    const bool is_p2p = op == OpType::Send || op == OpType::Recv;
    PG_CHECK(isValidGroup() || !is_p2p,
             "P2P is unavailable for an invalid Mooncake group");
    if (!is_p2p) {
        PG_CHECK(mode != CollectiveExtensionState::Quiescing,
                 "rank is quiescing and cannot issue collectives");
        PG_CHECK(mode == CollectiveExtensionState::Isolated ||
                     meta_->activeRanks[rank_],
                 "rank is not active in this group");
    }
}

std::shared_ptr<MooncakeCompletion> MooncakeCommunicator::send(
    const void* buffer, size_t bytes, int peer, cudaStream_t stream,
    int32_t* failed_ranks_hint) {
    PG_CHECK(buffer || bytes == 0, "send buffer is null");
    PG_CHECK(peer >= 0 && peer < max_group_size_,
             "P2P send peer is out of range");
    prepareOp(OpType::Send);
    PG_CHECK(failed_ranks_hint, "failed-ranks hint is null");
    std::fill_n(failed_ranks_hint, max_group_size_, int32_t{0});
    auto completion = std::make_shared<std::promise<void>>();
    auto future = completion->get_future().share();
    auto result = std::make_shared<MooncakeCompletion>(std::move(future));
    p2p_proxy_->enqueueSend(P2PProxy::SendOp{
        .buffer_ = buffer,
        .size_ = bytes,
        .peer_rank_ = peer,
        .cuda_stream_ = is_cpu_ ? nullptr : stream,
        .completion_ = completion,
        .failed_ranks_hint_ = failed_ranks_hint,
    });
    return result;
}

std::shared_ptr<MooncakeCompletion> MooncakeCommunicator::recv(
    void* buffer, size_t bytes, int peer, cudaStream_t stream,
    int32_t* failed_ranks_hint) {
    PG_CHECK(buffer || bytes == 0, "recv buffer is null");
    PG_CHECK(peer >= 0 && peer < max_group_size_,
             "P2P recv peer is out of range");
    prepareOp(OpType::Recv);
    PG_CHECK(failed_ranks_hint, "failed-ranks hint is null");
    std::fill_n(failed_ranks_hint, max_group_size_, int32_t{0});
    auto completion = std::make_shared<std::promise<void>>();
    auto future = completion->get_future().share();
    auto result = std::make_shared<MooncakeCompletion>(std::move(future));
    p2p_proxy_->enqueueRecv(P2PProxy::RecvOp{
        .buffer_ = buffer,
        .size_ = bytes,
        .peer_rank_ = peer,
        .cuda_stream_ = is_cpu_ ? nullptr : stream,
        .completion_ = completion,
        .failed_ranks_hint_ = failed_ranks_hint,
    });
    return result;
}

std::shared_ptr<MooncakeCompletion> MooncakeCommunicator::broadcastCpu(
    const void* send_buffer, void* recv_buffer, size_t bytes, int root,
    int32_t* failed_ranks_hint) {
    PG_CHECK(is_cpu_, "broadcastCpu requires a CPU communicator");
    PG_CHECK(root >= 0 && root < max_group_size_,
             "broadcast root is out of range");
    prepareOp(OpType::Broadcast);
    const bool is_root = root == rank_;
    return worker_->putTaskCpu(
        OpType::Broadcast, bytes, root, meta_, failed_ranks_hint,
        [=](void* dst, size_t pos, size_t size) {
            if (is_root) {
                std::memcpy(dst, static_cast<const char*>(send_buffer) + pos,
                            size);
            }
        },
        [=](void* src, size_t pos, size_t size) {
            std::memcpy(static_cast<char*>(recv_buffer) + pos, src, size);
        });
}

void MooncakeCommunicator::broadcastGpu(const void* send_buffer,
                                        void* recv_buffer, size_t bytes,
                                        int root, cudaStream_t stream,
                                        int32_t* failed_ranks_hint) {
    PG_CHECK(!is_cpu_, "broadcastGpu requires a GPU communicator");
    PG_CHECK(root >= 0 && root < max_group_size_,
             "broadcast root is out of range");
    prepareOp(OpType::Broadcast);
    const bool is_root = root == rank_;
    worker_->putTaskCuda(
        OpType::Broadcast, bytes, root, meta_, stream, failed_ranks_hint,
        [=](void* dst, size_t pos, size_t size, cudaStream_t enqueue_stream) {
            if (is_root) {
                copyToDevice(dst, static_cast<const char*>(send_buffer) + pos,
                             size, enqueue_stream);
            }
        },
        [=](void* src, size_t pos, size_t size, cudaStream_t enqueue_stream) {
            copyToDevice(static_cast<char*>(recv_buffer) + pos, src, size,
                         enqueue_stream);
        });
}

std::shared_ptr<MooncakeCompletion> MooncakeCommunicator::allReduceCpu(
    const void* send_buffer, void* recv_buffer, size_t bytes, DataType datatype,
    ReduceOp op, int32_t* failed_ranks_hint) {
    PG_CHECK(is_cpu_, "allReduceCpu requires a CPU communicator");
    prepareOp(OpType::AllReduce);
    return worker_->putTaskCpu(
        OpType::AllReduce, bytes, 0, meta_, failed_ranks_hint,
        [=](void* dst, size_t pos, size_t size) {
            std::memcpy(dst, static_cast<const char*>(send_buffer) + pos, size);
        },
        [=, this](void* src, size_t pos, size_t size) {
            std::memset(static_cast<char*>(recv_buffer) + pos, 0, size);
            launchReduceCpu(recv_buffer, datatype, pos, size, src,
                            max_group_size_, op, meta_->activeRanks);
        });
}

void MooncakeCommunicator::allReduceGpu(const void* send_buffer,
                                        void* recv_buffer, size_t bytes,
                                        DataType datatype, ReduceOp op,
                                        cudaStream_t stream,
                                        int32_t* failed_ranks_hint) {
    PG_CHECK(!is_cpu_, "allReduceGpu requires a GPU communicator");
    prepareOp(OpType::AllReduce);
    worker_->putTaskCuda(
        OpType::AllReduce, bytes, 0, meta_, stream, failed_ranks_hint,
        [=](void* dst, size_t pos, size_t size, cudaStream_t enqueue_stream) {
            copyToDevice(dst, static_cast<const char*>(send_buffer) + pos, size,
                         enqueue_stream);
        },
        [=, this](void* src, size_t pos, size_t size,
                  cudaStream_t enqueue_stream) {
            checkCuda(cudaMemsetAsync(static_cast<char*>(recv_buffer) + pos, 0,
                                      size, enqueue_stream),
                      "cudaMemsetAsync");
            launchReduceKernel(recv_buffer, datatype, pos, size, src,
                               max_group_size_, op, meta_->activeRanksDevice,
                               enqueue_stream);
        });
}

std::shared_ptr<MooncakeCompletion> MooncakeCommunicator::allGatherCpu(
    const void* send_buffer, void* recv_buffer, size_t send_bytes,
    int32_t* failed_ranks_hint) {
    PG_CHECK(is_cpu_, "allGatherCpu requires a CPU communicator");
    prepareOp(OpType::AllGather);
    return worker_->putTaskCpu(
        OpType::AllGather, send_bytes, 0, meta_, failed_ranks_hint,
        [=](void* dst, size_t pos, size_t size) {
            std::memcpy(dst, static_cast<const char*>(send_buffer) + pos, size);
        },
        [=, this](void* src, size_t pos, size_t size) {
            for (int peer = 0; peer < max_group_size_; ++peer) {
                if (!meta_->activeRanks[peer]) continue;
                std::memcpy(
                    static_cast<char*>(recv_buffer) + peer * send_bytes + pos,
                    static_cast<char*>(src) + peer * size, size);
            }
        });
}

void MooncakeCommunicator::allGatherGpu(const void* send_buffer,
                                        void* recv_buffer, size_t send_bytes,
                                        cudaStream_t stream,
                                        int32_t* failed_ranks_hint) {
    PG_CHECK(!is_cpu_, "allGatherGpu requires a GPU communicator");
    prepareOp(OpType::AllGather);
    worker_->putTaskCuda(
        OpType::AllGather, send_bytes, 0, meta_, stream, failed_ranks_hint,
        [=](void* dst, size_t pos, size_t size, cudaStream_t enqueue_stream) {
            copyToDevice(dst, static_cast<const char*>(send_buffer) + pos, size,
                         enqueue_stream);
        },
        [=, this](void* src, size_t pos, size_t size,
                  cudaStream_t enqueue_stream) {
            for (int peer = 0; peer < max_group_size_; ++peer) {
                if (!meta_->activeRanks[peer]) continue;
                copyToDevice(
                    static_cast<char*>(recv_buffer) + peer * send_bytes + pos,
                    static_cast<char*>(src) + peer * size, size,
                    enqueue_stream);
            }
        });
}

std::shared_ptr<MooncakeCompletion> MooncakeCommunicator::reduceScatterCpu(
    const void* send_buffer, void* recv_buffer, size_t recv_bytes,
    DataType datatype, ReduceOp op, int32_t* failed_ranks_hint) {
    PG_CHECK(is_cpu_, "reduceScatterCpu requires a CPU communicator");
    prepareOp(OpType::ReduceScatter);
    return worker_->putTaskCpu(
        OpType::ReduceScatter, recv_bytes, 0, meta_, failed_ranks_hint,
        [=, this](void* dst, size_t pos, size_t size) {
            for (int peer = 0; peer < max_group_size_; ++peer) {
                if (!meta_->activeRanks[peer]) continue;
                std::memcpy(static_cast<char*>(dst) + peer * size,
                            static_cast<const char*>(send_buffer) +
                                peer * recv_bytes + pos,
                            size);
            }
        },
        [=, this](void* src, size_t pos, size_t size) {
            std::memset(static_cast<char*>(recv_buffer) + pos, 0, size);
            launchReduceCpu(recv_buffer, datatype, pos, size, src,
                            max_group_size_, op, meta_->activeRanks);
        });
}

void MooncakeCommunicator::reduceScatterGpu(const void* send_buffer,
                                            void* recv_buffer,
                                            size_t recv_bytes,
                                            DataType datatype, ReduceOp op,
                                            cudaStream_t stream,
                                            int32_t* failed_ranks_hint) {
    PG_CHECK(!is_cpu_, "reduceScatterGpu requires a GPU communicator");
    prepareOp(OpType::ReduceScatter);
    worker_->putTaskCuda(
        OpType::ReduceScatter, recv_bytes, 0, meta_, stream, failed_ranks_hint,
        [=, this](void* dst, size_t pos, size_t size,
                  cudaStream_t enqueue_stream) {
            for (int peer = 0; peer < max_group_size_; ++peer) {
                if (!meta_->activeRanks[peer]) continue;
                copyToDevice(static_cast<char*>(dst) + peer * size,
                             static_cast<const char*>(send_buffer) +
                                 peer * recv_bytes + pos,
                             size, enqueue_stream);
            }
        },
        [=, this](void* src, size_t pos, size_t size,
                  cudaStream_t enqueue_stream) {
            checkCuda(cudaMemsetAsync(static_cast<char*>(recv_buffer) + pos, 0,
                                      size, enqueue_stream),
                      "cudaMemsetAsync");
            launchReduceKernel(recv_buffer, datatype, pos, size, src,
                               max_group_size_, op, meta_->activeRanksDevice,
                               enqueue_stream);
        });
}

std::shared_ptr<MooncakeCompletion> MooncakeCommunicator::allToAllCpu(
    const void* send_buffer, void* recv_buffer, size_t peer_bytes,
    int32_t* failed_ranks_hint) {
    PG_CHECK(is_cpu_, "allToAllCpu requires a CPU communicator");
    prepareOp(OpType::AllToAll);
    return worker_->putTaskCpu(
        OpType::AllToAll, peer_bytes, 0, meta_, failed_ranks_hint,
        [=, this](void* dst, size_t pos, size_t size) {
            for (int peer = 0; peer < max_group_size_; ++peer) {
                std::memcpy(static_cast<char*>(dst) + peer * size,
                            static_cast<const char*>(send_buffer) +
                                peer * peer_bytes + pos,
                            size);
            }
        },
        [=, this](void* src, size_t pos, size_t size) {
            for (int peer = 0; peer < max_group_size_; ++peer) {
                std::memcpy(
                    static_cast<char*>(recv_buffer) + peer * peer_bytes + pos,
                    static_cast<char*>(src) + peer * size, size);
            }
        });
}

void MooncakeCommunicator::allToAllGpu(const void* send_buffer,
                                       void* recv_buffer, size_t peer_bytes,
                                       cudaStream_t stream,
                                       int32_t* failed_ranks_hint) {
    PG_CHECK(!is_cpu_, "allToAllGpu requires a GPU communicator");
    prepareOp(OpType::AllToAll);
    worker_->putTaskCuda(
        OpType::AllToAll, peer_bytes, 0, meta_, stream, failed_ranks_hint,
        [=, this](void* dst, size_t pos, size_t size,
                  cudaStream_t enqueue_stream) {
            for (int peer = 0; peer < max_group_size_; ++peer) {
                copyToDevice(static_cast<char*>(dst) + peer * size,
                             static_cast<const char*>(send_buffer) +
                                 peer * peer_bytes + pos,
                             size, enqueue_stream);
            }
        },
        [=, this](void* src, size_t pos, size_t size,
                  cudaStream_t enqueue_stream) {
            for (int peer = 0; peer < max_group_size_; ++peer) {
                copyToDevice(
                    static_cast<char*>(recv_buffer) + peer * peer_bytes + pos,
                    static_cast<char*>(src) + peer * size, size,
                    enqueue_stream);
            }
        });
}

std::shared_ptr<MooncakeCompletion> MooncakeCommunicator::barrierCpu(
    int32_t* failed_ranks_hint) {
    PG_CHECK(is_cpu_, "barrierCpu requires a CPU communicator");
    prepareOp(OpType::Barrier);
    return worker_->putTaskCpu(
        OpType::Barrier, kBarrierDummySize, 0, meta_, failed_ranks_hint,
        [](void*, size_t, size_t) {}, [](void*, size_t, size_t) {});
}

void MooncakeCommunicator::barrierGpu(cudaStream_t stream,
                                      int32_t* failed_ranks_hint) {
    PG_CHECK(!is_cpu_, "barrierGpu requires a GPU communicator");
    prepareOp(OpType::Barrier);
    worker_->putTaskCuda(
        OpType::Barrier, kBarrierDummySize, 0, meta_, stream, failed_ranks_hint,
        [](void*, size_t, size_t, cudaStream_t) {},
        [](void*, size_t, size_t, cudaStream_t) {});
}

std::shared_ptr<MooncakeCompletion> MooncakeCommunicator::reduceCpu(
    const void* send_buffer, void* recv_buffer, size_t bytes, DataType datatype,
    ReduceOp op, int root, int32_t* failed_ranks_hint) {
    PG_CHECK(is_cpu_, "reduceCpu requires a CPU communicator");
    PG_CHECK(root >= 0 && root < max_group_size_,
             "reduce root is out of range");
    prepareOp(OpType::Reduce);
    const bool is_root = root == rank_;
    return worker_->putTaskCpu(
        OpType::Reduce, bytes, root, meta_, failed_ranks_hint,
        [=](void* dst, size_t pos, size_t size) {
            std::memcpy(dst, static_cast<const char*>(send_buffer) + pos, size);
        },
        [=, this](void* src, size_t pos, size_t size) {
            if (!is_root) return;
            std::memset(static_cast<char*>(recv_buffer) + pos, 0, size);
            launchReduceCpu(recv_buffer, datatype, pos, size, src,
                            max_group_size_, op, meta_->activeRanks);
        });
}

void MooncakeCommunicator::reduceGpu(const void* send_buffer, void* recv_buffer,
                                     size_t bytes, DataType datatype,
                                     ReduceOp op, int root, cudaStream_t stream,
                                     int32_t* failed_ranks_hint) {
    PG_CHECK(!is_cpu_, "reduceGpu requires a GPU communicator");
    PG_CHECK(root >= 0 && root < max_group_size_,
             "reduce root is out of range");
    prepareOp(OpType::Reduce);
    const bool is_root = root == rank_;
    worker_->putTaskCuda(
        OpType::Reduce, bytes, root, meta_, stream, failed_ranks_hint,
        [=](void* dst, size_t pos, size_t size, cudaStream_t enqueue_stream) {
            copyToDevice(dst, static_cast<const char*>(send_buffer) + pos, size,
                         enqueue_stream);
        },
        [=, this](void* src, size_t pos, size_t size,
                  cudaStream_t enqueue_stream) {
            if (!is_root) return;
            checkCuda(cudaMemsetAsync(static_cast<char*>(recv_buffer) + pos, 0,
                                      size, enqueue_stream),
                      "cudaMemsetAsync");
            launchReduceKernel(recv_buffer, datatype, pos, size, src,
                               max_group_size_, op, meta_->activeRanksDevice,
                               enqueue_stream);
        });
}

std::shared_ptr<MooncakeCompletion> MooncakeCommunicator::gatherCpu(
    const void* send_buffer, void* recv_buffer, size_t send_bytes, int root,
    int32_t* failed_ranks_hint) {
    PG_CHECK(is_cpu_, "gatherCpu requires a CPU communicator");
    PG_CHECK(root >= 0 && root < max_group_size_,
             "gather root is out of range");
    prepareOp(OpType::Gather);
    const bool is_root = root == rank_;
    return worker_->putTaskCpu(
        OpType::Gather, send_bytes, root, meta_, failed_ranks_hint,
        [=](void* dst, size_t pos, size_t size) {
            std::memcpy(dst, static_cast<const char*>(send_buffer) + pos, size);
        },
        [=, this](void* src, size_t pos, size_t size) {
            if (!is_root) return;
            for (int peer = 0; peer < max_group_size_; ++peer) {
                std::memcpy(
                    static_cast<char*>(recv_buffer) + peer * send_bytes + pos,
                    static_cast<char*>(src) + peer * size, size);
            }
        });
}

void MooncakeCommunicator::gatherGpu(const void* send_buffer, void* recv_buffer,
                                     size_t send_bytes, int root,
                                     cudaStream_t stream,
                                     int32_t* failed_ranks_hint) {
    PG_CHECK(!is_cpu_, "gatherGpu requires a GPU communicator");
    PG_CHECK(root >= 0 && root < max_group_size_,
             "gather root is out of range");
    prepareOp(OpType::Gather);
    const bool is_root = root == rank_;
    worker_->putTaskCuda(
        OpType::Gather, send_bytes, root, meta_, stream, failed_ranks_hint,
        [=](void* dst, size_t pos, size_t size, cudaStream_t enqueue_stream) {
            copyToDevice(dst, static_cast<const char*>(send_buffer) + pos, size,
                         enqueue_stream);
        },
        [=, this](void* src, size_t pos, size_t size,
                  cudaStream_t enqueue_stream) {
            if (!is_root) return;
            for (int peer = 0; peer < max_group_size_; ++peer) {
                copyToDevice(
                    static_cast<char*>(recv_buffer) + peer * send_bytes + pos,
                    static_cast<char*>(src) + peer * size, size,
                    enqueue_stream);
            }
        });
}

std::shared_ptr<MooncakeCompletion> MooncakeCommunicator::scatterCpu(
    const void* send_buffer, void* recv_buffer, size_t recv_bytes, int root,
    int32_t* failed_ranks_hint) {
    PG_CHECK(is_cpu_, "scatterCpu requires a CPU communicator");
    PG_CHECK(root >= 0 && root < max_group_size_,
             "scatter root is out of range");
    prepareOp(OpType::Scatter);
    const bool is_root = root == rank_;
    return worker_->putTaskCpu(
        OpType::Scatter, recv_bytes, root, meta_, failed_ranks_hint,
        [=, this](void* dst, size_t pos, size_t size) {
            if (!is_root) return;
            for (int peer = 0; peer < max_group_size_; ++peer) {
                std::memcpy(static_cast<char*>(dst) + peer * size,
                            static_cast<const char*>(send_buffer) +
                                peer * recv_bytes + pos,
                            size);
            }
        },
        [=](void* src, size_t pos, size_t size) {
            std::memcpy(static_cast<char*>(recv_buffer) + pos, src, size);
        });
}

void MooncakeCommunicator::scatterGpu(const void* send_buffer,
                                      void* recv_buffer, size_t recv_bytes,
                                      int root, cudaStream_t stream,
                                      int32_t* failed_ranks_hint) {
    PG_CHECK(!is_cpu_, "scatterGpu requires a GPU communicator");
    PG_CHECK(root >= 0 && root < max_group_size_,
             "scatter root is out of range");
    prepareOp(OpType::Scatter);
    const bool is_root = root == rank_;
    worker_->putTaskCuda(
        OpType::Scatter, recv_bytes, root, meta_, stream, failed_ranks_hint,
        [=, this](void* dst, size_t pos, size_t size,
                  cudaStream_t enqueue_stream) {
            if (!is_root) return;
            for (int peer = 0; peer < max_group_size_; ++peer) {
                copyToDevice(static_cast<char*>(dst) + peer * size,
                             static_cast<const char*>(send_buffer) +
                                 peer * recv_bytes + pos,
                             size, enqueue_stream);
            }
        },
        [=](void* src, size_t pos, size_t size, cudaStream_t enqueue_stream) {
            copyToDevice(static_cast<char*>(recv_buffer) + pos, src, size,
                         enqueue_stream);
        });
}

void MooncakeCommunicator::shutdown() {
    if (is_shutdown_) return;
    is_shutdown_ = true;
    // Remove this communicator from AgentHost's callback lookup before teardown
    // so a concurrent ViewUpdate cannot call into it. Keep the group registered
    // locally and at the Coordinator while worker tasks are draining because
    // their failure path may still call syncAfterFailure().
    if (isValidGroup()) agent_.detachCommunicator(meta_->group_id);

    // If we encounter any hung operations, don't release resources to avoid a
    // potential crash. Instead, allow those resources to leak and rely on the
    // OS to reclaim them later.
    bool has_hung_operation = false;

    // Phase 1: Drain P2P tasks.
    if (p2p_device_worker_ && p2p_proxy_) {
        p2p_device_worker_->removeProxy(p2p_proxy_);
        has_hung_operation |= !p2p_proxy_->drainTasks();
    }
    // Phase 2: Drain collective tasks for this communicator.
    if (worker_ && meta_) {
        has_hung_operation |= !worker_->drainTasks(meta_.get());
    }
    // Phase 3: Device synchronization.
    if (!is_cpu_ && !has_hung_operation) cudaDeviceSynchronize();

    // Phase 4: Release resources.
    if (has_hung_operation && p2p_proxy_) p2p_proxy_->abandonResources();

    if (!has_hung_operation && meta_) {
        for (size_t index = 0; index < 2; ++index) {
            context_->engine->unregisterLocalMemory(
                cpu_sync_send_region_[index]);
            context_->engine->unregisterLocalMemory(
                cpu_sync_recv_region_[index]);
            context_->engine->unregisterLocalMemory(send_buffer_[index]);
            context_->engine->unregisterLocalMemory(recv_buffer_[index]);
            delete[] cpu_sync_send_region_[index];
            delete[] cpu_sync_recv_region_[index];
            if (is_cpu_) {
                std::free(send_buffer_[index]);
                std::free(recv_buffer_[index]);
            } else {
                cudaFree(send_buffer_[index]);
                cudaFree(recv_buffer_[index]);
            }
        }
        delete[] meta_->maybeActivatable;
        if (is_cpu_) {
            delete[] meta_->activeRanks;
        } else {
            cudaFreeHost(meta_->activeRanks);
        }
        meta_->activeRanks = nullptr;
        meta_->activeRanksDevice = nullptr;
        meta_->maybeActivatable = nullptr;
    }
    // Prevent zombie P2PProxy workers from dereferencing this communicator
    // after destruction. Must happen after drainTasks so in-flight failures can
    // still be reported during shutdown.
    if (meta_) meta_->communicator = nullptr;

    // The data-plane teardown has finished. Remove the group from the local
    // Agent and notify the Coordinator that this rank has left it.
    if (isValidGroup()) agent_.unregisterGroup(meta_->group_id);
}

std::string MooncakeCommunicator::getPreferredHca(
    const std::string& location) const {
    static std::once_flag topology_once;
    static std::shared_ptr<Topology> topology;
    static TopologyMatrix matrix;
    std::call_once(topology_once, [this] {
        // FIXME: getLocalTopology is deprecated in TENT
        topology = context_->engine->getLocalTopology();
        if (topology) matrix = topology->getMatrix();
        if (!topology || matrix.empty()) {
            topology = std::make_shared<Topology>();
            topology->discover();
            matrix = topology->getMatrix();
        }
    });
    const auto entry = matrix.find(location);
    if (entry == matrix.end() || entry->second.preferred_hca.empty()) return {};
    return entry->second.preferred_hca.front();
}

std::vector<int32_t> MooncakeCommunicator::getActiveRanks() const {
    std::vector<int32_t> result(max_group_size_, 0);
    if (!meta_ || !meta_->activeRanks) return result;
    for (int index = 0; index < max_group_size_; ++index) {
        result[index] = meta_->activeRanks[index] ? 1 : 0;
    }
    return result;
}

void MooncakeCommunicator::syncActiveRanksMirror() const {
    if (!active_ranks_mirror_) return;
    // The mirror is InGroupRank-indexed, in the same order as the caller-owned
    // storage.
    auto active_ranks = getActiveRanks();
    const size_t bytes = max_group_size_ * sizeof(int32_t);
    if (active_ranks_mirror_is_device_) {
        checkCuda(cudaMemcpy(active_ranks_mirror_, active_ranks.data(), bytes,
                             cudaMemcpyHostToDevice),
                  "copy active-ranks mirror");
    } else {
        std::memcpy(active_ranks_mirror_, active_ranks.data(), bytes);
    }
}

int MooncakeCommunicator::getNumSyncedRanks() const {
    if (!meta_ || !meta_->maybeActivatable) return 0;
    int count = 0;
    for (int index = 0; index < max_group_size_; ++index) {
        if (meta_->maybeActivatable[index]) ++count;
    }
    return count;
}

void MooncakeCommunicator::requireValidGroup(const char* operation) const {
    PG_CHECK(isValidGroup(), operation,
             " is unavailable because this communicator is running "
             "local-only collectives");
}

std::vector<bool> MooncakeCommunicator::getPeerState(
    const std::vector<int>& ranks) const {
    requireValidGroup("getPeerState");
    std::vector<bool> result;
    result.reserve(ranks.size());
    for (const int rank : ranks) {
        PG_CHECK(rank >= 0 && rank < max_group_size_,
                 "peer rank is out of range");
        result.push_back(meta_->maybeActivatable[rank]);
    }
    return result;
}

ProposeViewUpdateResponse MooncakeCommunicator::activateRanks(
    const std::vector<int>& ranks) {
    requireValidGroup("activateRanks");
    std::vector<InGroupRank> local_ranks(ranks.begin(), ranks.end());
    return agent_.proposeActivate(meta_->group_id, local_ranks);
}

ProposeViewUpdateResponse MooncakeCommunicator::deactivateRanks(
    const std::vector<int>& ranks) {
    requireValidGroup("deactivateRanks");
    std::vector<InGroupRank> local_ranks(ranks.begin(), ranks.end());
    return agent_.proposeDeactivate(meta_->group_id, local_ranks);
}

void MooncakeCommunicator::joinGroup() {
    requireValidGroup("joinGroup");
    const auto mode = meta_->extensionMode.load(std::memory_order_acquire);
    PG_CHECK(mode == CollectiveExtensionState::Isolated,
             "joinGroup may only be called on an isolated rank");
    // Stop admitting isolated collectives before advertising readiness.
    meta_->extensionMode.store(CollectiveExtensionState::Quiescing,
                               std::memory_order_release);
    PG_CHECK(worker_->drainTasks(meta_.get()),
             "timed out draining join preparation collectives");
    agent_.confirmReadyForActivation(meta_->group_id);
    // Block until the Coordinator activates this rank in the group.
    agent_.waitUntilRankActive(meta_->group_id, meta_->globalRank,
                               std::chrono::seconds(300));
    PG_CHECK(meta_->extensionMode.load(std::memory_order_acquire) ==
                     CollectiveExtensionState::Normal &&
                 meta_->activeRanks[rank_],
             "rank did not become active after joinGroup");
}

uint64_t MooncakeCommunicator::getCurrentEpoch() const {
    return meta_ ? meta_->epoch.load(std::memory_order_acquire) : 0;
}

SyncAfterFailureResponse MooncakeCommunicator::syncAfterFailure() {
    requireValidGroup("syncAfterFailure");
    return agent_.syncAfterFailure(meta_->group_id);
}

void MooncakeCommunicator::applyViewUpdate(
    const GroupView& view, const std::vector<RankState>& rank_states,
    const std::vector<uint64_t>& rank_epochs,
    const std::vector<bool>& activatable) {
    if (!meta_) return;
    // Ignore stale views that arrive out of order.
    const auto current_epoch = meta_->epoch.load(std::memory_order_acquire);
    if (view.epoch < current_epoch) return;

    // An authoritative view in which self is Active is the common commit point
    // for enabling normal collective execution:
    //
    //   founding ranks: Isolated  -> Normal
    //   joining ranks:  Quiescing -> Normal
    //
    // A non-Active view deliberately does not determine the local mode. A new
    // joiner must remain Isolated until joinGroup is called; a joiner awaiting
    // activation must remain Quiescing; and an auto-deactivated communicator
    // must remain Normal so its inactive self bit makes the next collective
    // fail fast.
    const auto mode = meta_->extensionMode.load(std::memory_order_acquire);
    auto next_mode = mode;
    if (view.members[meta_->globalRank].isActive()) {
        next_mode = CollectiveExtensionState::Normal;
    }
    PG_CHECK(static_cast<int>(view.rank_order.size()) <= max_group_size_,
             "group view exceeds communicator capacity");

    // Preserve stable in-group rank slots: activeSize is the upper bound of the
    // active rank space, not the number of set bits. For example, an active
    // mask of [true, false, true] has activeSize == 3.
    int active_size = 0;
    for (size_t local = 0; local < view.rank_order.size(); ++local) {
        if (view.members[view.rank_order[local]].isActive()) {
            active_size = static_cast<int>(local) + 1;
        }
    }

    std::vector<bool> previous(max_group_size_);

    // The execution mode determines the effective active ranks consumed by
    // kernels. Isolated and Quiescing use a local-only mask; Normal follows the
    // Coordinator's committed membership view.
    for (int local = 0; local < max_group_size_; ++local) {
        previous[local] = meta_->activeRanks[local];
        meta_->activeRanks[local] =
            next_mode == CollectiveExtensionState::Normal
                ? (local < static_cast<int>(view.rank_order.size()) &&
                   view.members[view.rank_order[local]].isActive())
                : local == rank_;
    }

    // Only a change in execution mode or effective participants starts a new
    // collective taskCount. Endpoint and AwaitingActivation updates keep the
    // current taskCount even though they advance the view epoch.
    bool reset_task_count = next_mode != mode;
    for (int local = 0; local < max_group_size_; ++local) {
        reset_task_count |= previous[local] != meta_->activeRanks[local];
    }
    if (reset_task_count) meta_->taskCount = 0;

    // Rank order and endpoint metadata.
    for (size_t local = 0; local < view.rank_order.size(); ++local) {
        const auto global = view.rank_order[local];
        // rank order
        meta_->rank_order[local] = global;
        if (view.members[global].endpoint) {
            meta_->segmentInfos[local] = *view.members[global].endpoint;
        }
    }
    // Rank states.
    for (size_t index = 0; index < rank_states.size() && index < kMaxNumRanks;
         ++index) {
        meta_->rankStates[index] = rank_states[index];
    }
    for (size_t index = 0; index < rank_epochs.size() && index < kMaxNumRanks;
         ++index) {
        meta_->rankEpochs[index] = rank_epochs[index];
    }
    // Best-effort Activatable.
    for (size_t index = 0; index < activatable.size() &&
                           index < static_cast<size_t>(max_group_size_);
         ++index) {
        meta_->maybeActivatable[index] = activatable[index];
    }

    // Keep the caller-visible active-ranks mirror in sync with the view.
    // FIXME: potential deadlock?
    syncActiveRanksMirror();

    // Publish the rank-space extent after the corresponding data-plane state.
    // getSize() reads this from the application thread.
    meta_->activeSize.store(active_size, std::memory_order_release);

    // Publish epoch AFTER all data-plane state (activeRanks, segmentInfos,
    // etc.) is updated. This ensures that a thread observing the new epoch via
    // getCurrentEpoch() (acquire) sees the complete membership state.
    if (view.epoch != current_epoch) {
        meta_->epoch.store(view.epoch, std::memory_order_release);
    }
    if (next_mode != mode) {
        meta_->extensionMode.store(next_mode, std::memory_order_release);
    }
}

void MooncakeCommunicator::onPeerLinkReset(InGroupRank peer) {
    if (is_shutdown_) return;
    if (p2p_proxy_) p2p_proxy_->resetPeerState(peer);
    if (peer >= 0 && peer < max_group_size_) {
        meta_->segmentIDs[peer] = static_cast<TransferMetadata::SegmentID>(-1);
    }
}

void MooncakeCommunicator::refreshSegmentID(InGroupRank local) {
    if (local < 0 || local >= max_group_size_) return;
    const auto handle =
        context_->link_manager.resolvePeer(meta_->rank_order[local]);
    meta_->segmentIDs[local] =
        handle ? *handle : static_cast<TransferMetadata::SegmentID>(-1);
}

GroupEndpointPublication MooncakeCommunicator::buildEndpointMetadata() const {
    return GroupEndpointPublication{
        .group_id = meta_->group_id,
        .endpoint_info = meta_->segmentInfos[meta_->rank]};
}

const GroupEndpointInfo& MooncakeCommunicator::getLocalEndpointInfo() const {
    return meta_->segmentInfos[meta_->rank];
}

}  // namespace mooncake
