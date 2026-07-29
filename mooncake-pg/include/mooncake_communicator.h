#ifndef MOONCAKE_PG_COMMUNICATOR_H
#define MOONCAKE_PG_COMMUNICATOR_H

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <transfer_engine.h>

#include "control_plane/agent_host.h"
#include "control_plane/coordinator_host.h"
#include "control_plane/link_manager.h"
#include "mooncake_pg.h"
#include "mooncake_worker.cuh"
#include "p2p_proxy.h"
#include "types.h"

namespace mooncake {

static constexpr size_t kDefaultCollectiveTimeoutUs =
    MOONCAKE_PG_DEFAULT_COLLECTIVE_TIMEOUT_US;
static constexpr int64_t kDefaultP2PTimeoutUs =
    MOONCAKE_PG_DEFAULT_P2P_TIMEOUT_US;

// Must be greater than collective_timeout_us so that timeout-based
// failure reporters can contribute before the reconciliation window
// expires. (Some ranks report failures based on timeout, while others
// report based on failure status.)
static constexpr int64_t kDefaultFaultReconciliationWindowUs =
    MOONCAKE_PG_DEFAULT_FAULT_RECONCILIATION_WINDOW_US;

struct MooncakePGContext {
    std::string host_ip = "127.0.0.1";
    size_t collective_timeout_us = kDefaultCollectiveTimeoutUs;
    int64_t p2p_timeout_us = kDefaultP2PTimeoutUs;
    int64_t fault_reconciliation_window_us =
        kDefaultFaultReconciliationWindowUs;

    std::unique_ptr<TransferEngine> owned_engine =
        std::make_unique<TransferEngine>(true);
    TransferEngine* engine = owned_engine.get();
    bool engine_initialized = false;
    int global_rank = -1;
    int max_world_size = 0;

    LinkManager link_manager;
    MooncakeWorkerManager worker_manager;
    P2PDeviceWorkerManager p2p_device_worker_manager;
    // Coordinator (rank 0 only).
    // It must be started before the local AgentHost connects to it.
    std::unique_ptr<CoordinatorHost> coordinator_host;
    std::unique_ptr<AgentHost> agent_host;

    MooncakePGContext() = default;
    ~MooncakePGContext();

    // Non-copyable: engine points to either owned_engine or an external engine
    // whose lifetime is controlled by the caller.
    MooncakePGContext(const MooncakePGContext&) = delete;
    MooncakePGContext& operator=(const MooncakePGContext&) = delete;

    void initializeDataPlane(int rank, int world_size);
    std::string launchCoordinator();
    AgentHost& connectCoordinator(const std::string& coordinator_address);
    void setHostIp(std::string value);
    void setExternalEngine(TransferEngine* transfer_engine);
    void setDeviceFilter(std::vector<std::string> filters);
    void setFaultReconciliationWindow(int64_t timeout_us);

   private:
    std::mutex initialize_mutex_;
};

struct MooncakeCommunicatorConfig {
    int rank = 0;
    int size = 1;
    int max_group_size = -1;
    std::vector<GlobalRank> global_ranks;
    GroupBootstrapId group_bootstrap_id;
    bool is_cpu = false;
    int device_index = -1;
    GroupBootstrapIdResolvePolicy group_resolve_policy =
        GroupBootstrapIdResolvePolicy::CreateOrAttach;
    bool auto_deactivate_on_failure = true;
    bool auto_sync_on_failure = true;

    // Optional caller-owned mirror of the communicator's active ranks.
    int32_t* active_ranks_mirror = nullptr;
    size_t active_ranks_mirror_count = 0;
    bool active_ranks_mirror_is_device = false;
};

class MooncakeCommunicator {
   public:
    MooncakeCommunicator(std::shared_ptr<MooncakePGContext> context,
                         MooncakeCommunicatorConfig config);
    ~MooncakeCommunicator();

    MooncakeCommunicator(const MooncakeCommunicator&) = delete;
    MooncakeCommunicator& operator=(const MooncakeCommunicator&) = delete;

    int getRank() const { return rank_; }
    int getSize() const;
    int getMaxGroupSize() const { return max_group_size_; }
    bool isCpu() const { return is_cpu_; }

    std::shared_ptr<MooncakeCompletion> send(const void* buffer, size_t bytes,
                                             int peer, cudaStream_t stream,
                                             int32_t* failed_ranks_hint);
    std::shared_ptr<MooncakeCompletion> recv(void* buffer, size_t bytes,
                                             int peer, cudaStream_t stream,
                                             int32_t* failed_ranks_hint);

    std::shared_ptr<MooncakeCompletion> broadcastCpu(
        const void* send_buffer, void* recv_buffer, size_t bytes, int root,
        int32_t* failed_ranks_hint);
    void broadcastGpu(const void* send_buffer, void* recv_buffer, size_t bytes,
                      int root, cudaStream_t stream,
                      int32_t* failed_ranks_hint);
    std::shared_ptr<MooncakeCompletion> allReduceCpu(
        const void* send_buffer, void* recv_buffer, size_t bytes,
        DataType datatype, ReduceOp op, int32_t* failed_ranks_hint);
    void allReduceGpu(const void* send_buffer, void* recv_buffer, size_t bytes,
                      DataType datatype, ReduceOp op, cudaStream_t stream,
                      int32_t* failed_ranks_hint);
    std::shared_ptr<MooncakeCompletion> allGatherCpu(
        const void* send_buffer, void* recv_buffer, size_t send_bytes,
        int32_t* failed_ranks_hint);
    void allGatherGpu(const void* send_buffer, void* recv_buffer,
                      size_t send_bytes, cudaStream_t stream,
                      int32_t* failed_ranks_hint);
    std::shared_ptr<MooncakeCompletion> reduceScatterCpu(
        const void* send_buffer, void* recv_buffer, size_t recv_bytes,
        DataType datatype, ReduceOp op, int32_t* failed_ranks_hint);
    void reduceScatterGpu(const void* send_buffer, void* recv_buffer,
                          size_t recv_bytes, DataType datatype, ReduceOp op,
                          cudaStream_t stream, int32_t* failed_ranks_hint);
    std::shared_ptr<MooncakeCompletion> allToAllCpu(const void* send_buffer,
                                                    void* recv_buffer,
                                                    size_t peer_bytes,
                                                    int32_t* failed_ranks_hint);
    void allToAllGpu(const void* send_buffer, void* recv_buffer,
                     size_t peer_bytes, cudaStream_t stream,
                     int32_t* failed_ranks_hint);
    std::shared_ptr<MooncakeCompletion> barrierCpu(int32_t* failed_ranks_hint);
    void barrierGpu(cudaStream_t stream, int32_t* failed_ranks_hint);
    std::shared_ptr<MooncakeCompletion> reduceCpu(
        const void* send_buffer, void* recv_buffer, size_t bytes,
        DataType datatype, ReduceOp op, int root, int32_t* failed_ranks_hint);
    void reduceGpu(const void* send_buffer, void* recv_buffer, size_t bytes,
                   DataType datatype, ReduceOp op, int root,
                   cudaStream_t stream, int32_t* failed_ranks_hint);
    std::shared_ptr<MooncakeCompletion> gatherCpu(const void* send_buffer,
                                                  void* recv_buffer,
                                                  size_t send_bytes, int root,
                                                  int32_t* failed_ranks_hint);
    void gatherGpu(const void* send_buffer, void* recv_buffer,
                   size_t send_bytes, int root, cudaStream_t stream,
                   int32_t* failed_ranks_hint);
    std::shared_ptr<MooncakeCompletion> scatterCpu(const void* send_buffer,
                                                   void* recv_buffer,
                                                   size_t recv_bytes, int root,
                                                   int32_t* failed_ranks_hint);
    void scatterGpu(const void* send_buffer, void* recv_buffer,
                    size_t recv_bytes, int root, cudaStream_t stream,
                    int32_t* failed_ranks_hint);

    void shutdown();
    std::string getPreferredHca(const std::string& location) const;
    std::vector<int32_t> getActiveRanks() const;
    int getNumSyncedRanks() const;
    std::vector<bool> getPeerState(const std::vector<int>& ranks) const;
    ProposeViewUpdateResponse activateRanks(const std::vector<int>& ranks);
    ProposeViewUpdateResponse deactivateRanks(const std::vector<int>& ranks);
    void joinGroup();

    // Returns the current GroupView epoch.
    // Epoch starts at 0 (bootstrap) and increments on membership changes,
    // auto-deactivation, and recovery.
    uint64_t getCurrentEpoch() const;

    // Notify the Coordinator of a detected failure and block until a membership
    // decision has been made and the Agent has ACKed the resulting ViewUpdate.
    SyncAfterFailureResponse syncAfterFailure();

    // Update the data-plane view. Called by AgentHost when a ViewUpdatePush is
    // received or rank states change. rank_states and activatable are computed
    // by the state machine.
    void applyViewUpdate(const GroupView& view,
                         const std::vector<RankState>& rank_states,
                         const std::vector<uint64_t>& rank_epochs,
                         const std::vector<bool>& activatable);
    // Called by AgentHost when a TE link to a peer comes back up.
    void onPeerLinkReset(InGroupRank peer);

    // Called by NotifyLinkRefreshed effect: refresh the cached TE segment ID
    // for `local` (InGroupRank) from the LinkManager. If the link is not up,
    // segmentID is set to -1.
    void refreshSegmentID(InGroupRank local);
    GroupEndpointPublication buildEndpointMetadata() const;

    AgentInterface& getAgent() { return agent_; }
    const AgentInterface& getAgent() const { return agent_; }

   private:
    // Guard: checks that the rank is Healthy (always) and, for collectives,
    // that it is active in this group. Called at the top of every operation.
    void prepareOp(OpType op) const;

    // Reject operations if this communicator is invalid.
    void requireValidGroup(const char* operation) const;

    // A rejected registration has no Coordinator-assigned group id and is
    // restricted to local-only collectives.
    bool isValidGroup() const { return meta_ && !meta_->group_id.empty(); }

    // Sync the caller-provided active-ranks mirror on CPU/GPU from the current
    // GroupView.
    void syncActiveRanksMirror() const;
    const GroupEndpointInfo& getLocalEndpointInfo() const;

    std::shared_ptr<MooncakePGContext> context_;
    AgentInterface& agent_;
    int rank_ = 0;
    int size_ = 1;
    int max_group_size_ =
        1;  // per-group capacity (max active members for this group)
    int device_index_ = -1;
    bool is_cpu_ = false;
    bool is_shutdown_ = false;
    int32_t* active_ranks_mirror_ = nullptr;
    size_t active_ranks_mirror_count_ = 0;
    bool active_ranks_mirror_is_device_ = false;

    std::shared_ptr<MooncakeWorker> worker_;
    std::array<void*, 2> send_buffer_{};
    std::array<void*, 2> recv_buffer_{};
    std::array<int32_t*, 2> cpu_sync_send_region_{};
    std::array<int32_t*, 2> cpu_sync_recv_region_{};
    std::shared_ptr<TransferGroupMeta> meta_;

    // P2P async infrastructure. p2p_proxy_ is created by this communicator but
    // can live longer because P2PDeviceWorker retains it until all transfers
    // complete.
    std::shared_ptr<P2PProxy> p2p_proxy_;

    // Created by P2PDeviceWorkerManager and shared between communicators on the
    // same device.
    std::shared_ptr<P2PDeviceWorker> p2p_device_worker_;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_COMMUNICATOR_H
