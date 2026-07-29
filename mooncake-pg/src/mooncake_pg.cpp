#include <mooncake_pg.h>

#include <mooncake_communicator.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct mooncakePgContext {
    std::shared_ptr<mooncake::MooncakePGContext> impl;
};

struct mooncakePgComm {
    std::shared_ptr<mooncake::MooncakeCommunicator> impl;
};

struct mooncakePgCompletion {
    std::shared_ptr<mooncake::MooncakeCompletion> impl;
    // Keep the communicator alive until the operation and its callbacks have
    // finished, even if the public communicator handle is destroyed first.
    std::shared_ptr<mooncake::MooncakeCommunicator> communicator;
};

namespace {

thread_local std::string g_last_error;

class TimeoutError : public std::runtime_error {
   public:
    using std::runtime_error::runtime_error;
};

void setLastError(const std::string& error) { g_last_error = error; }

template <typename Function>
mooncakePgResult_t translateExceptions(Function&& function) {
    try {
        function();
        return mooncakePgSuccess;
    } catch (const TimeoutError& error) {
        setLastError(error.what());
        return mooncakePgTimeout;
    } catch (const std::invalid_argument& error) {
        setLastError(error.what());
        return mooncakePgInvalidArgument;
    } catch (const std::exception& error) {
        setLastError(error.what());
        return mooncakePgSystemError;
    } catch (...) {
        setLastError("unknown Mooncake PG error");
        return mooncakePgInternalError;
    }
}

void validateContext(mooncakePgContext_t context) {
    if (!context || !context->impl) {
        throw std::invalid_argument("context is null");
    }
}

void validateProcessIdentity(int global_rank, int max_world_size) {
    if (global_rank < 0 || max_world_size <= 0 ||
        global_rank >= max_world_size ||
        max_world_size > MOONCAKE_PG_MAX_RANKS) {
        throw std::invalid_argument("invalid process rank or world size");
    }
}

void validateCommConfig(const mooncakePgCommConfig_t* config) {
    if (!config) throw std::invalid_argument("communicator config is null");
    if (config->structSize < sizeof(*config) ||
        config->magic != MOONCAKE_PG_COMM_CONFIG_MAGIC ||
        config->version != MOONCAKE_PG_COMM_CONFIG_VERSION) {
        throw std::invalid_argument(
            "communicator config was not initialized with "
            "MOONCAKE_PG_COMM_CONFIG_INITIALIZER");
    }
    if (!config->groupId || config->groupId[0] == '\0') {
        throw std::invalid_argument("communicator group id is empty");
    }
    if (config->rank < 0 || config->size <= 0 || config->rank >= config->size) {
        throw std::invalid_argument("invalid communicator rank or size");
    }
    const int max_group_size =
        config->maxGroupSize == MOONCAKE_PG_CONFIG_UNDEF_INT
            ? config->size
            : config->maxGroupSize;
    if (max_group_size < config->size ||
        max_group_size > MOONCAKE_PG_MAX_RANKS) {
        throw std::invalid_argument("invalid communicator max group size");
    }
    if (config->globalRankCount != 0) {
        if (!config->globalRanks) {
            throw std::invalid_argument("global ranks is null");
        }
        if (config->globalRankCount != static_cast<size_t>(config->size)) {
            throw std::invalid_argument(
                "global rank count must equal communicator size");
        }
    }
    if (config->activeRanksMirror &&
        config->activeRanksMirrorCount < static_cast<size_t>(max_group_size)) {
        throw std::invalid_argument("active-ranks mirror is too small");
    }
    if (!config->activeRanksMirror && config->activeRanksMirrorCount != 0) {
        throw std::invalid_argument("active-ranks mirror is null");
    }
}

void copyCoordinatorAddress(
    const std::string& value,
    mooncakePgCoordinatorAddress_t* coordinator_address) {
    if (!coordinator_address) {
        throw std::invalid_argument("coordinator address output is null");
    }
    if (value.empty() ||
        value.size() >= sizeof(coordinator_address->internal)) {
        throw std::invalid_argument("invalid coordinator address");
    }
    std::memset(coordinator_address->internal, 0,
                sizeof(coordinator_address->internal));
    std::memcpy(coordinator_address->internal, value.data(), value.size());
}

std::string coordinatorAddressToString(
    const mooncakePgCoordinatorAddress_t* coordinator_address) {
    if (!coordinator_address) {
        throw std::invalid_argument("coordinator address is null");
    }
    const char* begin = coordinator_address->internal;
    const char* end = begin + sizeof(coordinator_address->internal);
    const char* terminator = std::find(begin, end, '\0');
    if (terminator == begin || terminator == end) {
        throw std::invalid_argument("invalid coordinator address");
    }
    return std::string(begin, terminator);
}

mooncake::DataType convertDataType(mooncakePgDataType_t data_type) {
    using mooncake::DataType;
    switch (data_type) {
        case mooncakePgUint8:
            return DataType::Uint8;
        case mooncakePgInt8:
            return DataType::Int8;
        case mooncakePgInt16:
            return DataType::Int16;
        case mooncakePgInt32:
            return DataType::Int32;
        case mooncakePgInt64:
            return DataType::Int64;
        case mooncakePgFloat32:
            return DataType::Float32;
        case mooncakePgFloat64:
            return DataType::Float64;
        case mooncakePgBool:
            return DataType::Bool;
        case mooncakePgBfloat16:
            return DataType::BFloat16;
        default:
            throw std::invalid_argument("unsupported Mooncake PG datatype");
    }
}

mooncake::ReduceOp convertReduceOp(mooncakePgReduceOp_t reduce_op) {
    using mooncake::ReduceOp;
    switch (reduce_op) {
        case mooncakePgSum:
            return ReduceOp::Sum;
        case mooncakePgAvg:
            throw std::invalid_argument(
                "Mooncake PG does not support AVG reduction");
        case mooncakePgProduct:
            return ReduceOp::Product;
        case mooncakePgMin:
            return ReduceOp::Min;
        case mooncakePgMax:
            return ReduceOp::Max;
        default:
            throw std::invalid_argument(
                "unsupported Mooncake PG reduction operation");
    }
}

size_t dataTypeSize(mooncakePgDataType_t data_type) {
    switch (convertDataType(data_type)) {
        case mooncake::DataType::Uint8:
        case mooncake::DataType::Int8:
        case mooncake::DataType::Bool:
            return 1;
        case mooncake::DataType::Int16:
        case mooncake::DataType::BFloat16:
            return 2;
        case mooncake::DataType::Int32:
        case mooncake::DataType::Float32:
            return 4;
        case mooncake::DataType::Int64:
        case mooncake::DataType::Float64:
            return 8;
    }
    throw std::invalid_argument("unsupported Mooncake PG datatype");
}

size_t byteCount(size_t count, mooncakePgDataType_t data_type) {
    const size_t element_size = dataTypeSize(data_type);
    if (count > std::numeric_limits<size_t>::max() / element_size) {
        throw std::invalid_argument("collective element count overflows");
    }
    return count * element_size;
}

cudaStream_t convertStream(mooncakePgStream_t stream) {
    return reinterpret_cast<cudaStream_t>(stream);
}

void validateComm(mooncakePgComm_t comm) {
    if (!comm || !comm->impl) {
        throw std::invalid_argument("communicator is null");
    }
}

void validateCompletion(mooncakePgCompletion_t completion) {
    if (!completion || !completion->impl) {
        throw std::invalid_argument("completion is null");
    }
}

void validateBuffer(const void* buffer, size_t bytes, const char* name) {
    if (!buffer && bytes != 0) {
        throw std::invalid_argument(std::string(name) + " is null");
    }
}

void validateCommDevice(mooncakePgComm_t comm, bool expect_cpu) {
    validateComm(comm);
    if (comm->impl->isCpu() != expect_cpu) {
        throw std::invalid_argument(
            expect_cpu ? "operation requires a CPU communicator"
                       : "operation requires a GPU communicator");
    }
}

void initializeFailedRanksHint(mooncakePgComm_t comm,
                               int32_t* failed_ranks_hint,
                               size_t failed_ranks_hint_count) {
    if (!failed_ranks_hint) {
        throw std::invalid_argument("failed-ranks hint is null");
    }
    const size_t required_count =
        static_cast<size_t>(comm->impl->getMaxGroupSize());
    if (failed_ranks_hint_count < required_count) {
        throw std::invalid_argument("failed-ranks hint is too small");
    }
    std::fill_n(failed_ranks_hint, required_count, int32_t{0});
}

template <typename Launch>
mooncakePgResult_t launchCpuOperation(mooncakePgComm_t comm,
                                      int32_t* failed_ranks_hint,
                                      size_t failed_ranks_hint_count,
                                      mooncakePgCompletion_t* output_completion,
                                      Launch&& launch) {
    return translateExceptions([&] {
        if (!output_completion) {
            throw std::invalid_argument("completion output is null");
        }
        *output_completion = nullptr;
        validateCommDevice(comm, true);
        initializeFailedRanksHint(comm, failed_ranks_hint,
                                  failed_ranks_hint_count);
        auto output = std::make_unique<mooncakePgCompletion>();
        output->communicator = comm->impl;
        auto completion = launch(failed_ranks_hint);
        if (!completion) {
            throw std::runtime_error("operation returned no completion");
        }

        output->impl = std::move(completion);
        *output_completion = output.release();
    });
}

template <typename Launch>
mooncakePgResult_t launchGpuOperation(mooncakePgComm_t comm,
                                      int32_t* failed_ranks_hint,
                                      size_t failed_ranks_hint_count,
                                      Launch&& launch) {
    return translateExceptions([&] {
        validateCommDevice(comm, false);
        initializeFailedRanksHint(comm, failed_ranks_hint,
                                  failed_ranks_hint_count);
        launch(failed_ranks_hint);
    });
}

template <typename Launch>
mooncakePgResult_t launchP2POperation(mooncakePgComm_t comm, bool expect_cpu,
                                      int32_t* failed_ranks_hint,
                                      size_t failed_ranks_hint_count,
                                      mooncakePgCompletion_t* output_completion,
                                      Launch&& launch) {
    return translateExceptions([&] {
        if (!output_completion) {
            throw std::invalid_argument("completion output is null");
        }
        *output_completion = nullptr;
        validateCommDevice(comm, expect_cpu);
        initializeFailedRanksHint(comm, failed_ranks_hint,
                                  failed_ranks_hint_count);
        auto output = std::make_unique<mooncakePgCompletion>();
        output->communicator = comm->impl;
        auto completion = launch(failed_ranks_hint);
        if (!completion) {
            throw std::runtime_error("operation returned no completion");
        }

        output->impl = std::move(completion);
        *output_completion = output.release();
    });
}

std::vector<int> copyRanks(const int32_t* ranks, size_t rank_count) {
    if (rank_count != 0 && !ranks) {
        throw std::invalid_argument("ranks is null");
    }
    if (rank_count == 0) return {};
    return std::vector<int>(ranks, ranks + rank_count);
}

void copyErrorText(const std::string& text, char* output, size_t capacity) {
    if (!output || capacity == 0) return;
    std::snprintf(output, capacity, "%s", text.c_str());
}

void copyProposal(const mooncake::ProposeViewUpdateResponse& source,
                  mooncakePgProposalResponse_t* destination) {
    if (!destination) throw std::invalid_argument("proposal response is null");
    std::memset(destination, 0, sizeof(*destination));
    destination->status =
        static_cast<mooncakePgProposalStatus_t>(source.status);
    destination->newEpoch = source.new_epoch;
    destination->droppedRankCount =
        std::min(source.dropped_ranks.size(),
                 static_cast<size_t>(MOONCAKE_PG_MAX_RANKS));
    std::copy_n(source.dropped_ranks.begin(), destination->droppedRankCount,
                destination->droppedRanks);
    copyErrorText(source.reject_reason, destination->rejectReason,
                  sizeof(destination->rejectReason));
}

}  // namespace

const char* mooncakePgGetErrorString(mooncakePgResult_t result) {
    switch (result) {
        case mooncakePgSuccess:
            return "success";
        case mooncakePgSystemError:
            return "system error";
        case mooncakePgInternalError:
            return "internal error";
        case mooncakePgInvalidArgument:
            return "invalid argument";
        case mooncakePgTimeout:
            return "operation timed out";
        default:
            return "unknown result";
    }
}

const char* mooncakePgGetLastError(void) { return g_last_error.c_str(); }

mooncakePgResult_t mooncakePgContextCreate(mooncakePgContext_t* context) {
    return translateExceptions([&] {
        if (!context) throw std::invalid_argument("context output is null");
        *context = nullptr;

        auto output = std::make_unique<mooncakePgContext>();
        output->impl = std::make_shared<mooncake::MooncakePGContext>();
        *context = output.release();
    });
}

mooncakePgResult_t mooncakePgContextInitialize(mooncakePgContext_t context,
                                               int global_rank,
                                               int max_world_size) {
    return translateExceptions([&] {
        validateContext(context);
        validateProcessIdentity(global_rank, max_world_size);
        context->impl->initializeDataPlane(global_rank, max_world_size);
    });
}

mooncakePgResult_t mooncakePgContextLaunchCoordinator(
    mooncakePgContext_t context,
    mooncakePgCoordinatorAddress_t* coordinator_address) {
    return translateExceptions([&] {
        validateContext(context);
        copyCoordinatorAddress(context->impl->launchCoordinator(),
                               coordinator_address);
    });
}

mooncakePgResult_t mooncakePgContextConnectCoordinator(
    mooncakePgContext_t context,
    const mooncakePgCoordinatorAddress_t* coordinator_address) {
    return translateExceptions([&] {
        validateContext(context);
        const std::string address =
            coordinatorAddressToString(coordinator_address);
        (void)context->impl->connectCoordinator(address);
    });
}

mooncakePgResult_t mooncakePgContextSetHostIp(mooncakePgContext_t context,
                                              const char* host_ip) {
    return translateExceptions([&] {
        validateContext(context);
        if (!host_ip) throw std::invalid_argument("host IP is null");
        context->impl->setHostIp(host_ip);
    });
}

mooncakePgResult_t mooncakePgContextSetTransferEngine(
    mooncakePgContext_t context, void* transfer_engine) {
    return translateExceptions([&] {
        validateContext(context);
        context->impl->setExternalEngine(
            static_cast<mooncake::TransferEngine*>(transfer_engine));
    });
}

mooncakePgResult_t mooncakePgContextSetDeviceFilter(mooncakePgContext_t context,
                                                    const char* const* filters,
                                                    size_t filter_count) {
    return translateExceptions([&] {
        validateContext(context);
        if (filter_count != 0 && !filters) {
            throw std::invalid_argument("device filters is null");
        }
        std::vector<std::string> values;
        values.reserve(filter_count);
        for (size_t index = 0; index < filter_count; ++index) {
            if (!filters[index]) {
                throw std::invalid_argument("device filter entry is null");
            }
            values.emplace_back(filters[index]);
        }
        context->impl->setDeviceFilter(std::move(values));
    });
}

mooncakePgResult_t mooncakePgContextSetCollectiveTimeout(
    mooncakePgContext_t context, size_t timeout_us) {
    return translateExceptions([&] {
        validateContext(context);
        context->impl->collective_timeout_us = timeout_us;
    });
}

mooncakePgResult_t mooncakePgContextSetP2PTimeout(mooncakePgContext_t context,
                                                  int64_t timeout_us) {
    return translateExceptions([&] {
        validateContext(context);
        context->impl->p2p_timeout_us = timeout_us;
    });
}

mooncakePgResult_t mooncakePgContextSetFaultReconciliationWindow(
    mooncakePgContext_t context, int64_t timeout_us) {
    return translateExceptions([&] {
        validateContext(context);
        context->impl->setFaultReconciliationWindow(timeout_us);
    });
}

mooncakePgResult_t mooncakePgContextDestroy(mooncakePgContext_t context) {
    return translateExceptions([&] { delete context; });
}

mooncakePgResult_t mooncakePgCommCreate(mooncakePgContext_t context,
                                        const mooncakePgCommConfig_t* config,
                                        mooncakePgComm_t* comm) {
    return translateExceptions([&] {
        if (!comm) throw std::invalid_argument("communicator output is null");
        *comm = nullptr;
        validateContext(context);
        validateCommConfig(config);

        mooncake::MooncakeCommunicatorConfig internal;
        internal.rank = config->rank;
        internal.size = config->size;
        internal.max_group_size =
            config->maxGroupSize == MOONCAKE_PG_CONFIG_UNDEF_INT
                ? config->size
                : config->maxGroupSize;
        if (config->globalRankCount != 0) {
            internal.global_ranks.assign(
                config->globalRanks,
                config->globalRanks + config->globalRankCount);
        }
        internal.group_bootstrap_id = config->groupId;
        switch (config->deviceType) {
            case mooncakePgDeviceCpu:
                internal.is_cpu = true;
                break;
            case mooncakePgDeviceGpu:
                internal.is_cpu = false;
                break;
            default:
                throw std::invalid_argument("invalid communicator device type");
        }
        internal.device_index =
            config->deviceIndex == MOONCAKE_PG_CONFIG_UNDEF_INT
                ? -1
                : config->deviceIndex;
        switch (config->idResolvePolicy) {
            case mooncakePgIdResolveCreateOrAttach:
                internal.group_resolve_policy =
                    mooncake::GroupBootstrapIdResolvePolicy::CreateOrAttach;
                break;
            case mooncakePgIdResolveAttachOrExtend:
                internal.group_resolve_policy =
                    mooncake::GroupBootstrapIdResolvePolicy::AttachOrExtend;
                break;
            default:
                throw std::invalid_argument(
                    "invalid communicator group resolve policy");
        }
        internal.auto_deactivate_on_failure =
            config->autoDeactivateOnFailure != 0;
        internal.auto_sync_on_failure = config->autoSyncOnFailure != 0;
        internal.active_ranks_mirror = config->activeRanksMirror;
        internal.active_ranks_mirror_count = config->activeRanksMirrorCount;
        internal.active_ranks_mirror_is_device =
            config->activeRanksMirrorIsDevice != 0;

        auto output = std::make_unique<mooncakePgComm>();
        output->impl = std::make_shared<mooncake::MooncakeCommunicator>(
            context->impl, std::move(internal));
        *comm = output.release();
    });
}

mooncakePgResult_t mooncakePgCommDestroy(mooncakePgComm_t comm) {
    return translateExceptions([&] {
        std::unique_ptr<mooncakePgComm> holder(comm);
        if (holder && holder->impl) holder->impl->shutdown();
    });
}

mooncakePgResult_t mooncakePgCommGetRank(mooncakePgComm_t comm, int* rank) {
    return translateExceptions([&] {
        validateComm(comm);
        if (!rank) throw std::invalid_argument("rank output is null");
        *rank = comm->impl->getRank();
    });
}

mooncakePgResult_t mooncakePgCommGetSize(mooncakePgComm_t comm, int* size) {
    return translateExceptions([&] {
        validateComm(comm);
        if (!size) throw std::invalid_argument("size output is null");
        *size = comm->impl->getSize();
    });
}

mooncakePgResult_t mooncakePgCommGetMaxGroupSize(mooncakePgComm_t comm,
                                                 int* max_group_size) {
    return translateExceptions([&] {
        validateComm(comm);
        if (!max_group_size) {
            throw std::invalid_argument("max group size output is null");
        }
        *max_group_size = comm->impl->getMaxGroupSize();
    });
}

mooncakePgResult_t mooncakePgBroadcastGpu(const void* send_buffer,
                                          void* recv_buffer, size_t count,
                                          mooncakePgDataType_t data_type,
                                          int root, mooncakePgComm_t comm,
                                          mooncakePgStream_t stream,
                                          int32_t* failed_ranks_hint,
                                          size_t failed_ranks_hint_count) {
    return launchGpuOperation(
        comm, failed_ranks_hint, failed_ranks_hint_count,
        [&](int32_t* failed_ranks_hint_buffer) {
            const size_t bytes = byteCount(count, data_type);
            if (comm->impl->getRank() == root) {
                validateBuffer(send_buffer, bytes, "broadcast send buffer");
            }
            validateBuffer(recv_buffer, bytes, "broadcast receive buffer");
            comm->impl->broadcastGpu(send_buffer, recv_buffer, bytes, root,
                                     convertStream(stream),
                                     failed_ranks_hint_buffer);
        });
}

mooncakePgResult_t mooncakePgAllReduceGpu(
    const void* send_buffer, void* recv_buffer, size_t count,
    mooncakePgDataType_t data_type, mooncakePgReduceOp_t reduce_op,
    mooncakePgComm_t comm, mooncakePgStream_t stream,
    int32_t* failed_ranks_hint, size_t failed_ranks_hint_count) {
    return launchGpuOperation(
        comm, failed_ranks_hint, failed_ranks_hint_count,
        [&](int32_t* failed_ranks_hint_buffer) {
            const size_t bytes = byteCount(count, data_type);
            validateBuffer(send_buffer, bytes, "all-reduce send buffer");
            validateBuffer(recv_buffer, bytes, "all-reduce receive buffer");
            comm->impl->allReduceGpu(
                send_buffer, recv_buffer, bytes, convertDataType(data_type),
                convertReduceOp(reduce_op), convertStream(stream),
                failed_ranks_hint_buffer);
        });
}

mooncakePgResult_t mooncakePgAllGatherGpu(const void* send_buffer,
                                          void* recv_buffer, size_t count,
                                          mooncakePgDataType_t data_type,
                                          mooncakePgComm_t comm,
                                          mooncakePgStream_t stream,
                                          int32_t* failed_ranks_hint,
                                          size_t failed_ranks_hint_count) {
    return launchGpuOperation(
        comm, failed_ranks_hint, failed_ranks_hint_count,
        [&](int32_t* failed_ranks_hint_buffer) {
            const size_t send_bytes = byteCount(count, data_type);
            validateBuffer(send_buffer, send_bytes, "all-gather send buffer");
            validateBuffer(recv_buffer, send_bytes,
                           "all-gather receive buffer");
            comm->impl->allGatherGpu(send_buffer, recv_buffer, send_bytes,
                                     convertStream(stream),
                                     failed_ranks_hint_buffer);
        });
}

mooncakePgResult_t mooncakePgReduceScatterGpu(
    const void* send_buffer, void* recv_buffer, size_t count,
    mooncakePgDataType_t data_type, mooncakePgReduceOp_t reduce_op,
    mooncakePgComm_t comm, mooncakePgStream_t stream,
    int32_t* failed_ranks_hint, size_t failed_ranks_hint_count) {
    return launchGpuOperation(
        comm, failed_ranks_hint, failed_ranks_hint_count,
        [&](int32_t* failed_ranks_hint_buffer) {
            const size_t recv_bytes = byteCount(count, data_type);
            validateBuffer(send_buffer, recv_bytes,
                           "reduce-scatter send buffer");
            validateBuffer(recv_buffer, recv_bytes,
                           "reduce-scatter receive buffer");
            comm->impl->reduceScatterGpu(
                send_buffer, recv_buffer, recv_bytes,
                convertDataType(data_type), convertReduceOp(reduce_op),
                convertStream(stream), failed_ranks_hint_buffer);
        });
}

mooncakePgResult_t mooncakePgAllToAllGpu(const void* send_buffer,
                                         void* recv_buffer, size_t count,
                                         mooncakePgDataType_t data_type,
                                         mooncakePgComm_t comm,
                                         mooncakePgStream_t stream,
                                         int32_t* failed_ranks_hint,
                                         size_t failed_ranks_hint_count) {
    return launchGpuOperation(
        comm, failed_ranks_hint, failed_ranks_hint_count,
        [&](int32_t* failed_ranks_hint_buffer) {
            const size_t peer_bytes = byteCount(count, data_type);
            validateBuffer(send_buffer, peer_bytes, "all-to-all send buffer");
            validateBuffer(recv_buffer, peer_bytes,
                           "all-to-all receive buffer");
            comm->impl->allToAllGpu(send_buffer, recv_buffer, peer_bytes,
                                    convertStream(stream),
                                    failed_ranks_hint_buffer);
        });
}

mooncakePgResult_t mooncakePgReduceGpu(
    const void* send_buffer, void* recv_buffer, size_t count,
    mooncakePgDataType_t data_type, mooncakePgReduceOp_t reduce_op, int root,
    mooncakePgComm_t comm, mooncakePgStream_t stream,
    int32_t* failed_ranks_hint, size_t failed_ranks_hint_count) {
    return launchGpuOperation(
        comm, failed_ranks_hint, failed_ranks_hint_count,
        [&](int32_t* failed_ranks_hint_buffer) {
            const size_t bytes = byteCount(count, data_type);
            validateBuffer(send_buffer, bytes, "reduce send buffer");
            if (comm->impl->getRank() == root) {
                validateBuffer(recv_buffer, bytes, "reduce receive buffer");
            }
            comm->impl->reduceGpu(
                send_buffer, recv_buffer, bytes, convertDataType(data_type),
                convertReduceOp(reduce_op), root, convertStream(stream),
                failed_ranks_hint_buffer);
        });
}

mooncakePgResult_t mooncakePgGatherGpu(const void* send_buffer,
                                       void* recv_buffer, size_t count,
                                       mooncakePgDataType_t data_type, int root,
                                       mooncakePgComm_t comm,
                                       mooncakePgStream_t stream,
                                       int32_t* failed_ranks_hint,
                                       size_t failed_ranks_hint_count) {
    return launchGpuOperation(
        comm, failed_ranks_hint, failed_ranks_hint_count,
        [&](int32_t* failed_ranks_hint_buffer) {
            const size_t send_bytes = byteCount(count, data_type);
            validateBuffer(send_buffer, send_bytes, "gather send buffer");
            if (comm->impl->getRank() == root) {
                validateBuffer(recv_buffer, send_bytes,
                               "gather receive buffer");
            }
            comm->impl->gatherGpu(send_buffer, recv_buffer, send_bytes, root,
                                  convertStream(stream),
                                  failed_ranks_hint_buffer);
        });
}

mooncakePgResult_t mooncakePgScatterGpu(const void* send_buffer,
                                        void* recv_buffer, size_t count,
                                        mooncakePgDataType_t data_type,
                                        int root, mooncakePgComm_t comm,
                                        mooncakePgStream_t stream,
                                        int32_t* failed_ranks_hint,
                                        size_t failed_ranks_hint_count) {
    return launchGpuOperation(
        comm, failed_ranks_hint, failed_ranks_hint_count,
        [&](int32_t* failed_ranks_hint_buffer) {
            const size_t recv_bytes = byteCount(count, data_type);
            if (comm->impl->getRank() == root) {
                validateBuffer(send_buffer, recv_bytes, "scatter send buffer");
            }
            validateBuffer(recv_buffer, recv_bytes, "scatter receive buffer");
            comm->impl->scatterGpu(send_buffer, recv_buffer, recv_bytes, root,
                                   convertStream(stream),
                                   failed_ranks_hint_buffer);
        });
}

mooncakePgResult_t mooncakePgBarrierGpu(mooncakePgComm_t comm,
                                        mooncakePgStream_t stream,
                                        int32_t* failed_ranks_hint,
                                        size_t failed_ranks_hint_count) {
    return launchGpuOperation(comm, failed_ranks_hint, failed_ranks_hint_count,
                              [&](int32_t* failed_ranks_hint_buffer) {
                                  comm->impl->barrierGpu(
                                      convertStream(stream),
                                      failed_ranks_hint_buffer);
                              });
}

mooncakePgResult_t mooncakePgBroadcastCpu(const void* send_buffer,
                                          void* recv_buffer, size_t count,
                                          mooncakePgDataType_t data_type,
                                          int root, mooncakePgComm_t comm,
                                          int32_t* failed_ranks_hint,
                                          size_t failed_ranks_hint_count,
                                          mooncakePgCompletion_t* completion) {
    return launchCpuOperation(
        comm, failed_ranks_hint, failed_ranks_hint_count, completion,
        [&](int32_t* failed_ranks_hint_buffer) {
            const size_t bytes = byteCount(count, data_type);
            if (comm->impl->getRank() == root) {
                validateBuffer(send_buffer, bytes, "broadcast send buffer");
            }
            validateBuffer(recv_buffer, bytes, "broadcast receive buffer");
            return comm->impl->broadcastCpu(send_buffer, recv_buffer, bytes,
                                            root, failed_ranks_hint_buffer);
        });
}

mooncakePgResult_t mooncakePgAllReduceCpu(
    const void* send_buffer, void* recv_buffer, size_t count,
    mooncakePgDataType_t data_type, mooncakePgReduceOp_t reduce_op,
    mooncakePgComm_t comm, int32_t* failed_ranks_hint,
    size_t failed_ranks_hint_count, mooncakePgCompletion_t* completion) {
    return launchCpuOperation(
        comm, failed_ranks_hint, failed_ranks_hint_count, completion,
        [&](int32_t* failed_ranks_hint_buffer) {
            const size_t bytes = byteCount(count, data_type);
            validateBuffer(send_buffer, bytes, "all-reduce send buffer");
            validateBuffer(recv_buffer, bytes, "all-reduce receive buffer");
            return comm->impl->allReduceCpu(
                send_buffer, recv_buffer, bytes, convertDataType(data_type),
                convertReduceOp(reduce_op), failed_ranks_hint_buffer);
        });
}

mooncakePgResult_t mooncakePgAllGatherCpu(const void* send_buffer,
                                          void* recv_buffer, size_t count,
                                          mooncakePgDataType_t data_type,
                                          mooncakePgComm_t comm,
                                          int32_t* failed_ranks_hint,
                                          size_t failed_ranks_hint_count,
                                          mooncakePgCompletion_t* completion) {
    return launchCpuOperation(
        comm, failed_ranks_hint, failed_ranks_hint_count, completion,
        [&](int32_t* failed_ranks_hint_buffer) {
            const size_t send_bytes = byteCount(count, data_type);
            validateBuffer(send_buffer, send_bytes, "all-gather send buffer");
            validateBuffer(recv_buffer, send_bytes,
                           "all-gather receive buffer");
            return comm->impl->allGatherCpu(
                send_buffer, recv_buffer, send_bytes, failed_ranks_hint_buffer);
        });
}

mooncakePgResult_t mooncakePgReduceScatterCpu(
    const void* send_buffer, void* recv_buffer, size_t count,
    mooncakePgDataType_t data_type, mooncakePgReduceOp_t reduce_op,
    mooncakePgComm_t comm, int32_t* failed_ranks_hint,
    size_t failed_ranks_hint_count, mooncakePgCompletion_t* completion) {
    return launchCpuOperation(
        comm, failed_ranks_hint, failed_ranks_hint_count, completion,
        [&](int32_t* failed_ranks_hint_buffer) {
            const size_t recv_bytes = byteCount(count, data_type);
            validateBuffer(send_buffer, recv_bytes,
                           "reduce-scatter send buffer");
            validateBuffer(recv_buffer, recv_bytes,
                           "reduce-scatter receive buffer");
            return comm->impl->reduceScatterCpu(
                send_buffer, recv_buffer, recv_bytes,
                convertDataType(data_type), convertReduceOp(reduce_op),
                failed_ranks_hint_buffer);
        });
}

mooncakePgResult_t mooncakePgAllToAllCpu(const void* send_buffer,
                                         void* recv_buffer, size_t count,
                                         mooncakePgDataType_t data_type,
                                         mooncakePgComm_t comm,
                                         int32_t* failed_ranks_hint,
                                         size_t failed_ranks_hint_count,
                                         mooncakePgCompletion_t* completion) {
    return launchCpuOperation(
        comm, failed_ranks_hint, failed_ranks_hint_count, completion,
        [&](int32_t* failed_ranks_hint_buffer) {
            const size_t peer_bytes = byteCount(count, data_type);
            validateBuffer(send_buffer, peer_bytes, "all-to-all send buffer");
            validateBuffer(recv_buffer, peer_bytes,
                           "all-to-all receive buffer");
            return comm->impl->allToAllCpu(send_buffer, recv_buffer, peer_bytes,
                                           failed_ranks_hint_buffer);
        });
}

mooncakePgResult_t mooncakePgReduceCpu(
    const void* send_buffer, void* recv_buffer, size_t count,
    mooncakePgDataType_t data_type, mooncakePgReduceOp_t reduce_op, int root,
    mooncakePgComm_t comm, int32_t* failed_ranks_hint,
    size_t failed_ranks_hint_count, mooncakePgCompletion_t* completion) {
    return launchCpuOperation(
        comm, failed_ranks_hint, failed_ranks_hint_count, completion,
        [&](int32_t* failed_ranks_hint_buffer) {
            const size_t bytes = byteCount(count, data_type);
            validateBuffer(send_buffer, bytes, "reduce send buffer");
            if (comm->impl->getRank() == root) {
                validateBuffer(recv_buffer, bytes, "reduce receive buffer");
            }
            return comm->impl->reduceCpu(
                send_buffer, recv_buffer, bytes, convertDataType(data_type),
                convertReduceOp(reduce_op), root, failed_ranks_hint_buffer);
        });
}

mooncakePgResult_t mooncakePgGatherCpu(const void* send_buffer,
                                       void* recv_buffer, size_t count,
                                       mooncakePgDataType_t data_type, int root,
                                       mooncakePgComm_t comm,
                                       int32_t* failed_ranks_hint,
                                       size_t failed_ranks_hint_count,
                                       mooncakePgCompletion_t* completion) {
    return launchCpuOperation(
        comm, failed_ranks_hint, failed_ranks_hint_count, completion,
        [&](int32_t* failed_ranks_hint_buffer) {
            const size_t send_bytes = byteCount(count, data_type);
            validateBuffer(send_buffer, send_bytes, "gather send buffer");
            if (comm->impl->getRank() == root) {
                validateBuffer(recv_buffer, send_bytes,
                               "gather receive buffer");
            }
            return comm->impl->gatherCpu(send_buffer, recv_buffer, send_bytes,
                                         root, failed_ranks_hint_buffer);
        });
}

mooncakePgResult_t mooncakePgScatterCpu(const void* send_buffer,
                                        void* recv_buffer, size_t count,
                                        mooncakePgDataType_t data_type,
                                        int root, mooncakePgComm_t comm,
                                        int32_t* failed_ranks_hint,
                                        size_t failed_ranks_hint_count,
                                        mooncakePgCompletion_t* completion) {
    return launchCpuOperation(
        comm, failed_ranks_hint, failed_ranks_hint_count, completion,
        [&](int32_t* failed_ranks_hint_buffer) {
            const size_t recv_bytes = byteCount(count, data_type);
            if (comm->impl->getRank() == root) {
                validateBuffer(send_buffer, recv_bytes, "scatter send buffer");
            }
            validateBuffer(recv_buffer, recv_bytes, "scatter receive buffer");
            return comm->impl->scatterCpu(send_buffer, recv_buffer, recv_bytes,
                                          root, failed_ranks_hint_buffer);
        });
}

mooncakePgResult_t mooncakePgBarrierCpu(mooncakePgComm_t comm,
                                        int32_t* failed_ranks_hint,
                                        size_t failed_ranks_hint_count,
                                        mooncakePgCompletion_t* completion) {
    return launchCpuOperation(
        comm, failed_ranks_hint, failed_ranks_hint_count, completion,
        [&](int32_t* failed_ranks_hint_buffer) {
            return comm->impl->barrierCpu(failed_ranks_hint_buffer);
        });
}

mooncakePgResult_t mooncakePgSendGpu(const void* send_buffer, size_t count,
                                     mooncakePgDataType_t data_type, int peer,
                                     mooncakePgComm_t comm,
                                     mooncakePgStream_t stream,
                                     int32_t* failed_ranks_hint,
                                     size_t failed_ranks_hint_count,
                                     mooncakePgCompletion_t* completion) {
    return launchP2POperation(
        comm, false, failed_ranks_hint, failed_ranks_hint_count, completion,
        [&](int32_t* failed_ranks_hint_buffer) {
            const size_t bytes = byteCount(count, data_type);
            validateBuffer(send_buffer, bytes, "send buffer");
            return comm->impl->send(send_buffer, bytes, peer,
                                    convertStream(stream),
                                    failed_ranks_hint_buffer);
        });
}

mooncakePgResult_t mooncakePgRecvGpu(void* recv_buffer, size_t count,
                                     mooncakePgDataType_t data_type, int peer,
                                     mooncakePgComm_t comm,
                                     mooncakePgStream_t stream,
                                     int32_t* failed_ranks_hint,
                                     size_t failed_ranks_hint_count,
                                     mooncakePgCompletion_t* completion) {
    return launchP2POperation(
        comm, false, failed_ranks_hint, failed_ranks_hint_count, completion,
        [&](int32_t* failed_ranks_hint_buffer) {
            const size_t bytes = byteCount(count, data_type);
            validateBuffer(recv_buffer, bytes, "receive buffer");
            return comm->impl->recv(recv_buffer, bytes, peer,
                                    convertStream(stream),
                                    failed_ranks_hint_buffer);
        });
}

mooncakePgResult_t mooncakePgSendCpu(const void* send_buffer, size_t count,
                                     mooncakePgDataType_t data_type, int peer,
                                     mooncakePgComm_t comm,
                                     int32_t* failed_ranks_hint,
                                     size_t failed_ranks_hint_count,
                                     mooncakePgCompletion_t* completion) {
    return launchP2POperation(
        comm, true, failed_ranks_hint, failed_ranks_hint_count, completion,
        [&](int32_t* failed_ranks_hint_buffer) {
            const size_t bytes = byteCount(count, data_type);
            validateBuffer(send_buffer, bytes, "send buffer");
            return comm->impl->send(send_buffer, bytes, peer, nullptr,
                                    failed_ranks_hint_buffer);
        });
}

mooncakePgResult_t mooncakePgRecvCpu(void* recv_buffer, size_t count,
                                     mooncakePgDataType_t data_type, int peer,
                                     mooncakePgComm_t comm,
                                     int32_t* failed_ranks_hint,
                                     size_t failed_ranks_hint_count,
                                     mooncakePgCompletion_t* completion) {
    return launchP2POperation(
        comm, true, failed_ranks_hint, failed_ranks_hint_count, completion,
        [&](int32_t* failed_ranks_hint_buffer) {
            const size_t bytes = byteCount(count, data_type);
            validateBuffer(recv_buffer, bytes, "receive buffer");
            return comm->impl->recv(recv_buffer, bytes, peer, nullptr,
                                    failed_ranks_hint_buffer);
        });
}

mooncakePgResult_t mooncakePgCompletionIsCompleted(
    mooncakePgCompletion_t completion, int* completed) {
    return translateExceptions([&] {
        validateCompletion(completion);
        if (!completed) {
            throw std::invalid_argument("completed output is null");
        }
        *completed = completion->impl->isCompleted() ? 1 : 0;
    });
}

mooncakePgResult_t mooncakePgCompletionWait(mooncakePgCompletion_t completion,
                                            int64_t timeout_us) {
    return translateExceptions([&] {
        validateCompletion(completion);
        if (!completion->impl->wait(std::chrono::microseconds(timeout_us))) {
            throw TimeoutError("operation timed out");
        }
    });
}

mooncakePgResult_t mooncakePgCompletionDestroy(
    mooncakePgCompletion_t completion) {
    return translateExceptions([&] { delete completion; });
}

mooncakePgResult_t mooncakePgCommGetActiveRanks(mooncakePgComm_t comm,
                                                int32_t* active_ranks,
                                                size_t rank_count) {
    return translateExceptions([&] {
        validateComm(comm);
        const auto ranks = comm->impl->getActiveRanks();
        if (rank_count < ranks.size()) {
            throw std::invalid_argument("active-ranks output is too small");
        }
        if (!ranks.empty() && !active_ranks) {
            throw std::invalid_argument("active-ranks output is null");
        }
        if (!ranks.empty()) {
            std::copy(ranks.begin(), ranks.end(), active_ranks);
        }
    });
}

mooncakePgResult_t mooncakePgCommGetPeerState(mooncakePgComm_t comm,
                                              const int32_t* ranks,
                                              size_t rank_count,
                                              int32_t* peer_states) {
    return translateExceptions([&] {
        validateComm(comm);
        if (rank_count != 0 && !peer_states) {
            throw std::invalid_argument("peer-states output is null");
        }
        const auto states =
            comm->impl->getPeerState(copyRanks(ranks, rank_count));
        for (size_t index = 0; index < states.size(); ++index) {
            peer_states[index] = states[index] ? 1 : 0;
        }
    });
}

mooncakePgResult_t mooncakePgCommActivateRanks(
    mooncakePgComm_t comm, const int32_t* ranks, size_t rank_count,
    mooncakePgProposalResponse_t* response) {
    return translateExceptions([&] {
        validateComm(comm);
        copyProposal(comm->impl->activateRanks(copyRanks(ranks, rank_count)),
                     response);
    });
}

mooncakePgResult_t mooncakePgCommDeactivateRanks(
    mooncakePgComm_t comm, const int32_t* ranks, size_t rank_count,
    mooncakePgProposalResponse_t* response) {
    return translateExceptions([&] {
        validateComm(comm);
        copyProposal(comm->impl->deactivateRanks(copyRanks(ranks, rank_count)),
                     response);
    });
}

mooncakePgResult_t mooncakePgCommJoin(mooncakePgComm_t comm) {
    return translateExceptions([&] {
        validateComm(comm);
        comm->impl->joinGroup();
    });
}

mooncakePgResult_t mooncakePgCommSyncAfterFailure(
    mooncakePgComm_t comm, mooncakePgSyncAfterFailureResponse_t* response) {
    return translateExceptions([&] {
        validateComm(comm);
        if (!response) throw std::invalid_argument("sync response is null");
        const auto source = comm->impl->syncAfterFailure();
        std::memset(response, 0, sizeof(*response));
        response->status =
            static_cast<mooncakePgSyncAfterFailureStatus_t>(source.status);
        copyErrorText(source.reject_reason, response->rejectReason,
                      sizeof(response->rejectReason));
    });
}

mooncakePgResult_t mooncakePgCommGetEpoch(mooncakePgComm_t comm,
                                          uint64_t* epoch) {
    return translateExceptions([&] {
        validateComm(comm);
        if (!epoch) throw std::invalid_argument("epoch output is null");
        *epoch = comm->impl->getCurrentEpoch();
    });
}

mooncakePgResult_t mooncakePgCommGetNumSyncedRanks(mooncakePgComm_t comm,
                                                   int* count) {
    return translateExceptions([&] {
        validateComm(comm);
        if (!count) throw std::invalid_argument("count output is null");
        *count = comm->impl->getNumSyncedRanks();
    });
}

mooncakePgResult_t mooncakePgCommGetPreferredHca(mooncakePgComm_t comm,
                                                 const char* location,
                                                 char* hca_buf,
                                                 size_t hca_buf_size) {
    return translateExceptions([&] {
        validateComm(comm);
        if (!location || !hca_buf || hca_buf_size == 0) {
            throw std::invalid_argument("invalid preferred-HCA output");
        }
        const std::string value = comm->impl->getPreferredHca(location);
        if (value.size() >= hca_buf_size) {
            throw std::invalid_argument("preferred-HCA output is too small");
        }
        std::memcpy(hca_buf, value.c_str(), value.size() + 1);
    });
}
