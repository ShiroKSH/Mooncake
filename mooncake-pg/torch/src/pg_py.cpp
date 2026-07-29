#include <mooncake_backend.h>

#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/python.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace py = pybind11;

namespace mooncake {
namespace {

constexpr const char* kCoordinatorStoreKey = "coordinator_addr";

struct TorchContextHandle {
    mooncakePgContext_t handle = nullptr;

    TorchContextHandle() = default;
    ~TorchContextHandle() {
        if (handle) (void)mooncakePgContextDestroy(handle);
    }

    TorchContextHandle(const TorchContextHandle&) = delete;
    TorchContextHandle& operator=(const TorchContextHandle&) = delete;
};

TorchContextHandle g_context;
std::once_flag g_create_context_once;
std::once_flag g_init_context_once;

void checkResult(mooncakePgResult_t result, const char* operation) {
    TORCH_CHECK(result == mooncakePgSuccess, operation,
                " failed: ", mooncakePgGetErrorString(result), ": ",
                mooncakePgGetLastError());
}

mooncakePgContext_t getContext() {
    std::call_once(g_create_context_once, [] {
        int64_t fault_reconciliation_window_us =
            MOONCAKE_PG_DEFAULT_FAULT_RECONCILIATION_WINDOW_US;
        if (const char* value =
                std::getenv("MOONCAKE_PG_FAULT_RECONCILIATION_WINDOW_US")) {
            try {
                fault_reconciliation_window_us = std::stoll(value);
            } catch (...) {
                TORCH_WARN(
                    "Invalid MOONCAKE_PG_FAULT_RECONCILIATION_WINDOW_US: ",
                    value, "; using ", fault_reconciliation_window_us, " us");
            }
        }

        mooncakePgContext_t context = nullptr;
        checkResult(mooncakePgContextCreate(&context),
                    "mooncakePgContextCreate");
        try {
            checkResult(mooncakePgContextSetFaultReconciliationWindow(
                            context, fault_reconciliation_window_us),
                        "mooncakePgContextSetFaultReconciliationWindow");
            g_context.handle = context;
        } catch (...) {
            (void)mooncakePgContextDestroy(context);
            throw;
        }
    });
    return g_context.handle;
}

void setDeviceFilters(mooncakePgContext_t context,
                      const std::vector<std::string>& filters) {
    std::vector<const char*> filter_pointers;
    filter_pointers.reserve(filters.size());
    for (const auto& filter : filters) {
        filter_pointers.push_back(filter.c_str());
    }
    checkResult(mooncakePgContextSetDeviceFilter(
                    context, filter_pointers.data(), filter_pointers.size()),
                "mooncakePgContextSetDeviceFilter");
}

mooncakePgContext_t initializeContext(
    const c10::intrusive_ptr<c10d::Store>& store, int rank,
    int max_world_size) {
    auto context = getContext();
    std::call_once(g_init_context_once, [&] {
        // Ordering constraint: AgentHost::start() sends registerAgent
        // immediately, which includes LinkManager's localServerName() and
        // getWarmupRecvAddr(). These must be non-empty, so the engine and
        // LinkManager must be initialized before
        // mooncakePgContextConnectCoordinator starts AgentHost.
        checkResult(mooncakePgContextInitialize(context, rank, max_world_size),
                    "mooncakePgContextInitialize");

        // Rank 0 hosts the Coordinator in-process.
        if (rank == 0) {
            mooncakePgCoordinatorAddress_t address{};
            checkResult(mooncakePgContextLaunchCoordinator(context, &address),
                        "mooncakePgContextLaunchCoordinator");
            store->set(kCoordinatorStoreKey, std::string(address.internal));
        }

        store->wait({kCoordinatorStoreKey});
        const std::string value = store->get_to_str(kCoordinatorStoreKey);
        TORCH_CHECK(!value.empty() &&
                        value.size() < MOONCAKE_PG_COORDINATOR_ADDRESS_BYTES,
                    "invalid Mooncake coordinator address in Store");
        mooncakePgCoordinatorAddress_t address{};
        std::memcpy(address.internal, value.data(), value.size());
        checkResult(mooncakePgContextConnectCoordinator(context, &address),
                    "mooncakePgContextConnectCoordinator");
    });
    return context;
}

c10::intrusive_ptr<c10d::ProcessGroup> createMooncakeBackend(
    c10d::DistributedBackendOptions distBackendOpts,
    c10::intrusive_ptr<MooncakeBackend::MooncakeBackendOptions>
        backendOptions) {
    const int rank = distBackendOpts.group_rank;
    auto context =
        initializeContext(distBackendOpts.store, rank, MOONCAKE_PG_MAX_RANKS);
    return c10::make_intrusive<MooncakeBackend>(
        std::move(distBackendOpts), std::move(backendOptions), context);
}

c10::intrusive_ptr<c10d::ProcessGroup> createMooncakeCpuBackend(
    c10d::DistributedBackendOptions distBackendOpts,
    c10::intrusive_ptr<MooncakeBackend::MooncakeBackendOptions>
        backendOptions) {
    const int rank = distBackendOpts.group_rank;
    auto context =
        initializeContext(distBackendOpts.store, rank, MOONCAKE_PG_MAX_RANKS);
    return c10::make_intrusive<MooncakeBackend>(
        std::move(distBackendOpts), std::move(backendOptions), context, true);
}

__attribute__((constructor)) void registerMooncakeBackend() {
    py::object module = py::module::import("torch.distributed");
    py::object register_backend =
        module.attr("Backend").attr("register_backend");

    py::dict cpu_kwargs;
    cpu_kwargs["devices"] = py::make_tuple("cpu");
    register_backend("mooncake-cpu", py::cpp_function(createMooncakeCpuBackend),
                     /*extended_api=*/true, **cpu_kwargs);
#ifndef MOONCAKE_EP_USE_MUSA
    py::dict device_kwargs;
    device_kwargs["devices"] = py::make_tuple("cuda");
    register_backend("mooncake", py::cpp_function(createMooncakeBackend),
                     /*extended_api=*/true, **device_kwargs);
#else
    py::dict device_kwargs;
    device_kwargs["devices"] = py::make_tuple("musa");
    register_backend("mooncake", py::cpp_function(createMooncakeBackend),
                     /*extended_api=*/true, **device_kwargs);
#endif
}

MooncakeBackend& asMooncakeBackend(
    const c10::intrusive_ptr<c10d::ProcessGroup>& backend) {
    return *c10::static_intrusive_pointer_cast<MooncakeBackend>(backend);
}

std::string getPreferredHca(c10::intrusive_ptr<c10d::ProcessGroup> backend,
                            const std::string& location) {
    return asMooncakeBackend(backend).getPreferredHca(location);
}

at::Tensor getActiveRanks(c10::intrusive_ptr<c10d::ProcessGroup> backend) {
    return asMooncakeBackend(backend).getActiveRanksTensor();
}

int getNumSyncedRanks(c10::intrusive_ptr<c10d::ProcessGroup> backend) {
    return asMooncakeBackend(backend).getNumSyncedRanks();
}

void extendGroupSizeTo(c10::intrusive_ptr<c10d::ProcessGroup> backend,
                       int size) {
    asMooncakeBackend(backend).extendGroupSizeTo(size);
}

std::vector<bool> getPeerState(c10::intrusive_ptr<c10d::ProcessGroup> backend,
                               const std::vector<int>& ranks) {
    return asMooncakeBackend(backend).getPeerState(ranks);
}

mooncakePgProposalResponse_t activateRanks(
    c10::intrusive_ptr<c10d::ProcessGroup> backend,
    const std::vector<int>& ranks) {
    return asMooncakeBackend(backend).activateRanks(ranks);
}

mooncakePgProposalResponse_t deactivateRanks(
    c10::intrusive_ptr<c10d::ProcessGroup> backend,
    const std::vector<int>& ranks) {
    return asMooncakeBackend(backend).deactivateRanks(ranks);
}

void joinGroup(c10::intrusive_ptr<c10d::ProcessGroup> backend) {
    asMooncakeBackend(backend).joinGroup();
}

at::Tensor getFailedRanksHint(c10::intrusive_ptr<c10d::Work> work) {
    if (auto* value = dynamic_cast<MooncakeWorkCuda*>(work.get())) {
        return value->getFailedRanksHint();
    }
    if (auto* value = dynamic_cast<MooncakeWorkCpu*>(work.get())) {
        return value->getFailedRanksHint();
    }
    if (auto* value = dynamic_cast<MooncakeP2PWork*>(work.get())) {
        return value->getFailedRanksHint();
    }
    return at::Tensor();
}

bool getLocalSuccess(c10::intrusive_ptr<c10d::Work> work) {
    if (auto* value = dynamic_cast<MooncakeWorkCuda*>(work.get())) {
        return value->getLocalSuccess();
    }
    if (auto* value = dynamic_cast<MooncakeWorkCpu*>(work.get())) {
        return value->getLocalSuccess();
    }
    if (auto* value = dynamic_cast<MooncakeP2PWork*>(work.get())) {
        return value->getLocalSuccess();
    }
    return false;
}

int64_t getCurrentEpoch(c10::intrusive_ptr<c10d::ProcessGroup> backend) {
    return static_cast<int64_t>(asMooncakeBackend(backend).getCurrentEpoch());
}

/// Python-facing wrapper that extracts the raw TransferEngine* from a
/// mooncake.engine.TransferEngine Python object and makes it the process-wide
/// engine for all MooncakeBackend instances. The caller must ensure the
/// TransferEnginePy object outlives all MooncakeBackend instances.
void setTransferEnginePy(py::object engine_obj) {
    void* transfer_engine = nullptr;
    if (engine_obj.is_none()) {
        checkResult(mooncakePgContextSetTransferEngine(getContext(), nullptr),
                    "mooncakePgContextSetTransferEngine");
    } else {
        const uintptr_t pointer =
            engine_obj.attr("get_engine_ptr")().cast<uintptr_t>();
        transfer_engine = reinterpret_cast<void*>(pointer);
        checkResult(
            mooncakePgContextSetTransferEngine(getContext(), transfer_engine),
            "mooncakePgContextSetTransferEngine");
    }
}

std::vector<int> droppedRanks(const mooncakePgProposalResponse_t& response) {
    const size_t count = std::min(response.droppedRankCount,
                                  static_cast<size_t>(MOONCAKE_PG_MAX_RANKS));
    return std::vector<int>(response.droppedRanks,
                            response.droppedRanks + count);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("createMooncakeBackend", &createMooncakeBackend);
    module.def("createMooncakeCpuBackend", &createMooncakeCpuBackend);
    module.def("set_host_ip", [](const std::string& host) {
        checkResult(mooncakePgContextSetHostIp(getContext(), host.c_str()),
                    "mooncakePgContextSetHostIp");
    });
    module.def(
        "set_collective_timeout_us",
        [](size_t timeout_us) {
            checkResult(
                mooncakePgContextSetCollectiveTimeout(getContext(), timeout_us),
                "mooncakePgContextSetCollectiveTimeout");
        },
        py::arg("us"),
        "Set the default peer-liveness probe timeout (microseconds) for "
        "collective operations.");
    module.def(
        "set_p2p_timeout_us",
        [](int64_t timeout_us) {
            checkResult(
                mooncakePgContextSetP2PTimeout(getContext(), timeout_us),
                "mooncakePgContextSetP2PTimeout");
        },
        py::arg("us"), "Set the default P2P transfer timeout (microseconds).");
    module.def(
        "set_fault_reconciliation_window_us",
        [](int64_t timeout_us) {
            checkResult(mooncakePgContextSetFaultReconciliationWindow(
                            getContext(), timeout_us),
                        "mooncakePgContextSetFaultReconciliationWindow");
        },
        py::arg("us"),
        "Set the coordinator fault reconciliation window (microseconds).");
    module.def("set_device_filter", [](std::vector<std::string> filters) {
        setDeviceFilters(getContext(), filters);
    });
    module.def("set_transfer_engine", &setTransferEnginePy, py::arg("engine"),
               "Set an external TransferEngine to be used by MooncakeBackend. "
               "Must be called before init_process_group(). The engine must "
               "already be initialized. Pass None to reset to default "
               "behavior. The caller must ensure the TransferEngine object "
               "outlives all MooncakeBackend instances.");
    module.def("get_preferred_hca", &getPreferredHca);
    module.def("get_active_ranks", &getActiveRanks);
    module.def("get_num_synced_ranks", &getNumSyncedRanks);
    module.def("extend_group_size_to", &extendGroupSizeTo);
    module.def("get_peer_state", &getPeerState);
    // Alias to activateRanks.
    module.def("recover_ranks", &activateRanks);
    module.def("activate_ranks", &activateRanks);
    module.def("deactivate_ranks", &deactivateRanks, py::arg("backend"),
               py::arg("ranks"));
    module.def("join_group", &joinGroup);
    module.def("get_failed_ranks_hint", &getFailedRanksHint, py::arg("work"));
    module.def("get_local_success", &getLocalSuccess, py::arg("work"),
               "Return True iff all locally-attempted peers succeeded in "
               "this operation.");
    module.def("get_current_epoch", &getCurrentEpoch, py::arg("backend"),
               "Get the current GroupView epoch (monotonically increasing on "
               "membership changes).");
    module.def(
        "sync_after_failure",
        [](c10::intrusive_ptr<c10d::ProcessGroup> backend) {
            return asMooncakeBackend(backend).syncAfterFailure();
        },
        py::arg("backend"));

    py::enum_<mooncakePgSyncAfterFailureStatus_t>(module,
                                                  "SyncAfterFailureStatus")
        .value("Reconciled", mooncakePgSyncReconciled)
        .value("NoPending", mooncakePgSyncNoPending)
        .value("Rejected", mooncakePgSyncRejected);

    auto proposal_status =
        py::enum_<mooncakePgProposalStatus_t>(module, "ProposalStatus")
            .value("Rejected", mooncakePgProposalRejected)
            .value("Applied", mooncakePgProposalApplied)
            .value("AppliedWithDroppedRanks",
                   mooncakePgProposalAppliedWithDroppedRanks);
    // Keep existing Python callers source-compatible with the renamed enum.
    module.attr("ViewUpdateStatus") = proposal_status;

    py::class_<mooncakePgSyncAfterFailureResponse_t>(module,
                                                     "SyncAfterFailureResponse")
        .def_property_readonly(
            "status",
            [](const mooncakePgSyncAfterFailureResponse_t& value) {
                return value.status;
            })
        .def_property_readonly(
            "reject_reason",
            [](const mooncakePgSyncAfterFailureResponse_t& value) {
                return std::string(value.rejectReason);
            });

    py::class_<mooncakePgProposalResponse_t>(module,
                                             "ProposeViewUpdateResponse")
        .def_property_readonly("status",
                               [](const mooncakePgProposalResponse_t& value) {
                                   return value.status;
                               })
        .def_property_readonly("new_epoch",
                               [](const mooncakePgProposalResponse_t& value) {
                                   return value.newEpoch;
                               })
        .def_property_readonly("dropped_ranks", &droppedRanks)
        .def_property_readonly("reject_reason",
                               [](const mooncakePgProposalResponse_t& value) {
                                   return std::string(value.rejectReason);
                               });

    py::class_<MooncakeBackend::MooncakeBackendOptions,
               c10::intrusive_ptr<MooncakeBackend::MooncakeBackendOptions>>(
        module, "MooncakeBackendOptions")
        // IMPORTANT: these constructors with tensor MUST be registered before
        // the (int, ...) constructors. Otherwise, when a 1-element Tensor is
        // passed, pybind11 implicitly converts it to int and resolves to the
        // wrong overload:
        // e.g. MooncakeBackendOptions(tensor([1]), False) ->
        //      MooncakeBackendOptions(int maxGroupSize=1,
        //                            bool isExtension=False)
        // instead of the intended Tensor-based path.
        .def(py::init<at::Tensor>(), py::arg("active_ranks"))
        .def(py::init<at::Tensor, bool>(), py::arg("active_ranks"),
             py::arg("is_extension"))
        .def(py::init<at::Tensor, bool, int>(), py::arg("active_ranks"),
             py::arg("is_extension"), py::arg("max_group_size"))
        // Recommended constructors
        .def(py::init<int>(), py::arg("max_group_size"))
        .def(py::init<int, bool>(), py::arg("max_group_size"),
             py::arg("is_extension"))
        .def(py::init<int, bool, bool, bool>(), py::arg("max_group_size"),
             py::arg("is_extension"), py::arg("auto_deactivate_on_failure"),
             py::arg("auto_sync_on_failure"));
}

}  // namespace mooncake
