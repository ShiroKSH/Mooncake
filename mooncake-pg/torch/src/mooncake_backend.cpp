#include <mooncake_backend.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace mooncake {
namespace {

constexpr const char* kSingleTensorError =
    "Expecting one tensor only but got multiple.";
constexpr const char* kSparseError = "Sparse op not supported.";
constexpr size_t kPreferredHcaBytes = 256;

void checkResult(mooncakePgResult_t result, const char* operation) {
    TORCH_CHECK(result == mooncakePgSuccess, operation,
                " failed: ", mooncakePgGetErrorString(result), ": ",
                mooncakePgGetLastError());
}

mooncakePgDataType_t convertDataType(at::ScalarType data_type) {
    switch (data_type) {
        case at::kByte:
            return mooncakePgUint8;
        case at::kChar:
            return mooncakePgInt8;
        case at::kShort:
            return mooncakePgInt16;
        case at::kInt:
            return mooncakePgInt32;
        case at::kLong:
            return mooncakePgInt64;
        case at::kFloat:
            return mooncakePgFloat32;
        case at::kDouble:
            return mooncakePgFloat64;
        case at::kBool:
            return mooncakePgBool;
        case at::kBFloat16:
            return mooncakePgBfloat16;
        default:
            TORCH_CHECK(false, "Unsupported Mooncake PG datatype: ", data_type);
    }
}

mooncakePgReduceOp_t convertReduceOp(const c10d::ReduceOp& reduce_op) {
    switch (reduce_op) {
        case c10d::ReduceOp::SUM:
            return mooncakePgSum;
        case c10d::ReduceOp::AVG:
            TORCH_CHECK(false, "Mooncake PG does not support AVG reduction");
        case c10d::ReduceOp::PRODUCT:
            return mooncakePgProduct;
        case c10d::ReduceOp::MIN:
            return mooncakePgMin;
        case c10d::ReduceOp::MAX:
            return mooncakePgMax;
        default:
            TORCH_CHECK(false, "Mooncake PG only supports SUM/PRODUCT/MIN/MAX");
    }
}

size_t tensorCount(const at::Tensor& tensor) {
    TORCH_CHECK(tensor.numel() >= 0, "invalid Tensor element count");
    return static_cast<size_t>(tensor.numel());
}

at::cuda::CUDAStream currentCudaStream(const at::Tensor& tensor) {
    return at::cuda::getCurrentCUDAStream(tensor.device().index());
}

at::cuda::CUDAStream currentCudaStream() {
    return at::cuda::getCurrentCUDAStream();
}

mooncakePgStream_t coreStream(const at::cuda::CUDAStream& stream) {
    return reinterpret_cast<mooncakePgStream_t>(stream.stream());
}

std::shared_ptr<c10::Event> recordCompletionEvent(
    const at::cuda::CUDAStream& stream) {
    auto event = std::make_shared<c10::Event>(c10::DeviceType::CUDA);
    event->record(stream);
    return event;
}

bool envFlag(const char* name, bool fallback) {
    const char* value = std::getenv(name);
    if (!value) return fallback;
    const std::string text(value);
    return text == "1" || text == "true" || text == "TRUE" || text == "on" ||
           text == "ON";
}

void validatePeerTensors(const std::vector<at::Tensor>& tensors,
                         const at::Tensor& reference, int max_group_size) {
    TORCH_CHECK(tensors.size() <= static_cast<size_t>(max_group_size),
                "Tensor list exceeds max_group_size");
    for (const auto& tensor : tensors) {
        TORCH_CHECK(tensor.scalar_type() == reference.scalar_type(),
                    "All peer tensors must have the same dtype");
        TORCH_CHECK(tensor.device() == reference.device(),
                    "All peer tensors must be on the same device");
        TORCH_CHECK(tensor.numel() == reference.numel(),
                    "All peer tensors must have the same number of elements");
    }
}

at::Tensor packPeerTensors(const std::vector<at::Tensor>& tensors,
                           const at::Tensor& reference, int max_group_size) {
    validatePeerTensors(tensors, reference, max_group_size);
    const int64_t peer_count = reference.numel();
    auto packed = at::empty({peer_count * max_group_size}, reference.options());
    for (size_t index = 0; index < tensors.size(); ++index) {
        packed.narrow(0, static_cast<int64_t>(index) * peer_count, peer_count)
            .copy_(tensors[index].reshape({peer_count}));
    }
    return packed;
}

std::function<void()> unpackPeerTensors(at::Tensor packed,
                                        std::vector<at::Tensor> outputs) {
    return
        [packed = std::move(packed), outputs = std::move(outputs)]() mutable {
            if (outputs.empty()) return;
            const int64_t peer_count = outputs.front().numel();
            for (size_t index = 0; index < outputs.size(); ++index) {
                outputs[index].copy_(
                    packed
                        .narrow(0, static_cast<int64_t>(index) * peer_count,
                                peer_count)
                        .view(outputs[index].sizes()));
            }
        };
}

std::vector<at::Tensor> retain(std::initializer_list<at::Tensor> tensors) {
    std::vector<at::Tensor> result;
    result.reserve(tensors.size());
    for (const auto& tensor : tensors) {
        if (tensor.defined()) result.push_back(tensor);
    }
    return result;
}

void appendTensors(std::vector<at::Tensor>& destination,
                   const std::vector<at::Tensor>& source) {
    destination.insert(destination.end(), source.begin(), source.end());
}

std::vector<int32_t> toCoreRanks(const std::vector<int>& ranks) {
    return std::vector<int32_t>(ranks.begin(), ranks.end());
}

}  // namespace

/**
 * @brief Initialize Mooncake backend state from the PyTorch process-group
 * information and optional Mooncake-specific options.
 */
MooncakeBackend::MooncakeBackend(
    c10d::DistributedBackendOptions distBackendOpts,
    c10::intrusive_ptr<MooncakeBackendOptions> options,
    mooncakePgContext_t context, bool isCpu)
    : ProcessGroup(distBackendOpts.store, distBackendOpts.group_rank,
                   distBackendOpts.group_size),
      options_(std::move(options)),
      isCpu_(isCpu),
      work_tracker_(std::make_shared<MooncakeWorkTracker>()) {
    TORCH_CHECK(context, "Mooncake PG core context is null");

    const int rank = distBackendOpts.group_rank;
    const int size = distBackendOpts.group_size;
    max_group_size_ = options_ && options_->maxGroupSize_ > 0
                          ? options_->maxGroupSize_
                          : size;
    TORCH_CHECK(max_group_size_ >= size && max_group_size_ > 0 &&
                    max_group_size_ <= MOONCAKE_PG_MAX_RANKS,
                "max_group_size must be in [group_size, ",
                MOONCAKE_PG_MAX_RANKS, "]");
    TORCH_CHECK(rank >= 0 && rank < size, "rank out of valid range");
    TORCH_CHECK(!distBackendOpts.group_id.empty(),
                "MooncakeBackend: group_id must not be empty");

    // Use user-provided tensor memory if available. Only its storage is used;
    // the Coordinator populates its contents through the core communicator.
    if (options_ && options_->activeRanks_.defined()) {
        activeRanks_ = options_->activeRanks_;
        TORCH_CHECK(activeRanks_.scalar_type() == at::kInt,
                    "active_ranks must have dtype int32");
        TORCH_CHECK(activeRanks_.is_contiguous(),
                    "active_ranks must be contiguous");
        TORCH_CHECK(activeRanks_.numel() >= max_group_size_,
                    "active_ranks is smaller than max_group_size");
        TORCH_CHECK(activeRanks_.device().is_cpu() == isCpu_,
                    "active_ranks device does not match the backend device");
    } else {
        activeRanks_ =
            at::empty({max_group_size_},
                      torch::dtype(torch::kInt32)
                          .device(isCpu_ ? torch::kCPU : torch::kCUDA));
    }

    std::vector<int32_t> global_ranks;
    if (distBackendOpts.global_ranks_in_group.size() ==
        static_cast<size_t>(size)) {
        global_ranks.reserve(size);
        for (const auto global_rank : distBackendOpts.global_ranks_in_group) {
            global_ranks.push_back(static_cast<int32_t>(global_rank));
        }
    }

    mooncakePgCommConfig_t config = MOONCAKE_PG_COMM_CONFIG_INITIALIZER;

    // PyTorch's group_id is only a bootstrap id. The Coordinator resolves it
    // together with rank order into a process-lifetime GroupId. CPU and device
    // backends use independent namespaces.
    config.groupId = distBackendOpts.group_id.c_str();
    config.rank = rank;
    config.size = size;
    config.maxGroupSize = max_group_size_;
    config.globalRanks = global_ranks.empty() ? nullptr : global_ranks.data();
    config.globalRankCount = global_ranks.size();
    config.deviceIndex = isCpu_ ? -1 : at::cuda::current_device();
    config.deviceType = isCpu_ ? mooncakePgDeviceCpu : mooncakePgDeviceGpu;
    config.idResolvePolicy = options_ && options_->isExtension_
                                 ? mooncakePgIdResolveAttachOrExtend
                                 : mooncakePgIdResolveCreateOrAttach;
    config.autoDeactivateOnFailure =
        options_ ? options_->autoDeactivateOnFailure_
                 : envFlag("MOONCAKE_PG_AUTO_DEACTIVATE_ON_FAILURE", true);
    config.autoSyncOnFailure =
        options_ ? options_->autoSyncOnFailure_
                 : envFlag("MOONCAKE_PG_AUTO_SYNC_ON_FAILURE", true);
    // auto_sync_on_failure requires auto_deactivate_on_failure.
    TORCH_CHECK(!config.autoSyncOnFailure || config.autoDeactivateOnFailure,
                "auto_sync_on_failure requires "
                "auto_deactivate_on_failure=true");
    config.activeRanksMirror = activeRanks_.data_ptr<int32_t>();
    config.activeRanksMirrorCount = static_cast<size_t>(activeRanks_.numel());
    config.activeRanksMirrorIsDevice = isCpu_ ? 0 : 1;

    checkResult(mooncakePgCommCreate(context, &config, &comm_),
                "mooncakePgCommCreate");

    // Register a lightweight Backend shim so PyTorch dispatch can find a
    // registered Backend for this ProcessGroup. The shim delegates supported
    // P2P and collective operations back to this backend.
    const auto device_type =
        isCpu_ ? c10::DeviceType::CPU : c10::DeviceType::CUDA;
    auto shim = c10::make_intrusive<MooncakeP2PShim>(this, max_group_size_);
    setBackend(device_type, BackendType::CUSTOM, shim);
#ifndef MOONCAKE_EP_USE_MUSA
    setDefaultBackend(BackendType::CUSTOM);
#endif
}

MooncakeBackend::~MooncakeBackend() {
    try {
        shutdown();
    } catch (...) {
        // ProcessGroup destructors must not throw.
    }
}

const std::string MooncakeBackend::getBackendName() const { return "mooncake"; }

int MooncakeBackend::getSize() const {
    int size = 0;
    checkResult(mooncakePgCommGetSize(comm_, &size), "mooncakePgCommGetSize");
    return size;
}

c10::intrusive_ptr<c10d::Work> MooncakeBackend::wrapCpuCompletion(
    c10d::OpType opType, mooncakePgCompletion_t completion,
    FailedRanksHint failedRanksHint, std::vector<at::Tensor> keepAlive,
    std::function<void()> finalize) {
    work_tracker_->reapCompleted();
    return c10::make_intrusive<MooncakeWorkCpu>(
        opType, completion, std::move(failedRanksHint), work_tracker_,
        std::move(keepAlive), std::move(finalize));
}

c10::intrusive_ptr<c10d::Work> MooncakeBackend::wrapCudaEvent(
    c10d::OpType opType, std::shared_ptr<c10::Event> event,
    FailedRanksHint failedRanksHint, std::vector<at::Tensor> keepAlive,
    bool isBarrier) {
    if (at::cuda::currentStreamCaptureStatus() ==
        c10::cuda::CaptureStatus::None) {
        work_tracker_->reapCompleted();
    }
    if (isBarrier) {
        return c10::make_intrusive<MooncakeBarrierWorkCuda>(
            opType, std::move(event), std::move(failedRanksHint), work_tracker_,
            std::move(keepAlive));
    }
    return c10::make_intrusive<MooncakeWorkCuda>(
        opType, std::move(event), std::move(failedRanksHint), work_tracker_,
        std::move(keepAlive));
}

c10::intrusive_ptr<c10d::Work> MooncakeBackend::wrapP2PWork(
    mooncakePgCompletion_t completion, FailedRanksHint failedRanksHint,
    std::vector<at::Tensor> keepAlive, std::function<void()> finalize) {
    if (isCpu_ || at::cuda::currentStreamCaptureStatus() ==
                      c10::cuda::CaptureStatus::None) {
        work_tracker_->reapCompleted();
    }
    return c10::make_intrusive<MooncakeP2PWork>(
        completion, std::move(failedRanksHint), work_tracker_,
        std::move(keepAlive), std::move(finalize));
}

c10::intrusive_ptr<c10d::Work> MooncakeBackend::send(
    std::vector<at::Tensor>& tensors, int dstRank, int tag) {
    (void)tag;
    TORCH_CHECK(tensors.size() == 1, kSingleTensorError);
    auto tensor = tensors.back().contiguous();
    auto failed_ranks_hint = FailedRanksHint::allocate(max_group_size_);
    mooncakePgCompletion_t completion = nullptr;
    if (isCpu_) {
        checkResult(mooncakePgSendCpu(tensor.data_ptr(), tensorCount(tensor),
                                      convertDataType(tensor.scalar_type()),
                                      dstRank, comm_, failed_ranks_hint.data(),
                                      static_cast<size_t>(max_group_size_),
                                      &completion),
                    "mooncakePgSendCpu");
    } else {
        const auto stream = currentCudaStream(tensor);
        checkResult(mooncakePgSendGpu(
                        tensor.data_ptr(), tensorCount(tensor),
                        convertDataType(tensor.scalar_type()), dstRank, comm_,
                        coreStream(stream), failed_ranks_hint.data(),
                        static_cast<size_t>(max_group_size_), &completion),
                    "mooncakePgSendGpu");
    }
    return wrapP2PWork(completion, std::move(failed_ranks_hint),
                       retain({tensor}));
}

c10::intrusive_ptr<c10d::Work> MooncakeBackend::recv(
    std::vector<at::Tensor>& tensors, int srcRank, int tag) {
    (void)tag;
    TORCH_CHECK(tensors.size() == 1, kSingleTensorError);
    auto output = tensors.back();
    auto target = output.is_contiguous() ? output : output.contiguous();
    auto failed_ranks_hint = FailedRanksHint::allocate(max_group_size_);
    mooncakePgCompletion_t completion = nullptr;
    if (isCpu_) {
        checkResult(mooncakePgRecvCpu(target.data_ptr(), tensorCount(target),
                                      convertDataType(target.scalar_type()),
                                      srcRank, comm_, failed_ranks_hint.data(),
                                      static_cast<size_t>(max_group_size_),
                                      &completion),
                    "mooncakePgRecvCpu");
    } else {
        const auto stream = currentCudaStream(target);
        checkResult(mooncakePgRecvGpu(
                        target.data_ptr(), tensorCount(target),
                        convertDataType(target.scalar_type()), srcRank, comm_,
                        coreStream(stream), failed_ranks_hint.data(),
                        static_cast<size_t>(max_group_size_), &completion),
                    "mooncakePgRecvGpu");
    }

    std::function<void()> finalize;
    if (!output.is_contiguous()) {
        finalize = [output, target, is_cpu = isCpu_]() mutable {
            output.copy_(target);
            if (!is_cpu) currentCudaStream(output).synchronize();
        };
    }
    return wrapP2PWork(completion, std::move(failed_ranks_hint),
                       retain({output, target}), std::move(finalize));
}

c10::intrusive_ptr<c10d::Work> MooncakeBackend::broadcast(
    std::vector<at::Tensor>& tensors, const c10d::BroadcastOptions& opts) {
    TORCH_CHECK(tensors.size() == 1, kSingleTensorError);
    auto tensor = tensors.back();
    const int root = opts.rootRank + opts.rootTensor;
    auto failed_ranks_hint = FailedRanksHint::allocate(max_group_size_);
    if (isCpu_) {
        mooncakePgCompletion_t completion = nullptr;
        checkResult(
            mooncakePgBroadcastCpu(
                tensor.data_ptr(), tensor.data_ptr(), tensorCount(tensor),
                convertDataType(tensor.scalar_type()), root, comm_,
                failed_ranks_hint.data(), static_cast<size_t>(max_group_size_),
                &completion),
            "mooncakePgBroadcastCpu");
        return wrapCpuCompletion(c10d::OpType::BROADCAST, completion,
                                 std::move(failed_ranks_hint),
                                 retain({tensor}));
    }
    const auto stream = currentCudaStream(tensor);
    checkResult(mooncakePgBroadcastGpu(
                    tensor.data_ptr(), tensor.data_ptr(), tensorCount(tensor),
                    convertDataType(tensor.scalar_type()), root, comm_,
                    coreStream(stream), failed_ranks_hint.data(),
                    static_cast<size_t>(max_group_size_)),
                "mooncakePgBroadcastGpu");
    return wrapCudaEvent(c10d::OpType::BROADCAST, recordCompletionEvent(stream),
                         std::move(failed_ranks_hint), retain({tensor}));
}

c10::intrusive_ptr<c10d::Work> MooncakeBackend::allreduce(
    std::vector<at::Tensor>& tensors, const c10d::AllreduceOptions& opts) {
    TORCH_CHECK(tensors.size() == 1, kSingleTensorError);
    TORCH_CHECK(opts.sparseIndices == std::nullopt, kSparseError);
    auto tensor = tensors.back();
    auto failed_ranks_hint = FailedRanksHint::allocate(max_group_size_);
    if (isCpu_) {
        mooncakePgCompletion_t completion = nullptr;
        checkResult(
            mooncakePgAllReduceCpu(
                tensor.data_ptr(), tensor.data_ptr(), tensorCount(tensor),
                convertDataType(tensor.scalar_type()),
                convertReduceOp(opts.reduceOp), comm_, failed_ranks_hint.data(),
                static_cast<size_t>(max_group_size_), &completion),
            "mooncakePgAllReduceCpu");
        return wrapCpuCompletion(c10d::OpType::ALLREDUCE, completion,
                                 std::move(failed_ranks_hint),
                                 retain({tensor}));
    }
    const auto stream = currentCudaStream(tensor);
    checkResult(
        mooncakePgAllReduceGpu(
            tensor.data_ptr(), tensor.data_ptr(), tensorCount(tensor),
            convertDataType(tensor.scalar_type()),
            convertReduceOp(opts.reduceOp), comm_, coreStream(stream),
            failed_ranks_hint.data(), static_cast<size_t>(max_group_size_)),
        "mooncakePgAllReduceGpu");
    return wrapCudaEvent(c10d::OpType::ALLREDUCE, recordCompletionEvent(stream),
                         std::move(failed_ranks_hint), retain({tensor}));
}

c10::intrusive_ptr<c10d::Work> MooncakeBackend::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors, const c10d::AllgatherOptions&) {
    TORCH_CHECK(inputTensors.size() == 1, kSingleTensorError);
    TORCH_CHECK(outputTensors.size() == 1, kSingleTensorError);
    auto input = inputTensors.back();
    auto outputs = outputTensors.back();
    validatePeerTensors(outputs, input, max_group_size_);
    auto packed_output =
        at::empty({input.numel() * max_group_size_}, input.options());

    auto failed_ranks_hint = FailedRanksHint::allocate(max_group_size_);
    auto keep_alive = retain({input, packed_output});
    appendTensors(keep_alive, outputs);
    auto finalize = unpackPeerTensors(packed_output, std::move(outputs));
    if (isCpu_) {
        mooncakePgCompletion_t completion = nullptr;
        checkResult(
            mooncakePgAllGatherCpu(
                input.data_ptr(), packed_output.data_ptr(), tensorCount(input),
                convertDataType(input.scalar_type()), comm_,
                failed_ranks_hint.data(), static_cast<size_t>(max_group_size_),
                &completion),
            "mooncakePgAllGatherCpu");
        return wrapCpuCompletion(c10d::OpType::ALLGATHER, completion,
                                 std::move(failed_ranks_hint),
                                 std::move(keep_alive), std::move(finalize));
    }
    const auto stream = currentCudaStream(input);
    checkResult(
        mooncakePgAllGatherGpu(
            input.data_ptr(), packed_output.data_ptr(), tensorCount(input),
            convertDataType(input.scalar_type()), comm_, coreStream(stream),
            failed_ranks_hint.data(), static_cast<size_t>(max_group_size_)),
        "mooncakePgAllGatherGpu");
    finalize();
    return wrapCudaEvent(c10d::OpType::ALLGATHER, recordCompletionEvent(stream),
                         std::move(failed_ranks_hint), std::move(keep_alive));
}

c10::intrusive_ptr<c10d::Work> MooncakeBackend::_allgather_base(
    at::Tensor& outputBuffer, at::Tensor& inputBuffer,
    const c10d::AllgatherOptions&) {
    auto failed_ranks_hint = FailedRanksHint::allocate(max_group_size_);
    if (isCpu_) {
        mooncakePgCompletion_t completion = nullptr;
        checkResult(mooncakePgAllGatherCpu(
                        inputBuffer.data_ptr(), outputBuffer.data_ptr(),
                        tensorCount(inputBuffer),
                        convertDataType(inputBuffer.scalar_type()), comm_,
                        failed_ranks_hint.data(),
                        static_cast<size_t>(max_group_size_), &completion),
                    "mooncakePgAllGatherCpu");
        return wrapCpuCompletion(c10d::OpType::_ALLGATHER_BASE, completion,
                                 std::move(failed_ranks_hint),
                                 retain({inputBuffer, outputBuffer}));
    }
    const auto stream = currentCudaStream(inputBuffer);
    checkResult(mooncakePgAllGatherGpu(
                    inputBuffer.data_ptr(), outputBuffer.data_ptr(),
                    tensorCount(inputBuffer),
                    convertDataType(inputBuffer.scalar_type()), comm_,
                    coreStream(stream), failed_ranks_hint.data(),
                    static_cast<size_t>(max_group_size_)),
                "mooncakePgAllGatherGpu");
    return wrapCudaEvent(
        c10d::OpType::_ALLGATHER_BASE, recordCompletionEvent(stream),
        std::move(failed_ranks_hint), retain({inputBuffer, outputBuffer}));
}

c10::intrusive_ptr<c10d::Work> MooncakeBackend::_reduce_scatter_base(
    at::Tensor& outputBuffer, at::Tensor& inputBuffer,
    const c10d::ReduceScatterOptions& opts) {
    auto failed_ranks_hint = FailedRanksHint::allocate(max_group_size_);
    if (isCpu_) {
        mooncakePgCompletion_t completion = nullptr;
        checkResult(
            mooncakePgReduceScatterCpu(
                inputBuffer.data_ptr(), outputBuffer.data_ptr(),
                tensorCount(outputBuffer),
                convertDataType(outputBuffer.scalar_type()),
                convertReduceOp(opts.reduceOp), comm_, failed_ranks_hint.data(),
                static_cast<size_t>(max_group_size_), &completion),
            "mooncakePgReduceScatterCpu");
        return wrapCpuCompletion(c10d::OpType::_REDUCE_SCATTER_BASE, completion,
                                 std::move(failed_ranks_hint),
                                 retain({inputBuffer, outputBuffer}));
    }
    const auto stream = currentCudaStream(outputBuffer);
    checkResult(
        mooncakePgReduceScatterGpu(
            inputBuffer.data_ptr(), outputBuffer.data_ptr(),
            tensorCount(outputBuffer),
            convertDataType(outputBuffer.scalar_type()),
            convertReduceOp(opts.reduceOp), comm_, coreStream(stream),
            failed_ranks_hint.data(), static_cast<size_t>(max_group_size_)),
        "mooncakePgReduceScatterGpu");
    return wrapCudaEvent(
        c10d::OpType::_REDUCE_SCATTER_BASE, recordCompletionEvent(stream),
        std::move(failed_ranks_hint), retain({inputBuffer, outputBuffer}));
}

c10::intrusive_ptr<c10d::Work> MooncakeBackend::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors, const c10d::AllToAllOptions&) {
    TORCH_CHECK(!inputTensors.empty() && !outputTensors.empty(),
                "alltoall requires non-empty Tensor lists");
    const auto reference = inputTensors.front();
    validatePeerTensors(outputTensors, reference, max_group_size_);
    auto packed_input =
        packPeerTensors(inputTensors, reference, max_group_size_);
    auto packed_output =
        at::empty({reference.numel() * max_group_size_}, reference.options());

    auto failed_ranks_hint = FailedRanksHint::allocate(max_group_size_);
    auto keep_alive = retain({packed_input, packed_output});
    appendTensors(keep_alive, inputTensors);
    appendTensors(keep_alive, outputTensors);
    auto finalize = unpackPeerTensors(packed_output, outputTensors);
    if (isCpu_) {
        mooncakePgCompletion_t completion = nullptr;
        checkResult(mooncakePgAllToAllCpu(
                        packed_input.data_ptr(), packed_output.data_ptr(),
                        tensorCount(reference),
                        convertDataType(reference.scalar_type()), comm_,
                        failed_ranks_hint.data(),
                        static_cast<size_t>(max_group_size_), &completion),
                    "mooncakePgAllToAllCpu");
        return wrapCpuCompletion(c10d::OpType::ALLTOALL, completion,
                                 std::move(failed_ranks_hint),
                                 std::move(keep_alive), std::move(finalize));
    }
    const auto stream = currentCudaStream(reference);
    checkResult(
        mooncakePgAllToAllGpu(packed_input.data_ptr(), packed_output.data_ptr(),
                              tensorCount(reference),
                              convertDataType(reference.scalar_type()), comm_,
                              coreStream(stream), failed_ranks_hint.data(),
                              static_cast<size_t>(max_group_size_)),
        "mooncakePgAllToAllGpu");
    finalize();
    return wrapCudaEvent(c10d::OpType::ALLTOALL, recordCompletionEvent(stream),
                         std::move(failed_ranks_hint), std::move(keep_alive));
}

c10::intrusive_ptr<c10d::Work> MooncakeBackend::barrier(
    const c10d::BarrierOptions&) {
    auto failed_ranks_hint = FailedRanksHint::allocate(max_group_size_);
    if (isCpu_) {
        mooncakePgCompletion_t completion = nullptr;
        checkResult(mooncakePgBarrierCpu(comm_, failed_ranks_hint.data(),
                                         static_cast<size_t>(max_group_size_),
                                         &completion),
                    "mooncakePgBarrierCpu");
        return wrapCpuCompletion(c10d::OpType::BARRIER, completion,
                                 std::move(failed_ranks_hint));
    }
    const auto stream = currentCudaStream();
    checkResult(mooncakePgBarrierGpu(comm_, coreStream(stream),
                                     failed_ranks_hint.data(),
                                     static_cast<size_t>(max_group_size_)),
                "mooncakePgBarrierGpu");
    return wrapCudaEvent(c10d::OpType::BARRIER, recordCompletionEvent(stream),
                         std::move(failed_ranks_hint), {}, true);
}

c10::intrusive_ptr<c10d::Work> MooncakeBackend::reduce(
    std::vector<at::Tensor>& tensors, const c10d::ReduceOptions& opts) {
    TORCH_CHECK(tensors.size() == 1, kSingleTensorError);
    auto tensor = tensors.back();
    const int root = opts.rootRank + opts.rootTensor;
    auto failed_ranks_hint = FailedRanksHint::allocate(max_group_size_);
    if (isCpu_) {
        mooncakePgCompletion_t completion = nullptr;
        checkResult(mooncakePgReduceCpu(tensor.data_ptr(), tensor.data_ptr(),
                                        tensorCount(tensor),
                                        convertDataType(tensor.scalar_type()),
                                        convertReduceOp(opts.reduceOp), root,
                                        comm_, failed_ranks_hint.data(),
                                        static_cast<size_t>(max_group_size_),
                                        &completion),
                    "mooncakePgReduceCpu");
        return wrapCpuCompletion(c10d::OpType::REDUCE, completion,
                                 std::move(failed_ranks_hint),
                                 retain({tensor}));
    }
    const auto stream = currentCudaStream(tensor);
    checkResult(
        mooncakePgReduceGpu(
            tensor.data_ptr(), tensor.data_ptr(), tensorCount(tensor),
            convertDataType(tensor.scalar_type()),
            convertReduceOp(opts.reduceOp), root, comm_, coreStream(stream),
            failed_ranks_hint.data(), static_cast<size_t>(max_group_size_)),
        "mooncakePgReduceGpu");
    return wrapCudaEvent(c10d::OpType::REDUCE, recordCompletionEvent(stream),
                         std::move(failed_ranks_hint), retain({tensor}));
}

c10::intrusive_ptr<c10d::Work> MooncakeBackend::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors, const c10d::GatherOptions& opts) {
    TORCH_CHECK(inputTensors.size() == 1, kSingleTensorError);
    const int root = opts.rootRank;
    const bool is_root = root == rank_;
    if (is_root) {
        TORCH_CHECK(outputTensors.size() == 1, kSingleTensorError);
    }
    auto input = inputTensors.back();
    std::vector<at::Tensor> outputs;
    at::Tensor packed_output;
    if (is_root) {
        outputs = outputTensors.back();
        validatePeerTensors(outputs, input, max_group_size_);
        packed_output =
            at::empty({input.numel() * max_group_size_}, input.options());
    }

    auto failed_ranks_hint = FailedRanksHint::allocate(max_group_size_);
    auto keep_alive = retain({input, packed_output});
    appendTensors(keep_alive, outputs);
    auto finalize = packed_output.defined()
                        ? unpackPeerTensors(packed_output, outputs)
                        : std::function<void()>{};
    if (isCpu_) {
        mooncakePgCompletion_t completion = nullptr;
        checkResult(
            mooncakePgGatherCpu(
                input.data_ptr(),
                packed_output.defined() ? packed_output.data_ptr() : nullptr,
                tensorCount(input), convertDataType(input.scalar_type()), root,
                comm_, failed_ranks_hint.data(),
                static_cast<size_t>(max_group_size_), &completion),
            "mooncakePgGatherCpu");
        return wrapCpuCompletion(c10d::OpType::GATHER, completion,
                                 std::move(failed_ranks_hint),
                                 std::move(keep_alive), std::move(finalize));
    }
    const auto stream = currentCudaStream(input);
    checkResult(
        mooncakePgGatherGpu(
            input.data_ptr(),
            packed_output.defined() ? packed_output.data_ptr() : nullptr,
            tensorCount(input), convertDataType(input.scalar_type()), root,
            comm_, coreStream(stream), failed_ranks_hint.data(),
            static_cast<size_t>(max_group_size_)),
        "mooncakePgGatherGpu");
    if (finalize) finalize();
    return wrapCudaEvent(c10d::OpType::GATHER, recordCompletionEvent(stream),
                         std::move(failed_ranks_hint), std::move(keep_alive));
}

c10::intrusive_ptr<c10d::Work> MooncakeBackend::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const c10d::ScatterOptions& opts) {
    TORCH_CHECK(outputTensors.size() == 1, kSingleTensorError);
    const int root = opts.rootRank;
    const bool is_root = root == rank_;
    if (is_root) {
        TORCH_CHECK(inputTensors.size() == 1, kSingleTensorError);
    }
    auto output = outputTensors.back();
    at::Tensor packed_input;
    std::vector<at::Tensor> inputs;
    if (is_root) {
        inputs = inputTensors.back();
        packed_input = packPeerTensors(inputs, output, max_group_size_);
    }

    auto failed_ranks_hint = FailedRanksHint::allocate(max_group_size_);
    auto keep_alive = retain({output, packed_input});
    appendTensors(keep_alive, inputs);
    if (isCpu_) {
        mooncakePgCompletion_t completion = nullptr;
        checkResult(
            mooncakePgScatterCpu(
                packed_input.defined() ? packed_input.data_ptr() : nullptr,
                output.data_ptr(), tensorCount(output),
                convertDataType(output.scalar_type()), root, comm_,
                failed_ranks_hint.data(), static_cast<size_t>(max_group_size_),
                &completion),
            "mooncakePgScatterCpu");
        return wrapCpuCompletion(c10d::OpType::SCATTER, completion,
                                 std::move(failed_ranks_hint),
                                 std::move(keep_alive));
    }
    const auto stream = currentCudaStream(output);
    checkResult(mooncakePgScatterGpu(
                    packed_input.defined() ? packed_input.data_ptr() : nullptr,
                    output.data_ptr(), tensorCount(output),
                    convertDataType(output.scalar_type()), root, comm_,
                    coreStream(stream), failed_ranks_hint.data(),
                    static_cast<size_t>(max_group_size_)),
                "mooncakePgScatterGpu");
    return wrapCudaEvent(c10d::OpType::SCATTER, recordCompletionEvent(stream),
                         std::move(failed_ranks_hint), std::move(keep_alive));
}

void MooncakeBackend::shutdown() {
    if (isShutdown_) return;
    isShutdown_ = true;
    auto comm = std::exchange(comm_, nullptr);
    const auto result = comm ? mooncakePgCommDestroy(comm) : mooncakePgSuccess;
    work_tracker_->shutdown();
    checkResult(result, "mooncakePgCommDestroy");
}

std::string MooncakeBackend::getPreferredHca(
    const std::string& location) const {
    std::array<char, kPreferredHcaBytes> hca_buf{};
    checkResult(mooncakePgCommGetPreferredHca(comm_, location.c_str(),
                                              hca_buf.data(), hca_buf.size()),
                "mooncakePgCommGetPreferredHca");
    return hca_buf.data();
}

void MooncakeBackend::extendGroupSizeTo(int) {
    // Deprecated: in the Coordinator-based path, group size is determined by
    // GroupView.rank_order. This is a no-op stub kept for compatibility.
    TORCH_WARN(
        "MooncakeBackend::extendGroupSizeTo is deprecated; group size "
        "is controlled by the Coordinator's GroupView.");
}

int MooncakeBackend::getNumSyncedRanks() {
    int count = 0;
    checkResult(mooncakePgCommGetNumSyncedRanks(comm_, &count),
                "mooncakePgCommGetNumSyncedRanks");
    return count;
}

std::vector<bool> MooncakeBackend::getPeerState(const std::vector<int>& ranks) {
    const auto core_ranks = toCoreRanks(ranks);
    std::vector<int32_t> core_states(ranks.size(), 0);
    checkResult(
        mooncakePgCommGetPeerState(comm_, core_ranks.data(), core_ranks.size(),
                                   core_states.data()),
        "mooncakePgCommGetPeerState");
    std::vector<bool> result;
    result.reserve(core_states.size());
    for (const int32_t state : core_states) result.push_back(state != 0);
    return result;
}

mooncakePgProposalResponse_t MooncakeBackend::activateRanks(
    const std::vector<int>& ranks) {
    const auto core_ranks = toCoreRanks(ranks);
    mooncakePgProposalResponse_t response{};
    checkResult(mooncakePgCommActivateRanks(comm_, core_ranks.data(),
                                            core_ranks.size(), &response),
                "mooncakePgCommActivateRanks");
    return response;
}

mooncakePgProposalResponse_t MooncakeBackend::deactivateRanks(
    const std::vector<int>& ranks) {
    const auto core_ranks = toCoreRanks(ranks);
    mooncakePgProposalResponse_t response{};
    checkResult(mooncakePgCommDeactivateRanks(comm_, core_ranks.data(),
                                              core_ranks.size(), &response),
                "mooncakePgCommDeactivateRanks");
    return response;
}

void MooncakeBackend::joinGroup() {
    checkResult(mooncakePgCommJoin(comm_), "mooncakePgCommJoin");
}

uint64_t MooncakeBackend::getCurrentEpoch() const {
    uint64_t epoch = 0;
    checkResult(mooncakePgCommGetEpoch(comm_, &epoch),
                "mooncakePgCommGetEpoch");
    return epoch;
}

mooncakePgSyncAfterFailureResponse_t MooncakeBackend::syncAfterFailure() {
    mooncakePgSyncAfterFailureResponse_t response{};
    checkResult(mooncakePgCommSyncAfterFailure(comm_, &response),
                "mooncakePgCommSyncAfterFailure");
    return response;
}

// ---- MooncakeP2PShim implementation ----

MooncakeP2PShim::MooncakeP2PShim(MooncakeBackend* owner, int maxGroupSize)
    : Backend(owner->getRank(), maxGroupSize), owner_(owner) {}

const std::string MooncakeP2PShim::getBackendName() const { return "mooncake"; }

c10::intrusive_ptr<c10d::Work> MooncakeP2PShim::send(
    std::vector<at::Tensor>& tensors, int dstRank, int tag) {
    return owner_->send(tensors, dstRank, tag);
}

c10::intrusive_ptr<c10d::Work> MooncakeP2PShim::recv(
    std::vector<at::Tensor>& tensors, int srcRank, int tag) {
    return owner_->recv(tensors, srcRank, tag);
}

c10::intrusive_ptr<c10d::Work> MooncakeP2PShim::recvAnysource(
    std::vector<at::Tensor>& tensors, int tag) {
    // MooncakeBackend doesn't implement recvAnysource; fall back to the base
    // class which will raise a clear error.
    return ::c10d::Backend::recvAnysource(tensors, tag);
}

c10::intrusive_ptr<c10d::Work> MooncakeP2PShim::barrier(
    const c10d::BarrierOptions& opts) {
    return owner_->barrier(opts);
}

// ---- MooncakeP2PShim collective delegation ----
// PyTorch 2.13's all_gather_single and reduce_scatter_single entry points
// dispatch through the registered Backend. Delegate every supported collective
// so the shim exposes the same capabilities as its owning MooncakeBackend. See
// the class comment in mooncake_backend.h for the full rationale.
c10::intrusive_ptr<c10d::Work> MooncakeP2PShim::broadcast(
    std::vector<at::Tensor>& tensors, const c10d::BroadcastOptions& opts) {
    return owner_->broadcast(tensors, opts);
}

c10::intrusive_ptr<c10d::Work> MooncakeP2PShim::allreduce(
    std::vector<at::Tensor>& tensors, const c10d::AllreduceOptions& opts) {
    return owner_->allreduce(tensors, opts);
}

c10::intrusive_ptr<c10d::Work> MooncakeP2PShim::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors, const c10d::AllgatherOptions& opts) {
    return owner_->allgather(outputTensors, inputTensors, opts);
}

c10::intrusive_ptr<c10d::Work> MooncakeP2PShim::_allgather_base(
    at::Tensor& outputBuffer, at::Tensor& inputBuffer,
    const c10d::AllgatherOptions& opts) {
    return owner_->_allgather_base(outputBuffer, inputBuffer, opts);
}

c10::intrusive_ptr<c10d::Work> MooncakeP2PShim::_reduce_scatter_base(
    at::Tensor& outputBuffer, at::Tensor& inputBuffer,
    const c10d::ReduceScatterOptions& opts) {
    return owner_->_reduce_scatter_base(outputBuffer, inputBuffer, opts);
}

c10::intrusive_ptr<c10d::Work> MooncakeP2PShim::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors, const c10d::AllToAllOptions& opts) {
    return owner_->alltoall(outputTensors, inputTensors, opts);
}

c10::intrusive_ptr<c10d::Work> MooncakeP2PShim::reduce(
    std::vector<at::Tensor>& tensors, const c10d::ReduceOptions& opts) {
    return owner_->reduce(tensors, opts);
}

c10::intrusive_ptr<c10d::Work> MooncakeP2PShim::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors, const c10d::GatherOptions& opts) {
    return owner_->gather(outputTensors, inputTensors, opts);
}

c10::intrusive_ptr<c10d::Work> MooncakeP2PShim::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const c10d::ScatterOptions& opts) {
    return owner_->scatter(outputTensors, inputTensors, opts);
}

}  // namespace mooncake
