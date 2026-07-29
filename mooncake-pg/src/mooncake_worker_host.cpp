// mooncake_worker_host.cpp — Host-side code for PG collectives.
// Compiled by the host C++ compiler for CUDA, MUSA, and MACA builds. Uses
// kernel launch wrappers from mooncake_worker_kernels.cuh instead of <<<>>>
// syntax. This translation unit is deliberately free of ATen/c10 so the core
// can be built without PyTorch.

#include <mooncake_worker.cuh>
#include <mooncake_worker_kernels.cuh>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <thread>

#include "pg_utils.h"

namespace mooncake {
namespace {

class ScopedCudaDeviceGuard {
   public:
    explicit ScopedCudaDeviceGuard(int device) {
        PG_CHECK(device >= 0, "invalid collective CUDA device index");

        const cudaError_t get_device_error = cudaGetDevice(&previous_device_);
        PG_CHECK(get_device_error == cudaSuccess,
                 "failed to get current CUDA device: ",
                 cudaGetErrorString(get_device_error));

        if (previous_device_ == device) return;

        const cudaError_t set_device_error = cudaSetDevice(device);
        PG_CHECK(set_device_error == cudaSuccess,
                 "failed to select collective CUDA device: ",
                 cudaGetErrorString(set_device_error));
        restore_device_ = true;
    }

    ~ScopedCudaDeviceGuard() {
        if (!restore_device_) return;

        const cudaError_t restore_error = cudaSetDevice(previous_device_);
        if (restore_error != cudaSuccess) {
            std::fprintf(stderr,
                         "Mooncake PG failed to restore CUDA device %d: %s\n",
                         previous_device_, cudaGetErrorString(restore_error));
        }
    }

    ScopedCudaDeviceGuard(const ScopedCudaDeviceGuard&) = delete;
    ScopedCudaDeviceGuard& operator=(const ScopedCudaDeviceGuard&) = delete;

   private:
    int previous_device_ = -1;
    bool restore_device_ = false;
};

size_t elementSize(DataType datatype) {
    switch (datatype) {
        case DataType::Uint8:
        case DataType::Int8:
        case DataType::Bool:
            return 1;
        case DataType::Int16:
        case DataType::BFloat16:
            return 2;
        case DataType::Int32:
        case DataType::Float32:
            return 4;
        case DataType::Int64:
        case DataType::Float64:
            return 8;
    }
    throw std::invalid_argument("unsupported Mooncake datatype");
}

template <typename T>
T applyReduceOp(const T& lhs, const T& rhs, ReduceOp op) {
    switch (op) {
        case ReduceOp::Sum:
            return lhs + rhs;
        case ReduceOp::Avg:
            throw std::invalid_argument(
                "Mooncake PG does not support AVG reduction");
        case ReduceOp::Product:
            return lhs * rhs;
        case ReduceOp::Min:
            return std::min(lhs, rhs);
        case ReduceOp::Max:
            return std::max(lhs, rhs);
    }
    throw std::invalid_argument("unsupported Mooncake reduction operation");
}

template <typename T>
void reduceCpu(T* dst, const T* src, size_t num_elements, size_t num_ranks,
               ReduceOp op, const bool* active_ranks) {
    for (size_t element = 0; element < num_elements; ++element) {
        bool valid = false;
        T accumulator{};
        for (size_t rank = 0; rank < num_ranks; ++rank) {
            // Note: failed-ranks hint is intentionally NOT checked here
            // (same rationale as the GPU reduction kernel).
            if (!active_ranks[rank]) continue;
            const T value = src[element + rank * num_elements];
            accumulator = valid ? applyReduceOp(accumulator, value, op) : value;
            valid = true;
        }
        dst[element] = accumulator;
    }
}

}  // namespace

void launchReduceKernel(void* dst, DataType datatype, size_t pos,
                        size_t real_size, void* src, size_t num_ranks,
                        ReduceOp op, bool* active_ranks, cudaStream_t stream) {
    auto* output = static_cast<char*>(dst) + pos;
    const size_t count = real_size / elementSize(datatype);
    const int reduce_op = static_cast<int>(op);

    switch (datatype) {
        case DataType::Uint8:
            launchReduceKernel_uint8(
                static_cast<uint8_t*>(static_cast<void*>(output)),
                static_cast<uint8_t*>(src), count, num_ranks, reduce_op,
                active_ranks, stream);
            return;
        case DataType::Int8:
            launchReduceKernel_int8(
                static_cast<int8_t*>(static_cast<void*>(output)),
                static_cast<int8_t*>(src), count, num_ranks, reduce_op,
                active_ranks, stream);
            return;
        case DataType::Int16:
            launchReduceKernel_int16(
                static_cast<int16_t*>(static_cast<void*>(output)),
                static_cast<int16_t*>(src), count, num_ranks, reduce_op,
                active_ranks, stream);
            return;
        case DataType::Int32:
            launchReduceKernel_int32(
                static_cast<int*>(static_cast<void*>(output)),
                static_cast<int*>(src), count, num_ranks, reduce_op,
                active_ranks, stream);
            return;
        case DataType::Int64:
            launchReduceKernel_int64(
                static_cast<int64_t*>(static_cast<void*>(output)),
                static_cast<int64_t*>(src), count, num_ranks, reduce_op,
                active_ranks, stream);
            return;
        case DataType::Float32:
            launchReduceKernel_float(
                static_cast<float*>(static_cast<void*>(output)),
                static_cast<float*>(src), count, num_ranks, reduce_op,
                active_ranks, stream);
            return;
        case DataType::Float64:
            launchReduceKernel_double(
                static_cast<double*>(static_cast<void*>(output)),
                static_cast<double*>(src), count, num_ranks, reduce_op,
                active_ranks, stream);
            return;
        case DataType::Bool:
            launchReduceKernel_bool(
                static_cast<bool*>(static_cast<void*>(output)),
                static_cast<bool*>(src), count, num_ranks, reduce_op,
                active_ranks, stream);
            return;
        case DataType::BFloat16:
            launchReduceKernel_bf16(output, src, count, num_ranks, reduce_op,
                                    active_ranks, stream);
            return;
    }
    throw std::invalid_argument("unsupported GPU reduction datatype");
}

void launchReduceCpu(void* dst, DataType datatype, size_t pos, size_t real_size,
                     void* src, size_t num_ranks, ReduceOp op,
                     bool* active_ranks) {
    auto* output = static_cast<char*>(dst) + pos;
    const size_t count = real_size / elementSize(datatype);

    switch (datatype) {
        case DataType::Uint8:
            reduceCpu(static_cast<uint8_t*>(static_cast<void*>(output)),
                      static_cast<const uint8_t*>(src), count, num_ranks, op,
                      active_ranks);
            return;
        case DataType::Int8:
            reduceCpu(static_cast<int8_t*>(static_cast<void*>(output)),
                      static_cast<const int8_t*>(src), count, num_ranks, op,
                      active_ranks);
            return;
        case DataType::Int16:
            reduceCpu(static_cast<int16_t*>(static_cast<void*>(output)),
                      static_cast<const int16_t*>(src), count, num_ranks, op,
                      active_ranks);
            return;
        case DataType::Int32:
            reduceCpu(static_cast<int32_t*>(static_cast<void*>(output)),
                      static_cast<const int32_t*>(src), count, num_ranks, op,
                      active_ranks);
            return;
        case DataType::Int64:
            reduceCpu(static_cast<int64_t*>(static_cast<void*>(output)),
                      static_cast<const int64_t*>(src), count, num_ranks, op,
                      active_ranks);
            return;
        case DataType::Float32:
            reduceCpu(static_cast<float*>(static_cast<void*>(output)),
                      static_cast<const float*>(src), count, num_ranks, op,
                      active_ranks);
            return;
        case DataType::Float64:
            reduceCpu(static_cast<double*>(static_cast<void*>(output)),
                      static_cast<const double*>(src), count, num_ranks, op,
                      active_ranks);
            return;
        case DataType::Bool:
            reduceCpu(static_cast<bool*>(static_cast<void*>(output)),
                      static_cast<const bool*>(src), count, num_ranks, op,
                      active_ranks);
            return;
        case DataType::BFloat16:
            throw std::invalid_argument(
                "BFloat16 reduction is not supported by the CPU communicator");
    }
    throw std::invalid_argument("unsupported CPU reduction datatype");
}

MooncakeWorker::MooncakeWorker(int cuda_device_index)
    : cuda_device_index_(cuda_device_index) {
    int device_count = 0;
    const cudaError_t count_error = cudaGetDeviceCount(&device_count);
    if (count_error == cudaSuccess && device_count > 0) {
        PG_CHECK(cudaHostAlloc(&tasks_, kNumTasks_ * sizeof(Task),
                               cudaHostAllocMapped) == cudaSuccess,
                 "failed to allocate collective task slots");
        PG_CHECK(
            cudaHostGetDevicePointer(&tasks_device_, tasks_, 0) == cudaSuccess,
            "failed to map collective task slots");
    } else {
        tasks_ = new Task[kNumTasks_];
        tasks_device_ = tasks_;
    }

    if (cuda_device_index_ >= 0) {
        PG_CHECK(cudaSetDevice(cuda_device_index_) == cudaSuccess,
                 "failed to select collective worker device");
        PG_CHECK(cudaStreamCreateWithFlags(
                     &enqueue_stream_, cudaStreamNonBlocking) == cudaSuccess,
                 "failed to create collective enqueue stream");
    }

    for (size_t i = 0; i < kNumTasks_; ++i) {
        tasks_[i].active = false;
        tasks_[i].submitSequence = 0;
        tasks_[i].failedRanksHint = nullptr;
        tasks_[i].resetFailedRanksHint = false;
        submitted_task_sequence_[i].store(0, std::memory_order_relaxed);
    }
}

MooncakeWorker::~MooncakeWorker() {
    running_ = false;
    if (worker_thread_.joinable()) worker_thread_.join();
    if (enqueue_stream_) cudaStreamDestroy(enqueue_stream_);
}

std::shared_ptr<MooncakeCompletion> MooncakeWorker::putTaskCpu(
    OpType op_type, size_t data_size, int64_t broadcast_root,
    const std::shared_ptr<TransferGroupMeta>& meta, int32_t* failed_ranks_hint,
    const std::function<void(void*, size_t, size_t)>& copy_to_send_buffer,
    const std::function<void(void*, size_t, size_t)>& copy_from_recv_buffer) {
    PG_CHECK(failed_ranks_hint, "failed-ranks hint is null");
    const size_t chunk_size =
        ((kBufferSize - 1) / meta->maxGroupSize) & ~(size_t)7;
    auto completion = std::make_shared<std::promise<void>>();
    auto future = completion->get_future().share();
    auto result = std::make_shared<MooncakeCompletion>(std::move(future));

    struct IterState {
        size_t current_pos = 0;
    };
    auto state = std::make_shared<IterState>();
    auto process_next_chunk = std::make_shared<std::function<void()>>();
    std::weak_ptr<std::function<void()>> weak_process_next_chunk =
        process_next_chunk;

    *process_next_chunk = [this, weak_process_next_chunk, state, op_type,
                           data_size, chunk_size, broadcast_root, meta,
                           copy_to_send_buffer, copy_from_recv_buffer,
                           completion, failed_ranks_hint]() {
        auto next = weak_process_next_chunk.lock();
        if (state->current_pos >= data_size) {
            completion->set_value();
            return;
        }

        const int task_id = cpuTaskCount % 2;
        PG_CHECK(!tasks_[task_id].active,
                 "collective CPU task slot is still active");
        const size_t real_size =
            std::min(chunk_size, data_size - state->current_pos);
        const int buffer_offset = meta->taskCount % 2;
        tasks_[task_id].opType = static_cast<int>(op_type);
        tasks_[task_id].dataSize = real_size;
        tasks_[task_id].broadcastRoot = broadcast_root;
        tasks_[task_id].bufferOffset = buffer_offset;
        tasks_[task_id].submitSequence = 0;
        tasks_[task_id].failedRanksHint = failed_ranks_hint;
        tasks_[task_id].resetFailedRanksHint = state->current_pos == 0;
        tasks_[task_id].transferGroupMeta = meta.get();

        copy_to_send_buffer(
            reinterpret_cast<void*>(
                meta->segmentInfos[meta->rank].send_buffer[buffer_offset]),
            state->current_pos, real_size);
        hasCallback_[task_id] = true;
        callbacks_[task_id] = [next, state, meta, copy_from_recv_buffer,
                               buffer_offset, real_size, completion]() {
            copy_from_recv_buffer(
                reinterpret_cast<void*>(
                    meta->segmentInfos[meta->rank].recv_buffer[buffer_offset]),
                state->current_pos, real_size);
            state->current_pos += real_size;
            (*next)();
        };

        tasks_[task_id].active = true;
        ++cpuTaskCount;
        ++meta->taskCount;
    };

    (*process_next_chunk)();
    return result;
}

void MooncakeWorker::putTaskCuda(
    OpType op_type, size_t data_size, int64_t broadcast_root,
    const std::shared_ptr<TransferGroupMeta>& meta, cudaStream_t issue_stream,
    int32_t* failed_ranks_hint,
    const std::function<void(void*, size_t, size_t, cudaStream_t)>&
        copy_to_send_buffer,
    const std::function<void(void*, size_t, size_t, cudaStream_t)>&
        copy_from_recv_buffer) {
    PG_CHECK(failed_ranks_hint, "failed-ranks hint is null");

    // The null CUDA stream handle denotes the valid legacy/default stream.
    // Select the worker's device before interpreting it or creating events so
    // both default and explicit streams are used in the communicator's context.
    const ScopedCudaDeviceGuard device_guard(cuda_device_index_);

    const size_t chunk_size =
        ((kBufferSize - 1) / meta->maxGroupSize) & ~(size_t)7;

    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    const auto capture_error =
        cudaStreamIsCapturing(issue_stream, &capture_status);
    const bool is_captured = capture_error == cudaSuccess &&
                             capture_status != cudaStreamCaptureStatusNone;

    cudaEvent_t start_event = nullptr;
    PG_CHECK(cudaEventCreateWithFlags(&start_event, cudaEventDisableTiming) ==
                 cudaSuccess,
             "failed to create collective start event");
    PG_CHECK(cudaEventRecord(start_event, issue_stream) == cudaSuccess,
             "failed to record collective start event");
    PG_CHECK(
        cudaStreamWaitEvent(enqueue_stream_, start_event, 0) == cudaSuccess,
        "failed to order collective enqueue stream");
    cudaEventDestroy(start_event);

    std::vector<CudaTaskSubmissionToken> submitted_tasks;
    submitted_tasks.reserve((data_size + chunk_size - 1) / chunk_size);

    for (size_t pos = 0; pos < data_size; pos += chunk_size) {
        const size_t real_size = std::min(data_size, pos + chunk_size) - pos;
        const int task_id = cudaTaskCount % 2 + 2;
        const int buffer_offset = meta->taskCount % 2;
        const uint64_t task_sequence =
            next_cuda_task_sequence_.fetch_add(1, std::memory_order_relaxed);
        submitted_tasks.push_back({.task_id = static_cast<size_t>(task_id),
                                   .sequence = task_sequence});

        copy_to_send_buffer(
            reinterpret_cast<void*>(
                meta->segmentInfos[meta->rank].send_buffer[buffer_offset]),
            pos, real_size, enqueue_stream_);
        hasCallback_[task_id] = false;

        launchEnqueueTaskKernel(static_cast<int>(op_type), real_size,
                                broadcast_root, buffer_offset, task_sequence,
                                failed_ranks_hint, pos == 0, meta.get(),
                                tasks_device_, task_id, enqueue_stream_);
        copy_from_recv_buffer(
            reinterpret_cast<void*>(
                meta->segmentInfos[meta->rank].recv_buffer[buffer_offset]),
            pos, real_size, enqueue_stream_);
        ++cudaTaskCount;
        ++meta->taskCount;
    }

    cudaEvent_t end_event = nullptr;
    PG_CHECK(cudaEventCreateWithFlags(&end_event, cudaEventDisableTiming) ==
                 cudaSuccess,
             "failed to create collective completion event");
    PG_CHECK(cudaEventRecord(end_event, enqueue_stream_) == cudaSuccess,
             "failed to record collective completion event");

    // Wait until each task has been submitted to TransferEngine. This tries to
    // ensure that the device operations required for the transfer have been
    // launched before this GPU collective call returns.
    //
    // Why is this needed? PyTorch documentation implies that collective
    // operations should be enqueued when Work::wait() returns. The core now
    // provides the stronger guarantee before returning the collective call. In
    // practice, violating this ordering causes hangs.
    //
    // Our current hypothesis for the hang is: PyTorch assumes the operations
    // needed for the transfer are already launched when wait() returns. It may
    // then launch a subsequent operation such as `.cpu()`. Such operations may
    // acquire a process-wide CUDA runtime lock and synchronize the caller
    // stream, which is waiting for enqueue_stream_. Holding that runtime lock
    // can prevent cudaMemcpy(Async) in TE/TENT from launching, so the transfer
    // cannot finish and enqueue_stream_ cannot complete.
    //
    // In practice, replacing cudaMemcpyAsync in TENT with cuMemcpyAsync
    // alleviates this, which further suggests a CUDA runtime deadlock. That
    // change is too invasive for TE/TENT, so PG keeps the submission wait.
    //
    // Strictly speaking, the wait is also needed because the caller stream is
    // blocked on end_event below. Any subsequent work on the caller stream will
    // effectively wait for the task to finish. Therefore, all operations needed
    // by the transfer must be launched before blocking that stream.
    //
    // This relies on TE/TENT launching all device operations in submitTransfer.
    // TcpTransport in TE and TENT currently violates that assumption because a
    // cudaMemcpy(Async) may be called later from a callback. That can still
    // hang when an operation such as `x.cpu().item()` follows the collective.
    // TE's TcpTransport use of cudaMemcpy on the default stream may also
    // contribute.
    //
    // For CPU-only transports such as RdmaTransport this submission wait is
    // unnecessary, but it is retained for uniform behavior.
    if (!is_captured) {
        // Normal execution: block until tasks are submitted.
        waitUntilTasksSubmitted(submitted_tasks);
    } else {
        // During CUDA graph capture, kernels are recorded but not actually
        // executed. enqueueTaskKernel would never run, so waiting for task
        // submission would hang because the CPU worker thread never sees
        // task.active == true.
        //
        // This also means NvlinkTransport (and TcpTransport) cannot be fully
        // captured: kernels launched inside TE/TENT are not captured by this
        // graph and are not ordered with graph replay. This may trigger the
        // same deadlock described above.
    }

    // Once all tasks have been submitted, synchronize the caller stream with
    // the enqueue stream through the event without blocking the host.
    PG_CHECK(cudaStreamWaitEvent(issue_stream, end_event, 0) == cudaSuccess,
             "failed to order collective completion on caller stream");
    cudaEventDestroy(end_event);
}

}  // namespace mooncake
