#include <work_handles.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>

#include <algorithm>
#include <chrono>
#include <iterator>
#include <thread>
#include <utility>

namespace mooncake {
namespace {

void checkResult(mooncakePgResult_t result, const char* operation) {
    TORCH_CHECK(result == mooncakePgSuccess, operation,
                " failed: ", mooncakePgGetErrorString(result), ": ",
                mooncakePgGetLastError());
}

int64_t timeoutToMicroseconds(std::chrono::milliseconds timeout) {
    if (timeout.count() < 0) return -1;
    return std::chrono::duration_cast<std::chrono::microseconds>(timeout)
        .count();
}

}  // namespace

FailedRanksHint FailedRanksHint::allocate(int size) {
    auto options =
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    return FailedRanksHint(torch::zeros({size}, options));
}

bool FailedRanksHint::isLocalSuccess() const {
    const auto* values = data();
    return std::all_of(values, values + tensor.numel(),
                       [](int32_t value) { return value == 0; });
}

class MooncakeOperationState {
   public:
    MooncakeOperationState(std::vector<at::Tensor> keep_alive,
                           std::function<void()> finalize)
        : keep_alive_(std::move(keep_alive)), finalize_(std::move(finalize)) {}
    virtual ~MooncakeOperationState() = default;

    virtual bool isCompleted() = 0;
    virtual bool retainUntilShutdown() const noexcept { return false; }

   protected:
    void finalizeOnce() {
        std::call_once(finalize_once_, [this] {
            if (finalize_) finalize_();
        });
    }

   private:
    std::vector<at::Tensor> keep_alive_;
    std::function<void()> finalize_;
    std::once_flag finalize_once_;
};

class MooncakeCpuCompletionState final : public MooncakeOperationState {
   public:
    MooncakeCpuCompletionState(mooncakePgCompletion_t completion,
                               FailedRanksHint failed_ranks_hint,
                               std::vector<at::Tensor> keep_alive,
                               std::function<void()> finalize)
        : MooncakeOperationState(std::move(keep_alive), std::move(finalize)),
          completion_(completion),
          failed_ranks_hint_(std::move(failed_ranks_hint)) {
        TORCH_CHECK(completion_, "Mooncake PG core returned a null completion");
    }

    ~MooncakeCpuCompletionState() override {
        if (completion_) {
            (void)mooncakePgCompletionDestroy(completion_);
        }
    }

    bool isCompleted() override {
        int completed = 0;
        checkResult(mooncakePgCompletionIsCompleted(completion_, &completed),
                    "mooncakePgCompletionIsCompleted");
        if (completed) finalizeOnce();
        return completed != 0;
    }

    bool wait(int64_t timeout_us) {
        const auto result = mooncakePgCompletionWait(completion_, timeout_us);
        if (result == mooncakePgTimeout) return false;
        checkResult(result, "mooncakePgCompletionWait");
        finalizeOnce();
        return true;
    }

    at::Tensor getFailedRanksHint() const { return failed_ranks_hint_.tensor; }

    bool getLocalSuccess() const {
        int completed = 0;
        checkResult(mooncakePgCompletionIsCompleted(completion_, &completed),
                    "mooncakePgCompletionIsCompleted");
        return completed != 0 && failed_ranks_hint_.isLocalSuccess();
    }

   private:
    mooncakePgCompletion_t completion_ = nullptr;
    FailedRanksHint failed_ranks_hint_;
};

class MooncakeCudaWorkState final : public MooncakeOperationState {
   public:
    MooncakeCudaWorkState(std::shared_ptr<c10::Event> event, bool is_captured,
                          FailedRanksHint failed_ranks_hint,
                          std::vector<at::Tensor> keep_alive)
        : MooncakeOperationState(std::move(keep_alive), {}),
          event_(std::move(event)),
          is_captured_(is_captured),
          failed_ranks_hint_(std::move(failed_ranks_hint)) {
        TORCH_CHECK(event_, "Mooncake PG Torch event is null");
    }

    bool isCompleted() override { return event_->query(); }

    bool retainUntilShutdown() const noexcept override { return is_captured_; }

    bool blockCurrentStream() {
        // The core has already submitted the transfer and ordered its
        // completion onto the issue stream. Block the currently active stream
        // on the Torch event, but do not block the host.
        //
        // See PyTorch docs for more details:
        // https://docs.pytorch.org/docs/stable/distributed.html#synchronous-and-asynchronous-collective-operations
        //   "wait() - in the case of CPU collectives, will block the process
        //    until the operation is completed. In the case of CUDA collectives,
        //    will block the currently active CUDA stream until the operation
        //    is completed (but will not block the CPU)."
        event_->block(at::cuda::getCurrentCUDAStream());
        return true;
    }

    bool waitForHost(std::chrono::milliseconds timeout) {
        // Skip host-side synchronization during CUDA graph capture.
        // cudaEventSynchronize is not permitted while a stream is capturing.
        if (at::cuda::currentStreamCaptureStatus() !=
            c10::cuda::CaptureStatus::None) {
            // We still need stream-level synchronization so that subsequent
            // operations on the capture stream are ordered after the barrier
            // task on the enqueue stream.
            return blockCurrentStream();
        }
        if (timeout == kNoTimeout) {
            event_->synchronize();
            return true;
        }
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (!event_->query()) {
            if (std::chrono::steady_clock::now() >= deadline) return false;
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
        return true;
    }

    at::Tensor getFailedRanksHint() {
        synchronizeForHint();
        return failed_ranks_hint_.tensor;
    }

    bool getLocalSuccess() {
        synchronizeForHint();
        return failed_ranks_hint_.isLocalSuccess();
    }

   private:
    void synchronizeForHint() {
        // Ensure the worker thread has completed the task and written the
        // failed-ranks bitmap before returning the tensor.
        if (at::cuda::currentStreamCaptureStatus() ==
            c10::cuda::CaptureStatus::None) {
            event_->synchronize();
        }
    }

    std::shared_ptr<c10::Event> event_;
    bool is_captured_ = false;
    FailedRanksHint failed_ranks_hint_;
};

class MooncakeP2PWorkState final : public MooncakeOperationState {
   public:
    MooncakeP2PWorkState(mooncakePgCompletion_t completion,
                         FailedRanksHint failed_ranks_hint,
                         std::vector<at::Tensor> keep_alive,
                         std::function<void()> finalize)
        : MooncakeOperationState(std::move(keep_alive), std::move(finalize)),
          completion_(completion),
          failed_ranks_hint_(std::move(failed_ranks_hint)) {
        TORCH_CHECK(completion_,
                    "Mooncake PG core returned a null P2P completion");
    }

    ~MooncakeP2PWorkState() override {
        if (completion_) (void)mooncakePgCompletionDestroy(completion_);
    }

    bool isCompleted() override {
        int completed = 0;
        checkResult(mooncakePgCompletionIsCompleted(completion_, &completed),
                    "mooncakePgCompletionIsCompleted");
        if (completed) finalizeIfSuccessful();
        return completed != 0;
    }

    bool isSuccess() const {
        int completed = 0;
        checkResult(mooncakePgCompletionIsCompleted(completion_, &completed),
                    "mooncakePgCompletionIsCompleted");
        return completed != 0 && failed_ranks_hint_.isLocalSuccess();
    }

    bool wait(int64_t timeout_us) {
        const auto result = mooncakePgCompletionWait(completion_, timeout_us);
        if (result == mooncakePgTimeout) return false;
        checkResult(result, "mooncakePgCompletionWait");
        finalizeIfSuccessful();
        return true;
    }

    at::Tensor getFailedRanksHint() const { return failed_ranks_hint_.tensor; }

    bool getLocalSuccess() const { return isSuccess(); }

   private:
    void finalizeIfSuccessful() {
        if (isSuccess()) finalizeOnce();
    }

    mooncakePgCompletion_t completion_ = nullptr;
    FailedRanksHint failed_ranks_hint_;
};

MooncakeWorkTracker::MooncakeWorkTracker() = default;

MooncakeWorkTracker::~MooncakeWorkTracker() { shutdown(); }

void MooncakeWorkTracker::retire(
    std::shared_ptr<MooncakeOperationState> state) noexcept {
    if (!state) return;
    // Work destruction only transfers ownership. In particular, it does not
    // poll a P2P state whose completion finalizer may perform a device copy.
    std::lock_guard<std::mutex> lock(mutex_);
    if (is_shutdown_) return;
    retired_.push_back(std::move(state));
}

void MooncakeWorkTracker::reapCompleted() noexcept {
    std::vector<std::shared_ptr<MooncakeOperationState>> candidates;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (is_shutdown_ || retired_.empty()) return;
        candidates.swap(retired_);
    }

    std::vector<std::shared_ptr<MooncakeOperationState>> pending;
    pending.reserve(candidates.size());
    for (auto& state : candidates) {
        if (state->retainUntilShutdown()) {
            pending.push_back(std::move(state));
            continue;
        }
        try {
            if (state->isCompleted()) continue;
        } catch (...) {
            // Keep resources when the completion state cannot be queried.
        }
        pending.push_back(std::move(state));
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (is_shutdown_) return;
    retired_.insert(retired_.end(), std::make_move_iterator(pending.begin()),
                    std::make_move_iterator(pending.end()));
}

void MooncakeWorkTracker::shutdown() noexcept {
    std::vector<std::shared_ptr<MooncakeOperationState>> retired;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (is_shutdown_) return;
        is_shutdown_ = true;
        retired.swap(retired_);
    }

    // The core communicator is shut down before this method is called. Query
    // once to run any ready CPU/P2P finalizers, then release all retained
    // resources. Captured GPU operations have no host finalizer and are kept
    // until this point solely to cover graph replay.
    for (auto& state : retired) {
        if (state->retainUntilShutdown()) continue;
        try {
            (void)state->isCompleted();
        } catch (...) {
        }
    }
}

MooncakeWorkCpu::MooncakeWorkCpu(c10d::OpType opType,
                                 mooncakePgCompletion_t completion,
                                 FailedRanksHint failedRanksHint,
                                 std::shared_ptr<MooncakeWorkTracker> tracker,
                                 std::vector<at::Tensor> keepAlive,
                                 std::function<void()> finalize)
    : Work(-1, opType),
      tracker_(std::move(tracker)),
      state_(std::make_shared<MooncakeCpuCompletionState>(
          completion, std::move(failedRanksHint), std::move(keepAlive),
          std::move(finalize))) {}

MooncakeWorkCpu::~MooncakeWorkCpu() {
    if (tracker_) tracker_->retire(std::move(state_));
}

bool MooncakeWorkCpu::isCompleted() { return state_->isCompleted(); }

bool MooncakeWorkCpu::wait(std::chrono::milliseconds) {
    // Preserve the existing CPU Work behavior: its timeout argument is
    // ignored and wait blocks until the operation completes.
    return state_->wait(-1);
}

at::Tensor MooncakeWorkCpu::getFailedRanksHint() const {
    return state_->getFailedRanksHint();
}

bool MooncakeWorkCpu::getLocalSuccess() const {
    return state_->getLocalSuccess();
}

MooncakeWorkCuda::MooncakeWorkCuda(c10d::OpType opType,
                                   std::shared_ptr<c10::Event> event,
                                   FailedRanksHint failedRanksHint,
                                   std::shared_ptr<MooncakeWorkTracker> tracker,
                                   std::vector<at::Tensor> keepAlive)
    : Work(-1, opType),
      tracker_(std::move(tracker)),
      state_(std::make_shared<MooncakeCudaWorkState>(
          std::move(event),
          at::cuda::currentStreamCaptureStatus() !=
              c10::cuda::CaptureStatus::None,
          std::move(failedRanksHint), std::move(keepAlive))) {}

MooncakeWorkCuda::~MooncakeWorkCuda() {
    if (tracker_) tracker_->retire(std::move(state_));
}

bool MooncakeWorkCuda::isCompleted() { return state_->isCompleted(); }

bool MooncakeWorkCuda::wait(std::chrono::milliseconds) {
    return state_->blockCurrentStream();
}

at::Tensor MooncakeWorkCuda::getFailedRanksHint() const {
    return state_->getFailedRanksHint();
}

bool MooncakeWorkCuda::getLocalSuccess() const {
    return state_->getLocalSuccess();
}

bool MooncakeBarrierWorkCuda::wait(std::chrono::milliseconds timeout) {
    return state_->waitForHost(timeout);
}

MooncakeP2PWork::MooncakeP2PWork(mooncakePgCompletion_t completion,
                                 FailedRanksHint failedRanksHint,
                                 std::shared_ptr<MooncakeWorkTracker> tracker,
                                 std::vector<at::Tensor> keepAlive,
                                 std::function<void()> finalize)
    : Work(-1, c10d::OpType::UNKNOWN),
      tracker_(std::move(tracker)),
      state_(std::make_shared<MooncakeP2PWorkState>(
          completion, std::move(failedRanksHint), std::move(keepAlive),
          std::move(finalize))) {}

MooncakeP2PWork::~MooncakeP2PWork() {
    if (tracker_) tracker_->retire(std::move(state_));
}

bool MooncakeP2PWork::isCompleted() { return state_->isCompleted(); }

bool MooncakeP2PWork::isSuccess() const { return state_->isSuccess(); }

bool MooncakeP2PWork::wait(std::chrono::milliseconds timeout) {
    const int64_t timeout_us =
        timeout.count() > 0 ? timeoutToMicroseconds(timeout) : -1;
    return state_->wait(timeout_us);
}

at::Tensor MooncakeP2PWork::getFailedRanksHint() const {
    return state_->getFailedRanksHint();
}

bool MooncakeP2PWork::getLocalSuccess() const {
    return state_->getLocalSuccess();
}

}  // namespace mooncake
