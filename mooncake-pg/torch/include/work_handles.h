#ifndef MOONCAKE_WORK_HANDLES_H
#define MOONCAKE_WORK_HANDLES_H

#include <mooncake_pg.h>

#include <c10/core/Event.h>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/torch.h>

#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace mooncake {

// Per-operation failedRanksHint buffer
struct FailedRanksHint {
    at::Tensor tensor;

    FailedRanksHint() = default;
    explicit FailedRanksHint(at::Tensor tensor_in)
        : tensor(std::move(tensor_in)) {}

    int32_t* data() { return tensor.data_ptr<int32_t>(); }
    const int32_t* data() const { return tensor.data_ptr<int32_t>(); }

    bool isLocalSuccess() const;

    static FailedRanksHint allocate(int size);
};

class MooncakeOperationState;
class MooncakeCpuCompletionState;
class MooncakeCudaWorkState;
class MooncakeP2PWorkState;

// Owns resources for operations whose c10d::Work was released before the
// operation completed. Reaping is deliberately opportunistic so Tensor
// destruction and completion finalizers run on a user thread. Captured CUDA
// operations remain retained until shutdown because their graph may replay.
class MooncakeWorkTracker final {
   public:
    MooncakeWorkTracker();
    ~MooncakeWorkTracker();

    void reapCompleted() noexcept;
    void shutdown() noexcept;

   private:
    friend class MooncakeWorkCpu;
    friend class MooncakeWorkCuda;
    friend class MooncakeP2PWork;

    void retire(std::shared_ptr<MooncakeOperationState> state) noexcept;

    std::mutex mutex_;
    std::vector<std::shared_ptr<MooncakeOperationState>> retired_;
    bool is_shutdown_ = false;
};

// Collective Work handles
class MooncakeWorkCpu : public ::c10d::Work {
   public:
    MooncakeWorkCpu(c10d::OpType opType, mooncakePgCompletion_t completion,
                    FailedRanksHint failedRanksHint,
                    std::shared_ptr<MooncakeWorkTracker> tracker,
                    std::vector<at::Tensor> keepAlive = {},
                    std::function<void()> finalize = {});
    ~MooncakeWorkCpu() override;

    bool isCompleted() override;
    bool wait(std::chrono::milliseconds timeout) override;

    at::Tensor getFailedRanksHint() const;
    bool getLocalSuccess() const;

   private:
    std::shared_ptr<MooncakeWorkTracker> tracker_;
    std::shared_ptr<MooncakeCpuCompletionState> state_;
};

class MooncakeWorkCuda : public ::c10d::Work {
   public:
    MooncakeWorkCuda(c10d::OpType opType, std::shared_ptr<c10::Event> event,
                     FailedRanksHint failedRanksHint,
                     std::shared_ptr<MooncakeWorkTracker> tracker,
                     std::vector<at::Tensor> keepAlive = {});
    ~MooncakeWorkCuda() override;

    bool isCompleted() override;
    bool wait(std::chrono::milliseconds timeout) override;

    at::Tensor getFailedRanksHint() const;
    bool getLocalSuccess() const;

   protected:
    std::shared_ptr<MooncakeWorkTracker> tracker_;
    std::shared_ptr<MooncakeCudaWorkState> state_;
};

class MooncakeBarrierWorkCuda : public MooncakeWorkCuda {
   public:
    using MooncakeWorkCuda::MooncakeWorkCuda;
    bool wait(std::chrono::milliseconds timeout) override;
};

// P2P Work handle
class MooncakeP2PWork : public ::c10d::Work {
   public:
    MooncakeP2PWork(mooncakePgCompletion_t completion,
                    FailedRanksHint failedRanksHint,
                    std::shared_ptr<MooncakeWorkTracker> tracker,
                    std::vector<at::Tensor> keepAlive = {},
                    std::function<void()> finalize = {});
    ~MooncakeP2PWork() override;

    bool isCompleted() override;
    bool isSuccess() const override;
    bool wait(std::chrono::milliseconds timeout) override;
    at::Tensor getFailedRanksHint() const;
    bool getLocalSuccess() const;

   private:
    std::shared_ptr<MooncakeWorkTracker> tracker_;
    std::shared_ptr<MooncakeP2PWorkState> state_;
};

}  // namespace mooncake

#endif  // MOONCAKE_WORK_HANDLES_H
