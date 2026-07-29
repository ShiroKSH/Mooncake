#ifndef MOONCAKE_PG_TYPES_H
#define MOONCAKE_PG_TYPES_H

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <utility>

namespace mooncake {

enum class OpType : uint8_t {
    Unknown = 0,
    Broadcast,
    AllReduce,
    AllGather,
    ReduceScatter,
    AllToAll,
    Barrier,
    Reduce,
    Gather,
    Scatter,
    Send,
    Recv,
};

enum class DataType : uint8_t {
    Uint8 = 0,
    Int8,
    Int16,
    Int32,
    Int64,
    Float32,
    Float64,
    Bool,
    BFloat16,
};

enum class ReduceOp : uint8_t {
    Sum = 0,
    Avg = 1,
    Product = 2,
    Min = 3,
    Max = 4,
};

class MooncakeCompletion {
   public:
    explicit MooncakeCompletion(std::shared_future<void> completion)
        : completion_(std::move(completion)) {}

    bool isCompleted() const {
        if (completion_.wait_for(std::chrono::microseconds(0)) !=
            std::future_status::ready) {
            return false;
        }
        completion_.get();
        return true;
    }

    bool wait(std::chrono::microseconds timeout) const {
        if (timeout.count() < 0) {
            completion_.wait();
        } else if (completion_.wait_for(timeout) != std::future_status::ready) {
            return false;
        }
        completion_.get();
        return true;
    }

   private:
    std::shared_future<void> completion_;
};

struct CudaTaskSubmissionToken {
    size_t task_id;
    uint64_t sequence;
};

}  // namespace mooncake

#endif  // MOONCAKE_PG_TYPES_H
