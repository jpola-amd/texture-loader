#include "DemandLoading/Ticket.h"

#include <thread>

namespace hip_demand {

class TicketImpl : public std::enable_shared_from_this<TicketImpl> {
public:
    int numTasksTotal() const { return started_ ? 1 : -1; }

    int numTasksRemaining() const {
        if (!started_) return -1;
        return done_.load(std::memory_order_acquire) ? 0 : 1;
    }

    void wait(hipEvent_t* event) {
        started_ = true;
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return done_.load(std::memory_order_acquire); });
        if (event && stream_) {
            hipEventRecord(*event, stream_);
        }
    }

    explicit TicketImpl(hipStream_t stream) : stream_(stream) {}

    void markDone() {
        started_ = true;
        done_.store(true, std::memory_order_release);
        cv_.notify_all();
    }

    hipStream_t stream_ = nullptr;
    std::atomic<bool> done_{false};
    bool started_ = true;
    std::mutex mutex_;
    std::condition_variable cv_;
};

std::shared_ptr<TicketImpl> createTicketImpl(std::function<void()> task, hipStream_t stream) {
    auto impl = std::make_shared<TicketImpl>(stream);
    std::thread([impl, task = std::move(task)]() {
        task();
        impl->markDone();
    }).detach();
    return impl;
}

Ticket::Ticket() = default;
Ticket::Ticket(std::shared_ptr<TicketImpl> impl) : impl_(std::move(impl)) {}

int Ticket::numTasksTotal() const {
    return impl_ ? impl_->numTasksTotal() : 0;
}

int Ticket::numTasksRemaining() const {
    return impl_ ? impl_->numTasksRemaining() : 0;
}

void Ticket::wait(hipEvent_t* event) {
    if (impl_) {
        impl_->wait(event);
    }
}

}  // namespace hip_demand
