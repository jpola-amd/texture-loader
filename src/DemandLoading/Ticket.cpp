#include "DemandLoading/Ticket.h"

#include <thread>
#include <queue>

namespace hip_demand {

class TicketImpl : public std::enable_shared_from_this<TicketImpl> {
public:
    int numTasksTotal() const { return 1; }

    int numTasksRemaining() const {
        return done_.load(std::memory_order_acquire) ? 0 : 1;
    }

    void wait(hipEvent_t* event) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return done_.load(std::memory_order_acquire); });
        if (event && stream_) {
            hipEventRecord(*event, stream_);
        }
    }

    explicit TicketImpl(hipStream_t stream) : stream_(stream) {}

    void markDone() {
        done_.store(true, std::memory_order_release);
        cv_.notify_all();
    }

    hipStream_t stream_ = nullptr;
    std::atomic<bool> done_{false};
    std::mutex mutex_;
    std::condition_variable cv_;
};

// Single worker thread for all async ticket tasks
class TicketWorker {
public:
    static TicketWorker& instance() {
        static TicketWorker worker;
        return worker;
    }

    void enqueue(std::shared_ptr<TicketImpl> impl, std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks_.emplace(std::move(impl), std::move(task));
        }
        cv_.notify_one();
    }

    ~TicketWorker() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_one();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

private:
    TicketWorker() {
        worker_ = std::thread([this] {
            while (true) {
                std::pair<std::shared_ptr<TicketImpl>, std::function<void()>> task;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                    if (stop_ && tasks_.empty()) {
                        return;
                    }
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
                
                // Execute task and mark done (outside the lock)
                try {
                    task.second();
                } catch (...) {
                    // Prevent exception from escaping; task failures should be
                    // communicated through the loader's error reporting mechanism
                }
                task.first->markDone();
            }
        });
    }

    std::thread worker_;
    std::queue<std::pair<std::shared_ptr<TicketImpl>, std::function<void()>>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_ = false;
};

std::shared_ptr<TicketImpl> createTicketImpl(std::function<void()> task, hipStream_t stream) {
    auto impl = std::make_shared<TicketImpl>(stream);
    TicketWorker::instance().enqueue(impl, std::move(task));
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
