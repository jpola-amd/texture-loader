#pragma once

#include <hip/hip_runtime.h>
#include <memory>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <functional>

namespace hip_demand {

class TicketImpl;
std::shared_ptr<TicketImpl> createTicketImpl(std::function<void()> task, hipStream_t stream);

/// A Ticket tracks completion of a set of host-side tasks and can optionally record a HIP event
/// when finished.
class Ticket {
public:
    Ticket();

    // Construct from an existing implementation handle.
    explicit Ticket(std::shared_ptr<TicketImpl> impl);

    /// Returns total task count (always 1 for this loader); -1 if not started.
    int numTasksTotal() const;

    /// Returns remaining tasks (0 or 1). -1 if not started.
    int numTasksRemaining() const;

    /// Blocks until tasks finish. If an event pointer is provided, the event is recorded on the
    /// stream associated with this ticket after host work completes.
    void wait(hipEvent_t* event = nullptr);

private:
    std::shared_ptr<TicketImpl> impl_;

    friend class TicketImpl;
};

}  // namespace hip_demand
