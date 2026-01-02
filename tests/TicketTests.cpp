// SPDX-License-Identifier: MIT
// Unit tests for Ticket

#include "TestUtils.h"
#include <DemandLoading/Ticket.h>

#include <chrono>
#include <thread>

namespace hip_demand {
namespace test {

// ============================================================================
// Construction Tests
// ============================================================================

TEST(TicketTest, DefaultConstruction) {
    Ticket ticket;
    // Default ticket should report -1 for "not started"
    EXPECT_EQ(ticket.numTasksTotal(), 0);
    EXPECT_EQ(ticket.numTasksRemaining(), 0);
}

TEST(TicketTest, MoveConstruction) {
    Ticket ticket1;
    Ticket ticket2(std::move(ticket1));
    EXPECT_EQ(ticket2.numTasksTotal(), 0);
}

TEST(TicketTest, MoveAssignment) {
    Ticket ticket1;
    Ticket ticket2;
    ticket2 = std::move(ticket1);
    EXPECT_EQ(ticket2.numTasksTotal(), 0);
}

// ============================================================================
// Wait Tests
// ============================================================================

TEST(TicketTest, WaitOnDefaultTicket) {
    Ticket ticket;
    // Should not hang or throw on default ticket
    ticket.wait();
    EXPECT_EQ(ticket.numTasksTotal(), 0);
}

TEST(TicketTest, WaitWithEventOnDefaultTicket) {
    Ticket ticket;
    hipEvent_t event = nullptr;
    // Should not hang or throw
    ticket.wait(&event);
    // Event should remain null for invalid ticket
    EXPECT_EQ(event, nullptr);
}

}  // namespace test
}  // namespace hip_demand
