#include "TicketImpl.h"
#include <DemandLoading/VmmDemandTextureLoader.h>

namespace hip_demand::vmm {

int Ticket::numTasksTotal() const
{
    return impl_ ? impl_->numTasksTotal() : 0;
}

int Ticket::numTasksRemaining() const
{
    return impl_ ? impl_->numTasksRemaining() : 0;
}

void Ticket::wait( hipEvent_t* event )
{
    if( impl_ )
        impl_->wait( event );
}

}  // namespace hip_demand::vmm