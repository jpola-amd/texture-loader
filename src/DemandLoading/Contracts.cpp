// SPDX-License-Identifier: MIT
#include <DemandLoading/ContractState.h>

namespace hip_demand {
namespace contract_v1 {

Outcome allocateLoaderIncarnation(uint64_t& incarnation) {
    static NonWrappingCounter counter;
    return counter.next(incarnation);
}

} // namespace contract_v1
} // namespace hip_demand

extern "C" uint32_t hipDemandGetContractAbiV1(
    uint32_t requestedVersion, uint32_t outputBytes, hip_demand::contract_v1::AbiInfo* output) noexcept {
    using namespace hip_demand::contract_v1;
    if (requestedVersion != Version || outputBytes != sizeof(AbiInfo))
        return static_cast<uint32_t>(Outcome::AbiMismatch);
    if (!output)
        return static_cast<uint32_t>(Outcome::InvalidInput);
    *output = AbiInfo{};
    return static_cast<uint32_t>(Outcome::Success);
}
