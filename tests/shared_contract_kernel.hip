// SPDX-License-Identifier: MIT
#include <hip/hip_runtime.h>
#include "ContractDeviceTestData.h"
#include <DemandLoading/TextureSampling.h>

extern "C" __global__ void evaluateContract(
    const ContractDeviceCase* input, ContractDeviceResult* output) {
    using namespace hip_demand::contract_v1;
    DeviceContext context{{input->contextVersion, input->contextBytes}, input->incarnation,
                          &input->texture, 1, 0};
    const auto decision = resolve(context, input->key, input->revision, input->lod, input->descriptor);
    output->decision = decision;
    output->abi = AbiInfo{};
    output->needsRequest = decision.needsRequest();
    output->contributes = decision.contributesStrictSample();
}

extern "C" __global__ void sampleLegacyContract(hip_demand::DeviceContext context, float4* output) {
    hip_demand::tex2DLod(context, 0, 0.5f, 0.5f, 0, *output);
}
