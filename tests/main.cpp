// SPDX-License-Identifier: MIT
// Main entry point for unit tests

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <iostream>

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    // Print HIP device info
    int deviceCount = 0;
    hipError_t err = hipGetDeviceCount(&deviceCount);
    if (err == hipSuccess && deviceCount > 0) {
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, 0);
        std::cout << "Running tests on: " << prop.name << std::endl;
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << std::endl;
    } else {
        std::cerr << "Warning: No HIP devices found. GPU tests will be skipped." << std::endl;
    }
    
    return RUN_ALL_TESTS();
}
