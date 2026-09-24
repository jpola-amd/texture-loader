// SPDX-License-Identifier: MIT
// HIP Error Checking Utilities
#pragma once

#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>
#include <sstream>
#include <stdexcept>

inline void hipCheck( bool throwIfFail, hipError_t err, const char* expr, const char* file, int line )
{
    if( err != hipSuccess )
    {
        std::ostringstream message;
        if( throwIfFail )
        {
            message << "ERROR";
        }
        else
        {
            message << "WARN";
        }
        message << " at " << file << ":" << line
            << " - " << expr << " returned "
            << hipGetErrorString( err ) << " (code "<< err << ")";
        if( throwIfFail )
        {
            throw std::runtime_error( message.str() );
        }
        else
        {
            std::cerr << message.str() << std::endl;
        }
    }
}

// Macro to check HIP API calls and report errors
#define HIP_CHECK( call ) hipCheck( true, call, #call, __FILE__, __LINE__)
// Non-fatal version that just warns but doesn't exit
#define HIP_WARN( call ) hipCheck( false, call, #call, __FILE__, __LINE__ )
// Check last error (useful after kernel launches)
#define HIP_CHECK_LAST() HIP_CHECK( hipGetLastError() )
// Synchronize device and check for errors
#define HIP_CHECK_SYNC() HIP_CHECK( hipDeviceSynchronize() )
