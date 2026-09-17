// SPDX-License-Identifier: MIT
#include <DemandLoading/CubicContext.h>
#include <gtest/gtest.h>

TEST( CubicContract, AdditiveWireLayoutsAndDefaults )
{
    using namespace hip_demand;
    EXPECT_EQ( sizeof( DeviceContext ), 48u );
    EXPECT_EQ( sizeof( whole_mip_v1::DeviceContext ), 56u );
    EXPECT_EQ( sizeof( whole_mip_v1::Entry ), 128u );
    EXPECT_EQ( sizeof( cubic_v1::Config ), 24u );
    EXPECT_EQ( sizeof( cubic_v1::Entry ), 392u );
    EXPECT_EQ( sizeof( cubic_v1::DeviceContext ), 136u );
    const cubic_v1::Config config;
    EXPECT_EQ( config.abi.version, 1u );
    EXPECT_EQ( config.abi.byteSize, sizeof( config ) );
    EXPECT_EQ( config.spatial, cubic_v1::SpatialMode::Bicubic );
    EXPECT_EQ( config.maxAnisotropy, 1 );
    EXPECT_EQ( config.conservative, 0u );
    const cubic_v1::DeviceContext context;
    EXPECT_EQ( context.entries, nullptr );
    EXPECT_EQ( context.count, 0u );
}
