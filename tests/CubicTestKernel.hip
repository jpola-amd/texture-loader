// SPDX-License-Identifier: MIT
#include "CubicTestData.h"
#include <DemandLoading/CubicSampling.h>
extern "C" __global__ void sampleCubic( hip_demand::cubic_v1::DeviceContext context,
                                        const hip_demand::test::CubicInput* inputs,
                                        hip_demand::test::CubicOutput*      outputs,
                                        uint32_t                            count )
{
    const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if( id >= count )
        return;
    const auto                    in = inputs[id];
    hip_demand::test::CubicOutput out;
    float4*                       v = in.outputs & 1 ? &out.value : nullptr;
    float4*                       s = in.outputs & 2 ? &out.ds : nullptr;
    float4*                       t = in.outputs & 4 ? &out.dt : nullptr;
    if( v )
        *v = make_float4( 9, 9, 9, 9 );
    if( s )
        *s = make_float4( 9, 9, 9, 9 );
    if( t )
        *t = make_float4( 9, 9, 9, 9 );
    namespace c  = hip_demand::cubic_v1;
    out.decision = in.gradient == 2 ?
                       c::tex2D( context, in.key, in.revision, in.config, in.s, in.t, v, s, t ) :
                   in.gradient == 1 ?
                       c::tex2DGrad( context, in.key, in.revision, in.config, in.s, in.t, in.dx, in.dy, v, s, t, in.jitter ) :
                       c::tex2DLod( context, in.key, in.revision, in.config, in.s, in.t, in.lod, v, s, t, in.jitter );
    if( out.decision.contributesStrictSample() )
    {
        const auto& entry = context.entries[in.key.slot];
        const auto p = c::detail::plan( entry, in.config, in.gradient == 2 ? 0 : in.lod, in.gradient == 1, in.dx, in.dy, s || t );
        out.lod    = p.lod;
        out.low    = p.low;
        out.high   = p.high;
        out.weight = p.cubic;
        out.dx     = p.dx;
        out.dy     = p.dy;
        // Separate native-only negative control; never used as a smart oracle.
        if( in.nativeControl )
            out.nativeValue =
                ::tex2DLod<float4>( reinterpret_cast<hipTextureObject_t>( entry.texture.textureObject ), in.s, in.t, 0 );
    }

    outputs[id] = out;
}

// Independently supplied prepared gradients; no candidate smart implementation
// or candidate LOD/preparation function is called by this native-only control.
extern "C" __global__ void sampleCubicNativePrimitive( hipTextureObject_t                  object,
                                                       const hip_demand::test::CubicInput* inputs,
                                                       hip_demand::test::CubicOutput*      outputs,
                                                       uint32_t                            count )
{
    const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if( id >= count )
        return;
    const auto                    in = inputs[id];
    hip_demand::test::CubicOutput out;
    out.value   = ::tex2DGrad<float4>( object, in.s, in.t, in.dx, in.dy );
    outputs[id] = out;
}
