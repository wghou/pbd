/**
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 *  This file implements common mathematical operations on vector types
 *  (float3, float4 etc.) since these are not provided as standard by CUDA.
 *
 *  The syntax is modeled on the Cg standard library.
 *
 *  This is part of the Helper library includes
 *
 *    Thanks to Linh Hah for additions and fixes.
 */

#ifndef HELPER_MATH_H
#define HELPER_MATH_H

#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>
#include <thrust/functional.h>

//typedef unsigned int uint;
typedef uint32_t uint;
typedef unsigned short ushort;
const uint kMaxUint = 0xffffffff;
const  int kMaxInt  = 0x7fffffff;

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

#ifndef __CUDACC__
#include <math.h>

#ifndef __host__
#define __host__
#define __device__
#endif

////////////////////////////////////////////////////////////////////////////////
// host implementations of CUDA functions
////////////////////////////////////////////////////////////////////////////////

inline float fminf(float a, float b)
{
    return a < b ? a : b;
}

inline float fmaxf(float a, float b)
{
    return a > b ? a : b;
}

inline int max(int a, int b)
{
    return a > b ? a : b;
}

inline int min(int a, int b)
{
    return a < b ? a : b;
}

inline float rsqrtf(float x)
{
    return 1.0f / sqrtf(x);
}
#endif

////////////////////////////////////////////////////////////////////////////////
// constructors
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 make_float2(float s)
{
    return make_float2(s, s);
}
inline __host__ __device__ float2 make_float2(float3 a)
{
    return make_float2(a.x, a.y);
}
inline __host__ __device__ float2 make_float2(int2 a)
{
    return make_float2(float(a.x), float(a.y));
}
inline __host__ __device__ float2 make_float2(uint2 a)
{
    return make_float2(float(a.x), float(a.y));
}

inline __host__ __device__ int2 make_int2(int s)
{
    return make_int2(s, s);
}
inline __host__ __device__ int2 make_int2(int3 a)
{
    return make_int2(a.x, a.y);
}
inline __host__ __device__ int2 make_int2(uint2 a)
{
    return make_int2(int(a.x), int(a.y));
}
inline __host__ __device__ int2 make_int2(float2 a)
{
    return make_int2(int(a.x), int(a.y));
}

inline __host__ __device__ uint2 make_uint2(uint s)
{
    return make_uint2(s, s);
}
inline __host__ __device__ uint2 make_uint2(uint3 a)
{
    return make_uint2(a.x, a.y);
}
inline __host__ __device__ uint2 make_uint2(int2 a)
{
    return make_uint2(uint(a.x), uint(a.y));
}
inline __host__ __device__ uint2 make_uint2(float2 a)
{
    return make_uint2(make_int2(a));
}
inline __host__ __device__ float3 make_float3(float s)
{
    return make_float3(s, s, s);
}
inline __host__ __device__ float3 make_float3(float2 a)
{
    return make_float3(a.x, a.y, 0.0f);
}
inline __host__ __device__ float3 make_float3(float2 a, float s)
{
    return make_float3(a.x, a.y, s);
}
inline __host__ __device__ float3 make_float3(float4 a)
{
    return make_float3(a.x, a.y, a.z);
}
inline __host__ __device__ float3 make_float3(int3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}
inline __host__ __device__ float3 make_float3(uint3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}

inline __host__ __device__ int3 make_int3(int s)
{
    return make_int3(s, s, s);
}
inline __host__ __device__ int3 make_int3(int2 a)
{
    return make_int3(a.x, a.y, 0);
}
inline __host__ __device__ int3 make_int3(int2 a, int s)
{
    return make_int3(a.x, a.y, s);
}
inline __host__ __device__ int3 make_int3(uint3 a)
{
    return make_int3(int(a.x), int(a.y), int(a.z));
}
inline __host__ __device__ int3 make_int3(float3 a)
{
    return make_int3(int(a.x), int(a.y), int(a.z));
}

inline __host__ __device__ uint3 make_uint3(uint s)
{
    return make_uint3(s, s, s);
}
inline __host__ __device__ uint3 make_uint3(uint2 a)
{
    return make_uint3(a.x, a.y, 0);
}
inline __host__ __device__ uint3 make_uint3(uint2 a, uint s)
{
    return make_uint3(a.x, a.y, s);
}
inline __host__ __device__ uint3 make_uint3(uint4 a)
{
    return make_uint3(a.x, a.y, a.z);
}
inline __host__ __device__ uint3 make_uint3(int3 a)
{
    return make_uint3(uint(a.x), uint(a.y), uint(a.z));
}
inline __host__ __device__ uint3 make_uint3(float3 a)
{
    return make_uint3(make_int3(a));
}

inline __host__ __device__ float4 make_float4(float s)
{
    return make_float4(s, s, s, s);
}
inline __host__ __device__ float4 make_float4(float3 a)
{
    return make_float4(a.x, a.y, a.z, 0.0f);
}
inline __host__ __device__ float4 make_float4(float3 a, float w)
{
    return make_float4(a.x, a.y, a.z, w);
}
inline __host__ __device__ float4 make_float4(int4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}
inline __host__ __device__ float4 make_float4(uint4 a)
{
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}

inline __host__ __device__ int4 make_int4(int s)
{
    return make_int4(s, s, s, s);
}
inline __host__ __device__ int4 make_int4(int3 a)
{
    return make_int4(a.x, a.y, a.z, 0);
}
inline __host__ __device__ int4 make_int4(int3 a, int w)
{
    return make_int4(a.x, a.y, a.z, w);
}
inline __host__ __device__ int4 make_int4(uint4 a)
{
    return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));
}
inline __host__ __device__ int4 make_int4(float4 a)
{
    return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));
}


inline __host__ __device__ uint4 make_uint4(uint s)
{
    return make_uint4(s, s, s, s);
}
inline __host__ __device__ uint4 make_uint4(uint3 a)
{
    return make_uint4(a.x, a.y, a.z, 0);
}
inline __host__ __device__ uint4 make_uint4(uint3 a, uint w)
{
    return make_uint4(a.x, a.y, a.z, w);
}
inline __host__ __device__ uint4 make_uint4(int4 a)
{
    return make_uint4(uint(a.x), uint(a.y), uint(a.z), uint(a.w));
}
inline __host__ __device__ uint4 make_uint4(float4 a)
{
    return make_uint4(make_int4(a));
}
////////////////////////////////////////////////////////////////////////////////
// negate
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator-(float2 &a)
{
    return make_float2(-a.x, -a.y);
}
inline __host__ __device__ int2 operator-(int2 &a)
{
    return make_int2(-a.x, -a.y);
}
inline __host__ __device__ float3 operator-(float3 &a)
{
    return make_float3(-a.x, -a.y, -a.z);
}
inline __host__ __device__ int3 operator-(int3 &a)
{
    return make_int3(-a.x, -a.y, -a.z);
}
inline __host__ __device__ float4 operator-(float4 &a)
{
    return make_float4(-a.x, -a.y, -a.z, -a.w);
}
inline __host__ __device__ int4 operator-(int4 &a)
{
    return make_int4(-a.x, -a.y, -a.z, -a.w);
}

////////////////////////////////////////////////////////////////////////////////
// addition
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator+(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(float2 &a, float2 b)
{
    a.x += b.x;
    a.y += b.y;
}
inline __host__ __device__ float2 operator+(float2 a, float b)
{
    return make_float2(a.x + b, a.y + b);
}
inline __host__ __device__ float2 operator+(float b, float2 a)
{
    return make_float2(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(float2 &a, float b)
{
    a.x += b;
    a.y += b;
}

inline __host__ __device__ int2 operator+(int2 a, int2 b)
{
    return make_int2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(int2 &a, int2 b)
{
    a.x += b.x;
    a.y += b.y;
}
inline __host__ __device__ int2 operator+(int2 a, int b)
{
    return make_int2(a.x + b, a.y + b);
}
inline __host__ __device__ int2 operator+(int b, int2 a)
{
    return make_int2(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(int2 &a, int b)
{
    a.x += b;
    a.y += b;
}

inline __host__ __device__ uint2 operator+(uint2 a, uint2 b)
{
    return make_uint2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(uint2 &a, uint2 b)
{
    a.x += b.x;
    a.y += b.y;
}
inline __host__ __device__ uint2 operator+(uint2 a, uint b)
{
    return make_uint2(a.x + b, a.y + b);
}
inline __host__ __device__ uint2 operator+(uint b, uint2 a)
{
    return make_uint2(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(uint2 &a, uint b)
{
    a.x += b;
    a.y += b;
}


inline __host__ __device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(float3 &a, float3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline __host__ __device__ float3 operator+(float3 a, float b)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(float3 &a, float b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

inline __host__ __device__ int3 operator+(int3 a, int3 b)
{
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(int3 &a, int3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline __host__ __device__ int3 operator+(int3 a, int b)
{
    return make_int3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(int3 &a, int b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

inline __host__ __device__ uint3 operator+(uint3 a, uint3 b)
{
    return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(uint3 &a, uint3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline __host__ __device__ uint3 operator+(uint3 a, uint b)
{
    return make_uint3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(uint3 &a, uint b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

inline __host__ __device__ int3 operator+(int b, int3 a)
{
    return make_int3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ uint3 operator+(uint b, uint3 a)
{
    return make_uint3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ float3 operator+(float b, float3 a)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}

inline __host__ __device__ float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline __host__ __device__ void operator+=(float4 &a, float4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
inline __host__ __device__ float4 operator+(float4 a, float b)
{
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ float4 operator+(float b, float4 a)
{
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ void operator+=(float4 &a, float b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

inline __host__ __device__ int4 operator+(int4 a, int4 b)
{
    return make_int4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline __host__ __device__ void operator+=(int4 &a, int4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
inline __host__ __device__ int4 operator+(int4 a, int b)
{
    return make_int4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
inline __host__ __device__ int4 operator+(int b, int4 a)
{
    return make_int4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
inline __host__ __device__ void operator+=(int4 &a, int b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

inline __host__ __device__ uint4 operator+(uint4 a, uint4 b)
{
    return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
inline __host__ __device__ void operator+=(uint4 &a, uint4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
inline __host__ __device__ uint4 operator+(uint4 a, uint b)
{
    return make_uint4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
inline __host__ __device__ uint4 operator+(uint b, uint4 a)
{
    return make_uint4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
inline __host__ __device__ void operator+=(uint4 &a, uint b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    a.w += b;
}

////////////////////////////////////////////////////////////////////////////////
// subtract
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator-(float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(float2 &a, float2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
inline __host__ __device__ float2 operator-(float2 a, float b)
{
    return make_float2(a.x - b, a.y - b);
}
inline __host__ __device__ float2 operator-(float b, float2 a)
{
    return make_float2(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(float2 &a, float b)
{
    a.x -= b;
    a.y -= b;
}

inline __host__ __device__ int2 operator-(int2 a, int2 b)
{
    return make_int2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(int2 &a, int2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
inline __host__ __device__ int2 operator-(int2 a, int b)
{
    return make_int2(a.x - b, a.y - b);
}
inline __host__ __device__ int2 operator-(int b, int2 a)
{
    return make_int2(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(int2 &a, int b)
{
    a.x -= b;
    a.y -= b;
}

inline __host__ __device__ uint2 operator-(uint2 a, uint2 b)
{
    return make_uint2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(uint2 &a, uint2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
inline __host__ __device__ uint2 operator-(uint2 a, uint b)
{
    return make_uint2(a.x - b, a.y - b);
}
inline __host__ __device__ uint2 operator-(uint b, uint2 a)
{
    return make_uint2(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(uint2 &a, uint b)
{
    a.x -= b;
    a.y -= b;
}

inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(float3 &a, float3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline __host__ __device__ float3 operator-(float3 a, float b)
{
    return make_float3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ float3 operator-(float b, float3 a)
{
    return make_float3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(float3 &a, float b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline __host__ __device__ int3 operator-(int3 a, int3 b)
{
    return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(int3 &a, int3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline __host__ __device__ int3 operator-(int3 a, int b)
{
    return make_int3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ int3 operator-(int b, int3 a)
{
    return make_int3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(int3 &a, int b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline __host__ __device__ uint3 operator-(uint3 a, uint3 b)
{
    return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(uint3 &a, uint3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline __host__ __device__ uint3 operator-(uint3 a, uint b)
{
    return make_uint3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ uint3 operator-(uint b, uint3 a)
{
    return make_uint3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(uint3 &a, uint b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline __host__ __device__ float4 operator-(float4 a, float4 b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline __host__ __device__ void operator-=(float4 &a, float4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
inline __host__ __device__ float4 operator-(float4 a, float b)
{
    return make_float4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
inline __host__ __device__ void operator-=(float4 &a, float b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

inline __host__ __device__ int4 operator-(int4 a, int4 b)
{
    return make_int4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline __host__ __device__ void operator-=(int4 &a, int4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
inline __host__ __device__ int4 operator-(int4 a, int b)
{
    return make_int4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
inline __host__ __device__ int4 operator-(int b, int4 a)
{
    return make_int4(b - a.x, b - a.y, b - a.z, b - a.w);
}
inline __host__ __device__ void operator-=(int4 &a, int b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

inline __host__ __device__ uint4 operator-(uint4 a, uint4 b)
{
    return make_uint4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline __host__ __device__ void operator-=(uint4 &a, uint4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
inline __host__ __device__ uint4 operator-(uint4 a, uint b)
{
    return make_uint4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
inline __host__ __device__ uint4 operator-(uint b, uint4 a)
{
    return make_uint4(b - a.x, b - a.y, b - a.z, b - a.w);
}
inline __host__ __device__ void operator-=(uint4 &a, uint b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

////////////////////////////////////////////////////////////////////////////////
// multiply
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator*(float2 a, float2 b)
{
    return make_float2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(float2 &a, float2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}
inline __host__ __device__ float2 operator*(float2 a, float b)
{
    return make_float2(a.x * b, a.y * b);
}
inline __host__ __device__ float2 operator*(float b, float2 a)
{
    return make_float2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(float2 &a, float b)
{
    a.x *= b;
    a.y *= b;
}

inline __host__ __device__ int2 operator*(int2 a, int2 b)
{
    return make_int2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(int2 &a, int2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}
inline __host__ __device__ int2 operator*(int2 a, int b)
{
    return make_int2(a.x * b, a.y * b);
}
inline __host__ __device__ int2 operator*(int b, int2 a)
{
    return make_int2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(int2 &a, int b)
{
    a.x *= b;
    a.y *= b;
}

inline __host__ __device__ uint2 operator*(uint2 a, uint2 b)
{
    return make_uint2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(uint2 &a, uint2 b)
{
    a.x *= b.x;
    a.y *= b.y;
}
inline __host__ __device__ uint2 operator*(uint2 a, uint b)
{
    return make_uint2(a.x * b, a.y * b);
}
inline __host__ __device__ uint2 operator*(uint b, uint2 a)
{
    return make_uint2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(uint2 &a, uint b)
{
    a.x *= b;
    a.y *= b;
}

inline __host__ __device__ float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(float3 &a, float3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline __host__ __device__ float3 operator*(float3 a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ float3 operator*(float b, float3 a)
{
    return make_float3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(float3 &a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline __host__ __device__ int3 operator*(int3 a, int3 b)
{
    return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(int3 &a, int3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline __host__ __device__ int3 operator*(int3 a, int b)
{
    return make_int3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ int3 operator*(int b, int3 a)
{
    return make_int3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(int3 &a, int b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline __host__ __device__ uint3 operator*(uint3 a, uint3 b)
{
    return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(uint3 &a, uint3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline __host__ __device__ uint3 operator*(uint3 a, uint b)
{
    return make_uint3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ uint3 operator*(uint b, uint3 a)
{
    return make_uint3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(uint3 &a, uint b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline __host__ __device__ float4 operator*(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
inline __host__ __device__ void operator*=(float4 &a, float4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline __host__ __device__ float4 operator*(float4 a, float b)
{
    return make_float4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
inline __host__ __device__ float4 operator*(float b, float4 a)
{
    return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(float4 &a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

inline __host__ __device__ int4 operator*(int4 a, int4 b)
{
    return make_int4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
inline __host__ __device__ void operator*=(int4 &a, int4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline __host__ __device__ int4 operator*(int4 a, int b)
{
    return make_int4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
inline __host__ __device__ int4 operator*(int b, int4 a)
{
    return make_int4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(int4 &a, int b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

inline __host__ __device__ uint4 operator*(uint4 a, uint4 b)
{
    return make_uint4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}
inline __host__ __device__ void operator*=(uint4 &a, uint4 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
inline __host__ __device__ uint4 operator*(uint4 a, uint b)
{
    return make_uint4(a.x * b, a.y * b, a.z * b,  a.w * b);
}
inline __host__ __device__ uint4 operator*(uint b, uint4 a)
{
    return make_uint4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(uint4 &a, uint b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

////////////////////////////////////////////////////////////////////////////////
// divide
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator/(float2 a, float2 b)
{
    return make_float2(a.x / b.x, a.y / b.y);
}
inline __host__ __device__ void operator/=(float2 &a, float2 b)
{
    a.x /= b.x;
    a.y /= b.y;
}
inline __host__ __device__ float2 operator/(float2 a, float b)
{
    return make_float2(a.x / b, a.y / b);
}
inline __host__ __device__ void operator/=(float2 &a, float b)
{
    a.x /= b;
    a.y /= b;
}
inline __host__ __device__ float2 operator/(float b, float2 a)
{
    return make_float2(b / a.x, b / a.y);
}

inline __host__ __device__ float3 operator/(float3 a, float3 b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ void operator/=(float3 &a, float3 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}
inline __host__ __device__ float3 operator/(float3 a, float b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}
inline __host__ __device__ void operator/=(float3 &a, float b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
}
inline __host__ __device__ float3 operator/(float b, float3 a)
{
    return make_float3(b / a.x, b / a.y, b / a.z);
}

inline __host__ __device__ float4 operator/(float4 a, float4 b)
{
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z,  a.w / b.w);
}
inline __host__ __device__ void operator/=(float4 &a, float4 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    a.w /= b.w;
}
inline __host__ __device__ float4 operator/(float4 a, float b)
{
    return make_float4(a.x / b, a.y / b, a.z / b,  a.w / b);
}
inline __host__ __device__ void operator/=(float4 &a, float b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
    a.w /= b;
}
inline __host__ __device__ float4 operator/(float b, float4 a)
{
    return make_float4(b / a.x, b / a.y, b / a.z, b / a.w);
}

inline __host__ __device__ uint3 operator/(uint3 a,uint b)
{
    return make_uint3(a.x / b, a.y / b, a.z / b);
}

////////////////////////////////////////////////////////////////////////////////
// bitwise operator
////////////////////////////////////////////////////////////////////////////////
inline __host__ __device__
uint3
operator & (uint3 const &_v1, uint3 const &_v2)
{
    return make_uint3(_v1.x & _v2.x, _v1.y & _v2.y, _v1.z & _v2.z);
}

////////////////////////////////////////////////////////////////////////////////
// min
////////////////////////////////////////////////////////////////////////////////

inline  __host__ __device__ float2 fminf(float2 a, float2 b)
{
    return make_float2(fminf(a.x,b.x), fminf(a.y,b.y));
}
inline __host__ __device__ float3 fminf(float3 a, float3 b)
{
    return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
}
inline  __host__ __device__ float4 fminf(float4 a, float4 b)
{
    return make_float4(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z), fminf(a.w,b.w));
}

inline __host__ __device__ int2 min(int2 a, int2 b)
{
    return make_int2(min(a.x,b.x), min(a.y,b.y));
}
inline __host__ __device__ int3 min(int3 a, int3 b)
{
    return make_int3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
}
inline __host__ __device__ int4 min(int4 a, int4 b)
{
    return make_int4(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z), min(a.w,b.w));
}

inline __host__ __device__ uint2 min(uint2 a, uint2 b)
{
    return make_uint2(min(a.x,b.x), min(a.y,b.y));
}
inline __host__ __device__ uint3 min(uint3 a, uint3 b)
{
    return make_uint3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
}
inline __host__ __device__ uint4 min(uint4 a, uint4 b)
{
    return make_uint4(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z), min(a.w,b.w));
}

////////////////////////////////////////////////////////////////////////////////
// max
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 fmaxf(float2 a, float2 b)
{
    return make_float2(fmaxf(a.x,b.x), fmaxf(a.y,b.y));
}
inline __host__ __device__ float3 fmaxf(float3 a, float3 b)
{
    return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}
inline __host__ __device__ float4 fmaxf(float4 a, float4 b)
{
    return make_float4(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z), fmaxf(a.w,b.w));
}

inline __host__ __device__ int2 max(int2 a, int2 b)
{
    return make_int2(max(a.x,b.x), max(a.y,b.y));
}
inline __host__ __device__ int3 max(int3 a, int3 b)
{
    return make_int3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
}
inline __host__ __device__ int4 max(int4 a, int4 b)
{
    return make_int4(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z), max(a.w,b.w));
}

inline __host__ __device__ uint2 max(uint2 a, uint2 b)
{
    return make_uint2(max(a.x,b.x), max(a.y,b.y));
}
inline __host__ __device__ uint3 max(uint3 a, uint3 b)
{
    return make_uint3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
}
inline __host__ __device__ uint4 max(uint4 a, uint4 b)
{
    return make_uint4(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z), max(a.w,b.w));
}

////////////////////////////////////////////////////////////////////////////////
// max_comp
////////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ int max_comp(int3 v)
{
    return max(max(v.x, v.y), v.z);
}

inline __host__ __device__ uint max_comp(uint3 v)
{
    return max(max(v.x, v.y), v.z);
}

inline __host__ __device__ float max_comp(float3 v)
{
    return fmaxf(fmaxf(v.x, v.y), v.z);
}

////////////////////////////////////////////////////////////////////////////////
// lerp
// - linear interpolation between a and b, based on value t in [0, 1] range
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}
inline __device__ __host__ float2 lerp(float2 a, float2 b, float t)
{
    return a + t*(b-a);
}
inline __device__ __host__ float3 lerp(float3 a, float3 b, float t)
{
    return a + t*(b-a);
}
inline __device__ __host__ float4 lerp(float4 a, float4 b, float t)
{
    return a + t*(b-a);
}

////////////////////////////////////////////////////////////////////////////////
// clamp
// - clamp the value v to be in the range [a, b]
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}
inline __device__ __host__ int clamp(int f, int a, int b)
{
    return max(a, min(f, b));
}
inline __device__ __host__ uint clamp(uint f, uint a, uint b)
{
    return max(a, min(f, b));
}

inline __device__ __host__ float2 clamp(float2 v, float a, float b)
{
    return make_float2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline __device__ __host__ float2 clamp(float2 v, float2 a, float2 b)
{
    return make_float2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline __device__ __host__ float3 clamp(float3 v, float a, float b)
{
    return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __device__ __host__ float3 clamp(float3 v, float3 a, float3 b)
{
    return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline __device__ __host__ float4 clamp(float4 v, float a, float b)
{
    return make_float4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline __device__ __host__ float4 clamp(float4 v, float4 a, float4 b)
{
    return make_float4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

inline __device__ __host__ int2 clamp(int2 v, int a, int b)
{
    return make_int2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline __device__ __host__ int2 clamp(int2 v, int2 a, int2 b)
{
    return make_int2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline __device__ __host__ int3 clamp(int3 v, int a, int b)
{
    return make_int3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __device__ __host__ int3 clamp(int3 v, int3 a, int3 b)
{
    return make_int3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline __device__ __host__ int4 clamp(int4 v, int a, int b)
{
    return make_int4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline __device__ __host__ int4 clamp(int4 v, int4 a, int4 b)
{
    return make_int4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

inline __device__ __host__ uint2 clamp(uint2 v, uint a, uint b)
{
    return make_uint2(clamp(v.x, a, b), clamp(v.y, a, b));
}
inline __device__ __host__ uint2 clamp(uint2 v, uint2 a, uint2 b)
{
    return make_uint2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
inline __device__ __host__ uint3 clamp(uint3 v, uint a, uint b)
{
    return make_uint3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}
inline __device__ __host__ uint3 clamp(uint3 v, uint3 a, uint3 b)
{
    return make_uint3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
inline __device__ __host__ uint4 clamp(uint4 v, uint a, uint b)
{
    return make_uint4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}
inline __device__ __host__ uint4 clamp(uint4 v, uint4 a, uint4 b)
{
    return make_uint4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// dot product
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float dot(float2 a, float2 b)
{
    return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ float dot(float4 a, float4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline __host__ __device__ int dot(int2 a, int2 b)
{
    return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ int dot(int3 a, int3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ int dot(int4 a, int4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline __host__ __device__ uint dot(uint2 a, uint2 b)
{
    return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ uint dot(uint3 a, uint3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ uint dot(uint4 a, uint4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

////////////////////////////////////////////////////////////////////////////////
// length
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float length(float2 v)
{
    return sqrtf(dot(v, v));
}
inline __host__ __device__ float length(float3 v)
{
    return sqrtf(dot(v, v));
}
inline __host__ __device__ float length(float4 v)
{
    return sqrtf(dot(v, v));
}

////////////////////////////////////////////////////////////////////////////////
// distance
////////////////////////////////////////////////////////////////////////////////
template <typename T>
inline __host__ __device__
float
distance(T const &_p1, T const &_p2)
{
    return length(_p1 - _p2);
}


////////////////////////////////////////////////////////////////////////////////
// normalize
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 normalize(float2 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}
inline __host__ __device__ float3 normalize(float3 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}
inline __host__ __device__ float4 normalize(float4 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

template <typename T>
__host__ __device__ inline T SafeNormalize(const T& v, const T& fallback = T())
{
    float l = dot(v, v);
    if (l > 0.0f)
    {
        return v * rsqrtf(l);
    }
    else
        return fallback;
}

////////////////////////////////////////////////////////////////////////////////
// floor
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 floorf(float2 v)
{
    return make_float2(floorf(v.x), floorf(v.y));
}
inline __host__ __device__ float3 floorf(float3 v)
{
    return make_float3(floorf(v.x), floorf(v.y), floorf(v.z));
}
inline __host__ __device__ float4 floorf(float4 v)
{
    return make_float4(floorf(v.x), floorf(v.y), floorf(v.z), floorf(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// ceil
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 ceilf(float2 v)
{
    return make_float2(ceilf(v.x), ceilf(v.y));
}
inline __host__ __device__ float3 ceilf(float3 v)
{
    return make_float3(ceilf(v.x), ceilf(v.y), ceilf(v.z));
}
inline __host__ __device__ float4 ceilf(float4 v)
{
    return make_float4(ceilf(v.x), ceilf(v.y), ceilf(v.z), ceilf(v.w));
}


inline __host__ __device__ int ceil_div2(int v)
{
    return (v >> 1) + (v & 1);
}

////////////////////////////////////////////////////////////////////////////////
// frac - returns the fractional portion of a scalar or each vector component
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float fracf(float v)
{
    return v - floorf(v);
}
inline __host__ __device__ float2 fracf(float2 v)
{
    return make_float2(fracf(v.x), fracf(v.y));
}
inline __host__ __device__ float3 fracf(float3 v)
{
    return make_float3(fracf(v.x), fracf(v.y), fracf(v.z));
}
inline __host__ __device__ float4 fracf(float4 v)
{
    return make_float4(fracf(v.x), fracf(v.y), fracf(v.z), fracf(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// fmod
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 fmodf(float2 a, float2 b)
{
    return make_float2(fmodf(a.x, b.x), fmodf(a.y, b.y));
}
inline __host__ __device__ float3 fmodf(float3 a, float3 b)
{
    return make_float3(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z));
}
inline __host__ __device__ float4 fmodf(float4 a, float4 b)
{
    return make_float4(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z), fmodf(a.w, b.w));
}

////////////////////////////////////////////////////////////////////////////////
// absolute value
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 fabs(float2 v)
{
    return make_float2(fabs(v.x), fabs(v.y));
}
inline __host__ __device__ float3 fabs(float3 v)
{
    return make_float3(fabs(v.x), fabs(v.y), fabs(v.z));
}
inline __host__ __device__ float4 fabs(float4 v)
{
    return make_float4(fabs(v.x), fabs(v.y), fabs(v.z), fabs(v.w));
}

inline __host__ __device__ int2 abs(int2 v)
{
    return make_int2(abs(v.x), abs(v.y));
}
inline __host__ __device__ int3 abs(int3 v)
{
    return make_int3(abs(v.x), abs(v.y), abs(v.z));
}
inline __host__ __device__ int4 abs(int4 v)
{
    return make_int4(abs(v.x), abs(v.y), abs(v.z), abs(v.w));
}

////////////////////////////////////////////////////////////////////////////////
// reflect
// - returns reflection of incident ray I around surface normal N
// - N should be normalized, reflected vector's length is equal to length of I
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float3 reflect(float3 i, float3 n)
{
    return i - 2.0f * n * dot(n,i);
}

////////////////////////////////////////////////////////////////////////////////
// cross product
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float3 cross(float3 a, float3 b)
{
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

////////////////////////////////////////////////////////////////////////////////
// smoothstep
// - returns 0 if x < a
// - returns 1 if x > b
// - otherwise returns smooth interpolation between 0 and 1 based on x
////////////////////////////////////////////////////////////////////////////////

inline __device__ __host__ float smoothstep(float a, float b, float x)
{
    float y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y*y*(3.0f - (2.0f*y)));
}
inline __device__ __host__ float2 smoothstep(float2 a, float2 b, float2 x)
{
    float2 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y*y*(make_float2(3.0f) - (make_float2(2.0f)*y)));
}
inline __device__ __host__ float3 smoothstep(float3 a, float3 b, float3 x)
{
    float3 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y*y*(make_float3(3.0f) - (make_float3(2.0f)*y)));
}
inline __device__ __host__ float4 smoothstep(float4 a, float4 b, float4 x)
{
    float4 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
    return (y*y*(make_float4(3.0f) - (make_float4(2.0f)*y)));
}


////////////////////////////////////////////////////////////////////////////////
// sign function
////////////////////////////////////////////////////////////////////////////////

// sgn(0) = 0
template <typename T> 
__device__ __host__
int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

// sign(0) = 1
template <typename T> 
__device__ __host__
int sign(T val) {
    return (T(0) <= val) - (val < T(0));
}

inline
__device__ __host__
float clamp_to_zero(float val, float min = FLT_EPSILON) {
    return fabsf(val) > min ? val : 0.f;
}

////////////////////////////////////////////////////////////////////////////////
// bit operation
////////////////////////////////////////////////////////////////////////////////
// return an unsigned integer of type T with n least significant bits set to 1
// return 0 if n = 0
template <typename T>
__device__ __host__
T 
n_LSB(uint n)
{
    if (n >= sizeof(T) * 8) return ~static_cast<T>(0);
    return ~(~static_cast<T>(0) << n);
}


template <typename T>
__device__ __host__
T
nth_bit(T x, uint n)
{
    if (n >= sizeof(T) * 8) return static_cast<T>(0);
    return (x & (static_cast<T>(1) << n));
}

////////////////////////////////////////////////////////////////////////////////
// vector types for accessing components through indices
////////////////////////////////////////////////////////////////////////////////

template<typename T, typename Ts, int n>
union _cuTVec {
    T v;
    Ts a[n];

    __device__ __host__ _cuTVec() {}

    __device__ __host__ _cuTVec(T const &_v) {
        v = _v;
    }

    __device__ __host__ _cuTVec & operator = (T const &_v) {
        v = _v;
        return *this;
    }

    __device__ __host__ Ts& operator[] (int i) {
        return a[i];
    }

    __device__ __host__ Ts const &operator[] (int i) const {
        return a[i];
    }

};

template<typename T, typename Ts, int n>
__device__ __host__
_cuTVec<T, Ts, n> operator+ (_cuTVec<T, Ts, n> const &_v1, _cuTVec<T, Ts, n> const &_v2)
{
    return _cuTVec<T, Ts, n>{_v1.v + _v2.v};
}

template<typename T, typename Ts, int n>
__device__ __host__
_cuTVec<T, Ts, n> operator+ (_cuTVec<T, Ts, n> const &_v1,T const &_v2)
{
    return _cuTVec<T, Ts, n>{_v1.v + _v2};
}

template<typename T, typename Ts, int n>
__device__ __host__
_cuTVec<T, Ts, n> operator- (_cuTVec<T, Ts, n> const &_v1, T const &_v2)
{
    return _cuTVec<T, Ts, n>{_v1.v - _v2};
}

template<typename T, typename Ts, int n>
__device__ __host__
_cuTVec<T, Ts, n> operator- (_cuTVec<T, Ts, n> const &_v1, _cuTVec<T, Ts, n> const &_v2)
{
    return _cuTVec<T, Ts, n>{_v1.v - _v2.v};
}

template<typename T, typename Ts, int n>
__device__ __host__
_cuTVec<T, Ts, n> operator* (Ts _s, _cuTVec<T, Ts, n> const &_v)
{
    return _cuTVec<T, Ts, n>{_s * _v.v};
}

template<typename T, typename Ts, int n>
__device__ __host__
_cuTVec<T, Ts, n> operator* (_cuTVec<T, Ts, n> const &_v1, _cuTVec<T, Ts, n> const &_v2)
{
    return _cuTVec<T, Ts, n>{_v1.v * _v2.v};
}

template<typename T, typename Ts, int n>
__device__ __host__
_cuTVec<T, Ts, n> operator/ (_cuTVec<T, Ts, n> const &_v1, _cuTVec<T, Ts, n> const &_v2)
{
    return _cuTVec<T, Ts, n>{_v1.v / _v2.v};
}

template<typename T, typename Ts, int n>
__device__ __host__
Ts dot(_cuTVec<T, Ts, n> const &_v1, _cuTVec<T, Ts, n> const &_v2)
{
    return dot(_v1.v, _v2.v);
}

template<typename T, typename Ts, int n>
__device__ __host__
_cuTVec<T, Ts, n> cross(_cuTVec<T, Ts, n> const &_v1, _cuTVec<T, Ts, n> const &_v2)
{
    return _cuTVec<T, Ts, n>{cross(_v1.v, _v2.v)};
}

template<typename T, typename Ts, int n>
__device__ __host__
_cuTVec<T, Ts, n> fminf(_cuTVec<T, Ts, n> const &_v1, _cuTVec<T, Ts, n> const &_v2)
{
    return _cuTVec<T, Ts, n>{fminf(_v1.v, _v2.v)};
}

template<typename T, typename Ts, int n>
__device__ __host__
_cuTVec<T, Ts, n> fmin(_cuTVec<T, Ts, n> const &_v1, _cuTVec<T, Ts, n> const &_v2)
{
    return _cuTVec<T, Ts, n>{fmin(_v1.v, _v2.v)};
}

template<typename T, typename Ts, int n>
__device__ __host__
_cuTVec<T, Ts, n> min(_cuTVec<T, Ts, n> const &_v1, _cuTVec<T, Ts, n> const &_v2)
{
    return _cuTVec<T, Ts, n>{min(_v1.v, _v2.v)};
}

template<typename T, typename Ts, int n>
__device__ __host__
_cuTVec<T, Ts, n> fmaxf(_cuTVec<T, Ts, n> const &_v1, _cuTVec<T, Ts, n> const &_v2)
{
    return _cuTVec<T, Ts, n>{fmaxf(_v1.v, _v2.v)};
}

template<typename T, typename Ts, int n>
__device__ __host__
_cuTVec<T, Ts, n> fmax(_cuTVec<T, Ts, n> const &_v1, _cuTVec<T, Ts, n> const &_v2)
{
    return _cuTVec<T, Ts, n>{fmax(_v1.v, _v2.v)};
}

template<typename T, typename Ts, int n>
__device__ __host__
_cuTVec<T, Ts, n> max(_cuTVec<T, Ts, n> const &_v1, _cuTVec<T, Ts, n> const &_v2)
{
    return _cuTVec<T, Ts, n>{max(_v1.v, _v2.v)};
}

template<typename T, typename Ts, int n>
__device__ __host__
uint index_of_min(_cuTVec<T, Ts, n> const &_v)
{
    uint idx = 0;
    for (uint i = 1; i < n; ++i) {
        if (_v[i] < _v[idx]) idx = i;
    }
    return idx;
}

template<typename T, typename Ts, int n>
__device__ __host__
uint index_of_max(_cuTVec<T, Ts, n> const &_v)
{
    uint idx = 0;
    for (uint i = 1; i < n; ++i) {
        if (_v[i] > _v[idx]) idx = i;
    }
    return idx;
}

template<typename T, typename Ts, int n>
__device__ __host__
_cuTVec<T, Ts, n> normalize(_cuTVec<T, Ts, n> const &_v)
{
    return _cuTVec<T, Ts, n>{normalize(_v.v)};
}

typedef _cuTVec<float2, float, 2> cuVec2;
typedef _cuTVec<float3, float, 3> cuVec3;
typedef _cuTVec<float4, float, 4> cuVec4;

typedef _cuTVec<int2, int, 2> cuiVec2;
typedef _cuTVec<int3, int, 3> cuiVec3;
typedef _cuTVec<int4, int, 4> cuiVec4;

typedef _cuTVec<uint2, uint, 2> cuuVec2;
typedef _cuTVec<uint3, uint, 3> cuuVec3;
typedef _cuTVec<uint4, uint, 4> cuuVec4;

inline __device__ __host__
cuVec2 fmaxf(cuVec2 const& _v1, cuVec2 const& _v2, cuVec2 const& _v3)
{
    return cuVec2{ fmaxf(fmaxf(_v1.v, _v2.v), _v3.v) };
}

inline __device__ __host__
cuVec3 fmaxf(cuVec3 const& _v1, cuVec3 const& _v2, cuVec3 const& _v3)
{
    return cuVec3{ fmaxf(fmaxf(_v1.v, _v2.v), _v3.v) };
}

inline __device__ __host__
cuVec2 fminf(cuVec2 const& _v1, cuVec2 const& _v2, cuVec2 const& _v3)
{
    return cuVec2{ fminf(fminf(_v1.v, _v2.v), _v3.v) };
}

inline __device__ __host__
cuVec3 fminf(cuVec3 const& _v1, cuVec3 const& _v2, cuVec3 const& _v3)
{
    return cuVec3{ fminf(fminf(_v1.v, _v2.v), _v3.v) };
}

inline __device__ __host__
cuiVec3 floor(cuVec3 const& _v1)
{
    int3 vi = make_int3(floorf(_v1.v));
    return cuiVec3{ vi };
}

// return unsigned int vector
inline __device__ __host__
cuuVec3 flooru(cuVec3 const& _v1)
{
    uint3 vu = make_uint3(make_int3(floorf(_v1.v)));
    return cuuVec3{vu};
}

inline __device__ __host__
cuiVec2 floor(cuVec2 const& _v1)
{
    int2 vi = make_int2(floorf(_v1.v));
    return cuiVec2{ vi };
}

// return unsigned int vector
inline __device__ __host__
cuuVec2 flooru(cuVec2 const& _v1)
{
    uint2 vu = make_uint2(make_int2(floorf(_v1.v)));
    return cuuVec2{ vu };
}


////////////////////////////////////////////////////////////////////////////////
// array version of vec3
////////////////////////////////////////////////////////////////////////////////
// _v1 = _v0 * _s
inline __host__ __device__
void
vec3_by_scalar(float const _v[3], float _s, float _vp[3])
{
    for (int i = 0; i < 3; ++i) {
        _vp[i] = _v[i] * _s;
    }
}

// _v1 = _s * _v0
inline __host__ __device__
void
scalar_by_vec3(float _s, float const _v[3], float _vp[3])
{
    vec3_by_scalar(_v, _s, _vp);
}

inline __host__ __device__
void
vec3_by_vec3(float const _v1[3], float const _v2[3], float _vp[3])
{
    for (int i = 0; i < 3; ++i) {
        _vp[i] = _v1[i] * _v2[i];
    }
}

// _p = _v1[3] dot _v2[3]
inline __host__ __device__
void
vec3_dot_vec3(float const _v1[3], float const _v2[3], float *_p)
{
    *_p = 0.f;
    *_p = _v1[0] * _v2[0] + _v1[1] * _v2[1] + _v1[2] * _v2[2];
}

// _vp = _v1[3] corss _v2[3]
inline __host__ __device__
void
vec3_cross_vec3(float const _v1[3], float const _v2[3], float _vp[3])
{
    _vp[0] = _v1[1] * _v2[2] - _v2[1] * _v1[2];
    _vp[1] = _v1[2] * _v2[0] - _v2[2] * _v1[0];
    _vp[2] = _v1[0] * _v2[1] - _v2[0] * _v1[1];
}

////////////////////////////////////////////////////////////////////////////////
// array version of mat3
////////////////////////////////////////////////////////////////////////////////
inline __host__ __device__
float3
get_col(float const _m[3][3], int i)
{
    return make_float3(_m[0][i], _m[1][i], _m[2][i]);
}

inline __host__ __device__
void
set_col(float3 const &v, float _m[3][3], int i)
{
    _m[0][i] = v.x; 
    _m[1][i] = v.y; 
    _m[2][i] = v.z;
}

// _mT = _m^T
inline __host__ __device__
void transpose_mat3(float const _m[3][3], float _mT[3][3])
{
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            _mT[i][j] = _m[j][i];
        }
    }
}

inline __host__ __device__
void
identity(float _m[3][3])
{
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            _m[i][j] = (float)0.0;
        }
    }
    for (int i = 0; i < 3; ++i) {
        _m[i][i] = (float)1.0;
    }
}


inline __host__ __device__
void
copy_mat3(float _m_dest[3][3], float const _m_src[3][3])
{
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            _m_dest[i][j] = _m_src[i][j];
        }
    }
}

inline __host__ __device__
void
zeroes(float _m[3][3])
{
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            _m[i][j] = (float)0.0;
        }
    }
}

// _vp = _m * _v
inline __host__ __device__
void
mat3_by_vec3(float const _m[3][3], float const _v[3], float _vp[3])
{
    float p;
    for (int i = 0; i < 3; ++i) {
        vec3_dot_vec3(_m[i], _v, &p);
        _vp[i] = p;
    }
}

// _vp = _v^T * _m
inline __host__ __device__
void
vec3_by_mat3(float const _v[3], float const _m[3][3], float _vp[3])
{
    float p;
    for (int i = 0; i < 3; ++i) {
        float _col_i[3] = { _m[0][i], _m[1][i], _m[2][i] };
        vec3_dot_vec3(_v, _col_i, &p);
        _vp[i] = p;
    }
}

inline __host__ __device__
float3
mat3_by_float3(float const _m[3][3], float3 const &_v)
{
    float3 vp, row;

    row = make_float3(_m[0][0], _m[0][1], _m[0][2]);
    vp.x = dot(row, _v);
    row = make_float3(_m[1][0], _m[1][1], _m[1][2]);
    vp.y = dot(row, _v);
    row = make_float3(_m[2][0], _m[2][1], _m[2][2]);
    vp.z = dot(row, _v);

    return vp;
}

// _mp = _m1 * _m2
inline __host__ __device__
void
mat3_by_mat3(float const _m1[3][3], float const _m2[3][3], float _mp[3][3])
{
    for (int i = 0; i < 3; ++i) {
        vec3_by_mat3(_m1[i], _m2, _mp[i]);
    }
}

////////////////////////////////////////////////////////////////////////////////
// mat3, column major
////////////////////////////////////////////////////////////////////////////////
template <typename T>
struct _t_mat3
{
    T col[3];
};

template<typename T>
__device__ __host__
_t_mat3<T> operator+ (_t_mat3<T> const &_m1, _t_mat3<T> const &_m2)
{
    _t_mat3<T> m;
    m.col[0] = _m1.col[0] + _m2.col[0];
    m.col[1] = _m1.col[1] + _m2.col[1];
    m.col[2] = _m1.col[2] + _m2.col[2];
    return m;
}

template<typename T>
__device__ __host__
_t_mat3<T> operator- (_t_mat3<T> const &_m1, _t_mat3<T> const &_m2)
{
    _t_mat3<T> m;
    m.col[0] = _m1.col[0] - _m2.col[0];
    m.col[1] = _m1.col[1] - _m2.col[1];
    m.col[2] = _m1.col[2] - _m2.col[2];
    return m;
}

template<typename T>
__device__ __host__
_t_mat3<T> operator* (float const &_s, _t_mat3<T> const &_m)
{
    _t_mat3<T> m;
    m.col[0] = _s * _m.col[0];
    m.col[1] = _s * _m.col[1];
    m.col[2] = _s * _m.col[2];
    return m;
}

typedef _t_mat3<float3> mat3;

inline __host__ __device__
mat3 
make_mat3()
{
    mat3 m;
    m.col[0] = make_float3(1.f, 0.f, 0.f);
    m.col[1] = make_float3(0.f, 1.f, 0.f);
    m.col[2] = make_float3(0.f, 0.f, 1.f);
    return m;
}

inline __host__ __device__
mat3
make_mat3(float s)
{
    mat3 m;
    m.col[0] = make_float3(s);
    m.col[1] = make_float3(s);
    m.col[2] = make_float3(s);
    return m;
}

inline __host__ __device__ 
mat3 outer(float3 const &v1, float3 const &v2)
{
    mat3 prod;
    prod.col[0] = v1 * v2.x;
    prod.col[1] = v1 * v2.y;
    prod.col[2] = v1 * v2.z;
    return prod;
}

////////////////////////////////////////////////////////////////////////////////
// quaternion
////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct _t_quat
{
    T x, y, z, w;
    __host__ __device__ _t_quat() : x(0), y(0), z(0), w(1.f) {}
    __host__ __device__ _t_quat(T _x, T _y, T _z, T _w) : x(_x), y(_y), z(_z), w(_w) {}
    __host__ __device__ _t_quat<T> operator * (_t_quat<T> q) const 
    {
        return _t_quat(w * q.x + q.w * x + y * q.z - q.y * z, w * q.y + q.w * y + z * q.x - q.z * x,
            w * q.z + q.w * z + x * q.y - q.x * y, w * q.w - x * q.x - y * q.y - z * q.z);
    }
    __host__ __device__ _t_quat<T> operator * (T s) const
    {
        return _t_quat(s * x, s * y, s * z, s * w);
    }

};

typedef _t_quat<float> quat;

template <typename T>
__host__ __device__ inline 
_t_quat<T> 
quat_from_axis_angle(const float3& _axis, float _angle)
{
    float3 v = normalize(_axis);

    float half = _angle*0.5f;
    float w = cosf(half);

    const float sin_theta_over_two = sinf(half);
    v *= sin_theta_over_two;

    return _t_quat<T>(v.x, v.y, v.z, w);
}

inline __host__ __device__
quat 
normalize(const quat& q)
{
    float lSq = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w;

    if (lSq > 0.0f) {
        float invL = 1.0f / sqrtf(lSq);
        return q * invL;
    }
    else {
        return quat();
    }
}

inline __host__ __device__
float3 
rotate(const quat& q, const float3& v)
{
    float3 q_vec = make_float3(q.x, q.y, q.z);
    return v*(2.0f*q.w*q.w - 1.0f) + cross(q_vec, v) * q.w * 2.0f + q_vec * dot(q_vec, v) * 2.0f;
}

inline __host__ __device__ 
void 
extract_matrix(quat const &q, float _m[3][3])
{
    set_col(rotate(q, make_float3(1.f, 0.f, 0.f)), _m, 0);
    set_col(rotate(q, make_float3(0.f, 1.f, 0.f)), _m, 1);
    set_col(rotate(q, make_float3(0.f, 0.f, 1.f)), _m, 2);
}

inline __host__ __device__
mat3
extract_matrix(quat const &_q)
{
    mat3 m;
    m.col[0] = rotate(_q, make_float3(1.f, 0.f, 0.f));
    m.col[1] = rotate(_q, make_float3(0.f, 1.f, 0.f));
    m.col[2] = rotate(_q, make_float3(0.f, 0.f, 1.f));
    return m;
}


inline __host__ __device__
quat
make_quat(float const _m[3][3])
{
    quat q;
    float tr = _m[0][0] + _m[1][1] + _m[2][2], h;
    if (tr >= 0)
    {
        h = sqrtf(tr + 1);
        q.w = 0.5f * h;
        h = 0.5f / h;

        q.x = (_m[2][1] - _m[1][2]) * h;
        q.y = (_m[0][2] - _m[2][0]) * h;
        q.z = (_m[1][0] - _m[0][1]) * h;
    }
    else
    {
        unsigned int i = 0;
        if (_m[1][1] > _m[0][0])
            i = 1;
        if (_m[2][2] > _m[i][i])
            i = 2;
        switch (i)
        {
        case 0:
            h = sqrtf((_m[0][0] - (_m[1][1] + _m[2][2])) + 1);
            q.x = 0.5f * h;
            h = 0.5f / h;

            q.y = (_m[0][1] + _m[1][0]) * h;
            q.z = (_m[2][0] + _m[0][2]) * h;
            q.w = (_m[2][1] - _m[1][2]) * h;
            break;
        case 1:
            h = sqrtf((_m[1][1] - (_m[2][2] + _m[0][0])) + 1);
            q.y = 0.5f * h;
            h = 0.5f / h;

            q.z = (_m[1][2] + _m[2][1]) * h;
            q.x = (_m[0][1] + _m[1][0]) * h;
            q.w = (_m[0][2] - _m[2][0]) * h;
            break;
        case 2:
            h = sqrtf((_m[2][2] - (_m[0][0] + _m[1][1])) + 1);
            q.z = 0.5f * h;
            h = 0.5f / h;

            q.x = (_m[2][0] + _m[0][2]) * h;
            q.y = (_m[1][2] + _m[2][1]) * h;
            q.w = (_m[1][0] - _m[0][1]) * h;
            break;
        default: // Make compiler happy
            q.x = q.y = q.z = q.w = 0.f;
            break;
        }
    }
    return q;
}


////////////////////////////////////////////////////////////////////////////////
// functor 
////////////////////////////////////////////////////////////////////////////////
template<typename Tv, typename Ts>
struct weighted_log2
{
    __host__ __device__ weighted_log2(Ts _w) : w_(_w) {}
    __host__ __device__ Ts operator()(const Tv &s) const
    {
        float n = floorf(s.w / w_ + 0.5f);
        return log2(n) + 1;
    }
private:
    Ts w_;
};

template<typename Tv, typename Ts>
struct weighted_power
{
    __host__ __device__ weighted_power(Ts _w, Ts _exp) : w_(_w), exp_(_exp) {}
    __host__ __device__ Ts operator()(const Tv &s) const
    {
        float n = floorf(static_cast<float>(s.w) / w_ + 0.5f);
        return pow(n, exp_);
    }
private:
    Ts w_;
    Ts exp_;
};

template<typename Ts>
struct inverse
{
    __host__ __device__ Ts operator()(const Ts &s) const
    {
        return 1.f / s;
    }
};


template<typename Ts, typename Tv>
struct scalar_vec_multiply : public thrust::binary_function<Ts, Tv, Tv>
{
    __host__ __device__ Tv operator()(const Ts &s, const Tv &v) const
    {
        return s * v;
    }
};

template<typename Tv, typename Ts>
struct vec_scalar_multiply : public thrust::binary_function<Tv, Ts, Tv>
{
    __host__ __device__ Tv operator()(const Tv &v, const Ts &s) const
    {
        return s * v;
    }
};

template<typename Tv, typename Ts>
struct vec_scalar_divide : public thrust::binary_function<Tv, Ts, Tv>
{
    __host__ __device__ Tv operator()(const Tv &v, const Ts &s) const
    {
        return ((Ts)1/s) * v;
    }
};


struct vec3_outer_prod
{
    __host__ __device__ mat3 operator()(float3 const &v1, float3 const &v2) const
    {
        mat3 prod;
        prod.col[0] = v1 * v2.x;
        prod.col[1] = v1 * v2.y;
        prod.col[2] = v1 * v2.z;
        return prod;
    }

    __host__ __device__ mat3 operator()(float4 const &v1, float4 const &v2) const
    {
        mat3 prod;
        prod.col[0] = make_float3(v1 * v2.x);
        prod.col[1] = make_float3(v1 * v2.y);
        prod.col[2] = make_float3(v1 * v2.z);
        return prod;
    }
};

template<typename T>
struct plus_by
{
    __host__ __device__ plus_by(T _t) : t_(_t) {}
    __host__ __device__ T operator() (T const &x)  const
    {
        return x + t_;
    }
private:
    T t_;
};


template<typename T>
struct minus_by
{
    __host__ __device__ minus_by(T _t) : t_(_t) {}
    __host__ __device__ T operator() (T const &x)  const
    {
        return x - t_;
    }
private:
    T t_;
};

template<typename T, typename Ts>
struct multiply_by
{
    __host__ __device__ multiply_by(Ts _t) : t_(_t) {}
    __host__ __device__ T operator() (T const &x)  const
    {
        return t_ * x;
    }
private:
    Ts t_;
};


////////////////////////////////////////////////////////////////////////////////
// collision structure
////////////////////////////////////////////////////////////////////////////////

struct cuAABB
{
    float4 lower_;
    float4 upper_;

    inline __host__ __device__ cuAABB() {
        lower_ = make_float4(0.f);
        upper_ = make_float4(0.f);
    }
};

inline __host__ __device__
cuAABB calcSpheresAABB(float4 const &_p1, float4 const &_p2)
{
    cuAABB aabb;
    aabb.lower_.x = fminf(_p1.x - _p1.w, _p2.x - _p2.w);
    aabb.lower_.y = fminf(_p1.y - _p1.w, _p2.y - _p2.w);
    aabb.lower_.z = fminf(_p1.z - _p1.w, _p2.z - _p2.w);

    aabb.upper_.x = fmaxf(_p1.x + _p1.w, _p2.x + _p2.w);
    aabb.upper_.y = fmaxf(_p1.y + _p1.w, _p2.y + _p2.w);
    aabb.upper_.z = fmaxf(_p1.z + _p1.w, _p2.z + _p2.w);

    return aabb;
}

inline __host__ __device__
cuAABB calcSphereBoxAABB(float4 const &_sphere, cuAABB const &_box)
{
    cuAABB aabb;
    aabb.lower_.x = fminf(_sphere.x - _sphere.w, _box.lower_.x);
    aabb.lower_.y = fminf(_sphere.y - _sphere.w, _box.lower_.y);
    aabb.lower_.z = fminf(_sphere.z - _sphere.w, _box.lower_.z);

    aabb.upper_.x = fmaxf(_sphere.x + _sphere.w, _box.upper_.x);
    aabb.upper_.y = fmaxf(_sphere.y + _sphere.w, _box.upper_.y);
    aabb.upper_.z = fmaxf(_sphere.z + _sphere.w, _box.upper_.z);

    return aabb;
}

inline __host__ __device__
cuAABB calcBoxesAABB(cuAABB const &_b1, cuAABB const &_b2)
{
    cuAABB aabb;
    aabb.lower_.x = fminf(_b1.lower_.x, _b2.lower_.x);
    aabb.lower_.y = fminf(_b1.lower_.y, _b2.lower_.y);
    aabb.lower_.z = fminf(_b1.lower_.z, _b2.lower_.z);

    aabb.upper_.x = fmaxf(_b1.upper_.x, _b2.upper_.x);
    aabb.upper_.y = fmaxf(_b1.upper_.y, _b2.upper_.y);
    aabb.upper_.z = fmaxf(_b1.upper_.z, _b2.upper_.z);

    return aabb;
}

// q.xyz: closest point, q.w: square of distance
inline __host__ __device__
float4 findClosestPtPointAABB(float4 const &_p, cuAABB const &_aabb)
{
    float4 q = _p;
    q = fmaxf(q, _aabb.lower_);
    q = fminf(q, _aabb.upper_);
    float3 v = make_float3(q - _p);
    q.w = dot(v, v);
    return q;
}


inline __host__ __device__
float calcSqDistPointAABB(float4 const &_p, cuAABB const &_aabb)
{
    float sq_dist = 0.f;
    float v, delta;

    v = _p.x;
    delta = fmaxf(0.f, _aabb.lower_.x - v);
    sq_dist += delta * delta;
    delta = fmaxf(0.f, v - _aabb.upper_.x);
    sq_dist += delta * delta;

    v = _p.y;
    delta = fmaxf(0.f, _aabb.lower_.y - v);
    sq_dist += delta * delta;
    delta = fmaxf(0.f, v - _aabb.upper_.y);
    sq_dist += delta * delta;

    v = _p.z;
    delta = fmaxf(0.f, _aabb.lower_.z - v);
    sq_dist += delta * delta;
    delta = fmaxf(0.f, v - _aabb.upper_.z);
    sq_dist += delta * delta;

    return sq_dist;
}

inline __host__ __device__
bool intersectSphereAABB(float4 const &_p, cuAABB const &_aabb)
{
    float sq_dist = calcSqDistPointAABB(_p, _aabb);
    return sq_dist < _p.w * _p.w;
}

inline __host__ __device__
bool intersectSphereAABB(float4 const &_p, cuAABB const &_aabb, float &_sq_dist)
{
    _sq_dist = calcSqDistPointAABB(_p, _aabb);
    return _sq_dist < _p.w * _p.w;
}

// 
inline __host__ __device__
bool intersectSphereAABBInRange(float4 const &_p, cuAABB const &_aabb, float _search_range)
{
    float sq_dist = calcSqDistPointAABB(_p, _aabb);
    return sq_dist < _search_range *_search_range;
}

inline __host__ __device__
bool intersectSphereAABBInRange(float4 const &_p, cuAABB const &_aabb, float _search_range, float &_sq_dist)
{
    _sq_dist = calcSqDistPointAABB(_p, _aabb);
    return _sq_dist < _search_range * _search_range;
}

#endif
