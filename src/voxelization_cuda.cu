#pragma once
#include "voxelization_cuda.h"
#include <iostream>
#include <bitset>

#include <device_launch_parameters.h>
#include <stdio.h>
#include <float.h>
#include "cuda_math_helper.h"

struct VoxelParams
{
    cuVec3 mesh_scale_; // normalize to 1, then scale to volume_dim_
    cuVec3 inv_mesh_scale_; // 1 / mesh_scale_
    uint vert_offset_;
    uint num_vert_;
    uint index_offset_; // indices
    uint num_triangles_;
    cuuVec3 volume_dim_;
    cuVec3 lower_;

    uint proj_x_; // x: 0; y: 1; z: 2, projection along this axis
    uint proj_y_; // (proj_x_ + 1) mod 3, first axis of the projection plane
    uint proj_z_; // (proj_x_ + 2) mod 3, second axis of the projection plane

    cuuVec2 vol_dim_2d_;
    
    uint voxel_idx_width_; // number of voxels packed into one unsigned integer 
};

__constant__ VoxelParams kVoxelParams;

inline __device__ __host__
cuVec2 proj(cuVec3 const& _v0, uint _axis)
{
    cuVec2 v1;
    v1[0] = _v0[(_axis + 1) % 3];
    v1[1] = _v0[(_axis + 2) % 3];
    return v1;
}

inline __device__ __host__
cuiVec2 proj(cuiVec3 const& _v0, uint _axis)
{
    cuiVec2 v1;
    v1[0] = _v0[(_axis + 1) % 3];
    v1[1] = _v0[(_axis + 2) % 3];
    return v1;
}

inline __device__ __host__
cuuVec2 proj(cuuVec3 const& _v0, uint _axis)
{
    cuuVec2 v1;
    v1[0] = _v0[(_axis + 1) % 3];
    v1[1] = _v0[(_axis + 2) % 3];
    return v1;
}

inline __device__ __host__
cuVec2 edge_normal(cuVec3 const &_e, cuVec3 const &_n, uint _proj_axis)
{
    cuVec2 v1;
    v1[0] = -_e[(_proj_axis + 2) % 3];
    v1[1] = _e[(_proj_axis + 1) % 3];
    return static_cast<float>(sign(_n[_proj_axis])) * normalize(v1);
}

inline __device__ __host__
bool is_left_or_top_edge(cuVec2 const &_n)
{
    return _n[0] > 0.f ||
        (_n[0] == 0.f) && (_n[1] < 0.f);
}


////////////////////////////////////////////////////////////////////////////////
// device functions
////////////////////////////////////////////////////////////////////////////////

// encode order : proj_x_ < proj_y_ < proj_z_
inline __device__
uint voxel_idx(uint _x_proj, uint _y_proj, uint _z_proj)
{
    return _z_proj * kVoxelParams.volume_dim_[kVoxelParams.proj_y_] * kVoxelParams.volume_dim_[kVoxelParams.proj_x_] +
        _y_proj * kVoxelParams.volume_dim_[kVoxelParams.proj_x_] +
        _x_proj;
}


////////////////////////////////////////////////////////////////////////////////
// kernels
////////////////////////////////////////////////////////////////////////////////

__global__ void
prepare_voxelization_kernel(
    float3 * _positions
)
{
    uint vert_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (vert_id >= kVoxelParams.num_vert_) return;

    vert_id += kVoxelParams.vert_offset_;
    float3 pos = _positions[vert_id];
    pos -= kVoxelParams.lower_.v;
    pos *= kVoxelParams.mesh_scale_.v;
    _positions[vert_id] = pos;
}

__global__ void
finalize_voxelization_kernel(
    float3 * _positions
)
{
    uint vert_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (vert_id >= kVoxelParams.num_vert_) return;

    vert_id += kVoxelParams.vert_offset_;
    float3 pos = _positions[vert_id];
    pos *= kVoxelParams.inv_mesh_scale_.v;
    pos += kVoxelParams.lower_.v;
    _positions[vert_id] = pos;
}


//_voxels must be initialized with 0
// mesh is normalized to 1 <inside box (0, 0, 0) - (1, 1, 1)>, then scale to volume_dim_
__global__ void
voxelize_kernel (
    float3 *const _positions,
    uint *const _indices,
    uint *_voxels // every voxel is represented using one bit, encode order: proj_x_ < proj_y_ < proj_z_
)
{
    uint triangle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (triangle_id >= kVoxelParams.num_triangles_) return;

    uint index_idx0 = triangle_id * 3 + kVoxelParams.index_offset_;

    cuVec3 v0{ _positions[_indices[index_idx0 + 0]] };
    cuVec3 v1{ _positions[_indices[index_idx0 + 1]] };
    cuVec3 v2{ _positions[_indices[index_idx0 + 2]] };

    cuVec3 e0 = v1 - v0;
    cuVec3 e1 = v2 - v1;
    cuVec3 e2 = v0 - v2;

    cuVec3 n = normalize(cross(e0, e1));
    cuVec2 aa = proj(fminf(v0, v1, v2), kVoxelParams.proj_x_);
    cuVec2 bb = proj(fmaxf(v0, v1, v2), kVoxelParams.proj_x_);

    //printf("thead %u, normal: (%.1f, %.1f, %.1f), v0:(%.1f, %.1f, %.1f), v1:(%.1f, %.1f, %.1f)\n", triangle_id, n[0], n[1], n[2], v0[0], v0[1], v0[2], v1[0], v1[1], v1[2]);
    cuuVec2 voxel_col_begin = flooru(aa + make_float2(0.5f));
    cuuVec2 voxel_col_end = flooru(bb + make_float2(0.5f));

    // edge functions
    cuVec2 n_e0 = edge_normal(e0, n, kVoxelParams.proj_x_);
    cuVec2 n_e1 = edge_normal(e1, n, kVoxelParams.proj_x_);
    cuVec2 n_e2 = edge_normal(e2, n, kVoxelParams.proj_x_);

    float d_e0 = -dot(n_e0, proj(v0, kVoxelParams.proj_x_));
    float d_e1 = -dot(n_e1, proj(v1, kVoxelParams.proj_x_));
    float d_e2 = -dot(n_e2, proj(v2, kVoxelParams.proj_x_));

    const float posi_eps = FLT_EPSILON;
    const float nega_eps = 0;

    float f_e0 = is_left_or_top_edge(n_e0) ? posi_eps : nega_eps;
    float f_e1 = is_left_or_top_edge(n_e1) ? posi_eps : nega_eps;
    float f_e2 = is_left_or_top_edge(n_e2) ? posi_eps : nega_eps;

    for (uint z = voxel_col_begin[1]; z < voxel_col_end[1]; ++z) {
        for (uint y = voxel_col_begin[0]; y < voxel_col_end[0]; ++y) {

            cuVec3 p;
            p[kVoxelParams.proj_z_] = (z + 0.5f);
            p[kVoxelParams.proj_y_] = (y + 0.5f);
            cuVec2 p_2d = proj(p, kVoxelParams.proj_x_);

            float dist_0 = dot(n_e0, p_2d) + d_e0;
            float dist_1 = dot(n_e1, p_2d) + d_e1;
            float dist_2 = dot(n_e2, p_2d) + d_e2;

            if (dist_0 + f_e0 > 0 &&
                dist_1 + f_e1 > 0 &&
                dist_2 + f_e2 > 0)
            {
               // assume triangle plane is: dot(n, x) + w = 0
                float w = -dot(v0, n);
                p[kVoxelParams.proj_x_] = (-w - dot(proj(n, kVoxelParams.proj_x_), proj(p, kVoxelParams.proj_x_))) / n[kVoxelParams.proj_x_];
                uint x = (uint)floorf(p[kVoxelParams.proj_x_] + 0.5f);
                if (x == 0) continue;

                uint voxel_begin_idx = voxel_idx(0, y, z);
                // inclusive end
                uint voxel_end_idx = voxel_idx(x - 1, y, z);
                uint uint_begin_idx = voxel_begin_idx / kVoxelParams.voxel_idx_width_;
                uint uint_end_idx = voxel_end_idx / kVoxelParams.voxel_idx_width_;
                uint uint_mask_begin = ~ n_LSB<uint>(voxel_begin_idx % kVoxelParams.voxel_idx_width_);
                uint uint_mask = ~n_LSB<uint>(0u);
                uint uint_mask_end = n_LSB<uint>(voxel_end_idx % kVoxelParams.voxel_idx_width_ + 1);


                if (uint_begin_idx == uint_end_idx) {
                    atomicXor(&_voxels[uint_begin_idx], uint_mask_begin & uint_mask_end);
                }
                else {
                    atomicXor(&_voxels[uint_begin_idx], uint_mask_begin);
                    atomicXor(&_voxels[uint_end_idx], uint_mask_end);
                }
                //printf("thead %u, col_y_z_x:(%u, %u, 0 -- %u), mask_beg: %04x, mask_end: %04x\n", triangle_id, y, z, x, uint_mask_begin, uint_mask_end);
                
                for (uint i = uint_begin_idx + 1; i < uint_end_idx; ++i) {
                    atomicXor(&_voxels[i], uint_mask);
                }
            } // end if: overlap test

        } // column y
    } // column z

}



////////////////////////////////////////////////////////////////////////////////
// host functions
////////////////////////////////////////////////////////////////////////////////


void
Volume_CUDA::voxelize(float3 * _dMeshPositions, uint _vert_offset, uint _num_vert, uint * _dMeshIndices, uint _index_offset, uint _num_triangles, uint3 _volume_dim, float3 _lower, float3 _upper, float _extension)
{
    VoxelParams params;
    params.vert_offset_ = _vert_offset;
    params.num_vert_ = _num_vert;
    params.index_offset_ = _index_offset;
    params.num_triangles_ = _num_triangles;
    params.volume_dim_ = _volume_dim;
    params.lower_ = _lower;
    cuVec3 diag = _upper - _lower;
    params.mesh_scale_ = make_float3(_volume_dim) / make_float3(fmaxf(fmaxf(diag[0], diag[1]), diag[2]));
    float3 min_scale = 1.f / diag.v;
    params.mesh_scale_ = fmaxf(params.mesh_scale_.v, min_scale);
    params.inv_mesh_scale_ = 1.f / params.mesh_scale_.v;

    params.proj_x_ = index_of_min(diag);
    params.proj_y_ = (params.proj_x_ + 1) % 3;
    params.proj_z_ = (params.proj_x_ + 2) % 3;

    params.vol_dim_2d_ = proj(params.volume_dim_, params.proj_x_);
    params.voxel_idx_width_ = sizeof(uint) * 8;

    real_voxel_size_ = make_float3(fmaxf(fmaxf(diag[0], diag[1]), diag[2])) / (make_float3(_volume_dim) - _extension);
    real_vol_dim_ = floorf(diag.v * params.mesh_scale_.v + 0.5f);
    float3 delta_lower = real_vol_dim_ * real_voxel_size_ - diag.v;
    real_voxel_lower_ = params.lower_.v - 0.5f * delta_lower;
    proj_axis_ = params.proj_x_;
    volume_dim_ = _volume_dim;

    //std::cout << "real voxel size: " << real_voxel_size_ 
    //    << ", real vol dim: " << real_vol_dim_
    //    << ", real voxel lower: " << real_voxel_lower_
    //    <<", mesh scale: " << params.mesh_scale_.v
    //    << std::endl;

    uint num_voxels = params.volume_dim_[0] * params.volume_dim_[1] * params.volume_dim_[2];
    uint num_uints = num_voxels / params.voxel_idx_width_ + (num_voxels % params.voxel_idx_width_ > 0);

    //printf("num_uints: %u, num_voxels: %u\n", num_uints, num_voxels);

    allocateArray((void**)&dVoxels_, num_uints * sizeof(uint));
    checkCudaErrors(cudaMemset(dVoxels_, 0u, num_uints * sizeof(uint)));

    checkCudaErrors(cudaMemcpyToSymbol(kVoxelParams, &params, sizeof(VoxelParams)));

    uint num_threads, num_blocks;
    computeGridSize(params.num_vert_, kCudaBlockSize, num_blocks, num_threads);
    prepare_voxelization_kernel << < num_blocks, num_threads >> > (_dMeshPositions);
    getLastCudaError("prepare_voxelization_kernel");

    computeGridSize(params.num_triangles_, kCudaBlockSize, num_blocks, num_threads);
    voxelize_kernel << < num_blocks, num_threads >> > (_dMeshPositions, _dMeshIndices, dVoxels_);
    getLastCudaError("prepare_voxelization_kernel");

    computeGridSize(params.num_vert_, kCudaBlockSize, num_blocks, num_threads);
    finalize_voxelization_kernel << < num_blocks, num_threads >> > (_dMeshPositions);
    getLastCudaError("finalize_voxelization_kernel");

    //uint *voxels = (uint *)malloc(num_uints * sizeof(uint));
    //copyArrayFromDevice(voxels, dVoxels_, 0, num_uints * sizeof(uint));

    //std::cout << "voxels: " << std::flush;
    //for (size_t i = 0; i < num_uints; ++i) {
    //    std::cout << std::bitset<32>(voxels[i]) << " (" << voxels[i] << ")" << std::endl;
    //}

    //free(voxels);
}

Volume_CUDA::Volume_CUDA(float3 * _dMeshPositions, uint _vert_offset, uint _num_vert, uint * _dMeshIndices, uint _index_offset, uint _num_triangles, uint3 _volume_dim, float3 _lower, float3 _upper, float _extension)
{
    voxelize(_dMeshPositions, _vert_offset, _num_vert, _dMeshIndices, _index_offset, _num_triangles, _volume_dim, _lower, _upper, _extension);
}

Volume_CUDA::Volume_CUDA(float3 *_dMeshPositions, uint *_dMeshIndices, VoxelizationParam _args)
{
    //std::cout << "_args.vert_offset_: " << _args.vert_offset_ << "\n"
    //    << "_args.num_vert_: " << _args.num_vert_ << "\n"
    //    << "_args.index_offset_: " << _args.index_offset_ << "\n"
    //    << "_args.num_triangles_: " << _args.num_triangles_ << "\n"
    //    << "_args.volume_dim_: (" << _args.volume_dim_.x << ", " << _args.volume_dim_.y << ", " << _args.volume_dim_.z << ")" << "\n"
    //    << "_args.lower_: " << _args.lower_ << "\n"
    //    << "_args.upper_: " << _args.upper_ << "\n"
    //    << "_args.extension_: " << _args.extension_
    //    << std::endl;

    voxelize(_dMeshPositions, _args.vert_offset_, _args.num_vert_, _dMeshIndices, _args.index_offset_, _args.num_triangles_, _args.volume_dim_, _args.lower_, _args.upper_, _args.extension_);
}

Volume_CUDA::~Volume_CUDA()
{
    FREE_CUDA_ARRAY(dVoxels_);
}