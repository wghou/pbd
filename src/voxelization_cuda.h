#pragma once
#include "cuda_helper.h"

struct VoxelizationParam
{
    uint vert_offset_; 
    uint num_vert_; 
    uint index_offset_; 
    uint num_triangles_; 
    uint3 volume_dim_; 
    float3 lower_; 
    float3 upper_;
    float extension_ = 0.f;
};

struct Volume_CUDA
{
    uint proj_axis_;
    float3 real_voxel_size_;
    float3 real_vol_dim_;
    float3 real_voxel_lower_;
    uint3 volume_dim_;
    uint *dVoxels_ = 0;

    //_extension: with respect to real_voxel_size_ 
    Volume_CUDA(float3 *_dMeshPositions, uint _vert_offset, uint _num_vert, uint * _dMeshIndices, uint _index_offset, uint _num_triangles, uint3 _volume_dim, float3 _lower, float3 _upper, float _extension = 0.f);
    Volume_CUDA(float3 *_dMeshPositions, uint *_dMeshIndices, VoxelizationParam _args);
    ~Volume_CUDA();
private:
    void voxelize(float3 *_dMeshPositions, uint _vert_offset, uint _num_vert, uint * _dMeshIndices, uint _index_offset, uint _num_triangles, uint3 _volume_dim, float3 _lower, float3 _upper, float _extension = 0.f);
};