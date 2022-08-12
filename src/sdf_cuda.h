#pragma once
#include "cuda_helper.h"
#include <vector>

struct SDF_CUDA
{
    //uint proj_axis_;
    uint3 volume_dim_; // include vol_dim_ext_
    uint vol_dim_ext_;
    int *dNumRealVoxels_;
    std::vector<float4*> dSdfPyramid_;
    std::vector<int> num_real_voxels_;
    float4 *dVoxelPostions_ = 0; // (x, y, z): voxel lower; w: voxel radius
    float4 *dSdfs_ = 0;
    int num_particles_;
    SDF_CUDA(uint3 _volume_dim, uint _proj_axis, uint *_dVoxelBits);
    void genSdfPyramid(float _sdf_threshold = -1.f);
    void mergeSdfPyramid();
    ~SDF_CUDA();
};