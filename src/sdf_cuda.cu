#pragma once
#include "sdf_cuda.h"
#include <device_launch_parameters.h>
#include <stdio.h>
#include "cuda_math_helper.h"

struct SDFParams
{
    cuuVec3 volume_dim_;
    uint num_voxels_; // volume_dim_.x * volume_dim_.y * volume_dim_.z
    cuuVec3 original_vol_dim;
    uint vol_dim_ext_; // original volume dim = volume_dim_ - vol_dim_ext_ * 2;
    uint proj_x_; // x: 0; y: 1; z: 2, projection along this axis
    uint proj_y_; // (proj_x_ + 1) mod 3, first axis of the projection plane
    uint proj_z_; // (proj_x_ + 2) mod 3, second axis of the projection plane

    uint voxel_idx_width_; // number of voxels packed into one unsigned integer
};

__constant__ SDFParams kSDFParams;

////////////////////////////////////////////////////////////////////////////////
// device functions
////////////////////////////////////////////////////////////////////////////////

// (_x_proj, _y_proj, _z_proj) in orginal volume
__device__ uint
voxel_from_proj_coord(uint _x_proj, uint _y_proj, uint _z_proj, uint const *_voxel_bits)
{
    if (_x_proj >= kSDFParams.original_vol_dim[kSDFParams.proj_x_] ||
        _y_proj >= kSDFParams.original_vol_dim[kSDFParams.proj_y_] ||
        _z_proj >= kSDFParams.original_vol_dim[kSDFParams.proj_z_])
    {
        return 0;
    }

    // encode order : proj_x_ < proj_y_ < proj_z_
    uint voxel_idx = _z_proj * kSDFParams.original_vol_dim[kSDFParams.proj_y_] * kSDFParams.original_vol_dim[kSDFParams.proj_x_]
        + _y_proj * kSDFParams.original_vol_dim[kSDFParams.proj_x_]
        + _x_proj;

    uint uint_idx = voxel_idx / kSDFParams.voxel_idx_width_;
    uint uint_offset = voxel_idx % kSDFParams.voxel_idx_width_;

    return nth_bit<uint>(_voxel_bits[uint_idx], uint_offset) != 0;
}

// coord in extended volume
__device__ uint
voxel(cuuVec3 _coord, uint const *_voxel_bits)
{
    uint x = _coord[kSDFParams.proj_x_];
    uint y = _coord[kSDFParams.proj_y_];
    uint z = _coord[kSDFParams.proj_z_];

    if (x < kSDFParams.vol_dim_ext_ || x >= kSDFParams.original_vol_dim[kSDFParams.proj_x_] + kSDFParams.vol_dim_ext_ ||
        y < kSDFParams.vol_dim_ext_ || y >= kSDFParams.original_vol_dim[kSDFParams.proj_y_] + kSDFParams.vol_dim_ext_ ||
        z < kSDFParams.vol_dim_ext_ || z >= kSDFParams.original_vol_dim[kSDFParams.proj_z_] + kSDFParams.vol_dim_ext_)
    {
        return 0;
    }
    return voxel_from_proj_coord(x - kSDFParams.vol_dim_ext_, y - kSDFParams.vol_dim_ext_, z - kSDFParams.vol_dim_ext_, _voxel_bits);
}

// coord in extended volume
// _voxel_idx = z * dimy * dimx + y * dimx + x
__device__ cuuVec3
getCoord(uint _voxel_idx, cuuVec3 _vol_dim = kSDFParams.volume_dim_)
{
    cuuVec3 coord;
    uint size_xy = _vol_dim[1] * _vol_dim[0];
    coord[2] = _voxel_idx / size_xy;
    uint residual = _voxel_idx % size_xy;
    coord[1] = residual / _vol_dim[0];
    coord[0] = residual % _vol_dim[0];
    return coord;
}

// coord in extended volume
__device__ uint
voxel(uint _voxel_idx, uint const *_voxel_bits)
{
    return voxel(getCoord(_voxel_idx), _voxel_bits);
}

// coord in extended volume
__device__ uint
getVoxelIdx(uint3 _voxel_coord, cuuVec3 _vol_dim = kSDFParams.volume_dim_)
{
    return _voxel_coord.z * _vol_dim[1] * _vol_dim[0] + _voxel_coord.y * _vol_dim[0] + _voxel_coord.x;
}

// _column.x: (_column_axis + 1) % 3; _column.y: (_column_axis + 2) % 3
__device__ bool
isACDominateB(cuuVec4 _a, cuuVec4 _b, cuuVec4 _c, cuuVec2 _column, uint _column_axis)
{
    uint axis_0 = (_column_axis + 1) % 3;
    uint axis_1 = (_column_axis + 2) % 3;
    float a0 = _a[axis_0], a1 = _a[axis_1], ay = _a[_column_axis];
    float b0 = _b[axis_0], b1 = _b[axis_1], by = _b[_column_axis];
    float c0 = _c[axis_0], c1 = _c[axis_1], cy = _c[_column_axis];
    float p0 = _column[0], p1 = _column[1];

    float py = 0.5f * (powf((a0 - p0), 2.f) + powf((a1 - p1), 2.f) + ay * ay - powf((b0 - p0), 2.f) - powf((b1 - p1), 2.f) - by * by) / (ay - by);
    float qy = 0.5f * (powf((c0 - p0), 2.f) + powf((c1 - p1), 2.f) + cy * cy - powf((b0 - p0), 2.f) - powf((b1 - p1), 2.f) - by * by) / (cy - by);
    return py > qy;
}


////////////////////////////////////////////////////////////////////////////////
// kernels
////////////////////////////////////////////////////////////////////////////////

__global__ void
init_sdf_kernel(
    uint const *_voxel_bits,
    int *_num_real_voxels,
    uint *_voxels,
    float *_margins, // distance between centers of surface voxels and surface
    uint4 *_nearest_sites // (x, y, z) nearest site, w: dist^2
)
{
    // _voxel_idx = z * dimy * dimx + y * dimx + x
    uint voxel_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (voxel_id >= kSDFParams.num_voxels_) return;

    cuuVec3 coord = getCoord(voxel_id);
    uint voxel_bit = voxel(coord, _voxel_bits);
    if (voxel_bit == 1) {
        atomicAdd(_num_real_voxels, 1);
    }
    float min_dist = FLT_MAX;
    uint4 site = make_uint4(kMaxUint, kMaxUint, kMaxUint, kMaxUint);
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int3 coord_i_n = make_int3(coord.v) + make_int3(dx, dy, dz);
                cuuVec3 coord_n(make_uint3(coord_i_n));
                uint voxel_n_bit = voxel(coord_n, _voxel_bits);
                if (voxel_n_bit != voxel_bit) {
                    min_dist = fminf(sqrtf(float(dx*dx + dy*dy + dz * dz)) * 0.5f, min_dist);
                }
            }
        }
    }
    if (min_dist != FLT_MAX) {
        site = make_uint4(coord.v, 0);
    }
    _voxels[voxel_id] = voxel_bit;
    _margins[voxel_id] = min_dist;
    _nearest_sites[voxel_id] = site;

}

__global__ void
sweeping_row_kernel(
    uint4 *_nearest_sites //(x, y, z) nearest site, w: dist^2
)
{
    uint3 voxel_coord;
    // 2d grid in yz plane
    voxel_coord.y = blockIdx.x * blockDim.x + threadIdx.x;
    voxel_coord.z = blockIdx.y * blockDim.y + threadIdx.y;
    // process one row 
    if (voxel_coord.y >= kSDFParams.volume_dim_[1] || voxel_coord.z >= kSDFParams.volume_dim_[2]) return;

    uint voxel_id = voxel_coord.z * kSDFParams.volume_dim_[1] * kSDFParams.volume_dim_[0] + voxel_coord.y * kSDFParams.volume_dim_[0];

    uint3 left_site = make_uint3(kMaxUint);
    for (uint i = 0; i < kSDFParams.volume_dim_[0]; ++i) {
        voxel_coord.x = i;
        uint4 site = _nearest_sites[voxel_id];
        if (site.w == 0) {
            left_site = voxel_coord;
        }
        else if (site.w == kMaxUint && left_site.x < voxel_coord.x) {
            // x in ascending ==>  voxel_coord.x >= nearest_site.x
            uint left_dist = voxel_coord.x - left_site.x;
            _nearest_sites[voxel_id] = make_uint4(left_site, left_dist * left_dist);
        }
        ++voxel_id;
    }
  
    uint3 right_site = make_uint3(kMaxUint);
    for (int i = kSDFParams.volume_dim_[0] - 1; i >= 0; --i) {
        voxel_coord.x = i;
        --voxel_id;
        if (voxel_id >= kSDFParams.num_voxels_) break;
        uint4 site = _nearest_sites[voxel_id];
        if (site.w == 0) {
            right_site = voxel_coord;
        }
        else if (right_site.x < kSDFParams.volume_dim_[0]) {
            uint right_dist = right_site.x - voxel_coord.x;
            uint right_dist_sqr = right_dist * right_dist;
            if (right_dist_sqr < site.w) {
                _nearest_sites[voxel_id] = make_uint4(right_site, right_dist_sqr);
            }
        }
    }
}


__global__ void
sweeping_column_kernel(
    uint _column_axis,
    uint4 *_nearest_sites, //(x, y, z) nearest site, w: dist^2
    uint4 *_nearest_sites_aux // memory for saving stack
)
{
    uint axis_0 = (_column_axis + 1) % 3;
    uint axis_1 = (_column_axis + 2) % 3;
    cuuVec3 voxel_coord;
    // 2d grid in <axis_0>-<axis_1> plane
    voxel_coord[axis_0] = blockIdx.x * blockDim.x + threadIdx.x;
    voxel_coord[axis_1] = blockIdx.y * blockDim.y + threadIdx.y;
    // process one row 
    if (voxel_coord[axis_0] >= kSDFParams.volume_dim_[axis_0] || voxel_coord[axis_1] >= kSDFParams.volume_dim_[axis_1]) return;
 
    cuuVec3 stack_top = voxel_coord;
    stack_top[_column_axis] = (uint)-1;
    
    cuuVec3 delta = make_uint3(0);
    delta[_column_axis] = 1;

    // phase 2
    for (uint i = 0; i < kSDFParams.volume_dim_[_column_axis]; ++i) {
        voxel_coord[_column_axis] = i;
        uint4 site = _nearest_sites[getVoxelIdx(voxel_coord.v)];
        uint4 a = make_uint4(kMaxUint);
        uint4 b = a;
        
        if (site.w != kMaxUint) {
            if (stack_top[_column_axis] < kSDFParams.volume_dim_[_column_axis]) {
                a = _nearest_sites_aux[getVoxelIdx(stack_top.v)];
            }
           
            while (stack_top[_column_axis] > 0 && stack_top[_column_axis] < kSDFParams.volume_dim_[_column_axis]) {
                b = a;
                a = _nearest_sites_aux[getVoxelIdx(stack_top.v - delta.v)];
                if (isACDominateB(a, b, site, make_uint2(voxel_coord[axis_0], voxel_coord[axis_1]), _column_axis)) {
                    --stack_top[_column_axis];
                }
                else {
                    break;
                }
            }
            ++stack_top[_column_axis];
            _nearest_sites_aux[getVoxelIdx(stack_top.v)] = site;
        }
    }

    cuuVec3 head = stack_top;
    head[_column_axis] = 0;

    // phase 3
    for (uint i = 0; i < kSDFParams.volume_dim_[_column_axis]; ++i) {
        // return immediately if the stack is empty
        if (stack_top[_column_axis] > kSDFParams.volume_dim_[_column_axis]) {
            break;
        }
        voxel_coord[_column_axis] = i;
        uint3 a, b = make_uint3(_nearest_sites_aux[getVoxelIdx(head.v)]);
        uint dist_b_sqr = dot(b - voxel_coord.v, b - voxel_coord.v), dist_a_sqr;
        while (head[_column_axis] < stack_top[_column_axis]) {
            a = b;
            b = make_uint3(_nearest_sites_aux[getVoxelIdx(head.v + delta.v)]);
            dist_a_sqr = dist_b_sqr;
            dist_b_sqr = dot(b - voxel_coord.v, b - voxel_coord.v);
            // In case of a tie, the distance from the site with the smaller coordinate is considered smaller.
            if (dist_a_sqr <= dist_b_sqr) {
                _nearest_sites[getVoxelIdx(voxel_coord.v)] = make_uint4(a, dist_a_sqr);
                break;
            }
            else {
                ++head[_column_axis];
            }
        }
        if (head[_column_axis] == stack_top[_column_axis]) {
            // at this point, b points to stack_top
            uint dist_b_sqr = dot(b - voxel_coord.v, b - voxel_coord.v);
            _nearest_sites[getVoxelIdx(voxel_coord.v)] = make_uint4(b, dist_b_sqr);
        }
    }
}

__global__ void
level_0_sdf_mag_kernel(
    uint const *_voxels,
    float const *_margins,
    uint4 const *_nearest_sites,
    float4 *_sdf
)
{
    // _voxel_idx = z * dimy * dimx + y * dimx + x
    uint voxel_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (voxel_id >= kSDFParams.num_voxels_) return;

    uint voxel_value = _voxels[voxel_id];
    float4 sdf = make_float4(FLT_MAX);
    uint4 nearest_site = _nearest_sites[voxel_id];
    float margin = _margins[getVoxelIdx(make_uint3(nearest_site))];
    sdf.w = (0.5f - (float)voxel_value) * 2.f * (sqrtf(nearest_site.w) + margin);
    _sdf[voxel_id] = sdf;
}

__global__ void
level_0_sdf_grad_kernel(
    float4 *_sdf
)
{
    uint voxel_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (voxel_id >= kSDFParams.num_voxels_) return;

    int3 coord = make_int3(getCoord(voxel_id).v);
    const int3 max_coord = make_int3(kSDFParams.volume_dim_.v - 1);
    const int3 min_coord = make_int3(0);
    uint3 c_xp = make_uint3(min(coord + make_int3(1, 0, 0), max_coord));
    uint3 c_xn = make_uint3(max(coord - make_int3(1, 0, 0), min_coord));
    uint3 c_yp = make_uint3(min(coord + make_int3(0, 1, 0), max_coord));
    uint3 c_yn = make_uint3(max(coord - make_int3(0, 1, 0), min_coord));
    uint3 c_zp = make_uint3(min(coord + make_int3(0, 0, 1), max_coord));
    uint3 c_zn = make_uint3(max(coord - make_int3(0, 0, 1), min_coord));

    float3 grad = make_float3(_sdf[getVoxelIdx(c_xp)].w - _sdf[getVoxelIdx(c_xn)].w,
                              _sdf[getVoxelIdx(c_yp)].w - _sdf[getVoxelIdx(c_yn)].w,
                              _sdf[getVoxelIdx(c_zp)].w - _sdf[getVoxelIdx(c_zn)].w);
    
    grad = SafeNormalize(grad);
    _sdf[voxel_id] = make_float4(grad, _sdf[voxel_id].w);
}


__global__ void
level_n_sdf_kernel(
    float _sdf_threshold,
    uint3 _dim_prev,
    uint3 _dim_next,
    uint _num_voxels_next,
    float4 *_sdf_prev,
    float4 *_sdf_next,
    int *_delta_num_voxels // _delta_num_voxels[0]: prev; _delta_num_voxels[1]: next
)
{
    uint voxel_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (voxel_id >= _num_voxels_next) return;

    float4 sdf_next = make_float4(-FLT_MAX);
    uint3 coord_next = getCoord(voxel_id, _dim_next).v;
    bool merged = true;

    for (uint z = 0; z <= 1; ++z) {
        for (uint y = 0; y <= 1; ++y) {
            for (uint x = 0; x <= 1; ++x) {
                uint3 coord_prev = 2 * coord_next + make_uint3(x, y, z);
                uint voxel_idx_prev = getVoxelIdx(coord_prev, _dim_prev);
                float4 sdf_prev = _sdf_prev[voxel_idx_prev];
                if (sdf_prev.w >= _sdf_threshold) {
                    merged = false;
                    break;
                }
                else if (sdf_prev.w > sdf_next.w) {
                    // 0.8660254f = sqrt(3)/2
                    sdf_next = sdf_prev.w + make_float4(0.f, 0.f, 0.f, -0.8660254f);
                }
            }
        }
    }

    if (merged) {
        atomicAdd(&_delta_num_voxels[1], 1);
        for (uint z = 0; z <= 1; ++z) {
            for (uint y = 0; y <= 1; ++y) {
                for (uint x = 0; x <= 1; ++x) {
                    uint3 coord_prev = 2 * coord_next + make_uint3(x, y, z);
                    uint voxel_idx_prev = getVoxelIdx(coord_prev, _dim_prev);
                    _sdf_prev[voxel_idx_prev] = make_float4(FLT_MAX);
                }
            }
        }
        atomicAdd(&_delta_num_voxels[0], 8);
    }
    else {
        sdf_next = make_float4(FLT_MAX);
    }

    _sdf_next[voxel_id] = sdf_next;
}


__global__ void
merge_sdf_pyramid_kernel(
    uint _vol_dim_ext,
    uint _num_voxels,
    float _level_n_radius,
    uint3 _dim,
    float4 const *_level_n_sdf,
    int *_offset,
    float4 *_voxel_postions,
    float4 *_sdfs
)
{
    uint voxel_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (voxel_id >= _num_voxels) return;

    float4 sdf = _level_n_sdf[voxel_id];
    if (sdf.w <= 0.f) {
        uint3 coord = getCoord(voxel_id, _dim).v;
        float3 pos = make_float3(coord) * 2.f * _level_n_radius - make_float3(_vol_dim_ext);
        int pos_idx = atomicAdd(_offset, 1);
        _voxel_postions[pos_idx] = make_float4(pos, _level_n_radius);
        _sdfs[pos_idx] = sdf;
    }
    
}

////////////////////////////////////////////////////////////////////////////////
// host functions
////////////////////////////////////////////////////////////////////////////////

SDF_CUDA::SDF_CUDA(uint3 _volume_dim, uint _proj_axis, uint *_dVoxelBits)
{
    SDFParams params;
    vol_dim_ext_ = 2;
    volume_dim_ = _volume_dim + 2 * vol_dim_ext_;
    params.vol_dim_ext_ = vol_dim_ext_;
    params.original_vol_dim = _volume_dim;
    params.volume_dim_ = volume_dim_;
    params.num_voxels_ = params.volume_dim_[0] * params.volume_dim_[1] * params.volume_dim_[2];
    params.proj_x_ = _proj_axis;
    params.proj_y_ = (_proj_axis + 1) % 3;
    params.proj_z_ = (_proj_axis + 2) % 3;
    params.voxel_idx_width_ = sizeof(uint) * 8;

    dSdfPyramid_.resize(1);
    num_real_voxels_.resize(1);

    uint4 *dNearestSites = 0;
    uint *dVoxels = 0;

    checkCudaErrors(cudaMemcpyToSymbol(kSDFParams, &params, sizeof(kSDFParams)));
    allocateArray((void**)&dNearestSites, params.num_voxels_ * sizeof(uint4));
    allocateArray((void**)&dVoxels, params.num_voxels_ * sizeof(uint));
    allocateArray((void**)&dNumRealVoxels_, sizeof(int));
    
    //distance between centers of surface voxels and surface
    float *dMargins;
    allocateArray((void**)&dMargins, params.num_voxels_ * sizeof(float));

    dim3 num_threads, num_blocks;
    computeGridSize(params.num_voxels_, kCudaBlockSize, num_blocks.x, num_threads.x);
    checkCudaErrors(cudaMemset(dNumRealVoxels_, 0, sizeof(int)));
    init_sdf_kernel <<<num_blocks, num_threads >>> (_dVoxelBits, dNumRealVoxels_, dVoxels, dMargins, dNearestSites);
    getLastCudaError("init_sdf_kernel");

    copyArrayFromDevice(&num_real_voxels_[0], dNumRealVoxels_, 0, sizeof(int));

    computeGridSize(params.volume_dim_[1], kCuda2DBlockSize, num_blocks.x, num_threads.x);
    computeGridSize(params.volume_dim_[2], kCuda2DBlockSize, num_blocks.y, num_threads.y);
    sweeping_row_kernel <<<num_blocks, num_threads>>> (dNearestSites);
    getLastCudaError("sweeping_row_kernel");

    uint4 *dNearestSitesAux;
    allocateArray((void**)&dNearestSitesAux, params.num_voxels_ * sizeof(uint4));

    uint column_axis = 1; // y-axis
    computeGridSize(params.volume_dim_[2], kCuda2DBlockSize, num_blocks.x, num_threads.x);
    computeGridSize(params.volume_dim_[0], kCuda2DBlockSize, num_blocks.y, num_threads.y);
    sweeping_column_kernel <<<num_blocks, num_threads>>> (column_axis, dNearestSites, dNearestSitesAux);
    getLastCudaError("sweeping_column_kernel: y");

    column_axis = 2;// z-axis
    computeGridSize(params.volume_dim_[0], kCuda2DBlockSize, num_blocks.x, num_threads.x);
    computeGridSize(params.volume_dim_[1], kCuda2DBlockSize, num_blocks.y, num_threads.y);
    sweeping_column_kernel <<<num_blocks, num_threads >>> (column_axis, dNearestSites, dNearestSitesAux);
    getLastCudaError("sweeping_column_kernel: z");

    FREE_CUDA_ARRAY(dNearestSitesAux);

    allocateArray((void**)&dSdfPyramid_[0], params.num_voxels_ * sizeof(float4));
    num_threads = make_uint3(1);
    num_blocks = make_uint3(1);
    computeGridSize(params.num_voxels_, kCudaBlockSize, num_blocks.x, num_threads.x);
    level_0_sdf_mag_kernel <<<num_blocks, num_threads >>> (dVoxels, dMargins, dNearestSites, dSdfPyramid_[0]);
    getLastCudaError("level_0_sdf_mag_kernel");
    
    level_0_sdf_grad_kernel <<<num_blocks, num_threads>>> (dSdfPyramid_[0]);
    getLastCudaError("level_0_sdf_grad_kernel");
    cudaDeviceSynchronize();

    FREE_CUDA_ARRAY(dVoxels);
    FREE_CUDA_ARRAY(dMargins);
    FREE_CUDA_ARRAY(dNearestSites);
    FREE_CUDA_ARRAY(dNumRealVoxels_);

}

void
SDF_CUDA::genSdfPyramid(float _sdf_threshold)
{
    allocateArray((void**)&dNumRealVoxels_, 2 * sizeof(int));
    int delta_num_voxels[2]; // delta_num_voxels[0]: prev; delta_num_voxels[1]: next
    size_t num_levels = 1;
    dim3 num_threads = make_uint3(1);
    dim3 num_blocks = make_uint3(1);
    while (num_real_voxels_[num_levels - 1] >= 4) {
        uint3 dim_prev = volume_dim_ / (1u << (num_levels - 1));
        uint3 dim_next = volume_dim_ / (1u << (num_levels));
        // printf("level %zu: dim = %u, %u, %u\n", num_levels, dim_next.x, dim_next.y, dim_next.z);
        ++num_levels;
        uint num_voxels_next = dim_next.x * dim_next.y * dim_next.z;
        num_real_voxels_.resize(num_levels);
        dSdfPyramid_.resize(num_levels);
        allocateArray((void**)&dSdfPyramid_[num_levels - 1], num_voxels_next * sizeof(float4));
        checkCudaErrors(cudaMemset(dNumRealVoxels_, 0, 2 * sizeof(int)));
        
        computeGridSize(num_voxels_next, kCudaBlockSize, num_blocks.x, num_threads.x);
        level_n_sdf_kernel <<<num_blocks, num_threads>>>(_sdf_threshold, dim_prev, dim_next, num_voxels_next, dSdfPyramid_[num_levels - 2], dSdfPyramid_[num_levels - 1], dNumRealVoxels_);
        getLastCudaError("level_n_sdf_kernel");

        copyArrayFromDevice(delta_num_voxels, dNumRealVoxels_, 0, 2 * sizeof(int));
        num_real_voxels_[num_levels - 2] -= delta_num_voxels[0];
        num_real_voxels_[num_levels - 1]  = delta_num_voxels[1];
 
        if (num_levels > 10) {
            printf("num_levels cannot be so large, somthing went wrong!\n");
            break;
        }
    }
    for (size_t i = 0; i < num_real_voxels_.size(); ++i) {
        // printf("num real voxels for level %zu: %d\n", i, num_real_voxels_[i]);
    }

    if (num_real_voxels_.back() == 0) {
        num_real_voxels_.pop_back();
        FREE_CUDA_ARRAY(dSdfPyramid_.back());
        dSdfPyramid_.pop_back();
    }

    FREE_CUDA_ARRAY(dNumRealVoxels_);

}

void
SDF_CUDA::mergeSdfPyramid()
{
    num_particles_ = 0;
    for (int & n : num_real_voxels_) {
        num_particles_ += n;
    }
    allocateArray((void**)&dVoxelPostions_, num_particles_ * sizeof(float4));
    allocateArray((void**)&dSdfs_, num_particles_ * sizeof(float4));
    int *dOffset;
    allocateArray((void**)&dOffset, sizeof(int));
    checkCudaErrors(cudaMemset(dOffset, 0, sizeof(int)));

    dim3 num_threads = make_uint3(1);
    dim3 num_blocks = make_uint3(1);
    
    for (size_t i = 0; i < num_real_voxels_.size(); ++i) {
        uint3 dim = volume_dim_ / (1u << i);
        uint num_voxels = dim.x * dim.y * dim.z;
        float radius = 0.5f * (float)(1u << i);

        computeGridSize(num_voxels, kCudaBlockSize, num_blocks.x, num_threads.x);
        merge_sdf_pyramid_kernel <<<num_blocks, num_threads>>> (vol_dim_ext_, num_voxels, radius, dim, dSdfPyramid_[i], dOffset, dVoxelPostions_, dSdfs_);
        getLastCudaError("merge_sdf_pyramid_kernel");
    }
    int num_voxel_pos = 0;
    copyArrayFromDevice(&num_voxel_pos, dOffset, 0, sizeof(int));
    //printf("there are %d voxels after merge.\n", num_voxel_pos);
    FREE_CUDA_ARRAY(dOffset);

    for (float4 * ptr : dSdfPyramid_) {
        FREE_CUDA_ARRAY(ptr);
    }
    //printf("merge finished.\n");
}

SDF_CUDA::~SDF_CUDA()
{
    FREE_CUDA_ARRAY(dVoxelPostions_);
    FREE_CUDA_ARRAY(dSdfs_);
}