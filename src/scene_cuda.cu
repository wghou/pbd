#include "scene_cuda.h"
#include <device_launch_parameters.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <stdio.h>

//
////////////////////////////////////////////////////////////////////////////////
// global functions
////////////////////////////////////////////////////////////////////////////////

__global__ void
createShapeParticles_kernel(
    float3 _lower,
    float3 _velocity,
    float _L0_radius,
    float _inv_mass,
    int _flags,
    int _group,
    int _particle_offset,
    int _rigid_particle_offset,
    int _num_voxels,
    float4 const *_voxel_positions,
    float4 const *_voxel_sdfs,
    float4 *_particle_positions,
    float  *_particle_inv_masses,
    float4 *_particle_velocities,
    int *_particle_phases,
    float4 *_rigid_particle_sdfs
)
{
    uint voxel_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (voxel_id >= _num_voxels) return;

    int particle_idx = voxel_id + _particle_offset;
    
    float scale = _L0_radius * 2.f;
    float4 voxel_position = _voxel_positions[voxel_id];
    _particle_velocities[particle_idx] = make_float4(_velocity, 0.f);
    _particle_inv_masses[particle_idx] = _inv_mass / powf((voxel_position.w * 2.f), 3.f);
    float3 voxel_center = make_float3(voxel_position + voxel_position.w);
    float4 particle_pos = scale * make_float4(voxel_center, voxel_position.w) + make_float4(_lower, 0.f);
    //float4 p_pos = scale * voxel_position + make_float4(_lower + _L0_radius, 0.f);
    if (_flags & ~kPhaseGroupMask) {
        _particle_phases[particle_idx] = _flags | _group;
    }
    else {
        particle_pos.w *= 1.1f;

        _particle_phases[particle_idx] = _group;
        int rigid_particle_idx = voxel_id + _rigid_particle_offset;
        float4 sdf = _voxel_sdfs[voxel_id];
        sdf.w *= scale;
        _rigid_particle_sdfs[rigid_particle_idx] = sdf;
    }
    _particle_positions[particle_idx] = particle_pos;
}


__global__ void
calcParticleHashes_kernel(
    int _particle_begin,
    int _particle_end, // exclusive
    int3 _hash_grid_dim, // vol_dim + 1
    float3 _hash_cell_size,
    float3 _lower,
    float4 const *_particle_positions,
    uint *_grid_particle_hashes,
    int *_grid_particle_indices
)
{
    uint cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint particle_idx = _particle_begin + cell_idx;
    if (particle_idx >= _particle_end) return;

    float3 particle_pos = make_float3(_particle_positions[particle_idx]);
    int3 grid_pos = make_int3(floorf((particle_pos - _lower) / _hash_cell_size));
    
    if (grid_pos.x < 0 || grid_pos.x >= _hash_grid_dim.x ||
        grid_pos.y < 0 || grid_pos.y >= _hash_grid_dim.y ||
        grid_pos.z < 0 || grid_pos.z >= _hash_grid_dim.z)
    {
        printf("particle %d is not inside hash grid\n", particle_idx);
    }

    int hash = grid_pos.z * _hash_grid_dim.y * _hash_grid_dim.x + grid_pos.y * _hash_grid_dim.x + grid_pos.x;
    _grid_particle_hashes[cell_idx] = hash;
    _grid_particle_indices[cell_idx] = particle_idx;
}

__global__ void
findHashCellOffsets_kernel(
    int _num_particles,
    uint const *_grid_particle_hashes,
    int *_cell_starts,
    int *_cell_ends // exclusive
)
{
    uint cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_idx >= _num_particles) return;

    if (cell_idx == _num_particles - 1) {
        uint first_hash = _grid_particle_hashes[0];
        uint last_hash = _grid_particle_hashes[cell_idx];
        _cell_starts[first_hash] = 0;
        _cell_ends[last_hash] = cell_idx + 1;
    }
    else {
        uint current_hash = _grid_particle_hashes[cell_idx];
        uint next_hash = _grid_particle_hashes[cell_idx + 1];
        if (current_hash != next_hash) {
            _cell_ends[current_hash] = cell_idx + 1;
            _cell_starts[next_hash] = cell_idx + 1;
        }
    }
}

__device__ void
searchCell(
    int const *_cell_starts,
    int const *_cell_ends, // exclusive
    int const *_grid_particle_indices,
    float4 const *_particle_positions,
    float3 _vert_pos,
    int _hash, // cell hash
    int *_indices,
    float *_dist_sqrs,
    int &_num_neighbors
)
{
    int cell_start = _cell_starts[_hash];
    if (cell_start < 0) return;

    int cell_end = _cell_ends[_hash];

    for (int i = cell_start; i < cell_end; ++i) {
        int particle_idx = _grid_particle_indices[i];
        float3 particle_pos = make_float3(_particle_positions[particle_idx]);
        float3 dist_vec = particle_pos - _vert_pos;
        float dist_sqr = dot(dist_vec, dist_vec);

        int j = 0;
        for (; j < 4; ++j) {
            if (dist_sqr < _dist_sqrs[j]) break;
        }
        if (j < 4) {
            for (int k = 3; k > j; --k) {
                _indices[k] = _indices[k - 1];
                _dist_sqrs[k] = _dist_sqrs[k - 1];
            }
            _dist_sqrs[j] = dist_sqr;
            _indices[j] = particle_idx;
        }
        ++_num_neighbors;
    }
}

__global__ void
calcSkinWeights_kernel(
    int3 _hash_grid_dim, // vol_dim + 4
    float3 _hash_cell_size,
    float3 _lower,
    int const *_cell_starts,
    int const *_cell_ends, // exclusive
    int const *_grid_particle_indices,
    int const _skin_pos_begin,
    int const _skin_pos_end, // exclusive
    float3 const *_skin_positions,
    float4 const *_particle_positions,
    int4 *_skin_particle_indices,
    float4 *_skin_particle_weights
)
{
    uint vert_idx = _skin_pos_begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (vert_idx >= _skin_pos_end) return;

    float3 vert_pos = _skin_positions[vert_idx];
    //int3 vert_grid_pos = make_int3(floorf((vert_pos - _lower) / _hash_cell_size + 0.5f));
    int3 vert_grid_pos = make_int3(floorf((vert_pos - _lower) / _hash_cell_size));

    if (vert_grid_pos.x < 0 || vert_grid_pos.x >= _hash_grid_dim.x ||
        vert_grid_pos.y < 0 || vert_grid_pos.y >= _hash_grid_dim.y ||
        vert_grid_pos.z < 0 || vert_grid_pos.z >= _hash_grid_dim.z)
    {
        printf("vertex %d is not inside hash grid (%d, %d, %d)\n", vert_idx, vert_grid_pos.x, vert_grid_pos.y, vert_grid_pos.z);
    }
    int hash = vert_grid_pos.z * _hash_grid_dim.y * _hash_grid_dim.x + vert_grid_pos.y * _hash_grid_dim.x + vert_grid_pos.x;
    int indices[4] = { -1, -1, -1, -1 };
    float dist_sqrs[4] = { FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX };
    int num_neighbors = 0;

    searchCell(_cell_starts, _cell_ends, _grid_particle_indices, _particle_positions, vert_pos, hash, indices, dist_sqrs, num_neighbors);
    
    int search_radius = 1;
    int max_radius = max_comp(max(vert_grid_pos, _hash_grid_dim - 1 - vert_grid_pos));
    // while (num_neighbors < 16 && search_radius <= max_radius) {
    while (num_neighbors < 4 && search_radius <= max_radius) {
        // two xy plane
        for (int dz = -search_radius; dz <= search_radius; dz += 2 * search_radius) {
            for (int dy = -search_radius; dy <= search_radius; ++dy) {
                for (int dx = -search_radius; dx <= search_radius; ++dx) {
                    int3 grid_pos = vert_grid_pos + make_int3(dx, dy, dz);
                    if (grid_pos.x >= 0 && grid_pos.x < _hash_grid_dim.x &&
                        grid_pos.y >= 0 && grid_pos.y < _hash_grid_dim.y &&
                        grid_pos.z >= 0 && grid_pos.z < _hash_grid_dim.z)
                    {
                        hash = grid_pos.z * _hash_grid_dim.y * _hash_grid_dim.x + grid_pos.y * _hash_grid_dim.x + grid_pos.x;
                        searchCell(_cell_starts, _cell_ends, _grid_particle_indices, _particle_positions, vert_pos, hash, indices, dist_sqrs, num_neighbors);
                    }
                }
            }
        }
        // two xz plane
        for (int dy = -search_radius; dy <= search_radius; dy += 2 * search_radius) {
            for (int dz = -search_radius + 1; dz < search_radius; ++dz) {
                for (int dx = -search_radius; dx <= search_radius; ++dx) {
                    int3 grid_pos = vert_grid_pos + make_int3(dx, dy, dz);
                    if (grid_pos.x >= 0 && grid_pos.x < _hash_grid_dim.x &&
                        grid_pos.y >= 0 && grid_pos.y < _hash_grid_dim.y &&
                        grid_pos.z >= 0 && grid_pos.z < _hash_grid_dim.z)
                    {
                        hash = grid_pos.z * _hash_grid_dim.y * _hash_grid_dim.x + grid_pos.y * _hash_grid_dim.x + grid_pos.x;
                        searchCell(_cell_starts, _cell_ends, _grid_particle_indices, _particle_positions, vert_pos, hash, indices, dist_sqrs, num_neighbors);
                    }
                }
            }
        }
        // two yz plane
        for (int dx = -search_radius; dx <= search_radius; dx += 2 * search_radius) {
            for (int dz = -search_radius + 1; dz < search_radius; ++dz) {
                for (int dy = -search_radius + 1; dy < search_radius; ++dy) {
                    int3 grid_pos = vert_grid_pos + make_int3(dx, dy, dz);
                    if (grid_pos.x >= 0 && grid_pos.x < _hash_grid_dim.x &&
                        grid_pos.y >= 0 && grid_pos.y < _hash_grid_dim.y &&
                        grid_pos.z >= 0 && grid_pos.z < _hash_grid_dim.z)
                    {
                        hash = grid_pos.z * _hash_grid_dim.y * _hash_grid_dim.x + grid_pos.y * _hash_grid_dim.x + grid_pos.x;
                        searchCell(_cell_starts, _cell_ends, _grid_particle_indices, _particle_positions, vert_pos, hash, indices, dist_sqrs, num_neighbors);
                    }
                }
            }
        }

        ++search_radius;
    } // end while

    if (num_neighbors < 4) {
        printf("vertex %d: not enough neighboring particles (%d < 4)\n", vert_idx, num_neighbors);
    }

    float sum_weights = 0.f;
    for (int i = 0; i < 4; ++i) {
        dist_sqrs[i] = 1.f / (0.1f + powf(dist_sqrs[i], 0.125f));
        sum_weights += dist_sqrs[i];
    }
    for (int i = 0; i < 4; ++i) {
        dist_sqrs[i] = dist_sqrs[i] / sum_weights;
    }
    _skin_particle_indices[vert_idx] = make_int4(indices[0], indices[1], indices[2], indices[3]);
    _skin_particle_weights[vert_idx] = make_float4(dist_sqrs[0], dist_sqrs[1], dist_sqrs[2], dist_sqrs[3]);
}


__global__ void
skinning_kernel(
    int _num_skin_vertices,
    float4 const *_particle_positions_init,
    float4 const *_particle_positions,
    int const *_particle_phases,
    quat const *_rigid_rotations,
    int4 const *_skin_particle_indices,
    float4 const *_skin_particle_weights,
    float3 const *_skin_positions_init,
    float3 *_skin_positions
)
{
    uint vert_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (vert_id >= _num_skin_vertices) return;

    cuiVec4 skin_particle_idx{ _skin_particle_indices[vert_id] };
    cuVec4 skin_particle_weight{ _skin_particle_weights[vert_id] };

    int rigid_idx = _particle_phases[skin_particle_idx[0]] & kPhaseGroupMask;
    float R[3][3];
    extract_matrix(_rigid_rotations[rigid_idx], R); // should be more efficient, num_ops(rotate(q, vec3)) = num_ops(q->mat3) + num_ops(mat3 * vec3)
    float3 vert_pos_init = _skin_positions_init[vert_id];
    float3 vert_pos = make_float3(0.f);
    
    float3 particle_pos_init;
    float3 particle_pos;

    for (int i = 0; i < 4; ++i) {
        if (skin_particle_idx[i] > -1) {
            particle_pos_init = make_float3(_particle_positions_init[skin_particle_idx[i]]);
            particle_pos = make_float3(_particle_positions[skin_particle_idx[i]]);
            vert_pos += (mat3_by_float3(R, vert_pos_init - particle_pos_init) + particle_pos) * skin_particle_weight[i];
        }
    }
    _skin_positions[vert_id] = vert_pos;

}

__global__ void
calcSkinNormals_kernel(
    int num_skin_faces,
    float3 const *_skin_positions,
    uint const *_skin_indices,
    float3 *_skin_normals
)
{
    uint face_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (face_id >= num_skin_faces) return;

    uint offset = face_id * 3;
    uint idx_a = _skin_indices[offset];
    uint idx_b = _skin_indices[offset + 1];
    uint idx_c = _skin_indices[offset + 2];

    float3 a = _skin_positions[idx_a];
    float3 b = _skin_positions[idx_b];
    float3 c = _skin_positions[idx_c];

    float3 normal = SafeNormalize(cross(b - a, c - a));

    atomicAdd(&_skin_normals[idx_a].x, normal.x);
    atomicAdd(&_skin_normals[idx_a].y, normal.y);
    atomicAdd(&_skin_normals[idx_a].z, normal.z);

    atomicAdd(&_skin_normals[idx_b].x, normal.x);
    atomicAdd(&_skin_normals[idx_b].y, normal.y);
    atomicAdd(&_skin_normals[idx_b].z, normal.z);

    atomicAdd(&_skin_normals[idx_c].x, normal.x);
    atomicAdd(&_skin_normals[idx_c].y, normal.y);
    atomicAdd(&_skin_normals[idx_c].z, normal.z);
}



////////////////////////////////////////////////////////////////////////////////
// host functions
////////////////////////////////////////////////////////////////////////////////
void
createParticles(
    ParticlizeConfig _config,
    float3 _lower,
    int _group,
    int _particle_offset,
    int _rigid_particle_offset,
    int _num_voxels,
    float4 *_voxel_positions,
    float4 *_voxel_sdfs,
    float4 *_particle_positions,
    float  *_particle_inv_masses,
    float4 *_particle_velocities,
    int *_particle_phases,
    float4 *_rigid_particle_sdfs
)
{
    dim3 num_threads = make_uint3(1);
    dim3 num_blocks = make_uint3(1);
    computeGridSize(_num_voxels, kCudaBlockSize, num_blocks.x, num_threads.x);
    createShapeParticles_kernel <<<num_blocks, num_threads >>> (
        _lower,
        _config.velocity_,
        _config.L0_radius_,
        _config.inv_mass_,
        _config.flags_,
        _group,
        _particle_offset,
        _rigid_particle_offset,
        _num_voxels,
        _voxel_positions,
        _voxel_sdfs,
        _particle_positions,
        _particle_inv_masses,
        _particle_velocities,
        _particle_phases,
        _rigid_particle_sdfs
        );
    getLastCudaError("createShapeParticles_kernel");
}

void
calcSkinWeights(
    int _skin_pos_begin,
    int _skin_pos_end, // exclusive
    int _particle_begin,
    int _particle_end, // exclusive
    int3 _hash_grid_dim, // vol_dim + 1
    float3 _hash_cell_size,
    float3 _lower,
    float3 *_skin_positions,
    float4 *_particle_positions,
    int4 *_skin_particle_indices,
    float4 *_skin_particle_weights
)
{
    int num_particles = _particle_end - _particle_begin;
    int num_cells = _hash_grid_dim.x * _hash_grid_dim.y * _hash_grid_dim.z;
    int num_vertices = _skin_pos_end - _skin_pos_begin;
    uint *dGridParticleHashes;
    int *dGridParticleIndices, *dCellStarts, *dCellEnds;

    allocateArray((void**)&dGridParticleHashes, num_particles * sizeof(int));
    allocateArray((void**)&dGridParticleIndices, num_particles * sizeof(int));

    allocateArray((void**)&dCellStarts, num_cells * sizeof(int));
    allocateArray((void**)&dCellEnds, num_cells * sizeof(int));

    dim3 num_threads = make_uint3(1);
    dim3 num_blocks = make_uint3(1);
    computeGridSize(num_particles, kCudaBlockSize, num_blocks.x, num_threads.x);

    calcParticleHashes_kernel <<<num_blocks, num_threads>>> (
        _particle_begin,
        _particle_end,
        _hash_grid_dim,
        _hash_cell_size,
        _lower,
        _particle_positions,
        dGridParticleHashes,
        dGridParticleIndices
        );
    getLastCudaError("calcParticleHashes_kernel");

    thrust::sort_by_key(thrust::device,
        thrust::device_ptr<uint>(dGridParticleHashes),
        thrust::device_ptr<uint>(dGridParticleHashes + num_particles),
        thrust::device_ptr<int>(dGridParticleIndices));

    thrust::fill(thrust::device,
        thrust::device_ptr<int>(dCellStarts),
        thrust::device_ptr<int>(dCellStarts + num_cells),
        -1);

    computeGridSize(num_particles, kCudaBlockSize, num_blocks.x, num_threads.x);
    findHashCellOffsets_kernel <<<num_blocks, num_threads>>> (
        num_particles,
        dGridParticleHashes,
        dCellStarts,
        dCellEnds // exclusive
        );
    getLastCudaError("findHashCellOffsets_kernel");

    computeGridSize(num_vertices, kCudaBlockSize, num_blocks.x, num_threads.x);
    calcSkinWeights_kernel <<<num_blocks, num_threads>>> (
        _hash_grid_dim, // vol_dim + 1
        _hash_cell_size,
        _lower,
        dCellStarts,
        dCellEnds, // exclusive
        dGridParticleIndices,
        _skin_pos_begin,
        _skin_pos_end, // exclusive
        _skin_positions,
        _particle_positions,
        _skin_particle_indices,
        _skin_particle_weights
    );
    getLastCudaError("calcSkinWeights_kernel");
    
    FREE_CUDA_ARRAY(dGridParticleHashes);
    FREE_CUDA_ARRAY(dGridParticleIndices);
    FREE_CUDA_ARRAY(dCellStarts);
    FREE_CUDA_ARRAY(dCellEnds);
}

void
skinning(
    int _num_skin_vertices,
    int _num_skin_faces,
    float4 const *_particle_positions_init,
    float4 const *_particle_positions,
    int const *_particle_phases,
    quat const *_rigid_rotations,
    int4 const *_skin_particle_indices,
    float4 const *_skin_particle_weights,
    float3 const *_skin_positions_init,
    uint const *_skin_indices,
    float3 *_skin_positions,
    float3 *_skin_normals
)
{
    dim3 num_threads = make_uint3(1);
    dim3 num_blocks = make_uint3(1);
    computeGridSize(_num_skin_vertices, kCudaBlockSize, num_blocks.x, num_threads.x);

    BEGIN_TIMER("skinning_kernel");
    skinning_kernel <<<num_blocks, num_threads>>> (
        _num_skin_vertices,
        _particle_positions_init,
        _particle_positions,
        _particle_phases,
        _rigid_rotations,
        _skin_particle_indices,
        _skin_particle_weights,
        _skin_positions_init,
        _skin_positions
    );
    getLastCudaError("skinning_kernel");
    STOP_TIMER("skinning_kernel");

    thrust::fill(thrust::device,
        thrust::device_ptr<float3>(_skin_normals),
        thrust::device_ptr<float3>(_skin_normals + _num_skin_vertices),
        make_float3(0.f));

    computeGridSize(_num_skin_faces, kCudaBlockSize, num_blocks.x, num_threads.x);
    BEGIN_TIMER("calcSkinNormals_kernel");
    calcSkinNormals_kernel <<<num_blocks, num_threads>>> (
        _num_skin_faces,
        _skin_positions,
        _skin_indices,
        _skin_normals
    );
    getLastCudaError("calcSkinNormals_kernel");
    STOP_TIMER("calcSkinNormals_kernel");
}

void
setPlane(
    float4 *_planes,
    int _idx,
    float4 const &_plane
)
{
    thrust::fill(thrust::device, _planes + _idx, _planes + _idx + 1, _plane);
}

void 
dev_copy_float3(float3 * _first, float3 * _last, float3 * _result)
{
    thrust::copy(thrust::device, thrust::device_ptr<float3>(_first), thrust::device_ptr<float3>(_last), thrust::device_ptr<float3>(_result));
}


void 
dev_copy_float4(float4 * _first, float4 * _last, float4 * _result)
{
    thrust::copy(thrust::device, thrust::device_ptr<float4>(_first), thrust::device_ptr<float4>(_last), thrust::device_ptr<float4>(_result));
}



