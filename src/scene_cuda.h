#pragma once
//#define _PROFILE_SOLVER_
#include "cuda_helper.h"

// for shape particlization
struct ParticlizeConfig
{
    float3 velocity_{ 0.f };
    float inv_mass_ = 1.f;
    float stiffness_ = 1.f;

    uint3 volume_dim;
    float L0_radius_; // radius of the finest level
    float sdf_merge_threshold = -1.f; // (assume radius = 0.5) for grain: -FLT_MAX; for rigid: generally: -1
    int flags_ = 0;
    bool skinning_ = true;

};

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
);

void
calcSkinWeights(
    int _skin_pos_begin,
    int _skin_pos_end,
    int _particle_begin,
    int _particle_end, // exclusive
    int3 _hash_grid_dim,
    float3 _hash_cell_size,
    float3 _lower,
    float3 *_skin_positions,
    float4 *_particle_positions,
    int4 *_skin_particle_indices,
    float4 *_skin_particle_weights
);

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
);

void
setPlane(
    float4 *_planes,
    int _idx,
    float4 const &_plane
);

void
dev_copy_float3(float3* _first, float3* _last, float3* _result);

void
dev_copy_float4(float4* _first, float4* _last, float4* _result);