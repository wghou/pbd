#pragma once
//#define _PROFILE_SOLVER_
#include "numerics.h"
#include <math_constants.h>

#define kBRTreeChildLeafFlag 0x80000000
#define kBRTreeChildIndexMask 0x7fffffff
// solver params
struct Params
{
    // solver parameters
    float3 gravity_;
    float  fluid_rest_extent_; // (0, 1], w.r.t. radius, distance(p_f1, p_f2) >= fluid_rest_extent_ * 2.f *  r_fluid
    float  solid_rest_extent_; // (0, 1], w.r.t. radius, distance(p1, p2) >= solid_rest_extent_ * (r1 + r2)
    float  shape_rest_extent_; // w.r.t. radius, distance(p, s) >= shape_rest_extent_ * r;
    float  particle_collision_margin_; // for particle p and its contact neighbor pn, distance(p, pn) < (r + rn) + particle_margin_
    float  shape_collision_margin_; // for particle p and contact neighbor shape s (e.g. plane...), distance(p, s) < r + shape_margin
    float  dynamic_friction_; 
    float  static_friction_;
    float  particle_friction_;
    
    float  fluid_neighbor_search_range_; // distance(pn, p) - rn < neighbor_search_range_

    float  sleeping_threshold_quad_;
    float  prestabilize_velocity_threshold_quad_;
    float  shock_propagation_;
    float  damping_force_factor_;

    // scene parameters
    int    num_particles_;
    int    num_rigid_particles_;
    int    rigid_particles_begin_;
    int    num_granular_particles_;
    int    granular_particles_begin_;
    int    num_fluid_particles_;
    int    fluid_particles_begin_;
    int    num_rigids_;
    int    num_planes_;

    float  fluid_rest_density_; // kg per m^3
    float  fluid_cfm_eps_; // constraint force mixing: relaxation parameter
    float  fluid_corr_k_;
    float  fluid_corr_refer_q_; // reference point inside SPH kernel radius
    float  fluid_corr_refer_inv_W_; // W(refer_q, h)
    float  fluid_corr_n_; 

    float  inv_min_particle_radius_; // for morton code
    float  min_particle_radius_; // should be r_fluid
    float  min_rigid_surface_sdf_; // to identify surface particle, 2 * min_rigid_particle_radius

    // solver settings
    int    num_substeps_;
    int    num_iterations_;
    int    num_pre_stabilize_iterations_;

    // simulation params
    int    num_frames_;
    float  exec_time_;
    float  frame_delta_time_;
    float  step_delta_time_; // frame_delta_time / num_substeps

    // internal params
    // int    max_num_contact_constraints_; // 2 * max_num_particles; {n = num_rigids * (edge_len^3), num_rigids <= n / (2^3); num_contacts = num_rigids * 6 * (edge_len^2) / 2 = 3 * [num_rigids^(1/3)] * [n^(2/3)] <= 1.5 * n}
    int    max_num_neighbors_per_particle_;
    int    stack_size_per_particle_;
    int    morton_bits_per_axis_;
    float  particle_upper_; // (2^(morton_bits_per_axis_ + 1) - 1) * min_particle_radius_ / 2, lower = -upper
    int    max_iter_extract_rotation_;

#ifndef __CUDACC__
    // default constructor
    Params() 
    {
        gravity_ = make_float3(0.f, -10.f, 0.f);
        fluid_rest_extent_ = 0.55f;
        solid_rest_extent_ = 1.f;
        shape_rest_extent_ = 1.f;
        particle_collision_margin_ = 0.04f;
        shape_collision_margin_ = 0.f;
        dynamic_friction_ = 0.25f;
        static_friction_ = 0.f;
        particle_friction_ = 0.03f;
        
        fluid_neighbor_search_range_ = 0.f;

        sleeping_threshold_quad_ = 0.01f;
        prestabilize_velocity_threshold_quad_ = 0.16f;
        shock_propagation_ = 3.f;
        damping_force_factor_ = 0.f;

        num_particles_ = 0;
        num_rigid_particles_ = 0;
        rigid_particles_begin_ = 0;
        num_granular_particles_ = 0;
        granular_particles_begin_ = 0;
        num_fluid_particles_ = 0;
        fluid_particles_begin_ = 0;
        num_rigids_ = 0;
        num_planes_ = 0;

        fluid_cfm_eps_ = 600.f; // constraint force mixing: relaxation parameter
        fluid_corr_k_ = 0.015f;
        fluid_corr_n_ = 4.f;

        num_substeps_ = 1;
        num_iterations_ = 4;
        num_pre_stabilize_iterations_ = 0;

        num_frames_ = 0;
        exec_time_ = 0.f;
        frame_delta_time_ = 1.f / 60.f;
        step_delta_time_ = frame_delta_time_ / num_substeps_;

        max_num_neighbors_per_particle_ = 32;
        stack_size_per_particle_ = 64;

        //     45-bit morton + 17-bit particle_indices (duplicated morton)  
        // ==> tree depth: 46 + 18 = 64 (traveral: LevelIndex at least 64 bit)
        morton_bits_per_axis_ = 15;
        max_iter_extract_rotation_ = 5;
    }
#endif
};

struct CubicSmoothKernel {
public:
    float h_; // smooth radius
    float inv_h_;
    float w_0_;

    inline __host__ __device__
        float W(float3 const &_r_vec)
    {
        float weight = 0.f;
        const float q = length(_r_vec) / h_;
        if (q < 0.5f) {
            const float qq = q * q;
            const float qqq = q * qq;
            weight = K1_ * (6.f * qqq - 6.f * qq + 1.f);
        }
        else if (q < 1.f) {
            const float p = 1.f - q;
            weight = K1_ * (2.f * p * p * p);
        }
        return weight;
    }

    inline __host__ __device__
        float3 grad_W(float3 const &_r_vec)
    {
        float3 grad_weight = make_float3(0.f);
        const float r = length(_r_vec);
        const float q = r * inv_h_;
        if (r > 1e-6f) {
            const float3 grad_q = _r_vec * inv_h_ / r;
            if (q < 0.5f) {
                grad_weight = K2_ * q * (3.f * q - 2.f) * grad_q;
            }
            else if (q < 1.f) {
                const float p = 1.f - q;
                grad_weight = -K2_ * p * p * grad_q;
            }
        }
        return grad_weight;
    }

#ifndef __CUDACC__    
    CubicSmoothKernel(float _h) : h_(_h)
    {
        inv_h_ = 1.f / h_;
        const float inv_hhh = inv_h_ * inv_h_ * inv_h_;
        K1_ = 4.f * CUDART_2_OVER_PI_F * inv_hhh; // 8 / (pi * h * h * h)
        K2_ = 24.f * CUDART_2_OVER_PI_F * inv_hhh; // 48 / (pi * h * h * h)
        w_0_ = W(make_float3(0.f));
    }
#endif    


    float K1_;
    float K2_;
};

void setSmoothKernel(CubicSmoothKernel *_hSmoothKernel);

void postInitDevMem(
    int _num_particles,
    int _rigid_particles_begin,
    int _num_rigid_particles,
    int _num_rigids,
    float _min_rigid_particle_radius,
    float4 *_particle_positions,
    float4 *_particle_positions_init,
    int  *_rigid_particle_offsets,
    int *_rigid_internal_particle_offsets,
    float *_particle_inv_masses,
    float *_rigid_particle_weights,
    quat *_rigid_rotations
);

void setParameters(Params *_hParams);

void integrate(
    int _num_particles,
    float4 *_particle_positions,
    float4 const *_planes,
    int const *_particle_phases,
    float  *_particle_inv_masses,
    float4 *_particle_velocities,
    float4 *_particle_positions_pred
);

void swapParticleMass(
    int _num_particles,
    float4 *_particle_velocities,
    float *_particle_inv_masses
);

void findNeighbors(
    int _num_particles,
    int _max_num_neighbors_per_particle,
    int const* _particle_phases,
    float4 *_particle_positions_pred, // positions will be clamped
    uint64_t *_particle_morton_codes,
    int *_particle_indices,
    int2 *_BRTree_internal_nodes_children,
    int *_BRTree_internal_nodes_parents,
    int *_BRTree_leaf_nodes_parents,
    int *_atomic_counters_, // dConstraintCounts_
    float4 *_particle_bounding_spheres_aux, // dConstraintDeltas_
    float4 *_particle_bounding_spheres,
    int *_BRTree_traversal_stack,
    int *_particle_neighbors
);

void findNeighbors_AABB(
    int _num_particles,
    int _max_num_neighbors_per_particle,
    int const* _particle_phases,
    float4 *_particle_positions_pred, // positions will be clamped
    uint64_t *_particle_morton_codes,
    uint64_t *_particle_morton_codes_aux,
    int *_particle_indices,
    int *_particle_indices_aux,
    int2 *_BRTree_internal_nodes_children,
    int *_BRTree_internal_nodes_parents,
    int *_BRTree_leaf_nodes_parents,
    int *_atomic_counters_, // dConstraintCounts_
    cuAABB *_particle_AABBs,
    int *_BRTree_traversal_stack,
    int *_particle_neighbors
);

void solveContacts(
    int _num_particles,
    int _is_pre_stabilize_mode, // 1: pre-stabilization, 0: solver iteration
    int const *_particle_neighbors,
    int const *_particle_phases,
    float const *__particle_inv_masses,
    float4 const *_rigid_sdfs,
    quat const *_rigid_rotations,
    float4 const *_planes,
    float4 *_constraint_deltas,
    int *_constraint_counts,
    float4 *_particle_positions,
    float4 *_particle_positions_pred,
    float4 const *_particle_velocities
);

void matchRigidShapes_cub(
    int _num_rigids,
    int _num_rigid_particles,
    int _rigid_particles_begin,
    int const *_particle_phases,
    int *_rigid_particle_offsets, // cannot be const, requirement from cub
    int *_rigid_internal_particle_offsets,
    float4 *_particle_positions_init,
    float *_rigid_particle_weights,
    float *_rigid_stiffnesses,
    float4 *_rigid_centers_of_mass,
    float4 *_rigid_centers_of_mass_init,
    float *_rigid_weights,
    float4 *_rigid_relative_positions,
    float4 *_rigid_relative_positions_init,
    mat3 *_rigid_particle_cov_matrices,
    mat3 *_rigid_cov_matrices,
    quat *_rigid_rotations,
    float4 *_particle_positions_pred
);

void matchRigidShapes(
    int _num_rigids,
    int _num_rigid_particles,
    int const *_particle_phases,
    int const *_rigid_particle_offsets,
    float4 const *_particle_positions_init,
    float const *_rigid_particle_weights,
    float const *_rigid_stiffnesses,
    float4 *_rigid_centers_of_mass,
    float4 *_rigid_relative_positions,
    float4 *_rigid_relative_positions_init,
    quat *_rigid_rotations,
    float4 *_particle_positions_pred
);

void
solverFluidDensityConstraints(
    int _num_fluid_particles,
    float4 *_particle_positions_pred,
    int const *_particle_phases,
    int const *_particle_neighbors,
    float const *_particle_inv_masses,
    float  *_fluid_lambdas
);

void finalize(
    int _num_particles,
    float4 const *_particle_positions_pred,
    float4 *_particle_velocities,
    float4 *_particle_positions
);