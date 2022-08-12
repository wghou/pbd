#include "solver_cuda.h"
#include "cuda_helper.h"

#include <device_launch_parameters.h>

#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>

//#define CUB_STDERR
#include <cub/cub.cuh>
#include <iostream>

__constant__ Params kParams;
__constant__ CubicSmoothKernel kSmoothKernel;
  
////////////////////////////////////////////////////////////////////////////////
// device functions
////////////////////////////////////////////////////////////////////////////////

inline __device__
bool
isRigidParticle(int _phase)
{
    return (_phase & kPhaseGroupMask) == _phase;
}

inline __device__
bool
isSelfCollideOn(int _phase)
{
    return (_phase & kPhaseSelfCollideFlag) != 0;
}

inline __device__
bool
isFluid(int _phase)
{
    return (_phase & kPhaseFluidFlag) != 0;
}

// method to seperate bits from a given integer 3 positions apart
inline __device__ __host__
uint64_t splitBy3(unsigned int a) {
    uint64_t x = a & 0x1fffff; // we only look at the first 21 bits
    x = (x | x << 32) & 0x1f00000000ffff;  // shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
    x = (x | x << 16) & 0x1f0000ff0000ff;  // shift left 16 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
    x = (x | x << 8) & 0x100f00f00f00f00f; // shift left 8 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
    x = (x | x << 4) & 0x10c30c30c30c30c3; // shift left 4 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
    x = (x | x << 2) & 0x1249249249249249;
    return x;
}

inline __device__ __host__
uint64_t encodeMorton(unsigned int x, unsigned int y, unsigned int z) {
    uint64_t answer = 0;
    answer |= splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2;
    return answer;
}

inline __device__
int deltaMorton(int _i_idx, int _j_idx, uint64_t const *_particle_morton_codes)
{
    if (_j_idx < 0 || _j_idx > kParams.num_particles_ - 1) return -1;

    uint64_t i = _particle_morton_codes[_i_idx];
    uint64_t j = _particle_morton_codes[_j_idx];
    
    int augmented = (i == j ? __clz(_i_idx ^ _j_idx) : 0); // for duplicated keys
    return __clzll(i ^ j) + augmented;
}

inline __device__
float4 calcBoundingSphere(float4 const &_p1, float4 const &_p2)
{
    float3 v12 = make_float3(_p2 - _p1);
    float3 dir_12 = make_float3(0.f, 0.f, 1.f);
    float dist = length(v12);
    if (dist > 0.f) {
        dir_12 = v12 / dist;
    }
    float d_neg = fminf(-_p1.w, dist - _p2.w);
    float d_pos = fmaxf(_p1.w, dist + _p2.w);
    float r = 0.5f * (d_pos - d_neg);
    float3 c = make_float3(_p1) + 0.5f * (d_pos + d_neg) * dir_12;
    return make_float4(c, r);
}


inline __device__
bool isNeighbor(float4 const &_p, float4 const &_pn, float _range)
{
    float3 v = make_float3(_p - _pn);
    float dist_sqr = dot(v, v);
    float r_sum = _range + _pn.w;
    return dist_sqr < r_sum * r_sum;
}

inline __device__
bool isNeighbor(float4 const &_p, float4 const &_pn, float _range, float &_sq_dist)
{
    float3 v = make_float3(_p - _pn);
    _sq_dist = dot(v, v);
    float r_sum = _range + _pn.w;
    return _sq_dist < r_sum * r_sum;
}


inline __device__
bool isCollisionEnabled(int _idx, int _idx_n, int _phase, int _phase_n)
{
    if (_idx == _idx_n) return false;

    if ((_phase == _phase_n) && !isSelfCollideOn(_phase)) {
        return false;
    }

    if (!isFluid(_phase) && !isFluid(_phase_n) && _idx > _idx_n) {
        return false;
    }

    return true;
}

// confine particles to the valid space specified by the planes and eliminate the velocity in the normal direction
inline __device__
void confineParticle(float4 &_pos, float3 &_velocity, float4 const *_planes)
{
    for (int k = 0; k < kParams.num_planes_; ++k) {
        float4 plane = _planes[k];
        float3 N = make_float3(plane);
        float C = dot(make_float3(_pos), N) + plane.w - 1.2f * _pos.w;
        if (C < 0) {
            _pos -= make_float4(C * N, 0.f);
            //_velocity -= dot(_velocity, N) * N;
            _velocity = make_float3(0.f);
        }
    }
}
////////////////////////////////////////////////////////////////////////////////
// kernels
////////////////////////////////////////////////////////////////////////////////

__global__ void
integrate_kernel(
    float4 const *_particle_positions,
    float4 const *_planes,
    int const *_particle_phases,
    float *_particle_inv_masses,
    float4 *_particle_velocities,
    float4 *_particle_positions_pred)
{
    uint particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= kParams.num_particles_) return;

    float4 pos = _particle_positions[particle_id];
    float3 v = make_float3(_particle_velocities[particle_id]);
    float3 damping = -kParams.damping_force_factor_ * v;
    v += kParams.step_delta_time_ * (kParams.gravity_ + damping);
    pos += make_float4(kParams.step_delta_time_ * v, 0.f);
    
    float4 min_pos = make_float4(-kParams.particle_upper_);
    float4 max_pos = make_float4( kParams.particle_upper_);
    min_pos.w = pos.w;
    max_pos.w = pos.w;
    pos = clamp(pos, min_pos, max_pos);

    float inv_mass = _particle_inv_masses[particle_id];
    _particle_velocities[particle_id] = make_float4(v, inv_mass);
    double shock_factor = exp(-(double)kParams.shock_propagation_ * pos.y);
    _particle_inv_masses[particle_id] = fminf((1.0 / shock_factor) * inv_mass, FLT_MAX);
    _particle_positions_pred[particle_id] = pos;
}


__global__ void
swapParticleMass_kernel(
    float4 *_particle_velocities,
    float *_particle_inv_masses
)
{
    uint particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= kParams.num_particles_) return;
    
    float mass_init = _particle_velocities[particle_id].w;
    float mass_modified = _particle_inv_masses[particle_id];

    _particle_inv_masses[particle_id] = mass_init;
    _particle_velocities[particle_id].w = mass_modified;
}


__global__ void
calcMortonCode_kernel(
    float4 *_particle_positions_pred,
    uint64_t *_particle_morton_codes,
    int *_particle_indices
)
{
    uint particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= kParams.num_particles_) return;

    // pos_pred - lower = pos_pred - (-upper) = pos_pred + upper
    float3 morton_pos = make_float3(_particle_positions_pred[particle_id]) + kParams.particle_upper_;
    uint3 morton_coord = make_uint3(floorf(morton_pos * kParams.inv_min_particle_radius_));
    _particle_indices[particle_id] = particle_id;
    _particle_morton_codes[particle_id] = encodeMorton(morton_coord.x, morton_coord.y, morton_coord.z);
}


// sort ke_value_pair <_particle_morton_codes, _particle_indices>
// parents of internal nodes init with -1 (0xffffffff), parent(root) = -1, termination condition in buildBoundingSphereTree 
__global__ void
buildBRTree_kernel(
    uint64_t const *_particle_morton_codes,
    int2 *_BRTree_internal_nodes_children,
    int *_BRTree_internal_nodes_parents,
    int *_BRTree_leaf_nodes_parents
)
{
    int internal_node_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (internal_node_id >= kParams.num_particles_ - 1) return;

    int d = sign(deltaMorton(internal_node_id, internal_node_id + 1, _particle_morton_codes) - deltaMorton(internal_node_id, internal_node_id - 1, _particle_morton_codes));
    
    // Compute upper bound for the length of the range
    int delta_min = deltaMorton(internal_node_id, internal_node_id - d, _particle_morton_codes);
    int L_max = 2;
    while (deltaMorton(internal_node_id, internal_node_id + L_max * d, _particle_morton_codes) > delta_min) {
        L_max = L_max * 2;
    }

    // Find the other end using binary search
    int L = 0;
    for (int t = L_max / 2; t >= 1; t = t >> 1) {
        if (deltaMorton(internal_node_id, internal_node_id + (L + t) * d, _particle_morton_codes) > delta_min) {
            L = L + t;
        }
    }

    int j = internal_node_id + L * d;

    // Find the split position using binary search
    int delta_node = deltaMorton(internal_node_id, j, _particle_morton_codes);
    int s = 0;
    for (int t = ceil_div2(L); t >= 1; t = ceil_div2(t)) {
        if (deltaMorton(internal_node_id, internal_node_id + (s + t) * d, _particle_morton_codes) > delta_node) {
            s = s + t;
        }
        if (t == 1) break;
    }

    int gamma = internal_node_id + s * d + min(d, 0);

    // Output child pointers
    int2 children = make_int2(gamma, gamma + 1);
    if (min(internal_node_id, j) == gamma) {
        children.x |= kBRTreeChildLeafFlag;
    }
    if (max(internal_node_id, j) == gamma + 1) {
        children.y |= kBRTreeChildLeafFlag;
    }

    _BRTree_internal_nodes_children[internal_node_id] = children;
    
    // Output parent pointers
    int *parents[2] = { _BRTree_internal_nodes_parents, _BRTree_leaf_nodes_parents };
    int left_parent_ptr_idx = (children.x & kBRTreeChildLeafFlag) != 0;
    int right_parent_ptr_idx = (children.y & kBRTreeChildLeafFlag) != 0;
    parents[left_parent_ptr_idx][gamma] = internal_node_id;
    parents[right_parent_ptr_idx][gamma + 1] = internal_node_id;
}


// _atomic_counters_ init with 0
__global__ void
buildBoundingSphereTree_kernel(
    int const *_BRTree_internal_nodes_parents,
    int2 const *_BRTree_internal_nodes_children,
    int const *_BRTree_leaf_nodes_parents,
    int const* _particle_phases,
    int const *_particle_indices,
    float4 const *_particle_positions_pred,
    int *_atomic_counters_, // dConstraintCounts_, internal_phase
    //float4 *_particle_bounding_spheres_aux, // dConstraintDeltas_
    float4 *_particle_bounding_spheres,
    int *_BRTree_nodes_siblings // dParticleMortonCodes_
)
{
    uint particle_idx_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_idx_id >= kParams.num_particles_) return;

    int parent_idx = _BRTree_leaf_nodes_parents[particle_idx_id];
    int current_idx = particle_idx_id | kBRTreeChildLeafFlag;
    int particle_idx = _particle_indices[particle_idx_id];

    // assume num_particles > 1
    if (parent_idx == -1) {
        printf("Particle has no parent. There must be something wrong with buildBRTree\n");
    }

    if (particle_idx_id == 0) {
        _BRTree_nodes_siblings[0] = 0;
    }

    if (!atomicAdd(&_atomic_counters_[parent_idx], 1)) {
        return;
    }
    else {
        int2 siblings = _BRTree_internal_nodes_children[parent_idx];
        int sibling_idx = siblings.x != current_idx ? siblings.x : siblings.y;
        int curr_phase = _particle_phases[particle_idx];
        int curr_idx_for_sibling = particle_idx_id + kParams.num_particles_ - 1;
        
        int sibling_phase;
        int sibling_idx_for_sibling = sibling_idx & kBRTreeChildIndexMask;
        if (sibling_idx & kBRTreeChildLeafFlag) {
            int sibling_particle_idx = _particle_indices[sibling_idx_for_sibling];
            sibling_phase = _particle_phases[sibling_particle_idx];
            sibling_idx_for_sibling += kParams.num_particles_ - 1;
        }
        else {
            sibling_phase = _atomic_counters_[sibling_idx];
        }
        _atomic_counters_[parent_idx] = (sibling_phase == curr_phase && !isSelfCollideOn(curr_phase)) ? curr_phase : -1;

        _BRTree_nodes_siblings[curr_idx_for_sibling] = sibling_idx;
        _BRTree_nodes_siblings[sibling_idx_for_sibling] = current_idx;

        float4 sibling_bsphere;
        if (sibling_idx & kBRTreeChildLeafFlag) {
            int p1_idx = _particle_indices[sibling_idx & kBRTreeChildIndexMask];
            sibling_bsphere = _particle_positions_pred[p1_idx];
        }
        else {
            sibling_bsphere = _particle_bounding_spheres[sibling_idx];
        }
        _particle_bounding_spheres[parent_idx] = calcBoundingSphere(_particle_positions_pred[particle_idx], sibling_bsphere);
        current_idx = parent_idx;
        parent_idx = _BRTree_internal_nodes_parents[current_idx];
    }

    while (parent_idx != -1) {
        if (!atomicAdd(&_atomic_counters_[parent_idx], 1)) {
            return;
        }
        else {
            int curr_phase = _atomic_counters_[current_idx];

            int2 siblings = _BRTree_internal_nodes_children[parent_idx];
            int sibling_idx = siblings.x != current_idx ? siblings.x : siblings.y;
            float4 sibling_bsphere;
            int sibling_phase, sibling_idx_for_sibling;

            if (sibling_idx & kBRTreeChildLeafFlag) {
                sibling_idx_for_sibling = (sibling_idx & kBRTreeChildIndexMask) + kParams.num_particles_ - 1;
                int p1_idx = _particle_indices[sibling_idx & kBRTreeChildIndexMask];
                sibling_bsphere = _particle_positions_pred[p1_idx];
                sibling_phase = _particle_phases[p1_idx];
            }
            else {
                sibling_idx_for_sibling = sibling_idx;
                sibling_phase = _atomic_counters_[sibling_idx];
                sibling_bsphere = _particle_bounding_spheres[sibling_idx];
            }
            _atomic_counters_[parent_idx] = (sibling_phase == curr_phase && !isSelfCollideOn(curr_phase)) ? curr_phase : -1;
            
            _BRTree_nodes_siblings[current_idx] = sibling_idx;
            _BRTree_nodes_siblings[sibling_idx_for_sibling] = current_idx;

            _particle_bounding_spheres[parent_idx] = calcBoundingSphere(_particle_bounding_spheres[current_idx], sibling_bsphere);
            current_idx = parent_idx;
            parent_idx = _BRTree_internal_nodes_parents[current_idx];
        }
    }
}

// _atomic_counters_ init with 0
__global__ void
buildAABBTree_kernel(
    int const *_BRTree_internal_nodes_parents,
    int2 const *_BRTree_internal_nodes_children,
    int const *_BRTree_leaf_nodes_parents,
    int const* _particle_phases,
    int const *_particle_indices,
    float4 const *_particle_positions_pred,
    int *_atomic_counters_, // dConstraintCounts_, internal_phase
    cuAABB *_particle_AABBs,
    int *_BRTree_nodes_siblings // dParticleMortonCodes_
)
{
    uint particle_idx_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_idx_id >= kParams.num_particles_) return;

    int parent_idx = _BRTree_leaf_nodes_parents[particle_idx_id];
    int current_idx = particle_idx_id | kBRTreeChildLeafFlag;
    int particle_idx = _particle_indices[particle_idx_id];

    // assume num_particles > 1
    if (parent_idx == -1) {
        printf("Particle has no parent. There must be something wrong with buildBRTree\n");
    }

    if (particle_idx_id == 0) {
        _BRTree_nodes_siblings[0] = 0;
    }

    if (!atomicAdd(&_atomic_counters_[parent_idx], 1)) {
        return;
    }
    else {
        int2 siblings = _BRTree_internal_nodes_children[parent_idx];
        int sibling_idx = siblings.x != current_idx ? siblings.x : siblings.y;
        int curr_phase = _particle_phases[particle_idx];
        int curr_idx_for_sibling = particle_idx_id + kParams.num_particles_ - 1;

        int sibling_phase;
        int sibling_idx_for_sibling = sibling_idx & kBRTreeChildIndexMask;
        if (sibling_idx & kBRTreeChildLeafFlag) {
            int sibling_particle_idx = _particle_indices[sibling_idx_for_sibling];
            sibling_phase = _particle_phases[sibling_particle_idx];
            sibling_idx_for_sibling += kParams.num_particles_ - 1;
        }
        else {
            sibling_phase = _atomic_counters_[sibling_idx];
        }
        _atomic_counters_[parent_idx] = (sibling_phase == curr_phase && !isSelfCollideOn(curr_phase)) ? curr_phase : -1;

        _BRTree_nodes_siblings[curr_idx_for_sibling] = sibling_idx;
        _BRTree_nodes_siblings[sibling_idx_for_sibling] = current_idx;

        if (sibling_idx & kBRTreeChildLeafFlag) {
            int p1_idx = _particle_indices[sibling_idx & kBRTreeChildIndexMask];
            float4 sibling_bsphere = _particle_positions_pred[p1_idx];
            _particle_AABBs[parent_idx] = calcSpheresAABB(_particle_positions_pred[particle_idx], sibling_bsphere);
        }
        else {
            cuAABB sibling_AABB = _particle_AABBs[sibling_idx];
            _particle_AABBs[parent_idx] = calcSphereBoxAABB(_particle_positions_pred[particle_idx], sibling_AABB);
        }
        
        current_idx = parent_idx;
        parent_idx = _BRTree_internal_nodes_parents[current_idx];
    }

    while (parent_idx != -1) {
        if (!atomicAdd(&_atomic_counters_[parent_idx], 1)) {
            return;
        }
        else {
            int curr_phase = _atomic_counters_[current_idx];

            int2 siblings = _BRTree_internal_nodes_children[parent_idx];
            int sibling_idx = siblings.x != current_idx ? siblings.x : siblings.y;
            int sibling_phase, sibling_idx_for_sibling;

            if (sibling_idx & kBRTreeChildLeafFlag) {
                sibling_idx_for_sibling = (sibling_idx & kBRTreeChildIndexMask) + kParams.num_particles_ - 1;
                int p1_idx = _particle_indices[sibling_idx & kBRTreeChildIndexMask];
                sibling_phase = _particle_phases[p1_idx];
                float4 sibling_bsphere = _particle_positions_pred[p1_idx];
                _particle_AABBs[parent_idx] = calcSphereBoxAABB(sibling_bsphere, _particle_AABBs[current_idx]);
            }
            else {
                sibling_idx_for_sibling = sibling_idx;
                sibling_phase = _atomic_counters_[sibling_idx];
                cuAABB sibling_AABB = _particle_AABBs[sibling_idx];
                _particle_AABBs[parent_idx] = calcBoxesAABB(sibling_AABB, _particle_AABBs[current_idx]);
            }
            _atomic_counters_[parent_idx] = (sibling_phase == curr_phase && !isSelfCollideOn(curr_phase)) ? curr_phase : -1;

            _BRTree_nodes_siblings[current_idx] = sibling_idx;
            _BRTree_nodes_siblings[sibling_idx_for_sibling] = current_idx;

            
            current_idx = parent_idx;
            parent_idx = _BRTree_internal_nodes_parents[current_idx];
        }
    }
}

// _particle_neighbors, init with -1
// num_particles must > 1
__global__ void
findNeighbors_kernel(
    int const *_BRTree_internal_nodes_parents,
    int const *_BRTree_leaf_nodes_parents,
    int2 const *_BRTree_internal_nodes_children,
    //int const *_BRTree_nodes_siblings, // dParticleMortonCodes_
    int const *_particle_indices,
    float4 const *_particle_bounding_spheres,
    float4 const *_particle_positions_pred,
    int const* _particle_phases,
    int *_particle_neighbors
)
{
    uint particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= kParams.num_particles_) return;

    float4 pos = _particle_positions_pred[particle_id];
    int phase = _particle_phases[particle_id];
    float search_range = isFluid(phase) ? kParams.fluid_neighbor_search_range_ : (pos.w + kParams.particle_collision_margin_);
    int neighbor_offset = particle_id * kParams.max_num_neighbors_per_particle_;
    int neighbor_end = (particle_id + 1) * kParams.max_num_neighbors_per_particle_; // exclusive

    uint64_t level_idx = 0;
    int node = 0; // init with root (0), num_particles must > 1

    float4 const *pos_ptr[2] = { _particle_bounding_spheres, _particle_positions_pred };
    int const *parents_ptr[2] = { _BRTree_internal_nodes_parents, _BRTree_leaf_nodes_parents };

    do {
        if (node & kBRTreeChildLeafFlag) {
            int idx_n = _particle_indices[node & kBRTreeChildIndexMask];
            int phase_n = _particle_phases[idx_n];
            if ( isCollisionEnabled(particle_id, idx_n, phase, phase_n) /*&& isNeighbor(pos, pos_n, search_range)*/ ) {
                
                if (neighbor_offset == neighbor_end) {
                    //printf("particle %d: neighbor_offset %d too large\n", particle_id, neighbor_offset);
                    break;
                    //--neighbor_offset;
                }

                _particle_neighbors[neighbor_offset] = idx_n;
                ++neighbor_offset;

                // test here seems useless, always fails at neighor_offset == neighbor_end
#if 0
                if (neighbor_offset >= neighbor_end) {
                    printf("thread %d: array of neighbors is full!\n");
                    --neighbor_offset;
                }
#endif           
            }
        }
        else {
            int2 children = _BRTree_internal_nodes_children[node];

            int left_pos_ptr_idx = (children.x & kBRTreeChildLeafFlag) != 0;
            int right_pos_ptr_idx = (children.y & kBRTreeChildLeafFlag) != 0;
            int left_pos_idx = left_pos_ptr_idx ? _particle_indices[children.x & kBRTreeChildIndexMask] : children.x;
            int right_pos_idx = right_pos_ptr_idx ? _particle_indices[children.y & kBRTreeChildIndexMask] : children.y;

            float4 bs_left = pos_ptr[left_pos_ptr_idx][left_pos_idx];
            float4 bs_right = pos_ptr[right_pos_ptr_idx][right_pos_idx];

            bool test_left = isNeighbor(pos, bs_left, search_range);
            bool test_right = isNeighbor(pos, bs_right, search_range);
            if (test_left || test_right) { // any accepted
                level_idx <<= 1;
                if (test_left && test_right) {
                    float3 v_left = make_float3(pos - bs_left);
                    float3 v_right = make_float3(pos - bs_right);
                    node = dot(v_left, v_left) < dot(v_right, v_right) ? children.x : children.y; // choose the nearest child
                }
                else { // rejected one child
                    node = test_left ? children.x : children.y;
                    level_idx += 1;
                }
                continue;
            }
        }

        level_idx += 1;
        while ((level_idx & 1) == 0) {
            int parent_ptr_idx = (node & kBRTreeChildLeafFlag) != 0;
            node = parents_ptr[parent_ptr_idx][node & kBRTreeChildIndexMask];
            level_idx = level_idx >> 1;
        }
        if (node != 0) { // find sibling
            int parent_ptr_idx = (node & kBRTreeChildLeafFlag) != 0;
            int parent = parents_ptr[parent_ptr_idx][node & kBRTreeChildIndexMask];
            int2 siblings = _BRTree_internal_nodes_children[parent];
            node = (node != siblings.x) ? siblings.x : siblings.y;
        }
    } while (node != 0);

}


// _particle_neighbors, init with -1
// num_particles must > 1
__global__ void
findNeighbors_mbvh2_kernel(
    int const *_BRTree_internal_nodes_parents,
    int const *_BRTree_leaf_nodes_parents,
    int2 const *_BRTree_internal_nodes_children,
    int const *_BRTree_nodes_siblings, // dParticleMortonCodes_
    int const *_particle_indices,
    float4 const *_particle_bounding_spheres,
    float4 const *_particle_positions_pred,
    int const* _particle_phases,
    int *_particle_neighbors
)
{
    uint particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= kParams.num_particles_) return;

    float4 pos = _particle_positions_pred[particle_id];
    int phase = _particle_phases[particle_id];
    float search_range = isFluid(phase) ? kParams.fluid_neighbor_search_range_ : (pos.w + kParams.particle_collision_margin_);
    int neighbor_offset = particle_id * kParams.max_num_neighbors_per_particle_;
    int neighbor_end = (particle_id + 1) * kParams.max_num_neighbors_per_particle_; // exclusive

    uint64_t level_idx = 0;
    int node = 0; // init with root (0), num_particles must > 1
    bool is_traversal_over = false;

    float4 const *pos_ptr[2] = { _particle_bounding_spheres, _particle_positions_pred };
    int const *parents_ptr[2] = { _BRTree_internal_nodes_parents, _BRTree_leaf_nodes_parents };

    do {
        if (node & kBRTreeChildLeafFlag) {
            int idx_n = _particle_indices[node & kBRTreeChildIndexMask];
            int phase_n = _particle_phases[idx_n];
            if (isCollisionEnabled(particle_id, idx_n, phase, phase_n) /*&& isNeighbor(pos, pos_n, search_range)*/) {

                if (neighbor_offset == neighbor_end) {
                    //printf("particle %d: neighbor_offset %d too large\n", particle_id, neighbor_offset);
                    break;
                    //--neighbor_offset;
                }

                _particle_neighbors[neighbor_offset] = idx_n;
                ++neighbor_offset;         
            }
        }
        else {
            int2 children = _BRTree_internal_nodes_children[node];

            int left_pos_ptr_idx = (children.x & kBRTreeChildLeafFlag) != 0;
            int right_pos_ptr_idx = (children.y & kBRTreeChildLeafFlag) != 0;
            int left_pos_idx = left_pos_ptr_idx ? _particle_indices[children.x & kBRTreeChildIndexMask] : children.x;
            int right_pos_idx = right_pos_ptr_idx ? _particle_indices[children.y & kBRTreeChildIndexMask] : children.y;

            float4 bs_left = pos_ptr[left_pos_ptr_idx][left_pos_idx];
            float4 bs_right = pos_ptr[right_pos_ptr_idx][right_pos_idx];

            bool test_left = isNeighbor(pos, bs_left, search_range);
            bool test_right = isNeighbor(pos, bs_right, search_range);
            if (test_left || test_right) { // any accepted
                level_idx <<= 1;
                if (test_left && test_right) {
                    float3 v_left = make_float3(pos - bs_left);
                    float3 v_right = make_float3(pos - bs_right);
                    node = dot(v_left, v_left) < dot(v_right, v_right) ? children.x : children.y; // choose the nearest child
                    level_idx |= 1;
                }
                else { // rejected one child
                    node = test_left ? children.x : children.y;
                }
                continue;
            }
        }

        //level_idx += 1;
        while ((level_idx & 1) == 0) {
            if (level_idx == 0) {
                is_traversal_over = true;
                break;
            }
            int parent_ptr_idx = (node & kBRTreeChildLeafFlag) != 0;
            node = parents_ptr[parent_ptr_idx][node & kBRTreeChildIndexMask];
            level_idx = level_idx >> 1;
        }
        //if (node == 0) {
        //    printf("!!!!!!!!!!ERROR: backtrack node == 0 !!!!!!!!!!!!!!\n");
        //}

        // find sibling
        //int parent_ptr_idx = (node & kBRTreeChildLeafFlag) != 0;
        //int parent = parents_ptr[parent_ptr_idx][node & kBRTreeChildIndexMask];
        //int2 siblings = _BRTree_internal_nodes_children[parent];
        //node = (node != siblings.x) ? siblings.x : siblings.y;

        int sibling_idx_offset = (node & kBRTreeChildLeafFlag) != 0 ? (kParams.num_particles_ - 1) : 0;
        int idx_for_sibling = (node & kBRTreeChildIndexMask) + sibling_idx_offset;
        node = _BRTree_nodes_siblings[idx_for_sibling];

        level_idx ^= 1;

    } while (!is_traversal_over);
}


__global__ void
findNeighbors_bottom_up_kernel(
    int const *_BRTree_internal_nodes_parents,
    int const *_BRTree_leaf_nodes_parents,
    int2 const *_BRTree_internal_nodes_children,
    int const *_BRTree_nodes_siblings, // dParticleMortonCodes_
    int const *_particle_indices,
    float4 const *_particle_bounding_spheres,
    float4 const *_particle_positions_pred,
    int const *_particle_phases,
    int const *_BRTree_internal_nodes_phases, // dConstraintCounts_
    int *_particle_neighbors
)
{
    uint particle_idx_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_idx_id >= kParams.num_particles_) return;

    int particle_id = _particle_indices[particle_idx_id];
    float4 pos = _particle_positions_pred[particle_id];
    int phase = _particle_phases[particle_id];
    bool is_self_collision_on = isSelfCollideOn(phase);
    float search_range = isFluid(phase) ? kParams.fluid_neighbor_search_range_ : (pos.w + kParams.particle_collision_margin_);
    int neighbor_offset = particle_id * kParams.max_num_neighbors_per_particle_;
    int neighbor_end = (particle_id + 1) * kParams.max_num_neighbors_per_particle_; // exclusive

    float4 const *pos_ptr[2] = { _particle_bounding_spheres, _particle_positions_pred };
    int const *parents_ptr[2] = { _BRTree_internal_nodes_parents, _BRTree_leaf_nodes_parents };
    int const *phases_ptr[2] = { _BRTree_internal_nodes_phases, _particle_phases};

    int curr_root = _BRTree_nodes_siblings[particle_idx_id + kParams.num_particles_ - 1];

    if (!is_self_collision_on) {
        while (curr_root != 0) {
            uint64_t level_idx = 0;
            int node = curr_root;
            bool is_traversal_over = false;

            if (neighbor_offset == neighbor_end) {
                break;
            }

            do {
                if (node & kBRTreeChildLeafFlag) {
                    int idx_n = _particle_indices[node & kBRTreeChildIndexMask];
                    int phase_n = _particle_phases[idx_n];
                    if (isCollisionEnabled(particle_id, idx_n, phase, phase_n)) {
                        if (neighbor_offset == neighbor_end) {
                            break;
                        }
                        _particle_neighbors[neighbor_offset] = idx_n;
                        ++neighbor_offset;
                    }
                }
                else {
                    int2 children = _BRTree_internal_nodes_children[node];

                    int left_pos_ptr_idx = (children.x & kBRTreeChildLeafFlag) != 0;
                    int right_pos_ptr_idx = (children.y & kBRTreeChildLeafFlag) != 0;
                    int left_pos_idx = left_pos_ptr_idx ? _particle_indices[children.x & kBRTreeChildIndexMask] : children.x;
                    int right_pos_idx = right_pos_ptr_idx ? _particle_indices[children.y & kBRTreeChildIndexMask] : children.y;

                    float4 bs_left = pos_ptr[left_pos_ptr_idx][left_pos_idx];
                    float4 bs_right = pos_ptr[right_pos_ptr_idx][right_pos_idx];

                    int left_phase = phases_ptr[left_pos_ptr_idx][left_pos_idx];
                    int right_phase = phases_ptr[right_pos_ptr_idx][right_pos_idx];

                    bool test_left = (phase != left_phase) && isNeighbor(pos, bs_left, search_range);
                    bool test_right = (phase != right_phase) && isNeighbor(pos, bs_right, search_range);
                    
                    //bool test_left = isNeighbor(pos, bs_left, search_range);
                    //bool test_right = isNeighbor(pos, bs_right, search_range);

                    if (test_left || test_right) { // any accepted
                        level_idx <<= 1;
                        if (test_left && test_right) {
                            float3 v_left = make_float3(pos - bs_left);
                            float3 v_right = make_float3(pos - bs_right);
                            node = dot(v_left, v_left) < dot(v_right, v_right) ? children.x : children.y; // choose the nearest child
                            level_idx |= 1;
                        }
                        else { // rejected one child
                            node = test_left ? children.x : children.y;
                        }
                        continue;
                    }
                }
                //level_idx += 1;
                while ((level_idx & 1) == 0) {
                    if (level_idx == 0) {
                        is_traversal_over = true;
                        break;
                    }
                    int parent_ptr_idx = (node & kBRTreeChildLeafFlag) != 0;
                    node = parents_ptr[parent_ptr_idx][node & kBRTreeChildIndexMask];
                    level_idx = level_idx >> 1;
                }
                //if (node == 0) {
                //    printf("!!!!!!!!!!ERROR: backtrack node == 0 !!!!!!!!!!!!!!\n");
                //}
                int sibling_idx_offset = (node & kBRTreeChildLeafFlag) != 0 ? (kParams.num_particles_ - 1) : 0;
                int idx_for_sibling = (node & kBRTreeChildIndexMask) + sibling_idx_offset;
                node = _BRTree_nodes_siblings[idx_for_sibling];

                level_idx ^= 1;

            } while (!is_traversal_over);

            curr_root = parents_ptr[(curr_root & kBRTreeChildLeafFlag) != 0][curr_root & kBRTreeChildIndexMask];
            curr_root = _BRTree_nodes_siblings[curr_root];
        }
    } 
    else {
        while (curr_root != 0) {
            uint64_t level_idx = 0;
            int node = curr_root;
            bool is_traversal_over = false;

            if (neighbor_offset == neighbor_end) {
                break;
            }

            do {
                if (node & kBRTreeChildLeafFlag) {
                    int idx_n = _particle_indices[node & kBRTreeChildIndexMask];
                    int phase_n = _particle_phases[idx_n];
                    if (isCollisionEnabled(particle_id, idx_n, phase, phase_n)) {
                        if (neighbor_offset == neighbor_end) {
                            break;
                        }
                        _particle_neighbors[neighbor_offset] = idx_n;
                        ++neighbor_offset;
                    }
                }
                else {
                    int2 children = _BRTree_internal_nodes_children[node];

                    int left_pos_ptr_idx = (children.x & kBRTreeChildLeafFlag) != 0;
                    int right_pos_ptr_idx = (children.y & kBRTreeChildLeafFlag) != 0;
                    int left_pos_idx = left_pos_ptr_idx ? _particle_indices[children.x & kBRTreeChildIndexMask] : children.x;
                    int right_pos_idx = right_pos_ptr_idx ? _particle_indices[children.y & kBRTreeChildIndexMask] : children.y;

                    float4 bs_left = pos_ptr[left_pos_ptr_idx][left_pos_idx];
                    float4 bs_right = pos_ptr[right_pos_ptr_idx][right_pos_idx];


                    bool test_left = isNeighbor(pos, bs_left, search_range);
                    bool test_right = isNeighbor(pos, bs_right, search_range);
                    if (test_left || test_right) { // any accepted
                        level_idx <<= 1;
                        if (test_left && test_right) {
                            float3 v_left = make_float3(pos - bs_left);
                            float3 v_right = make_float3(pos - bs_right);
                            node = dot(v_left, v_left) < dot(v_right, v_right) ? children.x : children.y; // choose the nearest child
                            level_idx |= 1;
                        }
                        else { // rejected one child
                            node = test_left ? children.x : children.y;
                        }
                        continue;
                    }
                }
                //level_idx += 1;
                while ((level_idx & 1) == 0) {
                    if (level_idx == 0) {
                        is_traversal_over = true;
                        break;
                    }
                    int parent_ptr_idx = (node & kBRTreeChildLeafFlag) != 0;
                    node = parents_ptr[parent_ptr_idx][node & kBRTreeChildIndexMask];
                    level_idx = level_idx >> 1;
                }
                //if (node == 0) {
                //    printf("!!!!!!!!!!ERROR: backtrack node == 0 !!!!!!!!!!!!!!\n");
                //}
                int sibling_idx_offset = (node & kBRTreeChildLeafFlag) != 0 ? (kParams.num_particles_ - 1) : 0;
                int idx_for_sibling = (node & kBRTreeChildIndexMask) + sibling_idx_offset;
                node = _BRTree_nodes_siblings[idx_for_sibling];

                level_idx ^= 1;

            } while (!is_traversal_over);

            curr_root = parents_ptr[(curr_root & kBRTreeChildLeafFlag) != 0][curr_root & kBRTreeChildIndexMask];
            curr_root = _BRTree_nodes_siblings[curr_root];
        }
    }
}


__global__ void
findNeighbors_stack_kernel(
    int const *_BRTree_internal_nodes_parents,
    int const *_BRTree_leaf_nodes_parents,
    int2 const *_BRTree_internal_nodes_children,
    int const *_BRTree_nodes_siblings, // dParticleMortonCodes_
    int const *_particle_indices,
    float4 const *_particle_bounding_spheres,
    float4 const *_particle_positions_pred,
    int const *_particle_phases,
    int const *_BRTree_internal_nodes_phases, // dConstraintCounts_
    int *_BRTree_traversal_stack,
    int *_particle_neighbors
)
{
    uint particle_idx_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_idx_id >= kParams.num_particles_) return;

    int particle_id = _particle_indices[particle_idx_id];
    float4 pos = _particle_positions_pred[particle_id];
    int phase = _particle_phases[particle_id];
    bool is_self_collision_on = isSelfCollideOn(phase);
    float search_range = isFluid(phase) ? kParams.fluid_neighbor_search_range_ : (pos.w + kParams.particle_collision_margin_);
    int neighbor_offset = particle_id * kParams.max_num_neighbors_per_particle_;
    int neighbor_end = (particle_id + 1) * kParams.max_num_neighbors_per_particle_; // exclusive
    int stack_offset = particle_idx_id * kParams.stack_size_per_particle_;

    float4 const *pos_ptr[2] = { _particle_bounding_spheres, _particle_positions_pred };
    int const *parents_ptr[2] = { _BRTree_internal_nodes_parents, _BRTree_leaf_nodes_parents };
    int const *phases_ptr[2] = { _BRTree_internal_nodes_phases, _particle_phases };

    int curr_root = _BRTree_nodes_siblings[particle_idx_id + kParams.num_particles_ - 1];

    if (!is_self_collision_on) {
        while (curr_root != 0) {
            int pos_ptr_idx = (curr_root & kBRTreeChildLeafFlag) != 0;
            int pos_idx = pos_ptr_idx ? _particle_indices[curr_root & kBRTreeChildIndexMask] : curr_root;
            float4 bs_curr = pos_ptr[pos_ptr_idx][pos_idx];
            int phase_curr = phases_ptr[pos_ptr_idx][pos_idx];
            if ((phase != phase_curr) && isNeighbor(pos, bs_curr, search_range)) break;
            else {
                curr_root = parents_ptr[(curr_root & kBRTreeChildLeafFlag) != 0][curr_root & kBRTreeChildIndexMask];
                curr_root = _BRTree_nodes_siblings[curr_root];
            }
        }

        while (curr_root != 0) {
            if (neighbor_offset == neighbor_end) {
                break;
            }
            int top = curr_root;
            int top_idx = 0;
            while (top_idx >= 0) {
                --top_idx;
                if (top & kBRTreeChildLeafFlag) {
                    int idx_n = _particle_indices[top & kBRTreeChildIndexMask];
                    int phase_n = _particle_phases[idx_n];
                    if (isCollisionEnabled(particle_id, idx_n, phase, phase_n)) {
                        if (neighbor_offset == neighbor_end) {
                            break;
                        }
                        _particle_neighbors[neighbor_offset] = idx_n;
                        ++neighbor_offset;
                    }
                    if (top_idx >= 0) {
                        top = _BRTree_traversal_stack[stack_offset + top_idx];
                    }
                }
                else {
                    int2 children = _BRTree_internal_nodes_children[top];

                    int left_pos_ptr_idx = (children.x & kBRTreeChildLeafFlag) != 0;
                    int right_pos_ptr_idx = (children.y & kBRTreeChildLeafFlag) != 0;
                    int left_pos_idx = left_pos_ptr_idx ? _particle_indices[children.x & kBRTreeChildIndexMask] : children.x;
                    int right_pos_idx = right_pos_ptr_idx ? _particle_indices[children.y & kBRTreeChildIndexMask] : children.y;

                    float4 bs_left = pos_ptr[left_pos_ptr_idx][left_pos_idx];
                    float4 bs_right = pos_ptr[right_pos_ptr_idx][right_pos_idx];
                    int phase_left = phases_ptr[left_pos_ptr_idx][left_pos_idx];
                    int phase_right = phases_ptr[right_pos_ptr_idx][right_pos_idx];

                    bool test_left = false;
                    bool test_right = false;

                    if (phase != phase_left) {
                        test_left = isNeighbor(pos, bs_left, search_range);
                    }
                    if (phase != phase_right) {
                        test_right = isNeighbor(pos, bs_right, search_range);
                    }
                    if (test_left && test_right) {
                        float3 v_left = make_float3(pos - bs_left);
                        float3 v_right = make_float3(pos - bs_right);
                        top = dot(v_left, v_left) < dot(v_right, v_right) ? children.x : children.y;
                        int far_child = (top == children.x ? children.y : children.x);
                        _BRTree_traversal_stack[stack_offset + (++top_idx)] = far_child;
                        ++top_idx; // top is near, no need to put into global memory
                    }
                    else if (test_left || test_right) {
                        top = test_left ? children.x : children.y;
                        ++top_idx;
                    }
                    else if (top_idx >= 0) {
                        top = _BRTree_traversal_stack[stack_offset + top_idx];
                    }
                }
            }

            curr_root = parents_ptr[(curr_root & kBRTreeChildLeafFlag) != 0][curr_root & kBRTreeChildIndexMask];
            curr_root = _BRTree_nodes_siblings[curr_root];
        }
    }
    else {
        while (curr_root != 0) {
            int pos_ptr_idx = (curr_root & kBRTreeChildLeafFlag) != 0;
            int pos_idx = pos_ptr_idx ? _particle_indices[curr_root & kBRTreeChildIndexMask] : curr_root;
            float4 bs_curr = pos_ptr[pos_ptr_idx][pos_idx];
            if (isNeighbor(pos, bs_curr, search_range)) break;
            else {
                curr_root = parents_ptr[(curr_root & kBRTreeChildLeafFlag) != 0][curr_root & kBRTreeChildIndexMask];
                curr_root = _BRTree_nodes_siblings[curr_root];
            }
        }

        while (curr_root != 0) {
            if (neighbor_offset == neighbor_end) {
                break;
            }
            int top = curr_root;
            int top_idx = 0;
            while (top_idx >= 0) {
                --top_idx;
                if (top & kBRTreeChildLeafFlag) {
                    int idx_n = _particle_indices[top & kBRTreeChildIndexMask];
                    int phase_n = _particle_phases[idx_n];
                    if (isCollisionEnabled(particle_id, idx_n, phase, phase_n)) {
                        if (neighbor_offset == neighbor_end) {
                            break;
                        }
                        _particle_neighbors[neighbor_offset] = idx_n;
                        ++neighbor_offset;
                    }
                    if (top_idx >= 0) {
                        top = _BRTree_traversal_stack[stack_offset + top_idx];
                    }
                }
                else {
                    int2 children = _BRTree_internal_nodes_children[top];

                    int left_pos_ptr_idx = (children.x & kBRTreeChildLeafFlag) != 0;
                    int right_pos_ptr_idx = (children.y & kBRTreeChildLeafFlag) != 0;
                    int left_pos_idx = left_pos_ptr_idx ? _particle_indices[children.x & kBRTreeChildIndexMask] : children.x;
                    int right_pos_idx = right_pos_ptr_idx ? _particle_indices[children.y & kBRTreeChildIndexMask] : children.y;

                    float4 bs_left = pos_ptr[left_pos_ptr_idx][left_pos_idx];
                    float4 bs_right = pos_ptr[right_pos_ptr_idx][right_pos_idx];

                    bool test_left = isNeighbor(pos, bs_left, search_range);
                    bool test_right = isNeighbor(pos, bs_right, search_range);

                    if (test_left && test_right) {
                        float3 v_left = make_float3(pos - bs_left);
                        float3 v_right = make_float3(pos - bs_right);
                        top = dot(v_left, v_left) < dot(v_right, v_right) ? children.x : children.y;
                        int far_child = (top == children.x ? children.y : children.x);
                        _BRTree_traversal_stack[stack_offset + (++top_idx)] = far_child;
                        ++top_idx; // top is near, no need to put into global memory
                    }
                    else if (test_left || test_right) {
                        top = test_left ? children.x : children.y;
                        ++top_idx;
                    }
                    else if (top_idx >= 0) {
                        top = _BRTree_traversal_stack[stack_offset + top_idx];
                    }
                }
            }

            curr_root = parents_ptr[(curr_root & kBRTreeChildLeafFlag) != 0][curr_root & kBRTreeChildIndexMask];
            curr_root = _BRTree_nodes_siblings[curr_root];
        }
    }
}


__global__ void
findNeighbors_stack_AABB_kernel(
    int const *_BRTree_internal_nodes_parents,
    int const *_BRTree_leaf_nodes_parents,
    int2 const *_BRTree_internal_nodes_children,
    int const *_BRTree_nodes_siblings, // dParticleMortonCodes_
    int const *_particle_indices,
    cuAABB const *_particle_AABBs,
    float4 const *_particle_positions_pred,
    int const *_particle_phases,
    int const *_BRTree_internal_nodes_phases, // dConstraintCounts_
    int *_BRTree_traversal_stack,
    int *_particle_neighbors
)
{
    uint particle_idx_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_idx_id >= kParams.num_particles_) return;

    int particle_id = _particle_indices[particle_idx_id];
    float4 pos = _particle_positions_pred[particle_id];
    int phase = _particle_phases[particle_id];
    bool is_self_collision_on = isSelfCollideOn(phase);
    float search_range = isFluid(phase) ? (kParams.fluid_neighbor_search_range_ - kParams.min_particle_radius_) : (pos.w + kParams.particle_collision_margin_);
    int neighbor_offset = particle_id * kParams.max_num_neighbors_per_particle_;
    int neighbor_end = (particle_id + 1) * kParams.max_num_neighbors_per_particle_; // exclusive
    int stack_offset = particle_idx_id * kParams.stack_size_per_particle_;

    int const *parents_ptr[2] = { _BRTree_internal_nodes_parents, _BRTree_leaf_nodes_parents };
    int const *phases_ptr[2] = { _BRTree_internal_nodes_phases, _particle_phases };

    int curr_root = _BRTree_nodes_siblings[particle_idx_id + kParams.num_particles_ - 1];
    if (!is_self_collision_on) {
        while (curr_root != 0) {
            if (curr_root & kBRTreeChildLeafFlag) {
                int p1_idx = _particle_indices[curr_root & kBRTreeChildIndexMask];
                float4 curr_pos = _particle_positions_pred[p1_idx];
                int curr_phase = _particle_phases[p1_idx];
                if ((phase != curr_phase) && isNeighbor(pos, curr_pos, search_range)) {
                    break;
                }
            }
            else {
                cuAABB curr_AABB = _particle_AABBs[curr_root];
                int curr_phase = _BRTree_internal_nodes_phases[curr_root];
                if ((phase != curr_phase) && intersectSphereAABBInRange(pos, curr_AABB, search_range)) break;
            }
            curr_root = parents_ptr[(curr_root & kBRTreeChildLeafFlag) != 0][curr_root & kBRTreeChildIndexMask];
            curr_root = _BRTree_nodes_siblings[curr_root];
        }

        while (curr_root != 0) {
            if (neighbor_offset == neighbor_end) {
                break;
            }
            int top = curr_root;
            int top_idx = 0;
            while (top_idx >= 0) {
                --top_idx;
                if (top & kBRTreeChildLeafFlag) {
                    int idx_n = _particle_indices[top & kBRTreeChildIndexMask];
                    int phase_n = _particle_phases[idx_n];
                    if (isCollisionEnabled(particle_id, idx_n, phase, phase_n)) {
                        if (neighbor_offset == neighbor_end) {
                            break;
                        }
                        _particle_neighbors[neighbor_offset] = idx_n;
                        ++neighbor_offset;
                    }
                    if (top_idx >= 0) {
                        top = _BRTree_traversal_stack[stack_offset + top_idx];
                    }
                }
                else {
                    int2 children = _BRTree_internal_nodes_children[top];

                    int left_pos_ptr_idx = (children.x & kBRTreeChildLeafFlag) != 0;
                    int right_pos_ptr_idx = (children.y & kBRTreeChildLeafFlag) != 0;
                    int left_pos_idx = left_pos_ptr_idx ? _particle_indices[children.x & kBRTreeChildIndexMask] : children.x;
                    int right_pos_idx = right_pos_ptr_idx ? _particle_indices[children.y & kBRTreeChildIndexMask] : children.y;

                    bool test_left = false;
                    bool test_right = false;
                    float sq_dist_left, sq_dist_right;

                    int phase_left = phases_ptr[left_pos_ptr_idx][left_pos_idx];
                    int phase_right = phases_ptr[right_pos_ptr_idx][right_pos_idx];
                    bool test_phase_left = phase != phase_left;
                    bool test_phase_right = phase != phase_right;

                    if (test_phase_left) {
                        if (left_pos_ptr_idx) {
                            float4 pos_left = _particle_positions_pred[left_pos_idx];
                            test_left = isNeighbor(pos, pos_left, search_range, sq_dist_left);
                        }
                        else {
                            cuAABB left_AABB = _particle_AABBs[left_pos_idx];
                            test_left = intersectSphereAABBInRange(pos, left_AABB, search_range, sq_dist_left);
                        }
                    }
                    
                    if (test_phase_right) {
                        if (right_pos_ptr_idx) {
                            float4 pos_right = _particle_positions_pred[right_pos_idx];
                            test_right = isNeighbor(pos, pos_right, search_range, sq_dist_right);
                        }
                        else {
                            cuAABB right_AABB = _particle_AABBs[right_pos_idx];
                            test_right = intersectSphereAABBInRange(pos, right_AABB, search_range, sq_dist_right);
                        }
                    }

                    if (test_left && test_right) {
                        top = sq_dist_left < sq_dist_right ? children.x : children.y;
                        int far_child = (top == children.x ? children.y : children.x);
                        _BRTree_traversal_stack[stack_offset + (++top_idx)] = far_child;
                        ++top_idx; // top is near, no need to put into global memory
                    }
                    else if (test_left || test_right) {
                        top = test_left ? children.x : children.y;
                        ++top_idx;
                    }
                    else if (top_idx >= 0) {
                        top = _BRTree_traversal_stack[stack_offset + top_idx];
                    }
                }
            }

            curr_root = parents_ptr[(curr_root & kBRTreeChildLeafFlag) != 0][curr_root & kBRTreeChildIndexMask];
            curr_root = _BRTree_nodes_siblings[curr_root];
        }
    }
    else {
        while (curr_root != 0) {
            if (curr_root & kBRTreeChildLeafFlag) {
                int p1_idx = _particle_indices[curr_root & kBRTreeChildIndexMask];
                float4 curr_pos = _particle_positions_pred[p1_idx];
                if (isNeighbor(pos, curr_pos, search_range)) {
                    break;
                }
            }
            else {
                cuAABB curr_AABB = _particle_AABBs[curr_root];
                if (intersectSphereAABBInRange(pos, curr_AABB, search_range)) break;
            }
            curr_root = parents_ptr[(curr_root & kBRTreeChildLeafFlag) != 0][curr_root & kBRTreeChildIndexMask];
            curr_root = _BRTree_nodes_siblings[curr_root];
        }

        while (curr_root != 0) {
            if (neighbor_offset == neighbor_end) {
                break;
            }
            int top = curr_root;
            int top_idx = 0;
            while (top_idx >= 0) {
                --top_idx;
                if (top & kBRTreeChildLeafFlag) {
                    int idx_n = _particle_indices[top & kBRTreeChildIndexMask];
                    int phase_n = _particle_phases[idx_n];
                    if (isCollisionEnabled(particle_id, idx_n, phase, phase_n)) {
                        if (neighbor_offset == neighbor_end) {
                            break;
                        }
                        _particle_neighbors[neighbor_offset] = idx_n;
                        ++neighbor_offset;
                    }
                    if (top_idx >= 0) {
                        top = _BRTree_traversal_stack[stack_offset + top_idx];
                    }
                }
                else {
                    int2 children = _BRTree_internal_nodes_children[top];

                    int left_pos_ptr_idx = (children.x & kBRTreeChildLeafFlag) != 0;
                    int right_pos_ptr_idx = (children.y & kBRTreeChildLeafFlag) != 0;
                    int left_pos_idx = left_pos_ptr_idx ? _particle_indices[children.x & kBRTreeChildIndexMask] : children.x;
                    int right_pos_idx = right_pos_ptr_idx ? _particle_indices[children.y & kBRTreeChildIndexMask] : children.y;

                    bool test_left, test_right;
                    float sq_dist_left, sq_dist_right;
                    if (left_pos_ptr_idx) {
                        float4 pos_left = _particle_positions_pred[left_pos_idx];
                        test_left = isNeighbor(pos, pos_left, search_range, sq_dist_left);
                    }
                    else {
                        cuAABB left_AABB = _particle_AABBs[left_pos_idx];
                        test_left = intersectSphereAABBInRange(pos, left_AABB, search_range, sq_dist_left);
                    }

                    if (right_pos_ptr_idx) {
                        float4 pos_right = _particle_positions_pred[right_pos_idx];
                        test_right = isNeighbor(pos, pos_right, search_range, sq_dist_right);
                    }
                    else {
                        cuAABB right_AABB = _particle_AABBs[right_pos_idx];
                        test_right = intersectSphereAABBInRange(pos, right_AABB, search_range, sq_dist_right);
                    }

                    if (test_left && test_right) {
                        top = sq_dist_left < sq_dist_right ? children.x : children.y;
                        int far_child = (top == children.x ? children.y : children.x);
                        _BRTree_traversal_stack[stack_offset + (++top_idx)] = far_child;
                        ++top_idx; // top is near, no need to put into global memory
                    }
                    else if (test_left || test_right) {
                        top = test_left ? children.x : children.y;
                        ++top_idx;
                    }
                    else if (top_idx >= 0) {
                        top = _BRTree_traversal_stack[stack_offset + top_idx];
                    }
                }
            }

            curr_root = parents_ptr[(curr_root & kBRTreeChildLeafFlag) != 0][curr_root & kBRTreeChildIndexMask];
            curr_root = _BRTree_nodes_siblings[curr_root];
        }
    }
}

//before exec kernel, set 1) _constraint_deltas, 2) _constraint_counts to 0
__global__ void
solveContacts_kernel(
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
    float4 *_particle_positions_pred
)
{
    uint idx_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_i >= kParams.num_particles_) return;

    int neighbor_offset = idx_i * kParams.max_num_neighbors_per_particle_;
    int neighbor_end = (idx_i + 1) * kParams.max_num_neighbors_per_particle_; // exclusive
    float4 *contact_pos_ptr = _is_pre_stabilize_mode ? _particle_positions : _particle_positions_pred;
    float4 pos_i = contact_pos_ptr[idx_i];
    float w_i = __particle_inv_masses[idx_i];
    float4 pos_i0 = _particle_positions[idx_i];
    int phase_i = _particle_phases[idx_i];
    float3 delta_i = make_float3(0.f);
    int delta_count_i = 0;

    for (; neighbor_offset < neighbor_end; ++neighbor_offset) {
        int idx_j = _particle_neighbors[neighbor_offset];
        if (isFluid(phase_i) || idx_j == -1) { // no collision handling for fluid particles
            break;
        }
        int phase_j = _particle_phases[idx_j];
        float4 pos_j = contact_pos_ptr[idx_j];
        float w_j = __particle_inv_masses[idx_j];
        float4 pos_j0 = _particle_positions[idx_j];
        if (isNeighbor(pos_i, pos_j, pos_i.w)) {
            float3 N, total_delta, delta_j; // contact normal
            float3 v_ij = make_float3(pos_j - pos_i);
            float len_v_ij = sqrtf(dot(v_ij, v_ij)), inv_w_sum = 1.f / (w_i + w_j);
            float3 dir_ij = (len_v_ij == 0.f) ? make_float3(0.f) : (v_ij / len_v_ij); // TODO: change backup dir to float3(1.f, 0.f, 0.f)

            if (isRigidParticle(phase_i) && isRigidParticle(phase_j)) {
                float4 init_sdf = _rigid_sdfs[idx_i - kParams.rigid_particles_begin_];
                float mag_i = fabsf(init_sdf.w);
                float3 grad_i = rotate(_rigid_rotations[phase_i], make_float3(init_sdf));
                init_sdf = _rigid_sdfs[idx_j - kParams.rigid_particles_begin_];
                float mag_j = fabsf(init_sdf.w);
                float3 grad_j = rotate(_rigid_rotations[phase_j], make_float3(init_sdf));

                N = SafeNormalize(mag_i < mag_j ? grad_i : -grad_j);
                float d = fmaxf(mag_i, mag_j);

                if (d < kParams.min_rigid_surface_sdf_) {
                    float dot_prod = fminf(dot(dir_ij, N), 0.f);
                    N = SafeNormalize(dir_ij - 2.f * dot_prod * N);
                    d = (pos_i.w + pos_j.w) - len_v_ij;
                }
                total_delta = d * N * inv_w_sum;
            }

            else if (isRigidParticle(phase_i)) {
                float4 init_sdf = _rigid_sdfs[idx_i - kParams.rigid_particles_begin_];
                float d = fabsf(init_sdf.w);
                N = dir_ij;
                if (d < kParams.min_rigid_surface_sdf_) {
                    N = SafeNormalize(rotate(_rigid_rotations[phase_i], make_float3(init_sdf)));
                    float dot_prod = fminf(dot(dir_ij, N), 0.f);
                    N = SafeNormalize(dir_ij - 2.f * dot_prod * N);
                    d = (pos_i.w + pos_j.w) - len_v_ij;
                }
                total_delta = d * N * inv_w_sum;
            }

            else if (isRigidParticle(phase_j)) {
                float4 init_sdf = _rigid_sdfs[idx_j - kParams.rigid_particles_begin_];
                float d = fabsf(init_sdf.w);
                N = -dir_ij;
                if (d < kParams.min_rigid_surface_sdf_) {
                    N = -SafeNormalize(rotate(_rigid_rotations[phase_j], make_float3(init_sdf)));
                    float dot_prod = fminf(dot(dir_ij, N), 0.f);
                    N = SafeNormalize(dir_ij - 2.f * dot_prod * N);
                    d = (pos_i.w + pos_j.w) - len_v_ij;
                }
                total_delta = d * N * inv_w_sum;
            }

            else { // fluid particles treated as solid particles
                total_delta = (pos_i.w + pos_j.w - len_v_ij) * dir_ij * inv_w_sum;
            }
            delta_i -= w_i * total_delta;
            delta_j = w_j * total_delta;

            if (!_is_pre_stabilize_mode) {
                N = dir_ij; // TODO: try sdf gradient for rigid particles
                float penetration_depth = fmaxf(pos_i.w + pos_j.w - len_v_ij, 0.f);
                float3 tan_disp = -make_float3((pos_i - pos_i0) - (pos_j - pos_j0));
                tan_disp = (tan_disp - dot(tan_disp, N) * N);
                float len_tan_disp = length(tan_disp);
                total_delta = tan_disp * inv_w_sum;
                if (len_tan_disp >= kParams.static_friction_ * penetration_depth) {
                    total_delta *= fminf(kParams.particle_friction_ * penetration_depth / len_tan_disp, 1.f);
                }
                delta_i += w_i * total_delta;
                delta_j -= w_j * total_delta;
            }

            atomicAdd(&_constraint_deltas[idx_j].x, delta_j.x);
            atomicAdd(&_constraint_deltas[idx_j].y, delta_j.y);
            atomicAdd(&_constraint_deltas[idx_j].z, delta_j.z);
            if (!isRigidParticle(phase_j)) atomicAdd(&_constraint_counts[idx_j], 1);
            delta_count_i += !isRigidParticle(phase_i);

        } // end if: isNeighbor
    } // end for: all neighbors

    float3 disp_i = make_float3(pos_i - pos_i0);
    for (int k = 0; k < kParams.num_planes_; ++k) {
        float4 plane = _planes[k];
        float3 N = make_float3(plane);
        float C = dot(make_float3(pos_i), N) + plane.w - kParams.shape_rest_extent_ * pos_i.w;
        if (C < 0) {
            delta_i -= C * N;
            delta_count_i += !isRigidParticle(phase_i);

            if (!_is_pre_stabilize_mode) {
                disp_i = -(disp_i - dot(disp_i, N) * N);
                float len_disp_i = length(disp_i);
                if (len_disp_i >= kParams.static_friction_ * -C) {
                    disp_i *= fminf(-C * kParams.dynamic_friction_ / len_disp_i, 1.f);
                }
                delta_i += disp_i;
            }
        }
    }

    atomicAdd(&_constraint_deltas[idx_i].x, delta_i.x);
    atomicAdd(&_constraint_deltas[idx_i].y, delta_i.y);
    atomicAdd(&_constraint_deltas[idx_i].z, delta_i.z);
    atomicAdd(&_constraint_counts[idx_i], delta_count_i);
}


__global__ void
prestabilize_kernel(
    float4 const *_constraint_deltas,
    int const *_constraint_counts,
    float4 const *_velocities,
    float4 *_particle_positions,
    float4 *_particle_positions_pred
)
{
    uint particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= kParams.num_particles_) return;

    float3 velocity = make_float3(_velocities[particle_id]);
    if (dot(velocity, velocity) < kParams.prestabilize_velocity_threshold_quad_) return;

    float num_collision = (float)max(_constraint_counts[particle_id], 1);
    float3 delta_pos = make_float3(_constraint_deltas[particle_id]) / num_collision;

    _particle_positions_pred[particle_id] += make_float4(delta_pos);
    _particle_positions[particle_id] += make_float4(delta_pos);
}

__global__ void
updatePosPred_kernel(
    float4 const *_constraint_deltas,
    int const *_constraint_counts,
    float4 *_particle_positions_pred
)
{
    uint particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= kParams.num_particles_) return;

    float num_collision = (float)max(_constraint_counts[particle_id], 1);
    float3 delta_pos = make_float3(_constraint_deltas[particle_id]) / num_collision;
    _particle_positions_pred[particle_id] += make_float4(delta_pos);
}

__global__ void
calcRigidParticleCovMatrices_kernel(
    int const *_particle_phases,
    float4 const *_rigid_centers_of_mass,
    float4 const *_rigid_centers_of_mass_init,
    float4 const *_particle_positions_init,
    float4 const *_particle_positions_pred,
    float const *_rigid_particle_weights,
    float const *_rigid_weights,
    quat const *_rigid_rotations,
    float4 *_rigid_relative_positions_init,
    mat3 *_rigid_particle_cov_matrices
)
{
    int rigid_internal_particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (rigid_internal_particle_id >= kParams.num_rigid_particles_) return;

    int rigid_particle_id = rigid_internal_particle_id + kParams.rigid_particles_begin_;
    int rigid_id = _particle_phases[rigid_particle_id];
    float rigid_particle_weight = _rigid_particle_weights[rigid_internal_particle_id];
    float4 pos_init = _particle_positions_init[rigid_particle_id];
    float4 pos_pred = _particle_positions_pred[rigid_particle_id];
    float4 CoM_init = _rigid_centers_of_mass_init[rigid_id];
    float4 CoM = _rigid_centers_of_mass[rigid_id];
    float4 pos_relative_init = pos_init - CoM_init;
    float4 pos_relative = pos_pred - CoM;
    mat3 cov_mat = rigid_particle_weight * outer(make_float3(pos_relative), make_float3(pos_relative_init));
    
    _rigid_relative_positions_init[rigid_internal_particle_id] = pos_relative_init;
    _rigid_particle_cov_matrices[rigid_internal_particle_id] = cov_mat;
}

__global__ void
calcRigidRotations_kernel(
    mat3 const *_rigid_cov_matrices,
    quat *_rigid_rotations
)
{
    int rigid_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (rigid_id >= kParams.num_rigids_) return;

    quat q = _rigid_rotations[rigid_id];
    mat3 A = _rigid_cov_matrices[rigid_id];
    extract_rotation(A, q, kParams.max_iter_extract_rotation_);
    _rigid_rotations[rigid_id] = q;
}

// before exec kernel, set _rigid_cov_matrices to 0
// _rigid_cov_matrices: save the cov matrices
__global__ void
calcRotations_kernel(
    int const *_rigid_particle_offsets,
    float4 const *_particle_positions_init,
    float4 const *_particle_positions_pred,
    float const *_rigid_particle_weights,
    float4 *_rigid_centers_of_mass,
    float4 *_rigid_relative_positions,
    float4 *_rigid_relative_positions_init,
    quat *_rigid_rotations
)
{
    int rigid_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (rigid_id >= kParams.num_rigids_) return;

    int begin_particle_idx = _rigid_particle_offsets[rigid_id];
    int end_particle_idx = _rigid_particle_offsets[rigid_id + 1];
    int num_particles = end_particle_idx - begin_particle_idx;
    int begin_rigid_particle_idx = begin_particle_idx - kParams.rigid_particles_begin_;
    int end_rigid_particle_idx = begin_rigid_particle_idx + num_particles;

    // TODO: change thrust::seq to thrust::device to use dynamic parallelism
    float4 weighted_sum = thrust::inner_product(
        thrust::seq,
        _particle_positions_pred + begin_particle_idx,
        _particle_positions_pred + end_particle_idx,
        _rigid_particle_weights + begin_rigid_particle_idx,
        make_float4(0.f),
        thrust::plus<float4>(),
        vec_scalar_multiply<float4, float>()
    );

    float4 weighted_sum_init = thrust::inner_product(
        thrust::seq,
        _particle_positions_init + begin_particle_idx,
        _particle_positions_init + end_particle_idx,
        _rigid_particle_weights + begin_rigid_particle_idx,
        make_float4(0.f),
        thrust::plus<float4>(),
        vec_scalar_multiply<float4, float>()
    );

    float inv_weight_sum = 1.f / thrust::reduce(
        thrust::seq,
        _rigid_particle_weights + begin_rigid_particle_idx,
        _rigid_particle_weights + begin_rigid_particle_idx + num_particles,
        0.f);

    //cudaDeviceSynchronize();
    float4 CoM = weighted_sum * inv_weight_sum;
    float4 CoM_init = weighted_sum_init * inv_weight_sum;
    thrust::transform(thrust::seq,
        _particle_positions_init + begin_particle_idx,
        _particle_positions_init + end_particle_idx,
        _rigid_relative_positions_init + begin_rigid_particle_idx,
        minus_by<float4>(CoM_init)
    );

    thrust::transform(thrust::seq,
        _particle_positions_pred + begin_particle_idx,
        _particle_positions_pred + end_particle_idx,
        _rigid_relative_positions + begin_rigid_particle_idx,
        minus_by<float4>(CoM)
    );

    // _rigid_relative_positions_init is not used, because it is needed for shape matching
    thrust::transform(thrust::seq,
        _rigid_relative_positions + begin_rigid_particle_idx,
        _rigid_relative_positions + end_rigid_particle_idx,
        _rigid_particle_weights   + begin_rigid_particle_idx,
        _rigid_relative_positions + begin_rigid_particle_idx,
        vec_scalar_multiply<float4, float>()
    );

    mat3 A = thrust::inner_product(
        thrust::seq,
        _rigid_relative_positions + begin_rigid_particle_idx,
        _rigid_relative_positions + end_rigid_particle_idx,
        _rigid_relative_positions_init + begin_rigid_particle_idx,
        make_mat3(0.f),
        thrust::plus<mat3>(),
        vec3_outer_prod()
    );
    
    //cudaDeviceSynchronize();

    quat q = _rigid_rotations[rigid_id];
    extract_rotation(A, q, kParams.max_iter_extract_rotation_);
    _rigid_centers_of_mass[rigid_id] = CoM;
    _rigid_rotations[rigid_id] = q;
    
    // printf("rigid body %d....\n", rigid_id);
#if 0
#endif
}


__global__ void
matchShapes_kernel(
    float4 const *_rigid_centers_of_mass,
    quat const *_rigid_rotations,
    float const *_rigid_stiffnesses,
    float4 const *_rigid_relative_positions_init,
    int const *_particle_phases,
    float4 *_particle_positions_pred
)
{
    int rigid_particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (rigid_particle_id >= kParams.num_rigid_particles_) return;

    int particle_id = rigid_particle_id + kParams.rigid_particles_begin_;
    int rigid_idx = _particle_phases[particle_id]; // assume rigid particle phases has no phase flags set

    float3 relative_pos_init = make_float3(_rigid_relative_positions_init[rigid_particle_id]);
    quat q = _rigid_rotations[rigid_idx];

    float3 goal_pos = rotate(q, relative_pos_init) + make_float3(_rigid_centers_of_mass[rigid_idx]);
    float4 pred_pos = _particle_positions_pred[rigid_particle_id];
    float stiffness = _rigid_stiffnesses[rigid_idx];
    float3 delta_pos = stiffness * (goal_pos - make_float3(pred_pos));
    _particle_positions_pred[rigid_particle_id] += make_float4(delta_pos);
}

__global__ void
computeFluidLambda_kernel(
    float4 const *_particle_positions_pred,
    int const *_particle_phases,
    int const *_particle_neighbors,
    float const *_particle_inv_masses,
    float  *_fluid_lambdas
)
{
    int fluid_particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (fluid_particle_id >= kParams.num_fluid_particles_) return;

    int particle_id = fluid_particle_id + kParams.fluid_particles_begin_;
    float3 pos_i = make_float3(_particle_positions_pred[particle_id]);
    //int phase_i = _particle_phases[particle_id];
    int neighbor_begin_idx = kParams.max_num_neighbors_per_particle_ * particle_id;
    int neighbor_end_idx = neighbor_begin_idx + kParams.max_num_neighbors_per_particle_;

    float density = kSmoothKernel.w_0_ / _particle_inv_masses[particle_id];
    for (int i = neighbor_begin_idx; i < neighbor_end_idx; ++i) {
        int idx_n = _particle_neighbors[i];
        if (idx_n == -1) break;
        float4 pos_j = _particle_positions_pred[idx_n];
        float3 vec_ij = pos_i - make_float3(pos_j);
        density += kSmoothKernel.W(vec_ij) / _particle_inv_masses[idx_n];
    }
    // density constraint
    float C = density / kParams.fluid_rest_density_ - 1.f;
    //if (fluid_particle_id == 6384) {
    //    printf("C of particle %d: %f, density: %f\n", fluid_particle_id, C, density);
    //}
    float lambda = 0.f;
    if (C != 0.f) {
        float sum_grad_C_sq = 0.f;
        float3 grad_C_i = make_float3(0.f);
        for (int i = neighbor_begin_idx; i < neighbor_end_idx; ++i) {
            int idx_n = _particle_neighbors[i];
            if (idx_n == -1) break;
            //int phase_n = _particle_phases[idx_n];
            //if (phase_n != phase_i) continue;

            float4 pos_j = _particle_positions_pred[idx_n];
            float3 vec_ij = pos_i - make_float3(pos_j);
            float w_j = _particle_inv_masses[idx_n];
            const float3 grad_C_j = - kSmoothKernel.grad_W(vec_ij) / kParams.fluid_rest_density_ / w_j;
            sum_grad_C_sq += dot(grad_C_j, grad_C_j) * w_j;
            grad_C_i -= grad_C_j;
        }
        sum_grad_C_sq += dot(grad_C_i, grad_C_i) * _particle_inv_masses[particle_id];
        lambda = -C / (sum_grad_C_sq + kParams.fluid_cfm_eps_);
    }
    _fluid_lambdas[fluid_particle_id] = lambda;
}

__global__ void
solveFluidDensityConstraints_kernel(
    int const *_particle_phases,
    float const *_fluid_lambdas,
    int const *_particle_neighbors,
    float const *_particle_inv_masses,
    float4 *_particle_positions_pred
)
{
    int fluid_particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (fluid_particle_id >= kParams.num_fluid_particles_) return;

    int particle_id = fluid_particle_id + kParams.fluid_particles_begin_;
    float3 pos_i = make_float3(_particle_positions_pred[particle_id]);
    int phase_i = _particle_phases[particle_id];
    int neighbor_begin_idx = kParams.max_num_neighbors_per_particle_ * particle_id;
    int neighbor_end_idx = neighbor_begin_idx + kParams.max_num_neighbors_per_particle_;
    float lambda_i = _fluid_lambdas[fluid_particle_id];

    float3 delta_pos_i = make_float3(0.f);
    // S_corr
    float inv_w_dq = kParams.fluid_corr_refer_inv_W_;
    float k = kParams.fluid_corr_k_;
    float n = kParams.fluid_corr_n_;
    for (int i = neighbor_begin_idx; i < neighbor_end_idx; ++i) {
        int idx_n = _particle_neighbors[i];
        if (idx_n == -1) break;
        int phase_n = _particle_phases[idx_n];
        if (phase_n != phase_i) continue;
        const float3 vec_ij = pos_i - make_float3(_particle_positions_pred[idx_n]);
        //const float3 grad_C_j = kSmoothKernel.grad_W(vec_ij) / _particle_inv_masses[idx_n];
        //float s_corr = -k * powf(kSmoothKernel.W(vec_ij) * inv_w_dq, n);
        const float3 grad_W_j = kSmoothKernel.grad_W(vec_ij);
        float s_corr = -k * powf(kSmoothKernel.W(vec_ij) * inv_w_dq, n);
        float lambda_j = _fluid_lambdas[idx_n - kParams.fluid_particles_begin_];
        //delta_pos_i += (lambda_i + lambda_j + s_corr) * grad_C_j;
        delta_pos_i += (lambda_i + lambda_j + s_corr) * grad_W_j;
    }
    _particle_positions_pred[particle_id] += make_float4(delta_pos_i / kParams.fluid_rest_density_, 0.f);
}


__global__ void
finalize_kernel(
    float4 const *_particle_positions_pred,
    float4 *_particle_velocities,
    float4 *_particle_positions
)
{
    uint particle_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle_id >= kParams.num_particles_) return;

    float4 pos_pred = _particle_positions_pred[particle_id];
    float4 pos = _particle_positions[particle_id];
    float3 delta_pos = make_float3(pos_pred - pos);
    float3 v = delta_pos / kParams.step_delta_time_;
    if (dot(v, v) < kParams.sleeping_threshold_quad_) {
        delta_pos = make_float3(0.f);
        //v = make_float3(0.f);
    }
    _particle_velocities[particle_id] = make_float4(v);
    _particle_positions[particle_id] = pos + make_float4(delta_pos);
}


////////////////////////////////////////////////////////////////////////////////
// host functions
////////////////////////////////////////////////////////////////////////////////

cub::CachingDeviceAllocator  g_cub_allocator(true);  // Caching allocator for device memory

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
)
{
    thrust::copy(thrust::device,
        thrust::device_ptr<float4>(_particle_positions),
        thrust::device_ptr<float4>(_particle_positions +_num_particles),
        thrust::device_ptr<float4>(_particle_positions_init));

    if (_num_rigid_particles > 0) {
        thrust::transform(thrust::device,
            thrust::device_ptr<int>(_rigid_particle_offsets),
            thrust::device_ptr<int>(_rigid_particle_offsets + _num_rigids + 1),
            thrust::device_ptr<int>(_rigid_internal_particle_offsets),
            minus_by<int>(_rigid_particles_begin));

        thrust::transform(thrust::device,
            thrust::device_ptr<float4>(_particle_positions_init + _rigid_particles_begin),
            thrust::device_ptr<float4>(_particle_positions_init + _rigid_particles_begin + _num_rigid_particles),
            thrust::device_ptr<float>(_rigid_particle_weights),
            weighted_power<float4, float>(_min_rigid_particle_radius, 3.f));

        //thrust::fill(thrust::device,
        //    thrust::device_ptr<float>(_rigid_particle_weights),
        //    thrust::device_ptr<float>(_rigid_particle_weights + _num_rigid_particles),
        //    1.f);

        thrust::fill(thrust::device,
            thrust::device_ptr<quat>(_rigid_rotations),
            thrust::device_ptr<quat>(_rigid_rotations + _num_rigids),
            quat()); // default: quat(0, 0, 0, 1)
    }
}


void setParameters(Params *_hParams)
{
    checkCudaErrors(cudaMemcpyToSymbol(kParams, _hParams, sizeof(Params)));
}
  
void integrate(
    int _num_particles, 
    float4 *_particle_positions,
    float4 const *_planes,
    int const *_particle_phases,
    float  *_particle_inv_masses, 
    float4 *_particle_velocities, 
    float4 *_particle_positions_pred
)
{
    uint num_threads, num_blocks;
    computeGridSize(_num_particles, kCudaBlockSize, num_blocks, num_threads);
    integrate_kernel <<< num_blocks, num_threads >>> (
        _particle_positions,
        _planes,
        _particle_phases,
        _particle_inv_masses,
        _particle_velocities,
        _particle_positions_pred
    );
    getLastCudaError("integrate_kernel");
}

void swapParticleMass(
    int _num_particles, 
    float4 *_particle_velocities, 
    float *_particle_inv_masses
)
{
    uint num_threads, num_blocks;
    computeGridSize(_num_particles, kCudaBlockSize, num_blocks, num_threads);
    swapParticleMass_kernel <<< num_blocks, num_threads >>> (
        _particle_velocities, 
        _particle_inv_masses
    );
    getLastCudaError("swapParticleMass_kernel");
}

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
)
{
    uint num_threads, num_blocks;
    computeGridSize(_num_particles, kCudaBlockSize, num_blocks, num_threads);
    BEGIN_TIMER("calcMortonCode_kernel");
    calcMortonCode_kernel <<< num_blocks, num_threads >>> (
        _particle_positions_pred, 
        _particle_morton_codes, 
        _particle_indices
    );
    getLastCudaError("calcMortonCode_kernel");
    STOP_TIMER("calcMortonCode_kernel");

    BEGIN_TIMER("thrust::sort morton");
    thrust::sort_by_key(thrust::device,
        thrust::device_ptr<uint64_t>(_particle_morton_codes),
        thrust::device_ptr<uint64_t>(_particle_morton_codes + _num_particles),
        thrust::device_ptr<int>(_particle_indices));
    STOP_TIMER("thrust::sort morton");

    checkCudaErrors(cudaMemset(_BRTree_internal_nodes_parents, 0xff, (_num_particles - 1) * sizeof(int)));

    computeGridSize(_num_particles - 1, kCudaBlockSize, num_blocks, num_threads);
    BEGIN_TIMER("buildBRTree_kernel");
    buildBRTree_kernel <<< num_blocks, num_threads >>> (
        _particle_morton_codes, 
        _BRTree_internal_nodes_children, 
        _BRTree_internal_nodes_parents, 
        _BRTree_leaf_nodes_parents
    );
    getLastCudaError("buildBRTree_kernel");
    STOP_TIMER("buildBRTree_kernel");

    checkCudaErrors(cudaMemset(_atomic_counters_, 0, _num_particles * sizeof(int)));
    computeGridSize(_num_particles, kCudaBlockSize, num_blocks, num_threads);

    int *BRTree_nodes_siblings = (int*)_particle_morton_codes;
    BEGIN_TIMER("buildBoundingSphereTree_kernel");
    buildBoundingSphereTree_kernel <<< num_blocks, num_threads >>> (
        _BRTree_internal_nodes_parents, 
        _BRTree_internal_nodes_children,
        _BRTree_leaf_nodes_parents,
        _particle_phases,
        _particle_indices, 
        _particle_positions_pred,
        _atomic_counters_, // dConstraintCounts_
        //_particle_bounding_spheres_aux, // dConstraintDeltas_
        _particle_bounding_spheres,
        BRTree_nodes_siblings
    );
    getLastCudaError("buildBoundingSphereTree_kernel");
    STOP_TIMER("buildBoundingSphereTree_kernel");

    checkCudaErrors(cudaMemset(_particle_neighbors, 0xff, _num_particles * _max_num_neighbors_per_particle * sizeof(int)));

    computeGridSize(_num_particles, kCudaBlockSize, num_blocks, num_threads);
    //BEGIN_TIMER("findNeighbors_kernel");
    //findNeighbors_kernel <<< num_blocks, num_threads >>> (
    //    _BRTree_internal_nodes_parents,
    //    _BRTree_leaf_nodes_parents,
    //    _BRTree_internal_nodes_children,
    //    _particle_indices,
    //    _particle_bounding_spheres,
    //    _particle_positions_pred,
    //    _particle_phases,
    //    _particle_neighbors
    //);
    //getLastCudaError("findNeighbors_kernel");
    //STOP_TIMER("findNeighbors_kernel");

    //BEGIN_TIMER("findNeighbors_mbvh2_kernel");
    //findNeighbors_mbvh2_kernel <<< num_blocks, num_threads >>> (
    //    _BRTree_internal_nodes_parents,
    //    _BRTree_leaf_nodes_parents,
    //    _BRTree_internal_nodes_children,
    //     BRTree_nodes_siblings,
    //    _particle_indices,
    //    _particle_bounding_spheres,
    //    _particle_positions_pred,
    //    _particle_phases,
    //    _particle_neighbors
    //    );
    //getLastCudaError("findNeighbors_mbvh2_kernel");
    //STOP_TIMER("findNeighbors_mbvh2_kernel");

    int *BRTree_internal_nodes_phases = _atomic_counters_;

    //BEGIN_TIMER("findNeighbors_bottom_up_kernel");
    //findNeighbors_bottom_up_kernel <<< num_blocks, num_threads >>> (
    //    _BRTree_internal_nodes_parents,
    //    _BRTree_leaf_nodes_parents,
    //    _BRTree_internal_nodes_children,
    //    BRTree_nodes_siblings, // dParticleMortonCodes_
    //    _particle_indices,
    //    _particle_bounding_spheres,
    //    _particle_positions_pred,
    //    _particle_phases,
    //    BRTree_internal_nodes_phases, // dConstraintCounts_
    //    _particle_neighbors
    //);
    //getLastCudaError("findNeighbors_bottom_up_kernel");
    //STOP_TIMER("findNeighbors_bottom_up_kernel");

    BEGIN_TIMER("findNeighbors_stack_kernel");
    findNeighbors_stack_kernel <<< num_blocks, num_threads >>> (
        _BRTree_internal_nodes_parents,
        _BRTree_leaf_nodes_parents,
        _BRTree_internal_nodes_children,
         BRTree_nodes_siblings, // dParticleMortonCodes_
        _particle_indices,
        _particle_bounding_spheres,
        _particle_positions_pred,
        _particle_phases,
         BRTree_internal_nodes_phases, // dConstraintCounts_
        _BRTree_traversal_stack,
        _particle_neighbors
    );
    getLastCudaError("findNeighbors_stack_kernel");
    STOP_TIMER("findNeighbors_stack_kernel");

}


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
)
{
    uint num_threads, num_blocks;
    computeGridSize(_num_particles, kCudaBlockSize, num_blocks, num_threads);
    BEGIN_TIMER("calcMortonCode_kernel");
    calcMortonCode_kernel <<< num_blocks, num_threads >>> (
        _particle_positions_pred,
        _particle_morton_codes,
        _particle_indices
        );
    getLastCudaError("calcMortonCode_kernel");
    STOP_TIMER("calcMortonCode_kernel");

    //BEGIN_TIMER("thrust::sort morton");
    //thrust::sort_by_key(thrust::device,
    //    thrust::device_ptr<uint64_t>(_particle_morton_codes),
    //    thrust::device_ptr<uint64_t>(_particle_morton_codes + _num_particles),
    //    thrust::device_ptr<int>(_particle_indices));
    //STOP_TIMER("thrust::sort morton");

    BEGIN_TIMER("cub::sort morton");
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs<uint64_t, int>(
        d_temp_storage,
        temp_storage_bytes,
        _particle_morton_codes,
        _particle_morton_codes_aux,
        _particle_indices,
        _particle_indices_aux,
        _num_particles);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cub::DeviceRadixSort::SortPairs<uint64_t, int>(
        d_temp_storage,
        temp_storage_bytes,
        _particle_morton_codes,
        _particle_morton_codes_aux,
        _particle_indices,
        _particle_indices_aux,
        _num_particles);
    cudaFree(d_temp_storage);
    uint64_t *temp_morton = _particle_morton_codes;
    _particle_morton_codes = _particle_morton_codes_aux;
    _particle_morton_codes_aux = temp_morton;
    int *temp_indices = _particle_indices;
    _particle_indices = _particle_indices_aux;
    _particle_indices_aux = temp_indices;
    STOP_TIMER("cub::sort morton");
    

    checkCudaErrors(cudaMemset(_BRTree_internal_nodes_parents, 0xff, (_num_particles - 1) * sizeof(int)));

    computeGridSize(_num_particles - 1, kCudaBlockSize, num_blocks, num_threads);
    BEGIN_TIMER("buildBRTree_kernel");
    buildBRTree_kernel <<< num_blocks, num_threads >>> (
        _particle_morton_codes,
        _BRTree_internal_nodes_children,
        _BRTree_internal_nodes_parents,
        _BRTree_leaf_nodes_parents
        );
    getLastCudaError("buildBRTree_kernel");
    STOP_TIMER("buildBRTree_kernel");

    checkCudaErrors(cudaMemset(_atomic_counters_, 0, _num_particles * sizeof(int)));
    computeGridSize(_num_particles, kCudaBlockSize, num_blocks, num_threads);

    int *BRTree_nodes_siblings = (int*)_particle_morton_codes;
    BEGIN_TIMER("buildAABBTree_kernel");
    buildAABBTree_kernel <<< num_blocks, num_threads >>> (
        _BRTree_internal_nodes_parents,
        _BRTree_internal_nodes_children,
        _BRTree_leaf_nodes_parents,
        _particle_phases,
        _particle_indices,
        _particle_positions_pred,
        _atomic_counters_, // dConstraintCounts_
        _particle_AABBs,
        BRTree_nodes_siblings
        );
    getLastCudaError("buildAABBTree_kernel");
    STOP_TIMER("buildAABBTree_kernel");

    checkCudaErrors(cudaMemset(_particle_neighbors, 0xff, _num_particles * _max_num_neighbors_per_particle * sizeof(int)));

    computeGridSize(_num_particles, kCudaBlockSize, num_blocks, num_threads);
    int *BRTree_internal_nodes_phases = _atomic_counters_;

    BEGIN_TIMER("findNeighbors_stack_AABB_kernel");
    findNeighbors_stack_AABB_kernel <<< num_blocks, num_threads >>> (
        _BRTree_internal_nodes_parents,
        _BRTree_leaf_nodes_parents,
        _BRTree_internal_nodes_children,
        BRTree_nodes_siblings, // dParticleMortonCodes_
        _particle_indices,
        _particle_AABBs,
        _particle_positions_pred,
        _particle_phases,
        BRTree_internal_nodes_phases, // dConstraintCounts_
        _BRTree_traversal_stack,
        _particle_neighbors
        );
    getLastCudaError("findNeighbors_stack_AABB_kernel");
    STOP_TIMER("findNeighbors_stack_AABB_kernel");

}


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
)
{
    uint num_threads, num_blocks;
    computeGridSize(_num_particles, kCudaBlockSize, num_blocks, num_threads);
    
    checkCudaErrors(cudaMemset(_constraint_counts, 0, _num_particles * sizeof(uint)));
    thrust::fill(thrust::device,
        thrust::device_ptr<float4>(_constraint_deltas),
        thrust::device_ptr<float4>(_constraint_deltas + _num_particles),
        make_float4(0.f)
    );

    solveContacts_kernel <<< num_blocks, num_threads >>> (
        _is_pre_stabilize_mode, // 1: stabilization, 0: solver iteration
        _particle_neighbors,
        _particle_phases,
        __particle_inv_masses,
        _rigid_sdfs, // always keep initial sdf gradient, transformed into world space before using
        _rigid_rotations,
        _planes,
        _constraint_deltas,
        _constraint_counts,
        _particle_positions,
        _particle_positions_pred
    );
    getLastCudaError("solveContacts_kernel");

    if (_is_pre_stabilize_mode) {
        prestabilize_kernel <<< num_blocks, num_threads >>> (
            _constraint_deltas,
            _constraint_counts,
            _particle_velocities,
            _particle_positions,
            _particle_positions_pred
        );
        getLastCudaError("prestabilize_kernel");
    }
    else {
        updatePosPred_kernel <<< num_blocks, num_threads >>> (
            _constraint_deltas,
            _constraint_counts,
            _particle_positions_pred
        );
        getLastCudaError("prestabilize_kernel");
    }
}


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
)
{
    dim3 num_threads, num_blocks;
    computeGridSize(_num_rigids, kCudaBlockSize, num_blocks.x, num_threads.x);
    BEGIN_TIMER("calcRotations_kernel");
    calcRotations_kernel <<< num_blocks, num_threads >>> (
        _rigid_particle_offsets,
        _particle_positions_init,
        _particle_positions_pred,
        _rigid_particle_weights,
        _rigid_centers_of_mass,
        _rigid_relative_positions,
        _rigid_relative_positions_init,
        _rigid_rotations
    );
    getLastCudaError("calcRotations_kernel");
    STOP_TIMER("calcRotations_kernel");

    computeGridSize(_num_rigid_particles, kCudaBlockSize, num_blocks.x, num_threads.x);
    BEGIN_TIMER("matchShapes_kernel");
    matchShapes_kernel <<< num_blocks, num_threads >>> (
        _rigid_centers_of_mass,
        _rigid_rotations,
        _rigid_stiffnesses,
        _rigid_relative_positions_init,
        _particle_phases,
        _particle_positions_pred
    );
    getLastCudaError("matchShapes_kernel");
    STOP_TIMER("matchShapes_kernel");
}

// assume rigid particles are contiguously stored
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
)
{
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    dim3 num_threads, num_blocks;

    BEGIN_TIMER("calcRigidRotations");
    // weighted postions
    thrust::transform(
        thrust::device,
        _particle_positions_pred + _rigid_particles_begin,
        _particle_positions_pred + _rigid_particles_begin + _num_rigid_particles,
        _rigid_particle_weights,
        _rigid_relative_positions,
        vec_scalar_multiply<float4, float>()
    );

    // weighted postions init
    thrust::transform(
        thrust::device,
        _particle_positions_init + _rigid_particles_begin,
        _particle_positions_init + _rigid_particles_begin + _num_rigid_particles,
        _rigid_particle_weights,
        _rigid_relative_positions_init,
        vec_scalar_multiply<float4, float>()
    );

    cub::DeviceSegmentedReduce::Sum<float4*, float4*>(
        d_temp_storage, 
        temp_storage_bytes, 
        _rigid_relative_positions + _rigid_particles_begin,
        _rigid_centers_of_mass,
        _num_rigids,
        _rigid_particle_offsets, 
        _rigid_particle_offsets + 1
    );
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // sums of weighted postions
    cub::DeviceSegmentedReduce::Sum<float4*, float4*>(
        d_temp_storage,
        temp_storage_bytes,
        _rigid_relative_positions + _rigid_particles_begin,
        _rigid_centers_of_mass,
        _num_rigids,
        _rigid_particle_offsets,
        _rigid_particle_offsets + 1
    );

    // sums of weighted postions init
    cub::DeviceSegmentedReduce::Sum<float4*, float4*>(
        d_temp_storage,
        temp_storage_bytes,
        _rigid_relative_positions_init + _rigid_particles_begin,
        _rigid_centers_of_mass_init,
        _num_rigids,
        _rigid_particle_offsets,
        _rigid_particle_offsets + 1
    );

    cudaFree(d_temp_storage);
    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    
    cub::DeviceSegmentedReduce::Sum<float*, float*>(
        d_temp_storage,
        temp_storage_bytes,
        _rigid_particle_weights,
        _rigid_weights,
        _num_rigids,
        _rigid_internal_particle_offsets,
        _rigid_internal_particle_offsets + 1
    );
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // calc rigid weights
    cub::DeviceSegmentedReduce::Sum<float*, float*>(
        d_temp_storage,
        temp_storage_bytes,
        _rigid_particle_weights,
        _rigid_weights,
        _num_rigids,
        _rigid_internal_particle_offsets,
        _rigid_internal_particle_offsets + 1
    );
    cudaFree(d_temp_storage);
    
    thrust::transform(
        thrust::device,
        _rigid_centers_of_mass,
        _rigid_centers_of_mass + _num_rigids,
        _rigid_weights,
        _rigid_centers_of_mass,
        vec_scalar_divide<float4, float>()
    );

    thrust::transform(
        thrust::device,
        _rigid_centers_of_mass_init,
        _rigid_centers_of_mass_init + _num_rigids,
        _rigid_weights,
        _rigid_centers_of_mass_init,
        vec_scalar_divide<float4, float>()
    );

    computeGridSize(_num_rigid_particles, kCudaBlockSize, num_blocks.x, num_threads.x);
    
    
    calcRigidParticleCovMatrices_kernel <<< num_blocks, num_threads >>> (
        _particle_phases,
        _rigid_centers_of_mass,
        _rigid_centers_of_mass_init,
        _particle_positions_init,
        _particle_positions_pred,
        _rigid_particle_weights,
        _rigid_weights,
        _rigid_rotations,
        _rigid_relative_positions_init,
        _rigid_particle_cov_matrices
    );

    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    cub::DeviceSegmentedReduce::Sum<mat3*, mat3*>(
        d_temp_storage,
        temp_storage_bytes,
        _rigid_particle_cov_matrices,
        _rigid_cov_matrices,
        _num_rigids,
        _rigid_internal_particle_offsets,
        _rigid_internal_particle_offsets + 1
    );
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // calc cov matrices
    cub::DeviceSegmentedReduce::Sum<mat3*, mat3*>(
        d_temp_storage,
        temp_storage_bytes,
        _rigid_particle_cov_matrices,
        _rigid_cov_matrices,
        _num_rigids,
        _rigid_internal_particle_offsets,
        _rigid_internal_particle_offsets + 1
    );
    cudaFree(d_temp_storage);

    computeGridSize(_num_rigids, kCudaBlockSize, num_blocks.x, num_threads.x);
    calcRigidRotations_kernel <<< num_blocks, num_threads >>> (
        _rigid_cov_matrices,
        _rigid_rotations
    );
    STOP_TIMER("calcRigidRotations");

    computeGridSize(_num_rigid_particles, kCudaBlockSize, num_blocks.x, num_threads.x);
    BEGIN_TIMER("matchShapes_kernel");
    matchShapes_kernel << < num_blocks, num_threads >> > (
        _rigid_centers_of_mass,
        _rigid_rotations,
        _rigid_stiffnesses,
        _rigid_relative_positions_init,
        _particle_phases,
        _particle_positions_pred
        );
    getLastCudaError("matchShapes_kernel");
    STOP_TIMER("matchShapes_kernel");
}


void setSmoothKernel(CubicSmoothKernel *_hSmoothKernel)
{
    checkCudaErrors(cudaMemcpyToSymbol(kSmoothKernel, _hSmoothKernel, sizeof(CubicSmoothKernel)));
}

void
solverFluidDensityConstraints(
    int _num_fluid_particles,
    float4 *_particle_positions_pred,
    int const *_particle_phases,
    int const *_particle_neighbors,
    float const *_particle_inv_masses,
    float  *_fluid_lambdas
)
{
    uint num_threads, num_blocks;
    computeGridSize(_num_fluid_particles, kCudaBlockSize, num_blocks, num_threads);
    computeFluidLambda_kernel << < num_blocks, num_threads >> > (
        _particle_positions_pred,
        _particle_phases,
        _particle_neighbors,
        _particle_inv_masses,
        _fluid_lambdas
        );

    solveFluidDensityConstraints_kernel << < num_blocks, num_threads >> > (
        _particle_phases,
        _fluid_lambdas,
        _particle_neighbors,
        _particle_inv_masses,
        _particle_positions_pred
        );
}

void finalize(
    int _num_particles,
    float4 const *_particle_positions_pred,
    float4 *_particle_velocities,
    float4 *_particle_positions
)
{
    dim3 num_threads, num_blocks;
    computeGridSize(_num_particles, kCudaBlockSize, num_blocks.x, num_threads.x);
    finalize_kernel <<< num_blocks, num_threads >>> (
        _particle_positions_pred,
        _particle_velocities,
        _particle_positions
    );
    getLastCudaError("finalize_kernel");
}
