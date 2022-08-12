#pragma once
//#define _PROFILE_SOLVER_
extern const int INTERVAL;
//#define SOLVER_LOGGER

#include "gl_helper.h"
#include "misc.h"
#include "cuda_helper.h"
#include "solver_cuda.h"

#include <vector>
#include <memory>
#include <fstream>
#include <sstream>
#include <iomanip>


class Solver
{
public:
    void setupPlanes(std::vector<float4> const &_planes);

    Solver() {};
    void init(int _max_num_particles, float __min_particle_radius, float _min_rigid_surface_sdf);
    void postInit();
    void release();
    ~Solver() { release(); }
    void update();

    Params params_;
    
    float4 *dParticlePositions_ = 0; // map to opengl bo
    int *dParticlePhases_ = 0; // map to opengl bo
    
    int *dParticleNeighbors_ = 0; // max_num_neighbors_per_particle * max_num_particles

    // max_num_particles
    float  *dParticleInvMasses_ = 0;
    float4 *dParticleVelocities_ = 0;
    float4 *dParticlePositionsInit_ = 0;
    float4 *dParticlePositionsPred_ = 0;
    float4 *dConstraintDeltas_ = 0; // also used as _particle_bounding_spheres_aux during BVH construction
    int    *dConstraintCounts_ = 0; //  also used as atomic counter during BVH construction; internal nodes phase

    // binary radix tree
    int *dBRTreeTraversalStack_ = 0; // 64 * max_num_particles

    //max_num_particles
    int  *dBRTreeLeafNodesParents_ = 0; // parent(MSB of parent indicates left or right child)
    int *dParticleIndices_ = 0;
    int *dParticleIndicesAux_ = 0; // double buffer for sorting
    uint64_t *dParticleMortonCodes_ = 0; // 15 bits for each axis; also used as array of siblings
    uint64_t *dParticleMortonCodesAux_ = 0; // double buffer for sorting

    // max_num_particles - 1
    int  *dBRTreeInternalNodesParents_ = 0; // parent(MSB of parent indicates left or right child)
    int2 *dBRTreeInternalNodesChildren_ = 0; //left child, right child (MSB of children indicates Leaf or Internal Node)
    float4 *dParticleBoundingSpheres_ = 0; // (pos, radius), <also be used for prefix sum to find min>
    cuAABB *dParticleAABBs_ = 0; 
    
    // num_rigid_particles
    float4 *dRigidSdfs_ = 0; // allocated during rigid body init in scene
    float4 *dRigidRelativePositionsInit_ = 0;
    float4 *dRigidRelativePositions_ = 0; 
    float  *dRigidParticleWeights_ = 0; // for shape matching
    mat3   *dRigidParticleCovMatrices_ = 0; // num_rigid_particles_, mat3, column major
    
    int    *dRigidParticleOffsets_ = 0; // num_rigids_ + 1
    int    *dRigidInternalParticleOffsets_ = 0; // num_rigids_ + 1, 0th element: 0
    float  *dRigidStiffnesses_ = 0; // num_rigids_
    float4 *dRigidCentersOfMassInit_ = 0;
    float4 *dRigidCentersOfMass_ = 0;
    float  *dRigidWeights_ = 0; // num_rigids_, sum of particle weights
    quat   *dRigidRotations_ = 0; // num_rigids_ (quaternions)
    mat3   *dRigidCovMatrices_ = 0; // num_rigids_, mat3, column major

    // num_fluid_particles
    float *dFluidLambdas_ = 0;

    float4 *dPlanes_ = 0;

private:
    void updateOneStep();

    void printBRTree();
    void printAABBTree();
    void printNeighbors();
    std::fstream logger_;

};

/************************************
memory usage per particle estimation:
             num * bytes
neighbor:     64 * 4
pos:           4 * 4
pos init:      4 * 4
pos pred:      4 * 4
velocity:      4 * 4
delta pos:     4 * 4
bound sphere:  4 * 4
solid contact: 2 * 4
intern child:  2 * 4
morton code:   2 * 4 (or 1 * 8)
phase:         1 * 4
invMass:       1 * 4
num deltas:    1 * 4
leaf parent:   1 * 4
indices:       1 * 4
intern parent: 1 * 4
--------------------------
total:       100 * 4

rigid particle:
sdf:           4 * 4
rela pos:      4 * 4
rela pos init: 4 * 4
weight:        1 * 4
--------------------------
              13 * 4

rigid:
offsets:       1 * 4
stiffness:     1 * 4
CoM:           4 * 4
quat:          4 * 4
--------------------------
              10 * 4

=======================================================



************************************/