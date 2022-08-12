
#include <algorithm>
#include <glm/glm.hpp>
#include <thread>
#include <ctime>
#include <cmath>
#include <queue>

#include "solver.h"
#include "mesh.h"
#include "cg_math.h"
#include "misc.h"

const int INTERVAL = 200;

void
Solver::init(int _max_num_particles, float _min_particle_radius, float _min_rigid_surface_sdf)
{
    params_.inv_min_particle_radius_ = 1.f / _min_particle_radius;
    params_.min_particle_radius_ = _min_particle_radius;
    params_.min_rigid_surface_sdf_ = _min_rigid_surface_sdf;
    // precedence: "+" > "bit <<"
    params_.particle_upper_ = (float)(1 << params_.morton_bits_per_axis_) * _min_particle_radius * 0.5f;

    std::cout << "SOLVER: max total neighbors: " << _max_num_particles * params_.max_num_neighbors_per_particle_ << "(per particle: " << params_.max_num_neighbors_per_particle_ << ")" << std::endl;
    allocateArray((void**)&dParticleNeighbors_, _max_num_particles * params_.max_num_neighbors_per_particle_ * sizeof(int));
    allocateArray((void**)&dParticleInvMasses_, _max_num_particles * sizeof(float));
    allocateArray((void**)&dParticleVelocities_, _max_num_particles * sizeof(float4));
    allocateArray((void**)&dParticlePositionsInit_, _max_num_particles * sizeof(float4));
    allocateArray((void**)&dParticlePositionsPred_, _max_num_particles * sizeof(float4));
    allocateArray((void**)&dConstraintDeltas_, _max_num_particles * sizeof(float4));
    allocateArray((void**)&dConstraintCounts_, _max_num_particles * sizeof(int));

    allocateArray((void**)&dFluidLambdas_, _max_num_particles * sizeof(float));

    allocateArray((void**)&dBRTreeTraversalStack_, _max_num_particles * params_.stack_size_per_particle_ * sizeof(int));
    allocateArray((void**)&dParticleAABBs_, (_max_num_particles - 1) * 2 * sizeof(float4));
    allocateArray((void**)&dBRTreeLeafNodesParents_, _max_num_particles * sizeof(int));
    allocateArray((void**)&dParticleIndices_, _max_num_particles * sizeof(int));
    allocateArray((void**)&dParticleMortonCodes_, _max_num_particles * sizeof(uint64_t));
    allocateArray((void**)&dParticleIndicesAux_, _max_num_particles * sizeof(int));
    allocateArray((void**)&dParticleMortonCodesAux_, _max_num_particles * sizeof(uint64_t));

    allocateArray((void**)&dBRTreeInternalNodesParents_, (_max_num_particles - 1) * sizeof(int));
    allocateArray((void**)&dBRTreeInternalNodesChildren_, (_max_num_particles - 1) * sizeof(int2));
    allocateArray((void**)&dParticleBoundingSpheres_, (_max_num_particles - 1) * sizeof(float4));
    allocateArray((void**)&dParticleAABBs_, (_max_num_particles - 1) * sizeof(cuAABB));

#ifdef SOLVER_LOGGER
    std::string file_name("../data/log/log.txt");
    logger_.open(file_name, logger_.binary | logger_.out | logger_.trunc);
    if (!logger_.is_open()) {
        std::cout << "open " << file_name << " failed." << std::endl;
    }
#endif
}

void
Solver::setupPlanes(std::vector<float4> const &_planes)
{
    params_.num_planes_ = static_cast<int>(_planes.size());
    createArray((void**)&dPlanes_, _planes.data(), 0, params_.num_planes_ * sizeof(float4));
}

void
Solver::postInit()
{
    if (params_.num_rigid_particles_ > 0) {
        allocateArray((void**)&dRigidRelativePositionsInit_, params_.num_rigid_particles_ * sizeof(float4));
        allocateArray((void**)&dRigidRelativePositions_, params_.num_rigid_particles_ * sizeof(float4));
        allocateArray((void**)&dRigidParticleWeights_, params_.num_rigid_particles_ * sizeof(float));
        allocateArray((void**)&dRigidParticleCovMatrices_, params_.num_rigid_particles_ * sizeof(mat3));
        allocateArray((void**)&dRigidInternalParticleOffsets_, (params_.num_rigids_ + 1) * sizeof(int));
        allocateArray((void**)&dRigidCentersOfMassInit_, params_.num_rigids_ * sizeof(float4));
        allocateArray((void**)&dRigidCentersOfMass_, params_.num_rigids_ * sizeof(float4));
        allocateArray((void**)&dRigidWeights_, params_.num_rigids_ * sizeof(float));
        allocateArray((void**)&dRigidRotations_, params_.num_rigids_ * sizeof(quat));
        allocateArray((void**)&dRigidCovMatrices_, params_.num_rigids_ * sizeof(mat3));
    }

    postInitDevMem(
        params_.num_particles_,
        params_.rigid_particles_begin_,
        params_.num_rigid_particles_, 
        params_.num_rigids_,
        params_.min_rigid_surface_sdf_ * 0.5f, 
        dParticlePositions_,
        dParticlePositionsInit_,
        dRigidParticleOffsets_,
        dRigidInternalParticleOffsets_,
        dParticleInvMasses_,
        dRigidParticleWeights_, 
        dRigidRotations_
    );
}

void 
Solver::release()
{
    if (logger_.is_open()) {
        logger_.close();
    }

    FREE_CUDA_ARRAY(dParticleNeighbors_); // max_num_neighbors_per_particle * max_num_particles

    FREE_CUDA_ARRAY(dParticleInvMasses_);
    FREE_CUDA_ARRAY(dParticleVelocities_);
    FREE_CUDA_ARRAY(dParticlePositionsInit_);
    FREE_CUDA_ARRAY(dParticlePositionsPred_);
    FREE_CUDA_ARRAY(dConstraintDeltas_);
    FREE_CUDA_ARRAY(dConstraintCounts_);

    FREE_CUDA_ARRAY(dFluidLambdas_);

    FREE_CUDA_ARRAY(dBRTreeLeafNodesParents_);
    FREE_CUDA_ARRAY(dParticleIndicesAux_);
    FREE_CUDA_ARRAY(dParticleMortonCodesAux_);
    FREE_CUDA_ARRAY(dParticleIndices_);
    FREE_CUDA_ARRAY(dParticleMortonCodes_);

    FREE_CUDA_ARRAY(dBRTreeInternalNodesParents_);
    FREE_CUDA_ARRAY(dBRTreeInternalNodesChildren_);
    FREE_CUDA_ARRAY(dParticleBoundingSpheres_);

    FREE_CUDA_ARRAY(dParticleAABBs_);
    FREE_CUDA_ARRAY(dBRTreeTraversalStack_);

    if (params_.num_rigids_ > 0) {
        FREE_CUDA_ARRAY(dRigidSdfs_); // allocated during rigid body init in scene
        FREE_CUDA_ARRAY(dRigidRelativePositions_);
        FREE_CUDA_ARRAY(dRigidRelativePositionsInit_);
        FREE_CUDA_ARRAY(dRigidParticleWeights_); // for shape matching
        FREE_CUDA_ARRAY(dRigidParticleCovMatrices_);

        FREE_CUDA_ARRAY(dRigidParticleOffsets_); // num_rigids_ + 1
        FREE_CUDA_ARRAY(dRigidInternalParticleOffsets_);
        FREE_CUDA_ARRAY(dRigidStiffnesses_); // num_rigids_
        FREE_CUDA_ARRAY(dRigidCentersOfMassInit_);
        FREE_CUDA_ARRAY(dRigidCentersOfMass_);
        FREE_CUDA_ARRAY(dRigidWeights_);
        FREE_CUDA_ARRAY(dRigidRotations_); // num_rigids_ (quaternions)
        FREE_CUDA_ARRAY(dRigidCovMatrices_); // num_rigids_, column major
    }

    if (params_.num_planes_) {
        FREE_CUDA_ARRAY(dPlanes_);
    }
    std::cout << "solver resources released..." << std::endl;
}

void
Solver::update()
{
    // glFinish();
    BEGIN_TIMER("updateSolver");
    params_.step_delta_time_ = params_.frame_delta_time_ / params_.num_substeps_;
    params_.num_frames_ += 1;
    setParameters(&params_);
    
    CubicSmoothKernel smooth_kernel{ params_.fluid_neighbor_search_range_ };
    setSmoothKernel(&smooth_kernel);
    params_.fluid_corr_refer_inv_W_ = 1.f / smooth_kernel.W(make_float3(params_.fluid_corr_refer_q_ * smooth_kernel.h_, 0.f, 0.f));

    //std::cout << "set params of size " << sizeof(Params) << std::endl;
    for (int i = 0; i < params_.num_substeps_; ++i) {
        params_.exec_time_ += params_.step_delta_time_;
        BEGIN_TIMER("updateOneStep");
        updateOneStep();
        //std::cout << "updated step " << i << std::endl;
        STOP_TIMER("updateOneStep");
    }
    STOP_TIMER("updateSolver");
    
}

void
Solver::updateOneStep()
{
    integrate(
        params_.num_particles_,
        dParticlePositions_,
        dPlanes_,
        dParticlePhases_,
        dParticleInvMasses_,
        dParticleVelocities_,
        dParticlePositionsPred_
    );
    BEGIN_TIMER("findNeighbors");
#if 0
    findNeighbors(
        params_.num_particles_,
        params_.max_num_neighbors_per_particle_,
        dParticlePhases_,
        dParticlePositionsPred_, // positions will be clamped
        dParticleMortonCodes_,
        dParticleIndices_,
        dBRTreeInternalNodesChildren_,
        dBRTreeInternalNodesParents_,
        dBRTreeLeafNodesParents_,
        dConstraintCounts_, // dConstraintCounts_
        dConstraintDeltas_, // dConstraintDeltas_
        dParticleBoundingSpheres_,
        dBRTreeTraversalStack_,
        dParticleNeighbors_
    );

#else
    findNeighbors_AABB(
        params_.num_particles_,
        params_.max_num_neighbors_per_particle_,
        dParticlePhases_,
        dParticlePositionsPred_, // positions will be clamped
        dParticleMortonCodes_,
        dParticleMortonCodesAux_,
        dParticleIndices_,
        dParticleIndicesAux_,
        dBRTreeInternalNodesChildren_,
        dBRTreeInternalNodesParents_,
        dBRTreeLeafNodesParents_,
        dConstraintCounts_, // dConstraintCounts_
        dParticleAABBs_,
        dBRTreeTraversalStack_,
        dParticleNeighbors_
    );
#endif

    STOP_TIMER("findNeighbors");
    // deviceSync();
    // printBRTree();
    // printAABBTree();
    // deviceSync();
    // printNeighbors();
    // deviceSync();
    int is_pre_stabilize_mode = 1;
    for (int i = 0; i < params_.num_pre_stabilize_iterations_; ++i) {
        BEGIN_TIMER("solveContacts");
        solveContacts(
            params_.num_particles_,
            is_pre_stabilize_mode, // 1: pre-stabilization, 0: solver iteration
            dParticleNeighbors_,
            dParticlePhases_,
            dParticleInvMasses_,
            dRigidSdfs_,
            dRigidRotations_,
            dPlanes_,
            dConstraintDeltas_,
            dConstraintCounts_,
            dParticlePositions_,
            dParticlePositionsPred_,
            dParticleVelocities_
        );
        STOP_TIMER("solveContacts");
    }

    swapParticleMass(
        params_.num_particles_,
        dParticleVelocities_,
        dParticleInvMasses_
    );

    is_pre_stabilize_mode = 0;
    for (int i = 0; i < params_.num_iterations_; ++i) {
        swapParticleMass(
            params_.num_particles_,
            dParticleVelocities_,
            dParticleInvMasses_
        );

        BEGIN_TIMER("solveContacts");
        solveContacts(
            params_.num_particles_,
            is_pre_stabilize_mode, // 1: pre-stabilization, 0: solver iteration
            dParticleNeighbors_,
            dParticlePhases_,
            dParticleInvMasses_,
            dRigidSdfs_,
            dRigidRotations_,
            dPlanes_,
            dConstraintDeltas_,
            dConstraintCounts_,
            dParticlePositions_,
            dParticlePositionsPred_,
            dParticleVelocities_
        );
        STOP_TIMER("solveContacts");
        
        swapParticleMass(
            params_.num_particles_,
            dParticleVelocities_,
            dParticleInvMasses_
        );

        if (params_.num_rigids_ > 0) {
            BEGIN_TIMER("matchRigidShapes");
#if 0
            matchRigidShapes(
                params_.num_rigids_,
                params_.num_rigid_particles_,
                dParticlePhases_,
                dRigidParticleOffsets_,
                dParticlePositionsInit_,
                dRigidParticleWeights_,
                dRigidStiffnesses_,
                dRigidCentersOfMass_,
                dRigidRelativePositions_,
                dRigidRelativePositionsInit_,
                dRigidRotations_,
                dParticlePositionsPred_
            );
#else 
            matchRigidShapes_cub(
                params_.num_rigids_,
                params_.num_rigid_particles_,
                params_.rigid_particles_begin_,
                dParticlePhases_,
                dRigidParticleOffsets_, // cannot be const, requirement from cub
                dRigidInternalParticleOffsets_,
                dParticlePositionsInit_,
                dRigidParticleWeights_,
                dRigidStiffnesses_,
                dRigidCentersOfMass_,
                dRigidCentersOfMassInit_,
                dRigidWeights_,
                dRigidRelativePositions_,
                dRigidRelativePositionsInit_,
                dRigidParticleCovMatrices_,
                dRigidCovMatrices_,
                dRigidRotations_,
                dParticlePositionsPred_
            );
#endif
            STOP_TIMER("matchRigidShapes");
        }

        if (params_.num_fluid_particles_ > 0) {
            solverFluidDensityConstraints(
                params_.num_fluid_particles_,
                dParticlePositionsPred_,
                dParticlePhases_,
                dParticleNeighbors_,
                dParticleInvMasses_,
                dFluidLambdas_
            );
        }
        
    }

    BEGIN_TIMER("finalize");
    finalize(
        params_.num_particles_,
        dParticlePositionsPred_,
        dParticleVelocities_,
        dParticlePositions_
    );
    STOP_TIMER("finalize");
}


void 
Solver::printBRTree()
{
    std::vector<float4> part_pred{ (size_t) params_.num_particles_ };
    std::vector<float4> b_sphere{ (size_t) params_.num_particles_ - 1};
    std::vector<int> part_idx; part_idx.resize(params_.num_particles_);
    std::vector<int2> children_idx{ (size_t) params_.num_particles_ - 1};

    copyArrayFromDevice(part_pred.data(), dParticlePositionsPred_, 0, params_.num_particles_ * sizeof(float4));
    copyArrayFromDevice(b_sphere.data(), dParticleBoundingSpheres_, 0, (params_.num_particles_ -1) * sizeof(float4));
    copyArrayFromDevice(part_idx.data(), dParticleIndices_, 0, params_.num_particles_ * sizeof(int));
    copyArrayFromDevice(children_idx.data(), dBRTreeInternalNodesChildren_, 0, (params_.num_particles_ - 1) * sizeof(int2));

    std::queue<int> Q;
    Q.push(0);
    while (Q.size() > 0) {
        int node = Q.front();
        Q.pop();
        if (node & kBRTreeChildLeafFlag) {
            int node_idx = node & kBRTreeChildIndexMask;
            int particle_idx = part_idx[node_idx];
            std::cout << "leaf " << node_idx << " : particle " << particle_idx << " (" << part_pred[particle_idx] << ")" << std::endl;
        }
        else {
            int2 children = children_idx[node];
            std::string left_type = (children.x & kBRTreeChildLeafFlag) ? "leaf" : "intern";
            std::string right_type = (children.y & kBRTreeChildLeafFlag) ? "leaf" : "intern";;
            std::cout << "intern " << node << " (" << b_sphere[node] << "): left< " << left_type << ", " << (children.x  & kBRTreeChildIndexMask) << " >; right< " << right_type << ", " << (children.y  & kBRTreeChildIndexMask) << " >" << std::endl;
            Q.push(children.x);
            Q.push(children.y);
        }
    }
    std::cout << "print tree finished." << std::endl;
}


void
Solver::printAABBTree()
{
    std::vector<float4> part_pred{ (size_t)params_.num_particles_ };
    std::vector<cuAABB> AABBs{ (size_t)params_.num_particles_ - 1 };
    std::vector<int> part_idx; part_idx.resize(params_.num_particles_);
    std::vector<int2> children_idx{ (size_t)params_.num_particles_ - 1 };

    copyArrayFromDevice(part_pred.data(), dParticlePositionsPred_, 0, params_.num_particles_ * sizeof(float4));
    copyArrayFromDevice(AABBs.data(), dParticleAABBs_, 0, (params_.num_particles_ - 1) * sizeof(cuAABB));
    copyArrayFromDevice(part_idx.data(), dParticleIndices_, 0, params_.num_particles_ * sizeof(int));
    copyArrayFromDevice(children_idx.data(), dBRTreeInternalNodesChildren_, 0, (params_.num_particles_ - 1) * sizeof(int2));

    std::queue<int> Q;
    Q.push(0);
    while (Q.size() > 0) {
        int node = Q.front();
        Q.pop();
        if (node & kBRTreeChildLeafFlag) {
            int node_idx = node & kBRTreeChildIndexMask;
            int particle_idx = part_idx[node_idx];
            std::cout << "leaf " << node_idx << " : particle " << particle_idx << " (" << part_pred[particle_idx] << ")" << std::endl;
        }
        else {
            int2 children = children_idx[node];
            std::string left_type = (children.x & kBRTreeChildLeafFlag) ? "leaf" : "intern";
            std::string right_type = (children.y & kBRTreeChildLeafFlag) ? "leaf" : "intern";;
            std::cout << "intern " << node << " (" << AABBs[node].lower_ << ", " << AABBs[node].upper_ << "): left< " << left_type << ", " << (children.x  & kBRTreeChildIndexMask) << " >; right< " << right_type << ", " << (children.y  & kBRTreeChildIndexMask) << " >" << std::endl;
            Q.push(children.x);
            Q.push(children.y);
        }
    }
    std::cout << "print tree finished." << std::endl;
}

void
Solver::printNeighbors()
{
    std::vector<int> neighbor_idx; neighbor_idx.resize(params_.max_num_neighbors_per_particle_ * params_.num_particles_);
    copyArrayFromDevice(neighbor_idx.data(), dParticleNeighbors_, 0, params_.max_num_neighbors_per_particle_ * params_.num_particles_ * sizeof(int));
    for (int i = 0; i < params_.num_particles_; ++i) {
        std::cout << "particle " << i << " neighbors:" << std::flush;
        int offset = i * params_.max_num_neighbors_per_particle_;
        while (neighbor_idx[offset] != -1) {
            std::cout << neighbor_idx[offset] << ", " << std::flush;
            ++offset;
        }
        std::cout << std::endl;
    }
}