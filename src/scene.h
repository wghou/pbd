#pragma once

#include <memory>
#include "gl_helper.h"
#include "scene_cuda.h"
#include "mesh.h"
#include "solver.h"

extern int gNumSamples;
extern bool gToggleMSAA;
extern bool gToggleWall;

extern glm::vec3 gLightPos;

extern Camera gCamera;
extern const float kCameraLowSpeed;
extern const float kCameraSpeed;
extern const float kCameraHighSpeed;
extern const float kCameraRotationSpeed;
extern const glm::vec3 kCameraPos;
extern const glm::vec3 kLookAt;
extern const glm::vec3 kUp;

extern glm::mat4 gProjMat;

extern bool gToggleWireframe;
extern bool gToggleDrawParticle;
extern bool gToggleDrawMesh;
extern bool gToggleDrawFluid;
extern bool gToggleStepMode;
extern float gFluidParticleAlpha;

void setupLight();
void updateProjMat(int width, int height);

// initialize OpenGL Program Objects
void initPrograms();
void reloadShaders();

const glm::vec3 kColors[7] = {
    { 0.000f, 0.625f, 1.000f},
    { 1.000f, 0.442f, 0.000f},
    { 0.000f, 0.436f, 0.216f},
    { 1.000f, 0.891f, 0.058f},
    { 0.013f, 0.213f, 0.566f},
    { 0.841f, 0.138f, 0.000f},
    { 0.765f, 0.243f, 0.493f}
};


class Scene
{
public:
    Scene(const std::string & _name) : name_(_name){}
    virtual ~Scene() { release(); }

    std::string getName() const { return name_; }
    void init(bool reset_params = true);
    void resizeFramebuffers(int _width, int _height);
    void pickParticle(int _x, int _y); // GL window coord
    virtual void release();
    void update();
    void render();

    Params& getSolverParams() { return solver_.params_; }

private:
    void setupSamplers();
    void setupFramebuffers();
    void setupScreenRectVao();
    void restoreParams();
    void skinMesh();
    void drawPlanes();
    void drawSkin();
    void drawParticles(int _particle_offset, int _num_particles);
    void drawShadowPlanes();
    void drawShadowSkin();
    void drawShadowParticles(int _particle_offset, int _num_particles);
    void drawShadowMap();
    void drawScene();
    void drawFluid();
    void drawScreenRect();

    GLuint plane_vao_;
    GLuint plane_bos_[4]; // 0: position; 1: tex_uv; 2: normal; 3: indices
    int num_plane_indices_ = 0;

    GLuint skin_vao_;
    GLuint skin_bos_[4]; // 0: position; 1: color; 2: normal; 3: indices
    int num_skin_vertices_ = 0;
    int num_skin_indices_ = 0;

    struct cudaGraphicsResource *dSkinBOsRes_[3]; // 0: position; 1: normal; 2: indices
    float3 *dSkinPositionsInit_ = 0; // for skinning, vpos += (R * (init_vpos - init_particle[i]) + particle[i]) * w, \forall i \in [0, 3]
    float3 *dSkinPositions_ = 0;
    float3 *dSkinNormals_ = 0;
    uint *dSkinIndices_ = 0;

    int4 *dSkinParticleIndices_ = 0;
    float4 *dSkinParticleWeights_ = 0;

    GLuint particle_vao_;
    GLuint particle_bos_[2]; // all particle data (including solver part) should be pre-allocated during initialization
    struct cudaGraphicsResource *dParticleBOsRes_[2]; // 0: position; 1: phase

    GLuint framebuffers_[4]; // 0: shadow framebuffer; 1: G-Buffer for deferred shading; 2: picking framebuffer; 4. fluid framebuffer
    GLuint fbo_textures_[9]; // 0: shadow tex (depth, depth^2); 1: shadow fbo depth; 
                             // 2: view_pos tex; 3: view_normal tex;  4: color tex;  5: shadow coord; 6. geometry fbo depth;  
                             // 7: picking_id tex; 8: picking fbo depth 
    GLuint fluid_textures_[3]; // 0: fluid thickness; 1: fluid position; 2: fluid normal
    GLuint msaa_framebuffer_;
    GLuint msaa_fbo_textures_[5];// 0: view_pos tex; 1: view_normal tex;  2: color tex;  3: shadow coord; 5. depth

    GLuint tex_samplers_[3]; // 0: general; 1: mipmap; 2: mipmap_repeat
    GLuint picking_pixel_bo_; // pixel pack buffer obj for picking
    int    picked_particle_idx_;

    GLuint screen_rect_vao_;
    GLuint screen_rect_bos_[2]; // 0: position; 1: texture uv

protected:
    virtual void resetParams() = 0;
    virtual void setup() = 0;
    virtual void modify() = 0; // optionally modify the scene during simulation

    void addParticles(
        std::vector<float4> &_particle_positions,
        std::vector<float4> &_particle_velocities,
        std::vector<float> &_particle_inv_masses,
        int _particle_phase);

    void setupPlanes(std::vector<float4> const &_planes, std::vector<float4> &_rendered_planes = std::vector<float4>());
    void setupShapeParticles(std::vector<Mesh> &_meshes, std::vector<ParticlizeConfig> &_config);

    Solver solver_;
    Params saved_params_;

    std::string name_;
    float min_particle_radius_;
    float min_rigid_particle_radius_;
    int max_num_particles_ = 524288; // 2^19
};

class RigidPile : public Scene
{
public:
    RigidPile(
        std::string const &_name,
        std::string const &_mesh_path,
        float _min_rigid_particle_radius,
        uint3 _vol_dim,
        int3 _pile_dim,
        int _max_num_particles,
        float _particle_merge_threshold = -FLT_MAX);

protected:
    void resetParams() override;
    void setup() override;
    void modify() override;
    std::string mesh_path_;
    uint3 vol_dim_;
    int3 pile_dim_;
    float particle_merge_threshold_;
};

class TwoRigidBodies : public Scene
{
public:
    TwoRigidBodies(
        std::string const &_name, 
        std::string const &_mesh1_path, 
        std::string const &_mesh2_path,
        float _min_rigid_particle_radius,
        uint3 _vol_dim1,
        uint3 _vol_dim2,
        int _max_num_particles);

protected:
    void resetParams() override;
    void setup() override;
    void modify() override {}

    std::string mesh1_path_;
    std::string mesh2_path_;
    uint3 vol_dim1_;
    uint3 vol_dim2_;
};

class GranularPile : public Scene
{
public:
    GranularPile(
        std::string const &_name,
        std::string const &_mesh_path,
        float _min_rigid_particle_radius,
        uint3 _vol_dim,
        int _max_num_particles,
        float _particle_merge_threshold = -FLT_MAX);

protected:
    void resetParams() override;
    void setup() override;
    void modify() override;
    std::string mesh_path_;
    uint3 vol_dim_;
    float particle_merge_threshold_;
};

class BreakingDam : public Scene
{
public:
    BreakingDam(
        std::string const &_name,
        float _min_particle_radius,
        int3 _fluid_vol_dim,
        int _max_num_particles,
        float _fluid_rest_density);

protected:
    void resetParams() override;
    void setup() override;
    void modify() override;

    int3 fluid_vol_dim_;
    float fluid_rest_density_;
    float4 movable_wall_;
    int idx_movable_wall_;
    float wall_delta_;
    float2 wall_extreme_;
};

class RigidFluid : public Scene
{
public:
    RigidFluid(
        std::string const &_name,
        std::string const &_mesh_path,
        float _min_rigid_particle_radius,
        float _rigid_density_scale,
        int3 _rigid_vol_dim,
        float _min_particle_radius,
        int3 _fluid_vol_dim,
        int _max_num_particles,
        float _fluid_rest_density,
        float _particle_merge_threshold = -FLT_MAX);

protected:
    void resetParams() override;
    void setup() override;
    void modify() override;

    std::string mesh_path_;
    int3 rigid_vol_dim_;
    float rigid_density_scale_;
    float particle_merge_threshold_;

    int3 fluid_vol_dim_;
    float fluid_rest_density_;
    float4 movable_wall_;
    int idx_movable_wall_;
    float wall_delta_;
    float2 wall_extreme_;
};