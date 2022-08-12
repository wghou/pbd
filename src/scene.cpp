#include <algorithm>
#include <glm/ext.hpp>
#include "cg_math.h"
#include "misc.h"
#include "aabb.h"
#include "scene.h"
#include "voxelization_cuda.h"
#include "sdf_cuda.h"

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

extern bool gPause;
extern int gWindowWidth;
extern int gWindowHeight;

const GLint kInvalidParticleIdx = -1;
// perspective parameters
const float kFovY = glm::quarter_pi<float>() / 1.0f;
float gFovX = kFovY;
const float kNearPlane = 0.01f; 
const float kFarPlane = 4000.f;

// camera parameters
const float kCameraLowSpeed = 0.4f;
const float kCameraSpeed = 8.f;
const float kCameraHighSpeed = 16.f;
const float kCameraRotationSpeed = 1.0f;
const glm::vec3 kCameraPos(10.f, 7.5f, 25.f); // default camera position
const glm::vec3 kLookAt(-kCameraPos); // default lookAt
const glm::vec3 kUp(0.f,  1.f,  0.f); // default up, y axis

const int kCheckerTexSize = 4; // physical size/length

glm::vec3 gLightPos(15.f, 25.f, 15.f);
glm::vec3 gLightDir(glm::normalize(glm::vec3(4.0, 0.0, 4.0) - gLightPos));
glm::vec3 gLightImax(1.2f);
float gCosLightPenumbra = glm::cos(glm::pi<float>() / 8.f);
float gCosLightUmbra = glm::cos(1.f * glm::pi<float>() / 2.f);
float gLightExp = 2.f;
float gLightFov = 5.f * glm::pi<float>() / 8.f;

int kShadowMapWidth = 2048, kShadowMapHeight = 2048;
glm::vec2 kShadowSampleSize = 1.f / glm::vec2(kShadowMapWidth, kShadowMapHeight);

glm::mat4 gLightProjMat, gLightViewMat, gLightViewProjMat, gShadowMat; // ShadowMat = ScaleBiasMat * LightProjMat * LightViewMat * InvViewMat

// camera
Camera gCamera;

// perspective projection matrix
glm::mat4 gProjMat, gViewProjMat;
glm::mat4 const kScaleBiasMat {
    0.5, 0.0, 0.0, 0.0, // column 0
    0.0, 0.5, 0.0, 0.0, // column 1
    0.0, 0.0, 0.5, 0.0, // column 2
    0.5, 0.5, 0.5, 1.0  // column 3
};


// OpenGL program objects
GLProgram gMeshProg, gPlaneProg, gParticleProg, // G-buffer
          gShadowParticleProg, gShadowPlaneProg, gShadowMeshProg, // shadow
          gFluidThicknessProg, gFluidSurfaceProg, // fluid
          gScreenRectProg; // shading 

bool gToggleWireframe = false;
bool gToggleDrawParticle = false;
bool gToggleDrawMesh = true;
bool gToggleStepMode = false;
bool gToggleDrawFluid = false;
float gFluidParticleAlpha = 0.025f;

void
setupLight()
{
    gLightViewMat = glm::lookAt(gLightPos, gLightPos + gLightDir, kUp);
    gLightProjMat = glm::perspectiveFov(gLightFov,
        static_cast<float>(kShadowMapWidth),
        static_cast<float>(kShadowMapHeight),
        kNearPlane, kFarPlane);
    gLightViewProjMat = gLightProjMat * gLightViewMat;
}

void 
updateProjMat(int width, int height)
{
    float aspect = static_cast<float>(width) / static_cast<float>(height);
    gFovX = 2.f * glm::atan(glm::tan(kFovY / 2.f) * aspect);
    gProjMat = glm::perspectiveFov(kFovY, 
        static_cast<float>(width), 
        static_cast<float>(height),
        kNearPlane, kFarPlane);
}

void
initPrograms()
{
    std::unordered_map<std::string, GLenum> shader_path_types;
    std::vector<std::string> uniform_names;

    // gScreenRectProg
    shader_path_types["../data/shaders/screen_rect_vert.glsl"] = GL_VERTEX_SHADER;
    shader_path_types["../data/shaders/screen_rect_frag.glsl"] = GL_FRAGMENT_SHADER;

    uniform_names.push_back("ShadowMat");
    uniform_names.push_back("ShadowSampleSize");
    uniform_names.push_back("LightPosView");
    uniform_names.push_back("LightDirView");
    uniform_names.push_back("LightImax");
    uniform_names.push_back("LightParams");

    gScreenRectProg.init(shader_path_types, uniform_names, "screen rect program");
    shader_path_types.clear();
    uniform_names.clear();

    // gFluidThicknessProg
    shader_path_types["../data/shaders/fluid_thickness_vert.glsl"] = GL_VERTEX_SHADER;
    shader_path_types["../data/shaders/fluid_thickness_frag.glsl"] = GL_FRAGMENT_SHADER;
    uniform_names.push_back("MVP");
    uniform_names.push_back("ViewMat");
    uniform_names.push_back("ParticleScale");
    uniform_names.push_back("ScaleBiasProjMat");
    uniform_names.push_back("FluidParticleAlpha");
    gFluidThicknessProg.init(shader_path_types, uniform_names, "fluid thickness program");
    shader_path_types.clear();
    uniform_names.clear();

    // common
    uniform_names.push_back("MVP");
    size_t custom_uniforms_start = uniform_names.size();

    // gShadowParticleProg
    shader_path_types["../data/shaders/shadow_particle_vert.glsl"] = GL_VERTEX_SHADER;
    shader_path_types["../data/shaders/shadow_particle_frag.glsl"] = GL_FRAGMENT_SHADER;

    uniform_names.push_back("ViewMat");
    uniform_names.push_back("ParticleScale");
    uniform_names.push_back("ScaleBiasProjMat");
    gShadowParticleProg.init(shader_path_types, uniform_names, "shadow particle program");

    shader_path_types.clear();

    // gFluidSurfaceProg
    shader_path_types["../data/shaders/fluid_surface_vert.glsl"] = GL_VERTEX_SHADER;
    shader_path_types["../data/shaders/fluid_surface_frag.glsl"] = GL_FRAGMENT_SHADER;
    gFluidSurfaceProg.init(shader_path_types, uniform_names, "fluid surface program");

    shader_path_types.clear();

    // gParticleProg
    shader_path_types["../data/shaders/particle_vert.glsl"] = GL_VERTEX_SHADER;
    shader_path_types["../data/shaders/particle_frag.glsl"] = GL_FRAGMENT_SHADER;
    
    uniform_names.push_back("ShadowMat");
    gParticleProg.init(shader_path_types, uniform_names, "particle program");

    shader_path_types.clear();
    uniform_names.erase(uniform_names.begin() + custom_uniforms_start, uniform_names.end());

    // gShadowMeshProg
    shader_path_types["../data/shaders/shadow_mesh_vert.glsl"] = GL_VERTEX_SHADER;
    shader_path_types["../data/shaders/shadow_mesh_frag.glsl"] = GL_FRAGMENT_SHADER;
    gShadowMeshProg.init(shader_path_types, uniform_names, "shadow mesh program");
    shader_path_types.clear();

    // gMeshProg
    shader_path_types["../data/shaders/mesh_vert.glsl"] = GL_VERTEX_SHADER;
    shader_path_types["../data/shaders/mesh_frag.glsl"] = GL_FRAGMENT_SHADER;
    uniform_names.push_back("ViewMat");
    uniform_names.push_back("ShadowMat");
    gMeshProg.init(shader_path_types, uniform_names, "mesh program");

    shader_path_types.clear();
    uniform_names.erase(uniform_names.begin() + custom_uniforms_start, uniform_names.end());

    //gShadowPlaneProg
    shader_path_types["../data/shaders/plane_vert.glsl"] = GL_VERTEX_SHADER;
    shader_path_types["../data/shaders/plane_frag.glsl"] = GL_FRAGMENT_SHADER;
    gShadowPlaneProg.init(shader_path_types, uniform_names, "shadow plane program");

    shader_path_types.clear();
    // gPlaneProg
    shader_path_types["../data/shaders/plane_vert.glsl"] = GL_VERTEX_SHADER;
    shader_path_types["../data/shaders/plane_frag.glsl"] = GL_FRAGMENT_SHADER;

    uniform_names.push_back("ViewMat");
    uniform_names.push_back("ShadowMat");
    gPlaneProg.init(shader_path_types, uniform_names, "plane program");

    shader_path_types.clear();
    uniform_names.erase(uniform_names.begin() + custom_uniforms_start, uniform_names.end());
}

void reloadShaders()
{
    gPlaneProg.refresh();
    gMeshProg.refresh();
    gParticleProg.refresh();
    gShadowMeshProg.refresh();
    gShadowParticleProg.refresh();
    gFluidThicknessProg.refresh();
    gFluidSurfaceProg.refresh();
    gScreenRectProg.refresh();
}

void
Scene::setupSamplers()
{
    glGenSamplers(3, tex_samplers_);
    // general 
    glSamplerParameteri(tex_samplers_[0], GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glSamplerParameteri(tex_samplers_[0], GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glSamplerParameteri(tex_samplers_[0], GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glSamplerParameteri(tex_samplers_[0], GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    // mipmap
    glSamplerParameteri(tex_samplers_[1], GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glSamplerParameteri(tex_samplers_[1], GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glSamplerParameteri(tex_samplers_[1], GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glSamplerParameteri(tex_samplers_[1], GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    // mipmap repeat
    glSamplerParameteri(tex_samplers_[2], GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glSamplerParameteri(tex_samplers_[2], GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glSamplerParameteri(tex_samplers_[2], GL_TEXTURE_WRAP_S, GL_REPEAT);
    glSamplerParameteri(tex_samplers_[2], GL_TEXTURE_WRAP_T, GL_REPEAT);
}

void
Scene::resizeFramebuffers(int _width, int _height)
{
    if (_width <= 0 || _height <= 0) return;
    // 1: geometry framebuffer (G-Buffer); tex_samplers_[0]
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffers_[1]);

    glBindTexture(GL_TEXTURE_2D, fbo_textures_[2]); // view_pos tex
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB32F,
        _width, _height,
        0,
        GL_RGB, GL_FLOAT,
        NULL
    );

    glBindTexture(GL_TEXTURE_2D, fbo_textures_[3]); // view_normal tex
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB32F,
        _width, _height,
        0,
        GL_RGB, GL_FLOAT,
        NULL
    );

    glBindTexture(GL_TEXTURE_2D, fbo_textures_[4]); // color tex
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA32F,
        _width, _height,
        0,
        GL_RGBA, GL_FLOAT,
        NULL
    );

    glBindTexture(GL_TEXTURE_2D, fbo_textures_[5]); // shadow coord, shadow depth
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB32F,
        _width, _height,
        0,
        GL_RGB, GL_FLOAT,
        NULL
    );

    glBindTexture(GL_TEXTURE_2D, fbo_textures_[6]); // geometry fbo depth
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_DEPTH_COMPONENT32F,
        _width, _height,
        0,
        GL_DEPTH_COMPONENT, GL_FLOAT,
        NULL
    );

    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, fbo_textures_[2], 0); // pos
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, fbo_textures_[3], 0); // normal
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, fbo_textures_[4], 0); // color
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, fbo_textures_[5], 0); // shadow coord, shadow depth
    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,  fbo_textures_[6], 0); // depth
    checkFBO("geometry_fbo");

    // 2: picking framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffers_[2]);
    glBindTexture(GL_TEXTURE_2D, fbo_textures_[7]); // picking_id tex
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_R32I,
        _width, _height,
        0,
        GL_RED_INTEGER, GL_INT,
        NULL
    );
    glBindTexture(GL_TEXTURE_2D, fbo_textures_[8]); // picking fbo depth
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_DEPTH_COMPONENT32F,
        _width, _height,
        0,
        GL_DEPTH_COMPONENT, GL_FLOAT,
        NULL
    );

    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, fbo_textures_[7], 0); // picking_id 
    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,  fbo_textures_[8], 0); // depth
    checkFBO("picking_fbo");

    // 3: fluid framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffers_[3]);
    glBindTexture(GL_TEXTURE_2D, fluid_textures_[0]); // fluid thickness tex
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_R32F,
        _width, _height,
        0,
        GL_RED, GL_FLOAT,
        NULL
    );

    glBindTexture(GL_TEXTURE_2D, fluid_textures_[1]); // fluid position tex
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA32F,
        _width, _height,
        0,
        GL_RGBA, GL_FLOAT,
        NULL
    );

    glBindTexture(GL_TEXTURE_2D, fluid_textures_[2]); // fluid normal tex
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB32F,
        _width, _height,
        0,
        GL_RGB, GL_FLOAT,
        NULL
    );

    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, fluid_textures_[0], 0); // fluid thickness
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, fluid_textures_[1], 0); // fluid position
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, fluid_textures_[2], 0); // fluid normal
    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,  fbo_textures_[6], 0); // depth
    checkFBO("fluid_fbo");

    if (gToggleMSAA) {
        glBindFramebuffer(GL_FRAMEBUFFER, msaa_framebuffer_);
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, msaa_fbo_textures_[0]); // view_pos tex
        glTexImage2DMultisample(
            GL_TEXTURE_2D_MULTISAMPLE,
            gNumSamples,
            GL_RGB32F,
            _width, _height,
            GL_FALSE
        );

        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, msaa_fbo_textures_[1]); // view_normal tex
        glTexImage2DMultisample(
            GL_TEXTURE_2D_MULTISAMPLE,
            gNumSamples,
            GL_RGB32F,
            _width, _height,
            GL_FALSE
        );

        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, msaa_fbo_textures_[2]); // color tex
        glTexImage2DMultisample(
            GL_TEXTURE_2D_MULTISAMPLE,
            gNumSamples,
            GL_RGBA32F,
            _width, _height,
            GL_FALSE
        );

        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, msaa_fbo_textures_[3]); // shadow coord, shadow depth
        glTexImage2DMultisample(
            GL_TEXTURE_2D_MULTISAMPLE,
            gNumSamples,
            GL_RGB32F,
            _width, _height,
            GL_FALSE
        );

        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, msaa_fbo_textures_[4]); // geometry fbo depth
        glTexImage2DMultisample(
            GL_TEXTURE_2D_MULTISAMPLE,
            gNumSamples,
            GL_DEPTH_COMPONENT32F,
            _width, _height,
            GL_FALSE
        );

        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, msaa_fbo_textures_[0], 0); // pos
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, msaa_fbo_textures_[1], 0); // normal
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, msaa_fbo_textures_[2], 0); // color
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, msaa_fbo_textures_[3], 0); // shadow coord, shadow depth
        glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, msaa_fbo_textures_[4], 0); // depth
        checkFBO("msaa_fbo");
    }
}

void
Scene::pickParticle(int _x, int _y) // GL window coord
{
    if (_x < 0 || _x >= gWindowWidth || _y < 0 || _y >= gWindowHeight) {
        picked_particle_idx_ = kInvalidParticleIdx;
        return;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, framebuffers_[2]);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, picking_pixel_bo_);
    glBufferData(GL_PIXEL_PACK_BUFFER, gWindowWidth * gWindowHeight * sizeof(GLint), 0, GL_STREAM_READ);
    glReadPixels(0, 0, gWindowWidth, gWindowHeight, GL_RED_INTEGER, GL_INT, 0);
    GLint *pick_buffer_ptr = (GLint *)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
    if (pick_buffer_ptr) {
        picked_particle_idx_ = pick_buffer_ptr[_y * gWindowWidth + _x];
        glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
    }
    if (picked_particle_idx_ != kInvalidParticleIdx) {
        std::cout << "picked particle " << picked_particle_idx_ << std::endl;
    }
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void
Scene::setupFramebuffers()
{
    glGenFramebuffers(4, framebuffers_);
    glGenTextures(9, fbo_textures_);
    glGenTextures(3, fluid_textures_);
    glGenFramebuffers(1, &msaa_framebuffer_);
    glGenTextures(5, msaa_fbo_textures_);
    glGenBuffers(1, &picking_pixel_bo_);

    glGenVertexArrays(1, &screen_rect_vao_);
    glBindVertexArray(screen_rect_vao_);
    glGenBuffers(2, screen_rect_bos_);

    //0: shadow framebuffer; tex_samplers_[1]
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffers_[0]);

    glBindTexture(GL_TEXTURE_2D, fbo_textures_[0]); //shadow tex (depth, depth ^ 2)
    glTexStorage2D(
        GL_TEXTURE_2D, 
        1, //static_cast<GLsizei>(std::floor(std::log2(std::min(kShadowMapWidth, kShadowMapHeight)))),
        GL_RG32F,
        kShadowMapWidth, kShadowMapHeight
    );
    //glGenerateMipmap(GL_TEXTURE_2D);

    glBindTexture(GL_TEXTURE_2D, fbo_textures_[1]); // shadow fbo depth
    glTexStorage2D(
        GL_TEXTURE_2D, 1,
        GL_DEPTH_COMPONENT32F,
        kShadowMapWidth, kShadowMapHeight
    );

    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, fbo_textures_[0], 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, fbo_textures_[1], 0);
    checkFBO("shadow_fbo");

    // g-buffer
    std::cout << "init window size: (" << gWindowWidth << " x " << gWindowHeight << ")" << std::endl;
    resizeFramebuffers(gWindowWidth, gWindowHeight);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void 
Scene::setupScreenRectVao()
{
    glGenVertexArrays(1, &screen_rect_vao_);
    glBindVertexArray(screen_rect_vao_);
    glGenBuffers(2, screen_rect_bos_);

    glm::vec2 postions[4] = { {-1.f, -1.f}, {1.f, -1.f}, {1.f, 1.f}, {-1.f, 1.f} };
    glm::vec2 tex_uvs[4] =  { { 0.f,  0.f}, {1.f,  0.f}, {1.f, 1.f}, { 0.f, 1.f} };

    glBindBuffer(GL_ARRAY_BUFFER, screen_rect_bos_[0]);
    glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(glm::vec2), postions, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, screen_rect_bos_[1]);
    glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(glm::vec2), tex_uvs, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
    picked_particle_idx_ = kInvalidParticleIdx;
}

void
Scene::init(bool reset_params)
{
    std::cout << "releasing resources..." << std::endl;
    release();
    std::cout << "released resources..." << std::endl;

    gToggleMSAA = true;

    if (!reset_params) {
        restoreParams();
    }
    else {
        solver_.params_ = Params{};
        resetParams();
    }
    setupSamplers();
    setupFramebuffers();
    setupScreenRectVao();

    glGenVertexArrays(1, &particle_vao_);
    glBindVertexArray(particle_vao_);
    glGenBuffers(2, particle_bos_);

    glBindBuffer(GL_ARRAY_BUFFER, particle_bos_[0]); // position
    glBufferData(GL_ARRAY_BUFFER, max_num_particles_ * sizeof(float4), NULL, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, particle_bos_[1]); // phase
    glBufferData(GL_ARRAY_BUFFER, max_num_particles_ * sizeof(int), NULL, GL_STATIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribIPointer(1, 1, GL_INT, GL_FALSE, 0);

    solver_.init(max_num_particles_, min_particle_radius_, 2.f * min_rigid_particle_radius_);

    registerGLBufferObject(particle_bos_[0], &dParticleBOsRes_[0]); // position
    registerGLBufferObject(particle_bos_[1], &dParticleBOsRes_[1]); // phase
    
    solver_.dParticlePositions_ = (float4*)mapGLBufferObject(&dParticleBOsRes_[0]);
    solver_.dParticlePhases_ = (int*)mapGLBufferObject(&dParticleBOsRes_[1]);
    
    std::cout << "scene initializing..." << std::endl;
    setup();
    solver_.postInit();

    unmapGLBufferObject(dParticleBOsRes_[0]);
    unmapGLBufferObject(dParticleBOsRes_[1]);
    deviceSync();
    std::cout << "scene initialized..." << std::endl;
}

void
Scene::restoreParams()
{
    saved_params_.exec_time_ = 0.f;
    saved_params_.num_frames_ = 0;
    saved_params_.num_particles_ = 0;
    saved_params_.num_planes_ = 0;
    saved_params_.num_rigids_ = 0;
    saved_params_.num_rigid_particles_ = 0;
    saved_params_.rigid_particles_begin_ = -1;
    saved_params_.num_granular_particles_ = 0;
    saved_params_.granular_particles_begin_ = -1;
    saved_params_.num_fluid_particles_ = 0;
    saved_params_.fluid_particles_begin_ = -1;
    solver_.params_ = saved_params_;
}

void
Scene::release()
{
    saved_params_ = solver_.params_;
    solver_.release();
    if (solver_.params_.num_planes_ > 0) {
        RELEASE_GL_BOs(plane_bos_, 4);
        RELEASE_GL_VAO(plane_vao_);
    }
    if (num_skin_vertices_ > 0) {
        for (int i = 0; i < 3; ++i) {
            UNREGISTER_GL_BO(dSkinBOsRes_[i]);
        }
        RELEASE_GL_BOs(skin_bos_, 4);
        RELEASE_GL_VAO(skin_vao_);
        FREE_CUDA_ARRAY(dSkinParticleIndices_);
        FREE_CUDA_ARRAY(dSkinParticleWeights_);
    }

    if (solver_.params_.num_particles_ > 0) {
        UNREGISTER_GL_BO(dParticleBOsRes_[0]);
        UNREGISTER_GL_BO(dParticleBOsRes_[1]);
        RELEASE_GL_BOs(particle_bos_, 2);
        RELEASE_GL_VAO(particle_vao_);
    }

    RELEASE_GL_VAO(screen_rect_vao_);
    RELEASE_GL_BOs(screen_rect_bos_, 2);

    RELEASE_GL_FBO(msaa_framebuffer_);
    RELEASE_GL_TEXs(msaa_fbo_textures_, 5);
    RELEASE_GL_FBOs(framebuffers_, 4);
    RELEASE_GL_TEXs(fbo_textures_, 9);
    RELEASE_GL_TEXs(fluid_textures_, 3);
    RELEASE_GL_BO(picking_pixel_bo_);
    RELEASE_GL_Samplers(tex_samplers_, 3);
    
    glFinish();
    deviceSync();
}


void
Scene::setupPlanes(std::vector<float4> const &_planes, std::vector<float4> &_rendered_planes)
{
    if (_rendered_planes.size() == 0) _rendered_planes = _planes;
    if (_planes.size() > 0) {
        solver_.setupPlanes(_planes);
    }

    size_t num_planes = _rendered_planes.size();
    if (num_planes == 0) {
        num_plane_indices_ = 0;
        return;
    }

    // plane mesh
    GLsizei const positions_size = PlaneMesh::getSizePositions();
    GLsizei const normals_size = PlaneMesh::getSizeNormals();
    GLsizei const tex_uvs_size = PlaneMesh::getSizeTexUVs();
    GLsizei const indices_size = PlaneMesh::getSizeIndices();
    GLsizei const num_vert = PlaneMesh::getNumVert();
    GLsizei const num_indices = PlaneMesh::getNumIndices();
    num_plane_indices_ = (int)num_planes * num_indices;

    std::vector<glm::vec3> positions;
    std::vector<glm::vec2> tex_uvs;
    std::vector<glm::vec3> normals;
    std::vector<GLuint> indices;

    for (float4 const &plane : _rendered_planes) {
        PlaneMesh plane_mesh{ make_vec4(plane), (float)kCheckerTexSize };
        for (GLuint &idx : plane_mesh.indices_) {
            idx += static_cast<GLuint>(positions.size());
        }
        positions.insert(positions.end(), plane_mesh.positions_.begin(), plane_mesh.positions_.end());
        tex_uvs.insert(tex_uvs.end(), plane_mesh.tex_uvs_.begin(), plane_mesh.tex_uvs_.end());
        normals.insert(normals.end(), plane_mesh.normals_.begin(), plane_mesh.normals_.end());
        indices.insert(indices.end(), plane_mesh.indices_.begin(), plane_mesh.indices_.end());
    }

    glGenVertexArrays(1, &plane_vao_);
    glBindVertexArray(plane_vao_);
    glGenBuffers(4, plane_bos_);

    glBindBuffer(GL_ARRAY_BUFFER, plane_bos_[0]); // positions
    glBufferData(GL_ARRAY_BUFFER, num_planes * positions_size, positions.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, PlaneMesh::getAttrSizePosition(), GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, plane_bos_[1]); // tex_uvs
    glBufferData(GL_ARRAY_BUFFER, num_planes * tex_uvs_size, tex_uvs.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, PlaneMesh::getAttrSizeTexUV(), GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, plane_bos_[2]); // normals
    glBufferData(GL_ARRAY_BUFFER, num_planes * normals_size, normals.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, PlaneMesh::getAttrSizeNormal(), GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, plane_bos_[3]); // indices
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, num_planes * indices_size, indices.data(), GL_STATIC_DRAW);

    checkGLError("Scene:setupPlanes");
}

// can only add non-rigid particles
// when adding granualr particles, right after the granular particles created from shape if any, before particles of other type 
// similar for fluid particles
void
Scene::addParticles(
    std::vector<float4> &_particle_positions,
    std::vector<float4> &_particle_velocities,
    std::vector<float> &_particle_inv_masses,
    int _particle_phase
)
{
    int offset = solver_.params_.num_particles_;
    int delta_num = static_cast<int>(_particle_positions.size());

    std::vector<int> particle_phases;
    particle_phases.resize(delta_num, _particle_phase);

    copyArrayToDevice(
        solver_.dParticlePositions_, 
        _particle_positions.data(),
        offset * sizeof(float4), 
        delta_num * static_cast<int>(sizeof(float4))
    );

    copyArrayToDevice(
        solver_.dParticlePositionsInit_,
        _particle_positions.data(),
        offset * static_cast<int>(sizeof(float4)),
        delta_num * static_cast<int>(sizeof(float4))
    );

    copyArrayToDevice(
        solver_.dParticleVelocities_,
        _particle_velocities.data(),
        offset * sizeof(float4),
        delta_num * static_cast<int>(sizeof(float4))
    );

    copyArrayToDevice(
        solver_.dParticleInvMasses_,
        _particle_inv_masses.data(),
        offset * sizeof(float),
        delta_num * static_cast<int>(sizeof(float))
    );

    copyArrayToDevice(
        solver_.dParticlePhases_,
        particle_phases.data(),
        offset * sizeof(int),
        delta_num * static_cast<int>(sizeof(int))
    );

    // fluid
    if (_particle_phase & kPhaseFluidFlag) {
        if (solver_.params_.num_fluid_particles_ == 0) {
            solver_.params_.fluid_particles_begin_ = offset;
        }
        solver_.params_.num_fluid_particles_ += delta_num;
        solver_.params_.num_particles_ += delta_num;
    }
    else {
        // granular particles
        if (_particle_phase & kPhaseSelfCollideFlag) {
            if (solver_.params_.num_granular_particles_ == 0) {
                solver_.params_.granular_particles_begin_ = offset;
            }
            solver_.params_.num_granular_particles_ += delta_num;
            solver_.params_.num_particles_ += delta_num;
        }
    }

}


void 
Scene::setupShapeParticles(std::vector<Mesh> &_meshes, std::vector<ParticlizeConfig> &_config)
{
    std::vector<glm::vec3> skin_positions;
    std::vector<glm::vec3> skin_colors;
    std::vector<glm::vec3> skin_normals;
    std::vector<GLuint> skin_indices;
    std::vector<VoxelizationParam> skin_voxel_params;
    std::vector<ParticlizeConfig> skin_particle_configs;

    std::vector<glm::vec3> grain_positions;
    std::vector<GLuint> grain_indices;
    std::vector<VoxelizationParam> grain_voxel_params;
    std::vector<ParticlizeConfig> grain_particle_configs;

    for (size_t i = 0; i < _meshes.size(); ++i) {
        VoxelizationParam voxel_param;
        voxel_param.num_triangles_ = static_cast<uint>(_meshes[i].indices_.size() / 3);
        voxel_param.volume_dim_ = _config[i].volume_dim;
        voxel_param.lower_ = make_float3(_meshes[i].lower_);
        voxel_param.upper_ = make_float3(_meshes[i].upper_);
        voxel_param.num_vert_ = static_cast<uint>(_meshes[i].positions_.size());

        if (_config[i].skinning_) {
            skin_particle_configs.push_back(_config[i]);
            voxel_param.vert_offset_ = static_cast<uint>(skin_positions.size());
            voxel_param.index_offset_ = static_cast<uint>(skin_indices.size());
            voxel_param.extension_ = 0.01f;
            
            GLuint indices_offset = static_cast<GLuint>(skin_positions.size());
            skin_positions.insert(skin_positions.end(), _meshes[i].positions_.begin(), _meshes[i].positions_.end());
            skin_normals.insert(skin_normals.end(), _meshes[i].normals_.begin(), _meshes[i].normals_.end());
            for (GLuint &idx : _meshes[i].indices_) {
                idx += indices_offset;
            }
            skin_indices.insert(skin_indices.end(), _meshes[i].indices_.begin(), _meshes[i].indices_.end());
            skin_colors.insert(skin_colors.end(), _meshes[i].colors_.begin(), _meshes[i].colors_.end());
            skin_voxel_params.push_back(voxel_param);
        }
        else {
            grain_particle_configs.push_back(_config[i]);
            voxel_param.vert_offset_ = static_cast<uint>(grain_positions.size());
            voxel_param.index_offset_ = static_cast<uint>(grain_indices.size());
            voxel_param.extension_ = -0.0001f;

            GLuint indices_offset = static_cast<GLuint>(grain_positions.size());
            grain_positions.insert(grain_positions.end(), _meshes[i].positions_.begin(), _meshes[i].positions_.end());
            for (GLuint &idx : _meshes[i].indices_) {
                idx += indices_offset;
            }
            grain_indices.insert(grain_indices.end(), _meshes[i].indices_.begin(), _meshes[i].indices_.end());
            grain_voxel_params.push_back(voxel_param);
        }
    }

    num_skin_vertices_ = static_cast<int>(skin_positions.size());
    num_skin_indices_ = static_cast<int>(skin_indices.size());

    // all rigid bodies go here
    if (num_skin_vertices_ > 0) {
        glGenVertexArrays(1, &skin_vao_);
        glBindVertexArray(skin_vao_);

        glGenBuffers(4, skin_bos_);

        glBindBuffer(GL_ARRAY_BUFFER, skin_bos_[0]); // positions
        glBufferData(GL_ARRAY_BUFFER, skin_positions.size() * sizeof(glm::vec3), skin_positions.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, Mesh::getAttrSizePosition(), GL_FLOAT, GL_FALSE, 0, 0);

        glBindBuffer(GL_ARRAY_BUFFER, skin_bos_[1]); // colors
        glBufferData(GL_ARRAY_BUFFER, skin_colors.size() * sizeof(glm::vec3), skin_colors.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, Mesh::getAttrSizeColor(), GL_FLOAT, GL_FALSE, 0, 0);

        glBindBuffer(GL_ARRAY_BUFFER, skin_bos_[2]); // normals
        glBufferData(GL_ARRAY_BUFFER, skin_normals.size() * sizeof(glm::vec3), skin_normals.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, Mesh::getAttrSizeNormal(), GL_FLOAT, GL_FALSE, 0, 0);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, skin_bos_[3]); // indices
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, skin_indices.size() * sizeof(GLuint), skin_indices.data(), GL_STATIC_DRAW);

        registerGLBufferObject(skin_bos_[0], &dSkinBOsRes_[0]); // position
        registerGLBufferObject(skin_bos_[2], &dSkinBOsRes_[1], cudaGraphicsRegisterFlagsWriteDiscard); // normal
        registerGLBufferObject(skin_bos_[3], &dSkinBOsRes_[2], cudaGraphicsRegisterFlagsReadOnly); // index

        dSkinPositions_ = (float3*)mapGLBufferObject(&dSkinBOsRes_[0]);
        dSkinIndices_ = (uint*)mapGLBufferObject(&dSkinBOsRes_[2]);
        allocateArray((void**)&dSkinPositionsInit_, skin_positions.size() * sizeof(float3));
        dev_copy_float3(dSkinPositions_, dSkinPositions_ + num_skin_vertices_, dSkinPositionsInit_);
        //thrust::copy(thrust::device, thrust::device_ptr<float3>(dSkinPositions_), thrust::device_ptr<float3>(dSkinPositions_ + num_skin_vertices_), thrust::device_ptr<float3>(dSkinPositionsInit_));

        allocateArray((void**)&dSkinParticleIndices_, skin_positions.size() * sizeof(int4));
        allocateArray((void**)&dSkinParticleWeights_, skin_positions.size() * sizeof(float4));

        float4 *dRigidSdfs;
        allocateArray((void**)&dRigidSdfs, max_num_particles_ * sizeof(float4));

        std::vector<int> rigid_particle_offsets;
        std::vector<float> rigid_stiffness;

        solver_.params_.rigid_particles_begin_ = solver_.params_.num_particles_;
        for (int i = 0; i < skin_particle_configs.size(); ++i) {
            rigid_particle_offsets.push_back(solver_.params_.num_particles_);
            rigid_stiffness.push_back(skin_particle_configs[i].stiffness_);

            Volume_CUDA vol{ dSkinPositions_, dSkinIndices_, skin_voxel_params[i] };
            SDF_CUDA sdf_cuda{ vol.volume_dim_, vol.proj_axis_, vol.dVoxels_ };
            sdf_cuda.genSdfPyramid(skin_particle_configs[i].sdf_merge_threshold);
            sdf_cuda.mergeSdfPyramid();
            skin_particle_configs[i].L0_radius_ = vol.real_voxel_size_.x * 0.5f;

            createParticles(
                skin_particle_configs[i],
                vol.real_voxel_lower_,
                i,
                solver_.params_.num_particles_,
                solver_.params_.num_rigid_particles_,
                sdf_cuda.num_particles_,
                sdf_cuda.dVoxelPostions_,
                sdf_cuda.dSdfs_,
                solver_.dParticlePositions_,
                solver_.dParticleInvMasses_,
                solver_.dParticleVelocities_,
                solver_.dParticlePhases_,
                dRigidSdfs); // dRigidSdfs should be of the same size as dParticles, pre-allocated

            calcSkinWeights(
                skin_voxel_params[i].vert_offset_,
                skin_voxel_params[i].vert_offset_ + skin_voxel_params[i].num_vert_,
                solver_.params_.num_particles_,
                solver_.params_.num_particles_ + sdf_cuda.num_particles_, // exclusive
                make_int3(vol.real_vol_dim_) + 4,
                vol.real_voxel_size_,
                vol.real_voxel_lower_ - vol.real_voxel_size_,
                dSkinPositions_,
                solver_.dParticlePositions_,
                dSkinParticleIndices_,
                dSkinParticleWeights_
            );

            solver_.params_.num_particles_ += sdf_cuda.num_particles_;
            solver_.params_.num_rigid_particles_ += sdf_cuda.num_particles_;
            ++solver_.params_.num_rigids_;
        }
        

        rigid_particle_offsets.push_back(solver_.params_.num_particles_);
        int num_rigid_particles = solver_.params_.num_rigid_particles_;

        //std::cout << readbackVecNArray<float4, 4>(num_rigid_particles, dRigidSdfs, "rigid sdf original") << std::endl;

        allocateArray((void**)&solver_.dRigidSdfs_, num_rigid_particles * sizeof(float4));
        //deviceSync();
        //std::cout << "prepare thrust copy..." << std::endl;

        dev_copy_float4(dRigidSdfs, dRigidSdfs + num_rigid_particles, solver_.dRigidSdfs_);
        //thrust::copy(thrust::device, thrust::device_ptr<float4>(dRigidSdfs), thrust::device_ptr<float4>(dRigidSdfs + num_rigid_particles), thrust::device_ptr<float4>(solver_.dRigidSdfs_));
        FREE_CUDA_ARRAY(dRigidSdfs);

        createArray((void**)&solver_.dRigidParticleOffsets_, rigid_particle_offsets.data(), 0, rigid_particle_offsets.size() * sizeof(int));
        createArray((void**)&solver_.dRigidStiffnesses_, rigid_stiffness.data(), 0, rigid_stiffness.size() * sizeof(float));
        //deviceSync();

        unmapGLBufferObject(dSkinBOsRes_[2]); 
        unmapGLBufferObject(dSkinBOsRes_[0]);

        deviceSync();
        std::cout << "rigid particle created..." << std::endl;
    }

    if (grain_particle_configs.size() > 0) {
        std::cout << "creating grain particles..." << std::endl;
        solver_.params_.granular_particles_begin_ = solver_.params_.num_particles_;
        float3 *dGrainPositions;
        uint *dGrainIndices;
        createArray((void**)&dGrainPositions, grain_positions.data(), 0, grain_positions.size() * sizeof(float3));
        createArray((void**)&dGrainIndices, grain_indices.data(), 0, grain_indices.size() * sizeof(uint));
        for (int i = 0; i < grain_particle_configs.size(); ++i) {
            Volume_CUDA vol{ dGrainPositions, dGrainIndices, grain_voxel_params[i] };
            SDF_CUDA sdf_cuda{ vol.volume_dim_, vol.proj_axis_, vol.dVoxels_ };
            sdf_cuda.genSdfPyramid(grain_particle_configs[i].sdf_merge_threshold);
            sdf_cuda.mergeSdfPyramid();
            createParticles(
                grain_particle_configs[i],
                vol.real_voxel_lower_,
                i,
                solver_.params_.num_particles_,
                solver_.params_.num_rigid_particles_,
                sdf_cuda.num_particles_,
                sdf_cuda.dVoxelPostions_,
                sdf_cuda.dSdfs_,
                solver_.dParticlePositions_,
                solver_.dParticleInvMasses_,
                solver_.dParticleVelocities_,
                solver_.dParticlePhases_,
                solver_.dRigidSdfs_); // dRigidSdfs_ should be of the same size as dParticles, pre-allocated
            solver_.params_.num_particles_ += sdf_cuda.num_particles_;
            solver_.params_.num_granular_particles_ += sdf_cuda.num_particles_;
        }
        FREE_CUDA_ARRAY(dGrainPositions);
        FREE_CUDA_ARRAY(dGrainIndices);
        deviceSync();
        std::cout << "grain particle created..." << std::endl;
    }
}

void 
Scene::skinMesh()
{
    if (num_skin_vertices_ > 0) {
        dSkinPositions_ = (float3*)mapGLBufferObject(&dSkinBOsRes_[0]);
        dSkinNormals_ = (float3*)mapGLBufferObject(&dSkinBOsRes_[1]);
        dSkinIndices_ = (uint*)mapGLBufferObject(&dSkinBOsRes_[2]);

        skinning(
            num_skin_vertices_,
            num_skin_indices_ / 3,
            solver_.dParticlePositionsInit_,
            solver_.dParticlePositions_,
            solver_.dParticlePhases_,
            solver_.dRigidRotations_,
            dSkinParticleIndices_,
            dSkinParticleWeights_,
            dSkinPositionsInit_,
            dSkinIndices_,
            dSkinPositions_,
            dSkinNormals_
        );

        unmapGLBufferObject(dSkinBOsRes_[2]);
        unmapGLBufferObject(dSkinBOsRes_[1]);
        unmapGLBufferObject(dSkinBOsRes_[0]);
    }
}

void
Scene::update()
{
    if (!gPause) {
        solver_.dParticlePositions_ = (float4*)mapGLBufferObject(&dParticleBOsRes_[0]);
        solver_.dParticlePhases_ = (int*)mapGLBufferObject(&dParticleBOsRes_[1]);

        modify();
        solver_.update();
        skinMesh();

        unmapGLBufferObject(dParticleBOsRes_[0]);
        unmapGLBufferObject(dParticleBOsRes_[1]);

        if (gToggleStepMode) gPause = !gPause;
    }
}

void
Scene::drawShadowPlanes()
{
    glBindVertexArray(plane_vao_);
    gShadowPlaneProg.use();
    glUniformMatrix4fv(gShadowPlaneProg["MVP"], 1, GL_FALSE, &gLightViewProjMat[0][0]);

    glDrawElements(GL_TRIANGLES, num_plane_indices_, GL_UNSIGNED_INT, 0);
}

void
Scene::drawShadowSkin()
{
    glBindVertexArray(skin_vao_);
    gShadowMeshProg.use();
    glUniformMatrix4fv(gShadowMeshProg["MVP"], 1, GL_FALSE, &gLightViewProjMat[0][0]);

    glDrawElements(GL_TRIANGLES, num_skin_indices_, GL_UNSIGNED_INT, 0);
    checkGLError("scene:draw shadow skin elments");
}

void
Scene::drawShadowParticles(int _particle_offset, int _num_particles)
{
    glm::mat4 const & view_mat = gCamera.getViewMat();
    glm::mat4 const & scale_bias_proj = kScaleBiasMat * gProjMat;
    float particle_scale = 0.5f * static_cast<float>(kShadowMapHeight) / std::tan(gLightFov * 0.5f);

    glBindVertexArray(particle_vao_);
    gShadowParticleProg.use();
    glEnable(GL_PROGRAM_POINT_SIZE);
    glUniformMatrix4fv(gShadowParticleProg["MVP"], 1, GL_FALSE, &gLightViewProjMat[0][0]);
    glUniformMatrix4fv(gShadowParticleProg["ViewMat"], 1, GL_FALSE, &gLightViewMat[0][0]);
    glUniform1f(gShadowParticleProg["ParticleScale"], particle_scale);
    glUniformMatrix4fv(gShadowParticleProg["ScaleBiasProjMat"], 1, GL_FALSE, &scale_bias_proj[0][0]);

    glDrawArrays(GL_POINTS, _particle_offset, _num_particles);
    glDisable(GL_PROGRAM_POINT_SIZE);
}


void
Scene::drawPlanes()
{
    glm::mat4 const & view_mat = gCamera.getViewMat();
    glBindVertexArray(plane_vao_);
    gPlaneProg.use();

    glUniformMatrix4fv(gPlaneProg["MVP"], 1, GL_FALSE, &gViewProjMat[0][0]);
    glUniformMatrix4fv(gPlaneProg["ViewMat"], 1, GL_FALSE, &view_mat[0][0]);
    glUniformMatrix4fv(gPlaneProg["ShadowMat"], 1, GL_FALSE, &gShadowMat[0][0]);
    
    glDrawElements(GL_TRIANGLES, num_plane_indices_, GL_UNSIGNED_INT, 0);
}

void
Scene::drawSkin()
{
    //glDisable(GL_CULL_FACE);
    glm::mat4 const & view_mat = gCamera.getViewMat();

    glBindVertexArray(skin_vao_);
    gMeshProg.use();
    
    glUniformMatrix4fv(gMeshProg["MVP"], 1, GL_FALSE, &gViewProjMat[0][0]);
    glUniformMatrix4fv(gMeshProg["ViewMat"], 1, GL_FALSE, &view_mat[0][0]);
    glUniformMatrix4fv(gMeshProg["ShadowMat"], 1, GL_FALSE, &gShadowMat[0][0]);

    glDrawElements(GL_TRIANGLES, num_skin_indices_, GL_UNSIGNED_INT, 0);
    //glEnable(GL_CULL_FACE);
    checkGLError("scene:draw skin elments");
}


void
Scene::drawParticles(int _particle_offset, int _num_particles)
{
    glm::mat4 const & view_mat = gCamera.getViewMat();
    glm::mat4 const & scale_bias_proj = kScaleBiasMat * gProjMat;
    float particle_scale = 0.5f * static_cast<float>(gWindowHeight) / std::tan(kFovY * 0.5f);

    glBindVertexArray(particle_vao_);
    gParticleProg.use();
    glEnable(GL_PROGRAM_POINT_SIZE);
    glUniformMatrix4fv(gParticleProg["MVP"], 1, GL_FALSE, &gViewProjMat[0][0]);
    glUniformMatrix4fv(gParticleProg["ViewMat"], 1, GL_FALSE, &view_mat[0][0]);
    glUniform1f(gParticleProg["ParticleScale"], particle_scale);
    glUniformMatrix4fv(gParticleProg["ScaleBiasProjMat"], 1, GL_FALSE, &scale_bias_proj[0][0]);
    glUniformMatrix4fv(gParticleProg["ShadowMat"], 1, GL_FALSE, &gShadowMat[0][0]);

    glDrawArrays(GL_POINTS, _particle_offset, _num_particles);
    glDisable(GL_PROGRAM_POINT_SIZE);
}

void
Scene::drawFluid()
{
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffers_[3]);

    // default: pos.w = 0.f, not fluid surface
    const GLenum fluid_surface_draw_buffers[2] = { GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
    glDrawBuffers(2, fluid_surface_draw_buffers);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    if (gToggleDrawFluid && solver_.params_.num_fluid_particles_ > 0) {
        glEnable(GL_BLEND);
        glDepthMask(GL_FALSE);
        //glDisable(GL_DEPTH_TEST);
        glBlendFunc(GL_ONE, GL_ONE);
        glm::mat4 const & view_mat = gCamera.getViewMat();
        glm::mat4 const & scale_bias_proj = kScaleBiasMat * gProjMat;
        float particle_scale = 0.5f * static_cast<float>(gWindowHeight) / std::tan(kFovY * 0.5f);
        
        glBindVertexArray(particle_vao_);
        gFluidThicknessProg.use();
        glEnable(GL_PROGRAM_POINT_SIZE);
        glUniformMatrix4fv(gFluidThicknessProg["MVP"], 1, GL_FALSE, &gViewProjMat[0][0]);
        glUniformMatrix4fv(gFluidThicknessProg["ViewMat"], 1, GL_FALSE, &view_mat[0][0]);
        glUniform1f(gFluidThicknessProg["ParticleScale"], particle_scale);
        glUniformMatrix4fv(gFluidThicknessProg["ScaleBiasProjMat"], 1, GL_FALSE, &scale_bias_proj[0][0]);
        glUniform1f(gFluidThicknessProg["FluidParticleAlpha"], gFluidParticleAlpha);
        glDrawArrays(GL_POINTS, solver_.params_.fluid_particles_begin_, solver_.params_.num_fluid_particles_);
        //glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);

        glDrawBuffers(2, fluid_surface_draw_buffers);
        gFluidSurfaceProg.use();
        glUniformMatrix4fv(gFluidSurfaceProg["MVP"], 1, GL_FALSE, &gViewProjMat[0][0]);
        glUniformMatrix4fv(gFluidSurfaceProg["ViewMat"], 1, GL_FALSE, &view_mat[0][0]);
        glUniform1f(gFluidSurfaceProg["ParticleScale"], particle_scale);
        glUniformMatrix4fv(gFluidSurfaceProg["ScaleBiasProjMat"], 1, GL_FALSE, &scale_bias_proj[0][0]);
        glDrawArrays(GL_POINTS, solver_.params_.fluid_particles_begin_, solver_.params_.num_fluid_particles_);
        
        glDisable(GL_PROGRAM_POINT_SIZE);
        checkGLError("drawFluid:finished");
    }
}


// GL_COLOR_ATTACHMENT0, fbo_textures_[0]; // shadowmap
// GL_DEPTH_ATTACHMENT , fbo_textures_[1]; // depth
void
Scene::drawShadowMap()
{
    // default: GL_BACK
    glCullFace(GL_FRONT);

    glBindFramebuffer(GL_FRAMEBUFFER, framebuffers_[0]);
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    glClearColor(1, 1, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    //glEnable(GL_POLYGON_OFFSET_FILL);
    //glPolygonOffset(2.f, 4.f);

    if (solver_.params_.num_granular_particles_ > 0) {
        drawShadowParticles(solver_.params_.granular_particles_begin_, solver_.params_.num_granular_particles_);
    }
    if (gToggleDrawParticle) {
        if (solver_.params_.num_rigid_particles_ > 0) {
            drawShadowParticles(solver_.params_.rigid_particles_begin_, solver_.params_.num_rigid_particles_);
        }
    }

    if (!gToggleDrawFluid && solver_.params_.num_fluid_particles_ > 0) {
        drawShadowParticles(solver_.params_.fluid_particles_begin_, solver_.params_.num_fluid_particles_);
    }

    if (num_skin_vertices_ > 0 && gToggleDrawMesh) {
        drawShadowSkin();
    }

    //if (solver_.params_.num_planes_ > 0) {
    //    drawShadowPlanes();
    //}
    // glDisable(GL_POLYGON_OFFSET_FILL);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glCullFace(GL_BACK);
}

// GL_COLOR_ATTACHMENT0, fbo_textures_[2]; // pos
// GL_COLOR_ATTACHMENT1, fbo_textures_[3]; // normal
// GL_COLOR_ATTACHMENT2, fbo_textures_[4]; // color
// GL_COLOR_ATTACHMENT3, fbo_textures_[5]; // shadow coord, shadow depth
// GL_DEPTH_ATTACHMENT,  fbo_textures_[6]; // depth

// GL_COLOR_ATTACHMENT4, fbo_textures_[7]; // picking_id
// GL_COLOR_ATTACHMENT4, fbo_textures_[8]; // picking depth

void
Scene::drawScene()
{

    glBindFramebuffer(GL_FRAMEBUFFER, framebuffers_[2]);
    const GLenum picking_draw_buffers[] = { GL_NONE, GL_NONE, GL_NONE, GL_NONE, GL_COLOR_ATTACHMENT0 };
    glDrawBuffers(5, picking_draw_buffers);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearBufferiv(GL_COLOR, 4, &kInvalidParticleIdx); // picking buffer, draw buffer index, not one of the GL_COLOR_ATTACHMENTi
    drawParticles(0, solver_.params_.num_particles_);
    
    if (gToggleMSAA) {
        glBindFramebuffer(GL_FRAMEBUFFER, msaa_framebuffer_);
        glEnable(GL_MULTISAMPLE);
        glEnable(GL_SAMPLE_SHADING);
    }
    else {
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffers_[1]);
    }

    const GLenum geometry_draw_buffers[] = { 
        GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1,
        GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3 
    };
    glDrawBuffers(4, geometry_draw_buffers);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (solver_.params_.num_granular_particles_ > 0) {
        drawParticles(solver_.params_.granular_particles_begin_, solver_.params_.num_granular_particles_);
    }
    if (gToggleDrawParticle) {
        if (solver_.params_.num_rigid_particles_ > 0) {
            drawParticles(solver_.params_.rigid_particles_begin_, solver_.params_.num_rigid_particles_);
        }
    }

    if (!gToggleDrawFluid && solver_.params_.num_fluid_particles_ > 0) {
        drawParticles(solver_.params_.fluid_particles_begin_, solver_.params_.num_fluid_particles_);
    }
    
    if (solver_.params_.num_planes_ > 0) {
        drawPlanes();
    }
    if (num_skin_vertices_ > 0 && gToggleDrawMesh) {
        drawSkin();
    }

    if (gToggleMSAA) {
        glDisable(GL_SAMPLE_SHADING);
        glDisable(GL_MULTISAMPLE);

        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffers_[1]);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, msaa_framebuffer_);

        for (int i = 0; i < 4; ++i) {
            glReadBuffer(GL_COLOR_ATTACHMENT0 + i);
            glDrawBuffer(GL_COLOR_ATTACHMENT0 + i);
            glBlitFramebuffer(
                0, 0, gWindowWidth, gWindowHeight,
                0, 0, gWindowWidth, gWindowHeight,
                i == 0 ? (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) : GL_COLOR_BUFFER_BIT,
                GL_NEAREST
            );
        }
    }
    drawFluid();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    checkGLError("Scene:drawScene");
}


void
Scene::drawScreenRect()
{
    glm::vec3 light_pos_view = glm::vec3(gCamera.getViewMat() * glm::vec4(gLightPos, 1.f));
    glm::vec3 light_dir_view = glm::normalize(glm::vec3(gCamera.getViewMat() * glm::vec4(gLightDir, 0.f)));
    glm::vec3 light_params = glm::vec3(gCosLightPenumbra, gCosLightUmbra, gLightExp);

    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glBindVertexArray(screen_rect_vao_);
    gScreenRectProg.use();

    glUniformMatrix4fv(gScreenRectProg["ShadowMat"], 1, GL_FALSE, &gShadowMat[0][0]);
    glUniform2fv(gScreenRectProg["ShadowSampleSize"], 1, &kShadowSampleSize[0]);
    glUniform3fv(gScreenRectProg["LightPosView"], 1, &light_pos_view[0]);
    glUniform3fv(gScreenRectProg["LightDirView"], 1, &light_dir_view[0]);
    glUniform3fv(gScreenRectProg["LightImax"], 1, &gLightImax[0]);
    glUniform3fv(gScreenRectProg["LightParams"], 1, &light_params[0]);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, fbo_textures_[0]); // shadowmap
    glBindSampler(0, tex_samplers_[0]);

    for (int i = 1; i <= 4; ++i) {
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_2D, fbo_textures_[i + 1]);// binding_unit <--> tex_idx: i <--> (i + 1)
        glBindSampler(i, tex_samplers_[0]);
    }
    checkGLError("Scene:fbo_textures_");
    for (int i = 5; i < 8; ++i) {
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_2D, fluid_textures_[i - 5]); // fluid: thickness, pos, normal
        glBindSampler(i, tex_samplers_[0]);
    }
    checkGLError("Scene:fluid_textures_");
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindSampler(0, 0);
    checkGLError("Scene:drawScreenRect");
}

void
Scene::render()
{
    gViewProjMat = gProjMat * gCamera.getViewMat();
    gShadowMat = kScaleBiasMat * gLightViewProjMat * gCamera.getViewInv();

    glPolygonMode(GL_FRONT_AND_BACK, gToggleWireframe ? GL_LINE : GL_FILL);
    glViewport(0, 0, kShadowMapWidth, kShadowMapHeight);
    drawShadowMap();
    glViewport(0, 0, gWindowWidth, gWindowHeight);
    drawScene();
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    drawScreenRect();
    checkGLError("Scene:render");
}

RigidPile::RigidPile(
    std::string const &_name,
    std::string const &_mesh_path,
    float _min_rigid_particle_radius,
    uint3 _vol_dim,
    int3 _pile_dim,
    int _max_num_particles,
    float _particle_merge_threshold
) : Scene(_name)
{
    min_particle_radius_ = _min_rigid_particle_radius;
    min_rigid_particle_radius_ = _min_rigid_particle_radius;
    max_num_particles_ = _max_num_particles;
    mesh_path_ = _mesh_path;
    vol_dim_ = _vol_dim;
    pile_dim_ = _pile_dim;
    particle_merge_threshold_ = _particle_merge_threshold;

}

void
RigidPile::resetParams()
{
    solver_.params_.max_num_neighbors_per_particle_ = 32;
    solver_.params_.num_substeps_ = 1;
    solver_.params_.num_iterations_ = 6;
    solver_.params_.num_pre_stabilize_iterations_ = 2;
    solver_.params_.shock_propagation_ = 4.f;
    solver_.params_.dynamic_friction_ = 0.4f;
    solver_.params_.particle_friction_ = 0.04f;
    solver_.params_.sleeping_threshold_quad_ = 0.02f;
    solver_.params_.damping_force_factor_ = 0.16f;
    solver_.params_.particle_collision_margin_ = 0.04f;
}

void
RigidPile::setup()
{
    std::vector<float4> planes;
    std::vector<Mesh> meshes;
    std::vector<ParticlizeConfig> configs;

    planes.push_back(make_float4(0.f, 1.f, 0.f, 0.f));
    setupPlanes(planes);

    Mesh m{ mesh_path_ };
    glm::vec3 s = make_vec3(vol_dim_) * min_rigid_particle_radius_ * 2.f;
    m.scale(s);
    ParticlizeConfig cfg;

    cfg.flags_ = 0;
    cfg.skinning_ = true;
    cfg.sdf_merge_threshold = particle_merge_threshold_;
    cfg.L0_radius_ = min_rigid_particle_radius_;
    cfg.volume_dim = make_uint3(max_comp(vol_dim_)); // voxelization always inside a cube
    //cfg.flags_ |= kPhaseSelfCollideFlag;
    //cfg.skinning_ = false;
    
    for (int dz = 0; dz < pile_dim_.z; ++dz) {
        for (int dy = 0; dy < pile_dim_.y; ++dy) {
            for (int dx = 0; dx < pile_dim_.x; ++dx) {
                Mesh temp = m;
                temp.setColor(kColors[(4 * dx + 2 * dy + dz) % 7]);
                temp.translate(glm::vec3(dx, 1.1f * dy, dz) * s * 1.1f);
                meshes.push_back(temp);
                configs.push_back(cfg);
            }
        }
    }

    setupShapeParticles(meshes, configs);
}

void
RigidPile::modify()
{
    if (solver_.params_.num_frames_ != 300) return;

    std::vector<float4> positions, velocities;
    std::vector<float> inv_masses;
    float radius = min_rigid_particle_radius_ * max_comp(vol_dim_);
    positions.push_back(make_float4(2.f * pile_dim_.x * 2.f *vol_dim_.x * min_rigid_particle_radius_, 5.f * radius, 0.0f, radius));
    float v = 20.f;
    velocities.push_back(v * make_float4(-0.8f, 0.f, 0.7f, 0.f));
    inv_masses.push_back(1.f / pow(static_cast<float>(max_comp(vol_dim_)), 3.f));
    int phase = 2 | kPhaseSelfCollideFlag;

    int offset = solver_.params_.num_particles_;
    int delta_num = static_cast<int>(positions.size());

    addParticles(positions, velocities, inv_masses, phase);
    copyArrayToDevice(
        solver_.dParticlePositionsInit_,
        positions.data(),
        offset * static_cast<int>(sizeof(float4)),
        delta_num * static_cast<int>(sizeof(float4))
    );

}


TwoRigidBodies::TwoRigidBodies(
    std::string const &_name,
    std::string const &_mesh1_path,
    std::string const &_mesh2_path,
    float _min_rigid_particle_radius,
    uint3 _vol_dim1,
    uint3 _vol_dim2,
    int _max_num_particles)
    : Scene(_name)
{
    min_particle_radius_ = _min_rigid_particle_radius;
    min_rigid_particle_radius_ = _min_rigid_particle_radius;
    max_num_particles_ = _max_num_particles;
    mesh1_path_ = _mesh1_path;
    mesh2_path_ = _mesh2_path;
    vol_dim1_ = _vol_dim1;
    vol_dim2_ = _vol_dim2;
}

void
TwoRigidBodies::resetParams()
{
    solver_.params_.max_num_neighbors_per_particle_ = 32;
    solver_.params_.sleeping_threshold_quad_ = 0.02f;
}

void
TwoRigidBodies::setup()
{
    std::vector<float4> planes;
    planes.push_back(make_float4(0.f, 1.f, 0.f, 0.f));
    setupPlanes(planes);

    std::vector<Mesh> meshes; 
    std::vector<ParticlizeConfig> configs;

    Mesh m1{ mesh1_path_ };
    glm::vec3 s1 = make_vec3(vol_dim1_) * min_rigid_particle_radius_ * 2.f;
    m1.scale(s1);
    ParticlizeConfig cfg1, cfg2;

    //cfg1.flags_ |= kPhaseSelfCollideFlag;
    //cfg1.skinning_ = false;
    //cfg1.sdf_merge_threshold = -FLT_MAX;
    cfg1.L0_radius_ = min_rigid_particle_radius_;
    cfg1.volume_dim = make_uint3(max_comp(vol_dim1_)); // voxelization always inside a cube

    cfg1.flags_ = 0;
    cfg1.skinning_ = true;
    cfg1.velocity_ = { 3.f, 0.f, 0.f };
    cfg1.stiffness_ = 0.075f;

    m1.translate(glm::vec3(-5.f, 7.f, 0.f));
    m1.setColor(glm::vec4(0.9f));
    meshes.push_back(m1);
    configs.push_back(cfg1);
    
    Mesh m2{ mesh2_path_ };
    glm::vec3 s2 = make_vec3(vol_dim2_) * min_rigid_particle_radius_ * 2.f;
    m2.scale(s2);
    m2.translate(glm::vec3(5.f, 7.f, 0.f));
    m2.setColor({ 0.8f, 0.8f, 0.f });
    cfg2.stiffness_ = 1.0f;
    // cfg2.sdf_merge_threshold = -FLT_MAX;
    cfg2.L0_radius_ = min_rigid_particle_radius_;
    cfg2.volume_dim = make_uint3(max_comp(vol_dim2_));

    cfg2.flags_ = 0;
    cfg2.skinning_ = true;
    cfg2.velocity_ = { -3.f, 0.f, 0.f };

    meshes.push_back(m2);
    configs.push_back(cfg2);

    setupShapeParticles(meshes, configs);
}

GranularPile::GranularPile(
    std::string const &_name,
    std::string const &_mesh_path,
    float _min_rigid_particle_radius,
    uint3 _vol_dim,
    int _max_num_particles,
    float _particle_merge_threshold
) : Scene(_name)
{
    min_particle_radius_ = _min_rigid_particle_radius;
    min_rigid_particle_radius_ = _min_rigid_particle_radius;
    max_num_particles_ = _max_num_particles;
    mesh_path_ = _mesh_path;
    vol_dim_ = _vol_dim;
    particle_merge_threshold_ = _particle_merge_threshold;

}

void
GranularPile::resetParams()
{
    solver_.params_.max_num_neighbors_per_particle_ = 32;
    solver_.params_.num_substeps_ = 1;
    solver_.params_.num_iterations_ = 16;
    solver_.params_.num_pre_stabilize_iterations_ = 16;
    solver_.params_.shock_propagation_ = 6.f;
    solver_.params_.dynamic_friction_ = 0.5f;
    solver_.params_.static_friction_ = 0.68f;
    solver_.params_.particle_friction_ = 0.05f;
    solver_.params_.sleeping_threshold_quad_ = 0.02f;
    solver_.params_.damping_force_factor_ = 0.2f;
    solver_.params_.particle_collision_margin_ = 0.04f;
}

void
GranularPile::setup()
{
    gToggleMSAA = false;
    std::vector<float4> planes;
    std::vector<Mesh> meshes;
    std::vector<ParticlizeConfig> configs;

    planes.push_back(make_float4(0.f, 1.f, 0.f, 0.f));
    setupPlanes(planes);

    Mesh m{ mesh_path_ };
    glm::vec3 s = make_vec3(vol_dim_) * min_rigid_particle_radius_ * 2.f;
    m.scale(s);
    ParticlizeConfig cfg;

    cfg.flags_ |= kPhaseSelfCollideFlag;
    cfg.skinning_ = false;
    cfg.sdf_merge_threshold = particle_merge_threshold_;
    cfg.L0_radius_ = min_rigid_particle_radius_;
    cfg.volume_dim = make_uint3(max_comp(vol_dim_)); // voxelization always inside a cube
    meshes.push_back(m);
    configs.push_back(cfg);

    setupShapeParticles(meshes, configs);
}

void
GranularPile::modify()
{
    if (solver_.params_.num_frames_ != 300) return;
    
    std::vector<float4> positions, velocities;
    std::vector<float> inv_masses;
    positions.push_back(make_float4(10.f, 5.f, 4.5f, min_rigid_particle_radius_ * 8.f));
    velocities.push_back(make_float4(-15.f, 0.f, 0.f, 0.f));
    inv_masses.push_back(1.f / pow(8.f, 3.f));
    int phase = 2 | kPhaseSelfCollideFlag;

    addParticles(positions, velocities, inv_masses, phase);

}

BreakingDam::BreakingDam(
    std::string const &_name,
    float _min_particle_radius,
    int3 _fluid_vol_dim,
    int _max_num_particles,
    float _fluid_rest_density
) : Scene(_name)
{
    min_particle_radius_ = _min_particle_radius;
    max_num_particles_ = _max_num_particles;
    fluid_vol_dim_ = _fluid_vol_dim;
    fluid_rest_density_ = _fluid_rest_density;
}

void
BreakingDam::resetParams()
{
    solver_.params_.max_num_neighbors_per_particle_ = 48;
    solver_.params_.fluid_neighbor_search_range_ = 4.f * min_particle_radius_;
    solver_.params_.fluid_rest_density_ = fluid_rest_density_;
    solver_.params_.num_substeps_ = 2;
    solver_.params_.num_iterations_ = 2;
    solver_.params_.num_pre_stabilize_iterations_ = 2;
    //solver_.params_.shock_propagation_ = 6.f;
    solver_.params_.dynamic_friction_ = 0.02f;
    //solver_.params_.static_friction_ = 0.68f;
    solver_.params_.particle_friction_ = 0.02f;
    solver_.params_.sleeping_threshold_quad_ = 0.01f;
    solver_.params_.damping_force_factor_ = 0.025f;
    solver_.params_.particle_collision_margin_ = 0.04f;
    solver_.params_.fluid_corr_refer_q_ = 0.075f; 
    solver_.params_.fluid_corr_k_ = 0.02f;
}

void
BreakingDam::setup()
{
    gToggleMSAA = false;
    std::vector<float4> planes, rendered_planes, positions, velocities;
    std::vector<float> inv_masses;
    int phase = 4 | kPhaseSelfCollideFlag | kPhaseFluidFlag;

    float diam = 2.f * min_particle_radius_;
    float right = 60.f * diam;
    float front = 15.f * diam;

    planes.push_back(make_float4(0.f, 1.f, 0.f, 0.f));
    planes.push_back(make_float4( 1.f,  0.f,  0.f, right)); // left wall
    planes.push_back(make_float4(-1.f,  0.f,  0.f, right)); // right wall
    planes.push_back(make_float4( 0.f,  0.f,  1.f, front)); // back wall
    planes.push_back(make_float4( 0.f,  0.f, -1.f, front)); // front wall

    rendered_planes.push_back(make_float4(0.f, 1.f, 0.f, 0.f));
    rendered_planes.push_back(make_float4(0.f, 0.f, 1.f, front));
    setupPlanes(planes, rendered_planes);

    movable_wall_ = make_float4(1.f, 0.f, 0.f, right);
    idx_movable_wall_ = 1;
    wall_delta_ = 0.35f * diam;
    wall_extreme_.x = right * 1.5f; // w --> large, wall--> left
    wall_extreme_.y = right * 0.5f;

    float margin = min_particle_radius_;
    float4 lower = make_float4(-right + margin, margin, front - margin, min_particle_radius_);
    //lower = make_float4(0.f, diam, 0.f, min_particle_radius_);
    float inv_mass = 1.f / (1.f * powf(diam, 3) * fluid_rest_density_);
    float4 velocity = make_float4(0.f);

    for (int k = 0; k < fluid_vol_dim_.z; ++k) {
        for (int j = 0; j < fluid_vol_dim_.y; ++j) {
            for (int i = 0; i < fluid_vol_dim_.x; ++i) {
                float4 delta = diam * make_float4(glm::vec4{ i, j, -k, 0 });
                positions.push_back(lower + delta);
                velocities.push_back(velocity);
                inv_masses.push_back(inv_mass);
            }
        }
    }
    addParticles(positions, velocities, inv_masses, phase);
    cudaDeviceSynchronize();
    std::cout << "initialized breaking dam. num particles = " 
        << solver_.params_.num_particles_ 
        << ", num fluid particle = " << solver_.params_.num_fluid_particles_
        << std::endl;
}

void
BreakingDam::modify()
{
    if (gToggleWall) {
        movable_wall_.w += wall_delta_;
        if (movable_wall_.w > wall_extreme_.x || movable_wall_.w < wall_extreme_.y) {
            wall_delta_ *= -1.f;
            movable_wall_.w += wall_delta_;
        }
        setPlane(solver_.dPlanes_, idx_movable_wall_, movable_wall_);
    }

    if (solver_.params_.num_frames_ != 400) return;

    std::vector<float4> positions, velocities;
    std::vector<float> inv_masses;

    float scale = 4.f, base = 512.f;
    float radius = scale * min_particle_radius_;
    float mass_radius = (std::log(scale) / std::log(base) + 1.f) * min_particle_radius_;

    positions.push_back(make_float4(-30.f  * min_particle_radius_, 50.f * min_particle_radius_, 0.f, radius));
    velocities.push_back(make_float4(0.f, 0.f, 0.f, 0.f));
    float mass = powf(2.f * mass_radius, 3.f) * 0.2f * fluid_rest_density_;
    inv_masses.push_back(1.f / mass);

    positions.push_back(make_float4(0.f, 50.f * min_particle_radius_, 0.f, radius));
    velocities.push_back(make_float4(0.f, 0.f, 0.f, 0.f));
    mass = powf(2.f * mass_radius, 3.f) * fluid_rest_density_;
    inv_masses.push_back(1.f / mass);
    
    positions.push_back(make_float4(30.f  * min_particle_radius_, 50.f * min_particle_radius_, 0.f, radius));
    velocities.push_back(make_float4(0.f, 0.f, 0.f, 0.f));
    mass = powf(2.f * mass_radius, 3.f) * 4.f * fluid_rest_density_;
    inv_masses.push_back(1.f / mass);

    int phase = 2 | kPhaseSelfCollideFlag;

    addParticles(positions, velocities, inv_masses, phase);
}

RigidFluid::RigidFluid(
    std::string const &_name,
    std::string const &_mesh_path,
    float _min_rigid_particle_radius,
    float _rigid_density_scale,
    int3 _rigid_vol_dim,
    float _min_particle_radius,
    int3 _fluid_vol_dim,
    int _max_num_particles,
    float _fluid_rest_density,
    float _particle_merge_threshold
) : Scene(_name)
{
    mesh_path_ = _mesh_path;
    min_rigid_particle_radius_ = _min_rigid_particle_radius;
    rigid_vol_dim_ = _rigid_vol_dim;
    particle_merge_threshold_ = _particle_merge_threshold;
    rigid_density_scale_ = _rigid_density_scale;

    min_particle_radius_ = _min_particle_radius;
    fluid_vol_dim_ = _fluid_vol_dim;
    fluid_rest_density_ = _fluid_rest_density;

    max_num_particles_ = _max_num_particles;
}

void
RigidFluid::resetParams()
{
    solver_.params_.max_num_neighbors_per_particle_ = 48;
    solver_.params_.fluid_neighbor_search_range_ = 4.f * min_particle_radius_;
    solver_.params_.fluid_rest_density_ = fluid_rest_density_;
    solver_.params_.num_substeps_ = 2;
    solver_.params_.num_iterations_ = 2;
    solver_.params_.num_pre_stabilize_iterations_ = 2;
    //solver_.params_.shock_propagation_ = 6.f;
    solver_.params_.dynamic_friction_ = 0.02f;
    //solver_.params_.static_friction_ = 0.68f;
    solver_.params_.particle_friction_ = 0.02f;
    solver_.params_.sleeping_threshold_quad_ = 0.01f;
    solver_.params_.damping_force_factor_ = 0.025f;
    solver_.params_.particle_collision_margin_ = 0.04f;
    solver_.params_.fluid_corr_refer_q_ = 0.075f;
    solver_.params_.fluid_corr_k_ = 0.02f;
}

void
RigidFluid::setup()
{
    gToggleMSAA = false;
    std::vector<Mesh> meshes;
    std::vector<ParticlizeConfig> configs;

    float diam = 2.f * min_particle_radius_;
    float right = 60.f * diam;
    float front = 15.f * diam;

    Mesh m{ mesh_path_ };
    glm::vec3 s = make_vec3(rigid_vol_dim_) * min_rigid_particle_radius_ * 2.f;
    m.scale(s);
    m.translate({0.f, 0.1f * min_particle_radius_, -front + 2.f * diam});
    m.setColor({ 1.000f, 0.442f, 0.000f });
    ParticlizeConfig cfg;

    cfg.flags_ = 0;
    cfg.skinning_ = true;
    cfg.sdf_merge_threshold = particle_merge_threshold_;
    cfg.L0_radius_ = min_rigid_particle_radius_;
    cfg.inv_mass_ = 1.f / (rigid_density_scale_ * powf(2.f * min_rigid_particle_radius_, 3.f) * fluid_rest_density_);
    cfg.volume_dim = make_uint3(max_comp(rigid_vol_dim_)); // voxelization always inside a cube
    meshes.push_back(m);
    configs.push_back(cfg);

    setupShapeParticles(meshes, configs);

    std::vector<float4> planes, rendered_planes, positions, velocities;
    std::vector<float> inv_masses;
    int phase = 4 | kPhaseSelfCollideFlag | kPhaseFluidFlag;

    
    planes.push_back(make_float4(0.f, 1.f, 0.f, 0.f));
    planes.push_back(make_float4(1.f, 0.f, 0.f, right)); // left wall
    planes.push_back(make_float4(-1.f, 0.f, 0.f, right)); // right wall
    planes.push_back(make_float4(0.f, 0.f, 1.f, front)); // back wall
    planes.push_back(make_float4(0.f, 0.f, -1.f, front)); // front wall

    rendered_planes.push_back(make_float4(0.f, 1.f, 0.f, 0.f));
    rendered_planes.push_back(make_float4(0.f, 0.f, 1.f, front));
    setupPlanes(planes, rendered_planes);

    movable_wall_ = make_float4(1.f, 0.f, 0.f, right);
    idx_movable_wall_ = 1;
    wall_delta_ = 0.35f * diam;
    wall_extreme_.x = right * 1.5f; // w --> large, wall--> left
    wall_extreme_.y = right * 0.5f;

    float margin = min_particle_radius_;
    float4 lower = make_float4(-right + margin, margin, front - margin, min_particle_radius_);
    float inv_mass = 1.f / (1.f * powf(diam, 3) * fluid_rest_density_);
    float4 velocity = make_float4(0.f);

    for (int k = 0; k < fluid_vol_dim_.z; ++k) {
        for (int j = 0; j < fluid_vol_dim_.y; ++j) {
            for (int i = 0; i < fluid_vol_dim_.x; ++i) {
                float4 delta = diam * make_float4(glm::vec4{ i, j, -k, 0 });
                positions.push_back(lower + delta);
                velocities.push_back(velocity);
                inv_masses.push_back(inv_mass);
            }
        }
    }
    addParticles(positions, velocities, inv_masses, phase);
    cudaDeviceSynchronize();
    std::cout << "initialized rigid fluid. num particles = "
        << solver_.params_.num_particles_
        << ", num fluid particle = " << solver_.params_.num_fluid_particles_
        << std::endl;
}

void
RigidFluid::modify()
{
    if (gToggleWall) {
        movable_wall_.w += wall_delta_;
        if (movable_wall_.w > wall_extreme_.x || movable_wall_.w < wall_extreme_.y) {
            wall_delta_ *= -1.f;
            movable_wall_.w += wall_delta_;
        }
        setPlane(solver_.dPlanes_, idx_movable_wall_, movable_wall_);
    }
}