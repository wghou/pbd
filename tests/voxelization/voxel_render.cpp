#include <memory>
#include <glm/ext.hpp>
#include <cg_math.h>
#include <bitset>
#include "voxel_render.h"

extern bool gPause;
extern int gWindowWidth;
extern int gWindowHeight;

glm::vec3 gLightPos(6.f, 6.f, 6.f);

glm::uvec3 gVoxelDim(0);
std::vector<glm::uvec3> gVolPyramidDims;
glm::uvec3 gSelectedVoxel(0);
uint gProjAxis;
glm::vec3 gVoxelTranslation(0.f, 0.f, 0.f);
GLsizei gNumVolumeIndices;
std::vector<GLsizei> gNumVolPyramidIndices;
uint gNumSdfLevels, gSelectedLevel;

GLuint gPlaneTex;
const int kCheckerTexSize = 4; // physical size/length

// perspective parameters
glm::mat4 gProjMat;
const float kFovY = glm::quarter_pi<float>() / 1.0f;
float gFovX = kFovY;
const float kNearPlane = 0.01f;
const float kFarPlane = 1000.f;

// camera parameters
Camera gCamera;
const float kCameraLowSpeed = 0.1f;
const float kCameraSpeed = 2.f;
const float kCameraHighSpeed = 8.f;
const float kCameraRotationSpeed = 0.3f;
const glm::vec3 kCameraPos(1.5f, 1.5f, 5.f); // default camera position
const glm::vec3 kLookAt(-kCameraPos); // default lookAt
const glm::vec3 kUp(0.f, 1.f, 0.f); // default up, y axis

bool gToggleWireframe = false;
bool gToggleVolumeWireframe = false;
bool gToggleDrawMesh = true;
bool gToggleDrawVolume = false;

PlaneMesh gGround;
std::unique_ptr<Mesh> gMeshPtr;
std::unique_ptr<Volume_CUDA> gVolumePtr;
std::unique_ptr<SDF_CUDA> gSDFPtr;
float gMaxSDF = FLT_EPSILON;
float gMinSDF = FLT_MAX;

GLProgram gPlaneProg, gMeshProg, gVolumeProg;
GLuint gPlaneVao, gPlaneBos[4], 
       gMeshVao, gMeshBos[3], 
       gVolumeVao, gVolumeBos[5]; // 0: position; 1: normal; 2: voxel_coord; 3: index; 4: sdf
std::vector<GLuint> gVolPyramidVao;
std::vector<std::vector<GLuint>> gVolPyramidBos;

void setupVolumeVao(uint *_voxels, glm::uvec3 const &_volume_dim, glm::vec3 const &_voxel_size, glm::vec3 const &_lower, float4 *_sdf, uint _vol_dim_ext);
void setupVolPyramidVaos(glm::vec3 const &_voxel_size, glm::vec3 const &_lower);
void setupMergedVolVaos(glm::vec3 const &_voxel_size, glm::vec3 const &_lower);
bool isVoxel(uint *_voxels, glm::uvec3 const &_voxel_coord, glm::uvec3 const &_volume_dim, uint proj_axis);

void updateProjMat(int width, int height)
{
    float aspect = static_cast<float>(width) / static_cast<float>(height);
    gFovX = 2.f * glm::atan(glm::tan(kFovY / 2.f) * aspect);
    gProjMat = glm::perspectiveFov(kFovY,
        static_cast<float>(width),
        static_cast<float>(height),
        kNearPlane, kFarPlane);
}

void reloadShaders()
{
    gMeshProg.refresh();
    gPlaneProg.refresh();
    gVolumeProg.refresh();
}

void
initPrograms()
{
    std::unordered_map<std::string, GLenum> shader_path_types;
    std::vector<std::string> uniform_names;
    // common
    uniform_names.push_back("MVP");
    uniform_names.push_back("LightPos");
    uniform_names.push_back("CamPos");
    size_t custom_uniforms_start = uniform_names.size();

    //gMeshProg
    shader_path_types["../data/shaders_test/mesh_vert.glsl"] = GL_VERTEX_SHADER;
    shader_path_types["../data/shaders_test/mesh_frag.glsl"] = GL_FRAGMENT_SHADER;
    gMeshProg.init(shader_path_types, uniform_names, "mesh program");

    shader_path_types.clear();
    uniform_names.erase(uniform_names.begin() + custom_uniforms_start, uniform_names.end());

    // gPlaneProg
    shader_path_types["../data/shaders_test/plane_vert.glsl"] = GL_VERTEX_SHADER;
    shader_path_types["../data/shaders_test/plane_frag.glsl"] = GL_FRAGMENT_SHADER;

    uniform_names.push_back("checker_tex");
    gPlaneProg.init(shader_path_types, uniform_names, "plane program");

    shader_path_types.clear();
    uniform_names.erase(uniform_names.begin() + custom_uniforms_start, uniform_names.end());

    //gVolumeProg
    shader_path_types["../data/shaders_test/volume_vert.glsl"] = GL_VERTEX_SHADER;
    shader_path_types["../data/shaders_test/volume_frag.glsl"] = GL_FRAGMENT_SHADER;

    uniform_names.push_back("SelectedVoxel");
    uniform_names.push_back("VoxelTranslation");
    uniform_names.push_back("SDFMax");
    uniform_names.push_back("SDFMin");
    gVolumeProg.init(shader_path_types, uniform_names, "volume program");

    shader_path_types.clear();
    uniform_names.erase(uniform_names.begin() + custom_uniforms_start, uniform_names.end());

}

void setupTexture()
{

    const int TILE_DIM = 32;
    const int TEX_SIZE = kCheckerTexSize;
    const int TEX_DIM = TILE_DIM * TEX_SIZE;
    const GLubyte checker_color[] = { 0x7F, 0xFF };
    GLubyte tex_checkerboard_data[TEX_DIM * TEX_DIM];
    for (int i = 0; i < TEX_DIM; ++i) {
        for (int j = 0; j < TEX_DIM; ++j) {
            int grid_w = i / TILE_DIM;
            int grid_h = j / TILE_DIM;
            int idx = j * TEX_DIM + i;
            tex_checkerboard_data[idx] = checker_color[(grid_w + grid_h) % 2];
        }
    }

    glGenTextures(1, &gPlaneTex);
    glBindTexture(GL_TEXTURE_2D, gPlaneTex);
    glTexStorage2D(GL_TEXTURE_2D, static_cast<int>(std::log2(TEX_DIM)), GL_R8, TEX_DIM, TEX_DIM);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, TEX_DIM, TEX_DIM, GL_RED, GL_UNSIGNED_BYTE, tex_checkerboard_data);
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
}


static void set_mesh(const char* a_path, const glm::vec3& a_scale = glm::vec3(1.f), const glm::vec3& a_translation = glm::vec3(0.f))
{
    gMeshPtr.reset(new Mesh(a_path));
    gMeshPtr->scale(a_scale);
    gMeshPtr->translate(a_translation);
}

void
setupVaos()
{
    glGenVertexArrays(1, &gMeshVao);
    glBindVertexArray(gMeshVao);
    glGenBuffers(3, gMeshBos); // pos, normal, index

    //set_mesh("../data/assets/box.ply", { 0.6f, 0.3f, 0.3f });
    set_mesh("../data/assets/armadillo.ply", glm::vec3(5.f));

    glBindBuffer(GL_ARRAY_BUFFER, gMeshBos[0]); // positions
    glBufferData(GL_ARRAY_BUFFER, gMeshPtr->getSizePositions(), gMeshPtr->positions_.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, gMeshPtr->getAttrSizePosition(), GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, gMeshBos[1]); // normals
    glBufferData(GL_ARRAY_BUFFER, gMeshPtr->getSizeNormals(), gMeshPtr->normals_.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, gMeshPtr->getAttrSizeNormal(), GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gMeshBos[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, gMeshPtr->getSizeIndices(), gMeshPtr->indices_.data(), GL_STATIC_DRAW);

    glGenVertexArrays(1, &gPlaneVao);
    glBindVertexArray(gPlaneVao);
    glGenBuffers(4, gPlaneBos);

    gGround.init(0.f, 1.f, 0.f, 0.f, (float)kCheckerTexSize);
    GLsizei const pos_size = gGround.getSizePositions();
    GLsizei const num_vert = gGround.getNumVert();
    GLsizei const tex_uv_size = gGround.getSizeTexUVs();
    GLsizei const normal_size = gGround.getSizeNormals();
    GLsizei const idx_size = gGround.getSizeIndices();
    GLsizei const num_idx = gGround.getNumIndices();

    glBindBuffer(GL_ARRAY_BUFFER, gPlaneBos[0]); // positions
    glBufferData(GL_ARRAY_BUFFER, pos_size, gGround.positions_.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, gGround.getAttrSizePosition(), GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, gPlaneBos[1]); // tex
    glBufferData(GL_ARRAY_BUFFER, tex_uv_size, gGround.tex_uvs_.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, gGround.getAttrSizeTexUV(), GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, gPlaneBos[2]); // normals
    glBufferData(GL_ARRAY_BUFFER, normal_size, gGround.normals_.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, PlaneMesh::getAttrSizeNormal(), GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gPlaneBos[3]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx_size, gGround.indices_.data(), GL_STATIC_DRAW);

    checkGLError("voxel_render:setupVaos");
}

void
voxelize()
{
    struct cudaGraphicsResource *dMeshPositions_Res = 0;
    struct cudaGraphicsResource *dMeshIndices_Res = 0;
    registerGLBufferObject(gMeshBos[0], &dMeshPositions_Res);
    registerGLBufferObject(gMeshBos[2], &dMeshIndices_Res, cudaGraphicsRegisterFlagsReadOnly);
    float3 *dMeshPositions = (float3 *)mapGLBufferObject(&dMeshPositions_Res);
    uint *dMeshIndices = (uint *)mapGLBufferObject(&dMeshIndices_Res);
    
    float3 lower = make_float3(gMeshPtr->lower_);
    float3 upper = make_float3(gMeshPtr->upper_);

    uint num_triangles = gMeshPtr->getNumIndices() / 3;
    // due to lack of gpu memory, for sdf construction, maximal 413, independent of mesh
    uint dim_x = 64;
    uint3 volume_dim = make_uint3(64);
    std::cout << "on GPU (dim = " << dim_x << ")" << std::endl;

    deviceSync();
    Timer t;
    t.start();
    gVolumePtr.reset(new Volume_CUDA(dMeshPositions, 0, gMeshPtr->getNumVert(), dMeshIndices, 0, num_triangles, volume_dim, lower, upper, 0.f));
    deviceSync();
    t.stop();
    std::cout << "voxelization finished, took " << t.getElapsedMilliseconds() << " ms" << std::endl;
    unmapGLBufferObject(dMeshIndices_Res);
    unmapGLBufferObject(dMeshPositions_Res);
    UNREGISTER_GL_BO(dMeshIndices_Res);
    UNREGISTER_GL_BO(dMeshPositions_Res);

    deviceSync();
    t.start();
    gSDFPtr.reset(new SDF_CUDA(gVolumePtr->volume_dim_, gVolumePtr->proj_axis_, gVolumePtr->dVoxels_));
    deviceSync();
    t.stop();
    std::cout << "sdf construction finished, took " << t.getElapsedMilliseconds() << " ms" << std::endl;

    std::cout << "proj axis: " << gVolumePtr->proj_axis_ << std::endl;
    std::cout << "num_real_voxels: " << gSDFPtr->num_real_voxels_[0] << std::endl;
    uint num_voxels = volume_dim.x * volume_dim.y * volume_dim.z;
    uint voxel_idx_width = sizeof(uint) * 8;
    uint num_uints = num_voxels / voxel_idx_width + (num_voxels % voxel_idx_width > 0);
    
    uint *voxels = (uint *)malloc(num_uints * sizeof(uint));
    copyArrayFromDevice(voxels, gVolumePtr->dVoxels_, 0, num_uints * sizeof(uint));

    uint3 sdf_vol_dim = volume_dim + 2 * gSDFPtr->vol_dim_ext_;
    uint num_sdf_voxels = sdf_vol_dim.x * sdf_vol_dim.y * sdf_vol_dim.z;
    float4 *h_sdf = (float4 *)malloc(num_sdf_voxels * sizeof(float4));
    copyArrayFromDevice(h_sdf, gSDFPtr->dSdfPyramid_[0], 0, num_sdf_voxels * sizeof(float4));

    //int offset = gSDFPtr->vol_dim_ext_;
    //for (int k = 0; k < sdf_vol_dim.z; ++k) {
    //    for (int j = 0; j < sdf_vol_dim.y; ++j) {
    //        for (int i = 0; i < sdf_vol_dim.x; ++i) {
    //            uint sdf_idx = k * sdf_vol_dim.x * sdf_vol_dim.y + j * sdf_vol_dim.x + i;
    //            float4 sdf = h_sdf[sdf_idx];

    //            std::cout << "(" << i - offset << ", " << j - offset << ", " << k - offset << "): ["
    //                << sdf.x << ", " << sdf.y << ", " << sdf.z << "], dist_sqr = " << sdf.w 
    //                << std::endl;

    //        }
    //    }
    //}

    //std::cout << "voxels: " << std::flush;
    //for (size_t i = 0; i < num_uints; ++i) {
    //    std::cout << std::bitset<32>(voxels[i]) << " (" << voxels[i] << ")" << std::endl;
    //}
    std::cout << "lower: " << lower.x << ", " << lower.y << ", " << lower.z << std::endl;
    setupVolumeVao(voxels, make_uvec3(volume_dim), make_vec3(gVolumePtr->real_voxel_size_), make_vec3(gVolumePtr->real_voxel_lower_), h_sdf, gSDFPtr->vol_dim_ext_);
    free(voxels);
    free(h_sdf);
    gSelectedLevel = 9;
    std::cout << "generating sdf pyramid..." << std::endl;
    gSDFPtr->genSdfPyramid();
    //deviceSync();
    //setupVolPyramidVaos(make_vec3(gVolumePtr->real_voxel_size_), make_vec3(gVolumePtr->real_voxel_lower_));
    gSDFPtr->mergeSdfPyramid();
    deviceSync();
    std::cout << "setup merged volume vao begins..." << std::endl;
    setupMergedVolVaos(make_vec3(gVolumePtr->real_voxel_size_), make_vec3(gVolumePtr->real_voxel_lower_));
    std::cout << "setup merged volume vao ends..." << std::endl;
}

void
init()
{
    updateProjMat(gWindowWidth, gWindowHeight);
    initPrograms();
    gCamera.init(kCameraPos, kLookAt, kUp);
    setupTexture();
    setupVaos();
    voxelize();
    std::cout << "init finished\n" << std::endl;
#if 1    
    {
        Volume *v;
        int dim = 64;
        std::cout << "on cpu (dim = " << dim << ")" << std::endl;
        TIMING(
            v = new Volume(*gMeshPtr,
                glm::ivec3(dim),
                gMeshPtr->lower_ - 0.05f - kEpsilonF,
                gMeshPtr->upper_ + 0.05f + kEpsilonF
            ), "voxelization");

        TIMING(SDF sdf(*v), "sdf");
        delete v;
    }
#endif
}

void
drawMesh()
{
    glEnable(GL_CULL_FACE);
    glPolygonMode(GL_FRONT_AND_BACK, gToggleWireframe ? GL_LINE : GL_FILL);
    glm::mat4 const & view_mat = gCamera.getViewMat();
    glm::mat4 const & vp = gProjMat * view_mat;
    glm::vec3 const & cam_pos = gCamera.getPosition();
    glBindVertexArray(gMeshVao);
    gMeshProg.use();

    glUniform3fv(gMeshProg["CamPos"], 1, &cam_pos[0]);
    glUniform3fv(gMeshProg["LightPos"], 1, &gLightPos[0]);
    glUniformMatrix4fv(gMeshProg["MVP"], 1, GL_FALSE, &vp[0][0]);

    glDrawElements(GL_TRIANGLES, gMeshPtr->getNumIndices(), GL_UNSIGNED_INT, 0);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    checkGLError("voxel_render:drawMesh end");
}

void
drawGround()
{
    glEnable(GL_CULL_FACE);
    glPolygonMode(GL_FRONT_AND_BACK, gToggleWireframe ? GL_LINE : GL_FILL);
    glm::mat4 const & view_mat = gCamera.getViewMat();
    glm::mat4 const & view_inv_mat = gCamera.getViewInv();
    glm::mat4 const & vp = gProjMat * view_mat;
    glm::vec3 const & cam_pos = gCamera.getPosition();
    glBindVertexArray(gPlaneVao);
    gPlaneProg.use();

    glUniform1i(gPlaneProg["checker_tex"], 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, gPlaneTex);

    glUniform3fv(gPlaneProg["CamPos"], 1, &cam_pos[0]);
    glUniform3fv(gPlaneProg["LightPos"], 1, &gLightPos[0]);
    glUniformMatrix4fv(gPlaneProg["MVP"], 1, GL_FALSE, &vp[0][0]);
    glDrawElements(GL_TRIANGLES, gGround.getNumIndices(), GL_UNSIGNED_INT, 0);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void
drawVoxel()
{
    glEnable(GL_CULL_FACE);
    glPolygonMode(GL_FRONT_AND_BACK, gToggleVolumeWireframe ? GL_LINE : GL_FILL);
    glm::mat4 const & view_mat = gCamera.getViewMat();
    glm::mat4 const & vp = gProjMat * view_mat;
    glm::vec3 const & cam_pos = gCamera.getPosition();

    gSelectedLevel %= 10;
    if (gSelectedLevel != 9) {
        gSelectedLevel %= gNumSdfLevels;
    }

    if (gSelectedLevel == 9) {
        glBindVertexArray(gVolumeVao);
        gSelectedVoxel %= gVoxelDim + 1u;
    }
    else {
        glBindVertexArray(gVolPyramidVao[gSelectedLevel]);
        gSelectedVoxel %= gVolPyramidDims[gSelectedLevel] + 1u;
    }
    gVolumeProg.use();

    glUniform3fv(gVolumeProg["CamPos"], 1, &cam_pos[0]);
    glUniform3fv(gVolumeProg["LightPos"], 1, &gLightPos[0]);
    glUniform3uiv(gVolumeProg["SelectedVoxel"], 1, &gSelectedVoxel[0]);
    glUniform3fv(gVolumeProg["VoxelTranslation"], 1, &gVoxelTranslation[0]);
    glUniform1f(gVolumeProg["SDFMax"], gMaxSDF);
    glUniform1f(gVolumeProg["SDFMin"], gMinSDF);
    glUniformMatrix4fv(gVolumeProg["MVP"], 1, GL_FALSE, &vp[0][0]);

    if (gSelectedLevel == 9) {
        glDrawElements(GL_TRIANGLES, gNumVolumeIndices, GL_UNSIGNED_INT, 0);
    }
    else {
        glDrawElements(GL_TRIANGLES, gNumVolPyramidIndices[gSelectedLevel], GL_UNSIGNED_INT, 0);
    }
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    checkGLError("voxel_render:drawVolume end");
}


void
render()
{
    if (gToggleDrawMesh) {
        drawMesh();
    }
    if (gToggleDrawVolume) {
        drawVoxel();
    }
    drawGround();
    checkGLError("voxel_render:render end");
}

void
clear()
{
    RELEASE_GL_BOs(gPlaneBos, 4);
    RELEASE_GL_VAO(gPlaneVao);
    RELEASE_GL_BOs(gMeshBos, 3);
    RELEASE_GL_VAO(gMeshVao);
    RELEASE_GL_BOs(gVolumeBos, 5);
    RELEASE_GL_VAO(gVolumeVao);
    RELEASE_GL_TEX(gPlaneTex);
}


bool isVoxel(uint *_voxels, glm::uvec3 const &_voxel_coord, glm::uvec3 const &_volume_dim, uint proj_axis)
{
    uint proj_y = (proj_axis + 1) % 3;
    uint proj_z = (proj_axis + 2) % 3;
    uint x_max = _volume_dim[proj_axis];
    uint y_max = _volume_dim[proj_y];
    uint voxel_idx = _voxel_coord[proj_z] * y_max * x_max + _voxel_coord[proj_y] * x_max + _voxel_coord[proj_axis];
    uint uint_idx = voxel_idx / (sizeof(uint) * 8);
    uint bit_offset = voxel_idx % (sizeof(uint) * 8);
    //std::cout << "voxel_idx: " << voxel_idx << ", uint_idx: " << uint_idx 
    //    << ", bit_offset: " << bit_offset 
    //    << " <" << (nth_bit<uint>(_voxels[uint_idx], bit_offset) != 0) << ">"
    //    << std::endl;
    return nth_bit<uint>(_voxels[uint_idx], bit_offset) != 0;
}

void setupVolumeVao(uint *_voxels, glm::uvec3 const &_volume_dim, glm::vec3 const &_voxel_size, glm::vec3 const &_lower, float4 *_sdf, uint _vol_dim_ext)
{
    gVoxelDim = _volume_dim;
    gSelectedVoxel = glm::uvec3(0);
    gProjAxis = gVolumePtr->proj_axis_;
    gVoxelTranslation = glm::vec3(_volume_dim) * _voxel_size * 1.1f;
    gVoxelTranslation.y = 0.f;

    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::uvec3> voxel_coords;
    std::vector<glm::vec4> sdfs;
    std::vector<uint> indices;

    Mesh box("../data/assets/box.ply");
    box.scale(_voxel_size);
    box.translate(_lower);

    uint num_voxels = 0;
    
    for (uint z = 0; z < _volume_dim.z; ++z) {
        for (uint y = 0; y < _volume_dim.y; ++y) {
            for (uint x = 0; x < _volume_dim.x; ++x) {
                glm::uvec3 voxel_coord{ x, y, z };
                if (!isVoxel(_voxels, voxel_coord, _volume_dim, gVolumePtr->proj_axis_)) continue;
                glm::uvec3 sdf_vol_coord = voxel_coord + _vol_dim_ext;
                glm::uvec3 sdf_vol_dim = _volume_dim + _vol_dim_ext * 2;
                glm::vec4 sdf = make_vec4(_sdf[sdf_vol_coord.z * sdf_vol_dim.y * sdf_vol_dim.x + sdf_vol_coord.y * sdf_vol_dim.x + sdf_vol_coord.x]);
                sdf.w = fabsf(sdf.w);
                //std::cout << "sdf for (" << x << ", " << y << ", " << z << ") = " << sdf.w << std::endl;
                if (sdf.w > gMaxSDF) {
                    gMaxSDF = sdf.w;
                }
                if (sdf.w < gMinSDF) {
                    gMinSDF = sdf.w;
                }
                // std::cout << "mesh for voxel: " << voxel_coord.x << ", " << voxel_coord.y << ", " << voxel_coord.z << std::endl;
                Mesh b = box;
                glm::vec3 translation = glm::vec3(voxel_coord) * _voxel_size;
                b.translate(translation);
                for (int i = 0; i < b.getNumIndices(); ++i) {
                    b.indices_[i] += (num_voxels)* b.getNumVert();
                }
                ++num_voxels;
                positions.insert(positions.end(), b.positions_.begin(), b.positions_.end());
                normals.insert(normals.end(), b.normals_.begin(), b.normals_.end());
                for (int i = 0; i < b.getNumVert(); ++i) {
                    voxel_coords.push_back(voxel_coord);
                    sdfs.push_back(sdf);
                }
                indices.insert(indices.end(), b.indices_.begin(), b.indices_.end());
            }
        }
    }
    std::cout << "cpu num voxels: " << num_voxels << ", max sdf: " << gMaxSDF << ", min sdf: " << gMinSDF << std::endl;

    glGenVertexArrays(1, &gVolumeVao);
    glBindVertexArray(gVolumeVao);
    glGenBuffers(5, gVolumeBos); // pos, normal, index

    glBindBuffer(GL_ARRAY_BUFFER, gVolumeBos[0]); // positions
    glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(glm::vec3), positions.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, box.getAttrSizePosition(), GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, gVolumeBos[1]); // normals
    glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(glm::vec3), normals.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, box.getAttrSizeNormal(), GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, gVolumeBos[2]); // voxel coords
    glBufferData(GL_ARRAY_BUFFER, voxel_coords.size() * sizeof(glm::uvec3), voxel_coords.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(2);
    glVertexAttribIPointer(2, 3, GL_UNSIGNED_INT, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, gVolumeBos[4]); // sdfs
    glBufferData(GL_ARRAY_BUFFER, sdfs.size() * sizeof(glm::uvec4), sdfs.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVolumeBos[3]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(uint), indices.data(), GL_STATIC_DRAW);

    gNumVolumeIndices = static_cast<GLsizei>(indices.size());
}

void setupVolPyramidVaos(glm::vec3 const &_voxel_size, glm::vec3 const &_lower)
{
    gNumSdfLevels = (uint)gSDFPtr->num_real_voxels_.size();
    gVolPyramidVao.resize(gNumSdfLevels);
    gVolPyramidBos.resize(gNumSdfLevels);
    gNumVolPyramidIndices.resize(gNumSdfLevels);

    for (size_t i = 0; i < gNumSdfLevels; ++i) {
        float max_sdf = FLT_EPSILON;
        float min_sdf = FLT_MAX;
        glm::uvec3 dim = make_uvec3(gSDFPtr->volume_dim_ / (1u << i));
        uint num_voxels = dim.x * dim.y * dim.z;

        float4 *h_sdf = (float4 *)malloc(num_voxels * sizeof(float4));
        copyArrayFromDevice(h_sdf, gSDFPtr->dSdfPyramid_[i], 0, num_voxels * sizeof(float4));

        gVolPyramidDims.push_back(dim);
        gVolPyramidBos[i].resize(5);

        std::vector<glm::vec3> positions;
        std::vector<glm::vec3> normals;
        std::vector<glm::uvec3> voxel_coords;
        std::vector<glm::vec4> sdfs;
        std::vector<uint> indices;

        Mesh box("../data/assets/box.ply");
        box.scale(_voxel_size * (1u << i));
        box.translate(_lower - gSDFPtr->vol_dim_ext_ * _voxel_size);
        uint num_real_voxels = 0;

        for (uint z = 0; z < dim.z; ++z) {
            for (uint y = 0; y < dim.y; ++y) {
                for (uint x = 0; x < dim.x; ++x) {
                    glm::vec4 sdf = make_vec4(h_sdf[z * dim.y * dim.x + y * dim.x + x]);
                    if (sdf.w > 0.f) continue;
                    sdf.w = -sdf.w;
                    if (sdf.w > max_sdf) {
                        max_sdf = sdf.w;
                    }
                    if (sdf.w < min_sdf) {
                        min_sdf = sdf.w;
                    }
                    Mesh b = box;
                    glm::vec3 translation = glm::vec3(x, y, z) * _voxel_size * (1u << i);
                    b.translate(translation);
                    for (int j = 0; j < b.getNumIndices(); ++j) {
                        b.indices_[j] += (num_real_voxels)* b.getNumVert();
                    }
                    ++num_real_voxels;
                    for (int j = 0; j < b.getNumVert(); ++j) {
                        voxel_coords.push_back(glm::uvec3(x, y, z));
                        sdfs.push_back(sdf);
                    }
                    positions.insert(positions.end(), b.positions_.begin(), b.positions_.end());
                    normals.insert(normals.end(), b.normals_.begin(), b.normals_.end());
                    indices.insert(indices.end(), b.indices_.begin(), b.indices_.end());
                }
            }
        }
        free(h_sdf);

        std::cout << "cpu num voxels in level " << i << ": " << num_real_voxels << ", max sdf: " << max_sdf << ", min sdf: " << min_sdf << std::endl;
        glGenVertexArrays(1, &gVolPyramidVao[i]);
        glBindVertexArray(gVolPyramidVao[i]);
        glGenBuffers(5, &gVolPyramidBos[i][0]); // pos, normal, index

        glBindBuffer(GL_ARRAY_BUFFER, gVolPyramidBos[i][0]); // positions
        glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(glm::vec3), positions.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, box.getAttrSizePosition(), GL_FLOAT, GL_FALSE, 0, 0);

        glBindBuffer(GL_ARRAY_BUFFER, gVolPyramidBos[i][1]); // normals
        glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(glm::vec3), normals.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, box.getAttrSizeNormal(), GL_FLOAT, GL_FALSE, 0, 0);

        glBindBuffer(GL_ARRAY_BUFFER, gVolPyramidBos[i][2]); // voxel coords
        glBufferData(GL_ARRAY_BUFFER, voxel_coords.size() * sizeof(glm::uvec3), voxel_coords.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(2);
        glVertexAttribIPointer(2, 3, GL_UNSIGNED_INT, 0, 0);

        glBindBuffer(GL_ARRAY_BUFFER, gVolPyramidBos[i][4]); // sdfs
        glBufferData(GL_ARRAY_BUFFER, sdfs.size() * sizeof(glm::uvec4), sdfs.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 0, 0);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVolPyramidBos[i][3]);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(uint), indices.data(), GL_STATIC_DRAW);

        gNumVolPyramidIndices[i] = static_cast<GLsizei>(indices.size());
    }

}


void setupMergedVolVaos(glm::vec3 const &_voxel_size, glm::vec3 const &_lower)
{
    Mesh voxel_mesh("../data/assets/box.ply");
    gNumSdfLevels = (uint)gSDFPtr->num_real_voxels_.size();
    gVolPyramidVao.resize(gNumSdfLevels);
    gVolPyramidBos.resize(gNumSdfLevels);
    gNumVolPyramidIndices.resize(gNumSdfLevels);

    std::vector<std::vector<glm::vec3>> positions(gNumSdfLevels);
    std::vector<std::vector<glm::vec3>> normals(gNumSdfLevels);
    std::vector<std::vector<glm::uvec3>> voxel_coords(gNumSdfLevels);
    std::vector<std::vector<glm::vec4>> sdfs(gNumSdfLevels);
    std::vector<std::vector<uint>> indices(gNumSdfLevels);
    std::vector<Mesh> boxes(gNumSdfLevels, voxel_mesh);

    for (size_t i = 0; i < gNumSdfLevels; ++i) {
        glm::uvec3 dim = make_uvec3(gSDFPtr->volume_dim_ / (1u << i));
        gVolPyramidDims.push_back(dim);
        gVolPyramidBos[i].resize(5);
        glm::vec3 edge_len = _voxel_size * (float)(1u << i);
        boxes[i].scale(edge_len);
        boxes[i].translate(_lower);
    }

    float4 *h_sdf = (float4 *)malloc(gSDFPtr->num_particles_ * sizeof(float4));
    float4 *h_pos = (float4 *)malloc(gSDFPtr->num_particles_ * sizeof(float4));
    std::cout << "copy pos and sdf from device begins..." << std::endl;
    copyArrayFromDevice(h_sdf, gSDFPtr->dSdfs_, 0, gSDFPtr->num_particles_ * sizeof(float4));
    copyArrayFromDevice(h_pos, gSDFPtr->dVoxelPostions_, 0, gSDFPtr->num_particles_ * sizeof(float4));
    std::cout << "copy pos and sdf from device ends..." << std::endl;
    std::vector<float> max_sdf(gNumSdfLevels, FLT_EPSILON);
    std::vector<float> min_sdf(gNumSdfLevels, FLT_MAX);
    std::vector<int> num_real_voxels(gNumSdfLevels, 0);

    for (int i = 0; i < gSDFPtr->num_particles_; ++i) {
        glm::vec4 pos = make_vec4(h_pos[i]);
        glm::vec4 sdf = make_vec4(h_sdf[i]);
        sdf.w = -sdf.w;
        if (sdf.w < 0) {
            std::cout << "sdf.w < 0 " << std::endl;
        }
        assert(sdf.w >= 0.f);
        int level = (int)std::round(std::log2(2.f * pos.w));
        //std::cout << "level = " << level << std::endl;
        assert(level >= 0);

        if (sdf.w > max_sdf[level]) {
            max_sdf[level] = sdf.w;
        }
        if (sdf.w < min_sdf[level]) {
            min_sdf[level] = sdf.w;
        }
        ++num_real_voxels[level];
        float diameter = 2.f * pos.w;
        glm::uvec3 voxel_coord = glm::uround((glm::vec3(pos) + glm::vec3((float)(gSDFPtr->vol_dim_ext_))) / diameter);
        glm::vec3 translation = glm::vec3(pos) * _voxel_size;

        Mesh b = boxes[level];
        b.translate(translation);
        for (int j = 0; j < b.getNumIndices(); ++j) {
            b.indices_[j] += (uint)positions[level].size();
        }
        for (int j = 0; j < b.getNumVert(); ++j) {
            voxel_coords[level].push_back(voxel_coord);
            sdfs[level].push_back(sdf);
        }
        positions[level].insert(positions[level].end(), b.positions_.begin(), b.positions_.end());
        normals[level].insert(normals[level].end(), b.normals_.begin(), b.normals_.end());
        indices[level].insert(indices[level].end(), b.indices_.begin(), b.indices_.end());
    }
    free(h_sdf);
    free(h_pos);

    for (size_t i = 0; i < gNumSdfLevels; ++i) {
        std::cout << "cpu num voxels in level " << i << ": " << num_real_voxels[i] << ", max sdf: " << max_sdf[i] << ", min sdf: " << min_sdf[i] << std::endl;
    
        glGenVertexArrays(1, &gVolPyramidVao[i]);
        glBindVertexArray(gVolPyramidVao[i]);
        glGenBuffers(5, &gVolPyramidBos[i][0]); // pos, normal, index

        glBindBuffer(GL_ARRAY_BUFFER, gVolPyramidBos[i][0]); // positions
        glBufferData(GL_ARRAY_BUFFER, positions[i].size() * sizeof(glm::vec3), positions[i].data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, voxel_mesh.getAttrSizePosition(), GL_FLOAT, GL_FALSE, 0, 0);

        glBindBuffer(GL_ARRAY_BUFFER, gVolPyramidBos[i][1]); // normals
        glBufferData(GL_ARRAY_BUFFER, normals[i].size() * sizeof(glm::vec3), normals[i].data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, voxel_mesh.getAttrSizeNormal(), GL_FLOAT, GL_FALSE, 0, 0);

        glBindBuffer(GL_ARRAY_BUFFER, gVolPyramidBos[i][2]); // voxel coords
        glBufferData(GL_ARRAY_BUFFER, voxel_coords[i].size() * sizeof(glm::uvec3), voxel_coords[i].data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(2);
        glVertexAttribIPointer(2, 3, GL_UNSIGNED_INT, 0, 0);

        glBindBuffer(GL_ARRAY_BUFFER, gVolPyramidBos[i][4]); // sdfs
        glBufferData(GL_ARRAY_BUFFER, sdfs[i].size() * sizeof(glm::uvec4), sdfs[i].data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 0, 0);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gVolPyramidBos[i][3]);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices[i].size() * sizeof(uint), indices[i].data(), GL_STATIC_DRAW);

        gNumVolPyramidIndices[i] = static_cast<GLsizei>(indices[i].size());
    }

}