#pragma once
#include <voxelization_cuda.h>
#include <sdf_cuda.h>
#include <gl_helper.h>
#include <mesh.h>

const glm::vec4 gColors[7] = {
    { 0.000f, 0.625f, 1.000f, 1.f },
    { 1.000f, 0.442f, 0.000f, 1.f },
    { 0.000f, 0.436f, 0.216f, 1.f },
    { 1.000f, 0.891f, 0.058f, 1.f },
    { 0.013f, 0.213f, 0.566f, 1.f },
    { 0.841f, 0.138f, 0.000f, 1.f },
    { 0.765f, 0.243f, 0.493f, 1.f }
};

extern Camera gCamera;
extern const float kCameraLowSpeed;
extern const float kCameraSpeed;
extern const float kCameraHighSpeed;
extern const float kCameraRotationSpeed;
extern const glm::vec3 kCameraPos;
extern const glm::vec3 kLookAt;
extern const glm::vec3 kUp;

extern const glm::vec2 kPlaneTileSize;

extern bool gToggleDrawMesh;
extern bool gToggleWireframe;
extern bool gToggleVolumeWireframe;
extern bool gToggleDrawVolume;

extern glm::uvec3 gVoxelDim;
extern glm::uvec3 gSelectedVoxel;
extern uint gSelectedLevel;
extern uint gProjAxis;

void updateProjMat(int width, int height);

// initialize OpenGL Program Objects
void reloadShaders();
void initPrograms();
void setupTextures();
void init();
void render();
void clear();