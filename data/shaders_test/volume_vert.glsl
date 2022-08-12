#version 430 core

uniform mat4 MVP;
uniform uvec3 SelectedVoxel;
uniform vec3 VoxelTranslation;
uniform float SDFMax;
uniform float SDFMin;

const vec4 gColors[7] = {
    { 0.000f, 0.625f, 1.000f, 1.f },
    { 1.000f, 0.442f, 0.000f, 1.f },
    { 0.000f, 0.436f, 0.216f, 1.f },
    { 1.000f, 0.891f, 0.058f, 1.f },
    { 0.013f, 0.213f, 0.566f, 1.f },
    { 0.841f, 0.138f, 0.000f, 1.f },
    { 0.765f, 0.243f, 0.493f, 1.f }
};

layout(location = 0) in vec3 Position;
layout(location = 1) in vec3 Normal;
layout(location = 2) in uvec3 VoxelCoord;
layout(location = 3) in vec4 SDF;

out vec4 vColor;
out vec3 vNormal;
out vec3 vWorldPos;

void main(void)
{
    vec3 translation = vec3(0.0);
    // if (all(equal(VoxelCoord, SelectedVoxel))) {
        // translation -= VoxelTranslation;
    // }
    //else 
    if (any(lessThan(VoxelCoord, SelectedVoxel))) {
        translation -= VoxelTranslation;
    }
    //translation = vec3(0.0);
    vWorldPos = Position + translation;
    gl_Position = MVP * vec4(vWorldPos, 1.0);
    vNormal = normalize(Normal);
    //vColor = vec4(abs(vNormal), 1.0);
    vColor = vec4(gColors[(VoxelCoord.z + VoxelCoord.y + VoxelCoord.x) % 7].xyz, 1.0);
    //vColor.xyz = vec3(1.0 - SDF.w / SDFSqrMax);
    vColor.xyz = vec3(1.0 - (SDF.w - SDFMin)/(SDFMax - SDFMin + 1e-7));
    vColor.z = 1.0 - vColor.z;
    //vColor += 0.05 * vec4(gColors[(VoxelCoord.z + VoxelCoord.y + VoxelCoord.x) % 7].xyz, 1.0);
}