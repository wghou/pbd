#version 450 core

uniform mat4 MVP;
uniform mat4 ViewMat;
uniform float ParticleScale;

layout(location = 0) in vec4 Position;
layout(location = 1) in int Phase;

out vec4 vColor;
out vec4 vPosView;
out int  vPickingID;

const vec4 kColors[7] = {
    { 0.000f, 0.625f, 1.000f, 1.f },
    { 0.821f, 0.821f, 0.297f, 1.f },
    { 0.000f, 0.436f, 0.216f, 1.f },
    { 1.000f, 0.442f, 0.000f, 1.f },
    { 0.013f, 0.213f, 0.566f, 1.f },
    { 0.841f, 0.138f, 0.000f, 1.f },
    { 0.765f, 0.243f, 0.493f, 1.f }
};

void main(void)
{
    vPosView.xyz = vec3(ViewMat * vec4(Position.xyz, 1.0));
    vPosView.w = Position.w;
    gl_PointSize = ParticleScale * 2.f * Position.w / length(vPosView.z);
    gl_Position = MVP * vec4(Position.xyz, 1.0);
    vColor = kColors[Phase % 7];
    vPickingID = gl_VertexID;
}