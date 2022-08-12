#version 450 core

uniform mat4 MVP;
uniform mat4 ViewMat;
uniform float ParticleScale;

layout(location = 0) in vec4 Position;
layout(location = 1) in int Phase;

out vec4 vPosView;

void main(void)
{
    gl_Position = MVP * vec4(Position.xyz, 1.0);
    vPosView.xyz = vec3(ViewMat * vec4(Position.xyz, 1.0));
    vPosView.w = Position.w;
    gl_PointSize = ParticleScale * 2.f * Position.w / length(vPosView.z);
}