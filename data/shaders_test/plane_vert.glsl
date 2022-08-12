#version 430 core

uniform mat4 MVP;

layout(location = 0) in vec3 Position;
layout(location = 1) in vec2 TexUV;
layout(location = 2) in vec3 Normal;

out vec3 vWorldPos;
out vec2 vTexUV;
out vec3 vNormal;

void main(void)
{
    vWorldPos = Position;
    vTexUV = TexUV;
    vNormal = Normal;
    gl_Position = MVP * vec4(Position, 1.0);
}