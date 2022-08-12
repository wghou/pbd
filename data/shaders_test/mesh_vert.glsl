#version 430 core

uniform mat4 MVP;

layout(location = 0) in vec3 Position;
layout(location = 1) in vec3 Normal;

out vec4 vColor;
out vec3 vNormal;
out vec3 vWorldPos;

void main(void)
{
    vWorldPos = Position;
    gl_Position = MVP * vec4(vWorldPos, 1.0);
    vNormal = normalize(Normal);
    vColor = vec4(abs(vNormal), 1.0);
}