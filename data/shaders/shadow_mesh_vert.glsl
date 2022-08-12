#version 450 core

uniform mat4 MVP;

layout(location = 0) in vec3 Position;
layout(location = 1) in vec4 Color;
layout(location = 2) in vec3 Normal;

void main(void)
{
    gl_Position = MVP * vec4(Position, 1.0);
}