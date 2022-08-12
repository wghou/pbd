#version 450 core

layout(location = 0) in vec2 Position;
layout(location = 1) in vec2 TexUV;

out vec2 vTexUV;

void main(void)
{
    gl_Position = vec4(Position, 0.0, 1.0);
    vTexUV = TexUV;
}