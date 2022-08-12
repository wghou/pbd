#version 450 core

uniform mat4 MVP;
uniform mat4 ViewMat;
uniform mat4 ShadowMat; // ScaleBiasMat * LightProjMat * LightViewMat * InvViewMat

layout(location = 0) in vec3 Position;
layout(location = 1) in vec4 Color;
layout(location = 2) in vec3 Normal;

out vec4 vPosView;
out vec3 vNormalView;
out vec4 vColor;
out vec4 vShadowUVW;


void main(void)
{
    gl_Position = MVP * vec4(Position, 1.0);
    vPosView = ViewMat * vec4(Position, 1.0);
    vNormalView = normalize((ViewMat * vec4(Normal, 0.0)).xyz);
    vColor = Color;
    vShadowUVW = ShadowMat * vPosView;
}