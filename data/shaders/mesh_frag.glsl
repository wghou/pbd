#version 450 core

in vec4 vPosView;
in vec3 vNormalView;
in vec4 vColor;
in vec4 vShadowUVW;

layout(location = 0) out vec3 fPosition; // in camera space
layout(location = 1) out vec3 fNormal; // in camera space
layout(location = 2) out vec4 fColor;
layout(location = 3) out vec3 fShadowUVW;

vec4 divW(vec4 v)
{
    return v / v.w;
}

void main(void)
{
    fPosition = vPosView.xyz;
    fNormal = normalize(vNormalView);
    fColor = vColor;
    fShadowUVW = divW(vShadowUVW).xyz;
    gl_FragDepth = gl_FragCoord.z;
}
