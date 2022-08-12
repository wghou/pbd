#version 450 core

layout(location = 0) out vec2 fShadowMap; // shadow map: (depth, depth^2)

void main(void)
{
    fShadowMap.x = gl_FragCoord.z;
    fShadowMap.y = gl_FragCoord.z * gl_FragCoord.z;
    gl_FragDepth = gl_FragCoord.z;
}
