#version 450 core

uniform mat4 ShadowMat;

in vec4 vPosView;
in vec3 vNormalView;
in vec4 vShadowUVW;
in vec2 vTexUV;

layout(location = 0) out vec3 fPosition; // in camera space
layout(location = 1) out vec3 fNormal; // in camera space
layout(location = 2) out vec4 fColor;
layout(location = 3) out vec3 fShadowUVW;

vec4 divW(vec4 v)
{
    return v / v.w;
}

vec2 bump(vec2 x) 
{
    return (floor(0.5 * x) + 2.f * max(0.5 * x - floor(0.5 * x) - 0.5, 0.0)); 
}

float checker(vec2 uv)
{
    vec2 width = 2.0 * fwidthFine(uv);
    vec2 p0 = uv - 0.5 * width;
    vec2 p1 = uv + 0.5 * width;
      
    vec2 i = (bump(p1) - bump(p0)) / width;
    float k = i.x * i.y + (1 - i.x) * (1 - i.y);
    return 0.8 - 0.38 * k;
}

void main(void)
{
    fPosition = vPosView.xyz;
    fNormal = normalize(vNormalView);
    fColor = vec4(vec3(checker(vTexUV)), 1.0);
    fShadowUVW = divW(ShadowMat * vPosView).xyz;
    gl_FragDepth = gl_FragCoord.z;
}
