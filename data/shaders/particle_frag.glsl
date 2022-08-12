#version 450 core

uniform mat4 ViewMat;
uniform mat4 ScaleBiasProjMat;
uniform mat4 ShadowMat; // ScaleBiasMat * LightProjMat * LightViewMat * InvViewMat

flat in int vPickingID;
in vec4 vColor;
in vec4 vPosView;

layout(location = 0) out vec3 fPosition; // in camera space
layout(location = 1) out vec3 fNormal; // in camera space
layout(location = 2) out vec4 fColor;
layout(location = 3) out vec3 fShadowUVW;
layout(location = 4) out int  fPickingID;

vec4 divW(vec4 v)
{
    return v / v.w;
}

void main(void)
{
    vec3 normal;
    normal.xy = vec2(2.0, -2.0) * gl_PointCoord - vec2(1.0, -1.0);
    float mag = dot(normal.xy, normal.xy);
    if (mag > 1.0) discard;
    normal.z = sqrt(1.0 - mag);
    
    fColor = vColor;
    
    vec4 pos_view = vec4(vPosView.xyz + vPosView.w * normal, 1.0);
    fPosition = pos_view.xyz;
    fNormal = normal;
    fShadowUVW = divW(ShadowMat * pos_view).xyz;
    fPickingID = vPickingID;
    
    gl_FragDepth = divW(ScaleBiasProjMat * pos_view).z;
    
}
