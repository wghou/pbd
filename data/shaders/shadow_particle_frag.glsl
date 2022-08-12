#version 450 core

uniform mat4 ScaleBiasProjMat;

in vec4 vPosView;

layout(location = 0) out vec2 fShadowMap; // shadow map: (depth, depth^2)

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
    
    gl_FragDepth = divW(ScaleBiasProjMat * vec4(vPosView.xyz + vPosView.w * normal, 1.0)).z + 9e-6;
    fShadowMap.x = gl_FragDepth;
    fShadowMap.y = gl_FragDepth * gl_FragDepth;
}
