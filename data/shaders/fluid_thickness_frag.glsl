#version 450 core
uniform mat4 ScaleBiasProjMat;
uniform float FluidParticleAlpha;

in vec4 vPosView;

layout(location = 0) out float fAlpha; 

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
    
    fAlpha = FluidParticleAlpha;
    vec4 pos_view = vec4(vPosView.xyz + vPosView.w * normal, 1.0);
    gl_FragDepth = divW(ScaleBiasProjMat * pos_view).z;
}
