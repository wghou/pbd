#version 450 core

uniform mat4 ShadowMat;
uniform vec2 ShadowSampleSize; // 1 / shadow_dim

uniform vec3 LightPosView; // light pos in camera space
uniform vec3 LightDirView; // light dir in camera space
uniform vec3 LightImax;
uniform vec3 LightParams; //x: CosLightPenumbra; y: CosLightUmbra; z: LightExp; 

layout (binding = 0) uniform sampler2D ShadowMapTex;
layout (binding = 1) uniform sampler2D PosViewTex;
layout (binding = 2) uniform sampler2D NormalViewTex;
layout (binding = 3) uniform sampler2D ColorTex;
layout (binding = 4) uniform sampler2D ShadowUVWTex;
layout (binding = 5) uniform sampler2D FluidThicknessTex;
layout (binding = 6) uniform sampler2D FluidSurfacePositionTex;
layout (binding = 7) uniform sampler2D FluidSurfaceNormalTex;

in vec2 vTexUV;

out vec4 fColor;

vec4 divW(vec4 v)
{
    return v / v.w;
}

float pcf(vec3 curr_uvw, int radius)
{
    bvec2 lt0 = lessThanEqual(curr_uvw.xy, vec2(0.0));
    bvec2 gt1 = greaterThanEqual(curr_uvw.xy, vec2(1.0));
    if (any(lt0) || any(gt1)) {
        return 0.98;
    }
    int len = 2 * radius + 1;
    float inv_num_samples = 1.0 / (len * len);
    float visibility = 0.0;
    
    // if (texture(ShadowMapTex, curr_uvw.xy).x == 1.0) return 1.0;
    // return 0.0;
    
    // if (curr_uvw.z < 0.99994) return 1.0;
    // return 0.0;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            float depth_min = texture(ShadowMapTex, curr_uvw.xy + vec2(dx, dy) * ShadowSampleSize).x;
            visibility += curr_uvw.z - 5e-7 <= depth_min ? 1.0 : 0.0;
        }
    }
    
    return inv_num_samples * visibility;
}

vec2 vsm_moments(vec2 curr_uv, int radius)
{
    vec2 moments = vec2(0.0);
    int len = 2 * radius + 1;
    float inv_num_samples = 1.0 / (len * len);
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            moments += texture(ShadowMapTex, curr_uv + vec2(dx, dy) * ShadowSampleSize).xy;
        }
    }
    return inv_num_samples * moments;
}

//uv: tex uv; w: depth
float vsm(vec3 curr_uvw, int filter_radius) 
{
    vec2 moments = vsm_moments(curr_uvw.xy, filter_radius);
    float E_x2 = moments.y;
    float Ex_2 = moments.x * moments.x;
    float variance = E_x2 - Ex_2;
    float mD = moments.x - curr_uvw.z;
    float p = variance / (mD * mD + variance);
    return max(p, curr_uvw.z <= moments.x ? 1.0 : 0.0);
    //return curr_uvw.z - 1e-6< texture(ShadowMapTex, curr_uvw.xy).x ? 1.0 : 0.0;
}

// return fluid alpha
float coloringFluid(in vec3 light_intensity, out vec3 color)
{
    vec4 fluid_surface_pos = texture(FluidSurfacePositionTex, vTexUV);
    color = vec3(0.0);
    if (fluid_surface_pos.w < 0.5f) {
        return 0.0;
    }
    float fluid_thickness = texture(FluidThicknessTex, vTexUV).x;
    vec3 base_color = vec3(0.113f, 0.425f, 0.55f); //vec3(0.0, 0.7, 0.9);
    vec3 L = normalize(LightPosView - fluid_surface_pos.xyz);
    vec3 N = normalize(texture(FluidSurfaceNormalTex, vTexUV).xyz);
    
    vec3 diffuse = max(dot(N, L), 0.2) * base_color;
    //color = 1.2 * light_intensity * diffuse;
    color = base_color; 
    return 0.4 + fluid_thickness;
}

// in camera space
vec3 lighting()
{
    vec3 curr_uvw = texture(ShadowUVWTex, vTexUV).xyz;
    vec3 P = texture(PosViewTex, vTexUV).xyz;
    vec3 N = normalize(texture(NormalViewTex, vTexUV).xyz);
    vec3 color = texture(ColorTex, vTexUV).xyz;
    //vec3 color = vec3(1.0, 1.0, 1.0);
    
    vec3 L = normalize(LightPosView - P);
    vec3 diffuse = (0.6 + 0.4 * pcf(curr_uvw, 1)) * max(dot(N, L), 0.2) * color;
    vec3 ambient = 0.2 * color;
    
    float cos_theta_s = dot(-L, LightDirView);
    float penumbra_coeff = pow((cos_theta_s - LightParams.y) / (LightParams.x - LightParams.y), LightParams.z);
    float light_intensity_coeff = max(cos_theta_s >= LightParams.x ? 1.0 : 0.0,  max(0.0, penumbra_coeff));
    vec3 light_intensity = light_intensity_coeff * LightImax;
    
    vec3 original_color = (diffuse + ambient) * max(light_intensity, vec3(0.2));
    
    vec3 fluid_color;
    float fluid_alpha = coloringFluid(light_intensity, fluid_color);
    //return (1.0 - fluid_alpha) * original_color + fluid_alpha * fluid_color;
    return (1.0 - fluid_alpha) * original_color + original_color * fluid_alpha * fluid_color;
}


void main(void)
{
    
    //curr_uvw = divW(ShadowMat * vec4(texture(PosViewTex, vTexUV).xyz, 1.0)).xyz;
    
    fColor = vec4(lighting(), 1.0);

    //fColor = vec4(lighting() , 1.0);
    //fColor = vec4(step(vec3(0.1), texture(ColorTex, vTexUV).xyz), 1.0);
    //fColor = vec4(1000 * (texture(ShadowMapTex, vTexUV).xxx -vec3(0.997)), 1.0);
    //fColor = vec4(abs(texture(NormalViewTex, vTexUV).xyz), 1.0);
}
