#version 430 core

uniform vec3 LightPos;
uniform vec3 CamPos;
uniform sampler2D CheckerTex;

sample in vec3 vWorldPos;
sample in vec2 vTexUV;
sample in vec3 vNormal;

out vec4 fColor;

vec4 lighting(vec3 _world_pos, vec3 _color)
{
    vec3 k_d = _color;

    vec3 N = normalize(vNormal);
    vec3 V = normalize(CamPos - _world_pos);
    vec3 L = normalize(LightPos - _world_pos);
    vec3 R = reflect(-L, N);

    vec3 ambient = vec3(0.1, 0.1, 0.1);
    vec3 diffuse = max(dot(N, L), 0.0) * k_d;

    return vec4(ambient + diffuse, 1.0);
}


void main(void)
{
    vec3 color = vec3(texture(CheckerTex, vTexUV).r);
    fColor = vec4(color, 1.0);
}
