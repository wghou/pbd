#version 430 core

uniform vec3 LightPos;
uniform vec3 CamPos;

sample in vec4 vColor;
sample in vec3 vNormal;
sample in vec3 vWorldPos;

out vec4 fColor;

vec4 lighting(vec3 _world_pos, vec3 _light_pos, vec3 _cam_pos, vec3 _normal, vec3 _color)
{
    vec3 k_d = _color;

    vec3 N = normalize(_normal);
    vec3 V = normalize(_cam_pos - _world_pos);
    vec3 L = normalize(_light_pos - _world_pos);
    vec3 R = reflect(-L, N);

    vec3 ambient = vec3(0.1, 0.1, 0.1);
    vec3 diffuse = max(dot(N, L), 0.0) * k_d;

    return vec4(ambient + diffuse, 1.0);
}


void main(void)
{
     // fColor = lighting(vWorldPos, LightPos, CamPos, vNormal, vColor);
     fColor = vec4(abs(vNormal), vColor.w);
     // fColor = vec4(vColor.xyz, vColor.w);
     //fColor = vec4(1.0, 1.0, 0.0, 1.0);
}
