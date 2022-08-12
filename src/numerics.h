#pragma once
#include "cuda_math_helper.h"

inline __host__ __device__
void
extract_rotation(mat3 &_A, quat &_q, int _num_iter)
{
    for (int i = 0; i < _num_iter; ++i) {
        mat3 R = extract_matrix(_q);
        float3 omega = (cross(R.col[0], _A.col[0]) +
                        cross(R.col[1], _A.col[1]) +
                        cross(R.col[2], _A.col[2])
            )
            *
            (1.f
                /
                (fabsf(dot(R.col[0], _A.col[0]) +
                       dot(R.col[1], _A.col[1]) +
                       dot(R.col[2], _A.col[2])
                      )
                  +    FLT_EPSILON
                )
            );
        float w = length(omega);
        if (w < FLT_EPSILON) break;
        _q = quat_from_axis_angle<float>((1.f / w) * omega, w) * _q;
        _q = normalize(_q);
    }
}




inline __host__ __device__
void 
extract_rotation(float _A[3][3], quat &q, int num_iter)
{
    float R[3][3];
    for (int i = 0; i < num_iter; ++i) {
        extract_matrix(q, R);
        float3 omega =     ( cross(get_col(R, 0), get_col(_A, 0)) + 
                             cross(get_col(R, 1), get_col(_A, 1)) +
                             cross(get_col(R, 2), get_col(_A, 2))
                           ) 
                      *  
                           (         1.f 
                                /    
                                     (    fabsf(dot(get_col(R, 0), get_col(_A, 0)) +
                                                dot(get_col(R, 1), get_col(_A, 1)) +
                                                dot(get_col(R, 2), get_col(_A, 2))
                                           )
                                      +    FLT_EPSILON
                                     )
                           );
        float w = length(omega);
        if (w < FLT_EPSILON) break;
        q = quat_from_axis_angle<float>((1.f / w) * omega, w) * q;
        q = normalize(q);
    }
}


// zeroes _A[p][q] and _A[q][p], columns of _E are eigenvectors of _A
inline __host__ __device__ 
void 
jacobi_rotation(float _A[3][3], float _E[3][3], int _p, int _q)
{
    if (_A[_p][_q] == 0.f) return;
    float d = (_A[_p][_p] - _A[_q][_q]) / (2.f * _A[_p][_q]);
    float t = 1.f / (fabs(d) + sqrt(d * d + 1.f));
    if (d < (float)0.0) t = -t;
    float c = 1.f / sqrt(t * t + 1.f);
    float s = t * c;
    
    _A[_p][_p] += t * _A[_p][_q];
    _A[_q][_q] -= t * _A[_p][_q];
    _A[_p][_q] = _A[_q][_p] = 0.f;

    // transform A
    for (int k = 0; k < 3; ++k) {
        if (k != _p && k != _q) {
            float Akp =  c * _A[k][_p] + s * _A[k][_q];
            float Akq = -s * _A[k][_p] + c * _A[k][_q];
            _A[k][_p] = _A[_p][k] = Akp;
            _A[k][_q] = _A[_q][k] = Akq;
        }
    }

    // store rotation in _R
    for (int k = 0; k < 3; ++k) {
        float Ekp =  c * _E[k][_p] + s * _E[k][_q];
        float Ekq = -s * _E[k][_p] + c * _E[k][_q];
        _E[k][_p] = Ekp;
        _E[k][_q] = Ekq;
    }
}

inline __host__ __device__ 
void 
eigen_decomposition(float const _A[3][3], float _E[3][3], float _e[3], int _num_jacobi_iter)
{
    float const epsilon = (float)1e-15; //FLT_EPSILON;
    float D[3][3];
    copy_mat3(D, _A);

    identity(_E);
    for (int i = 0; i < _num_jacobi_iter; ++i) {
        float a, max;
        max = fabs(D[0][1]);
        a = fabs(D[0][2]);

        int p = 0, q = 1;
        if (a > max) {
            p = 0; q = 2; max = a;
        }
        a = fabs(D[1][2]);
        if (a > max) {
            p = 1; q = 2; max = a;
        }
        if (max < epsilon) break;
        jacobi_rotation(D, _E, p, q);
    }

    for (int i = 0; i < 3; ++i) {
        _e[i] = D[i][i];
    }
}

// A = U D U^T R = SR, S = (A A^T)^(1/2)
inline __host__ __device__ 
void 
polar_decomposition(float const _A[3][3], float _R[3][3], float _U[3][3], float _D[3][3], int _num_jacobi_iter)
{
    const float epsilon = (float)1e-15; //FLT_EPSILON;
    float AAT[3][3];

    AAT[0][0] = _A[0][0] * _A[0][0] + _A[0][1] * _A[0][1] + _A[0][2] * _A[0][2];
    AAT[1][1] = _A[1][0] * _A[1][0] + _A[1][1] * _A[1][1] + _A[1][2] * _A[1][2];
    AAT[2][2] = _A[2][0] * _A[2][0] + _A[2][1] * _A[2][1] + _A[2][2] * _A[2][2];

    AAT[0][1] = _A[0][0] * _A[1][0] + _A[0][1] * _A[1][1] + _A[0][2] * _A[1][2];
    AAT[0][2] = _A[0][0] * _A[2][0] + _A[0][1] * _A[2][1] + _A[0][2] * _A[2][2];
    AAT[1][2] = _A[1][0] * _A[2][0] + _A[1][1] * _A[2][1] + _A[1][2] * _A[2][2];

    AAT[1][0] = AAT[0][1];
    AAT[2][0] = AAT[0][2];
    AAT[2][1] = AAT[1][2];

    identity(_R);
    float e[3];
    eigen_decomposition(AAT, _U, e, _num_jacobi_iter);

    zeroes(_D);
    float d[3], l[3];
    for (int i = 0; i < 3; ++i) {
        d[i] = sqrt(e[i]);
        _D[i][i] = d[i];
        l[i] = e[i] > epsilon ? 1.f / d[i] : 0.f;
    }

    float S1[3][3];

    S1[0][0] = l[0] * _U[0][0] * _U[0][0] + l[1] * _U[0][1] * _U[0][1] + l[2] * _U[0][2] * _U[0][2];
    S1[1][1] = l[0] * _U[1][0] * _U[1][0] + l[1] * _U[1][1] * _U[1][1] + l[2] * _U[1][2] * _U[1][2];
    S1[2][2] = l[0] * _U[2][0] * _U[2][0] + l[1] * _U[2][1] * _U[2][1] + l[2] * _U[2][2] * _U[2][2];

    S1[0][1] = l[0] * _U[0][0] * _U[1][0] + l[1] * _U[0][1] * _U[1][1] + l[2] * _U[0][2] * _U[1][2];
    S1[0][2] = l[0] * _U[0][0] * _U[2][0] + l[1] * _U[0][1] * _U[2][1] + l[2] * _U[0][2] * _U[2][2];
    S1[1][2] = l[0] * _U[1][0] * _U[2][0] + l[1] * _U[1][1] * _U[2][1] + l[2] * _U[1][2] * _U[2][2];

    S1[1][0] = S1[0][1];
    S1[2][0] = S1[0][2];
    S1[2][1] = S1[1][2];

    mat3_by_mat3(S1, _A, _R);
    // stabilize
    float RT[3][3], squared_norm[2];
    transpose_mat3(_R, RT);
    vec3_dot_vec3(RT[0], RT[0], &squared_norm[0]);
    vec3_dot_vec3(RT[1], RT[1], &squared_norm[1]);
    if (squared_norm[0] < epsilon) vec3_cross_vec3(RT[1], RT[2], RT[0]);
    else if (squared_norm[1] < epsilon) vec3_cross_vec3(RT[2], RT[0], RT[1]);
    else vec3_cross_vec3(RT[0], RT[1], RT[2]);
    transpose_mat3(RT, _R);
}
