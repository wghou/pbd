#include <iomanip>
#include <glm/gtc/random.hpp>
#include "decomp/MathFunctions.h"
#include "misc.h"

// #define HIGH_PRECISION
#include "numerics.h"

#ifdef  HIGH_PRECISION
typedef double real;
#else
typedef float real;
#endif

using namespace PBD;

void rand_mat3(real _m[3][3])
{
    real rand_max = 1000.f;
    real rand_min = -1000.f;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            _m[i][j] = glm::linearRand(rand_min, rand_max);
        }
    }
}

Vector3r eigen_vec3(real const _vec[3])
{
    Vector3r v;
    for (int i = 0; i < 3; ++i) {
        v(i) = _vec[i];
    }
    return v;
}

Matrix3r eigen_mat3(real const _mat[3][3])
{
    Matrix3r m;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            m(i, j) = _mat[i][j];
        }
    }
    return m;
}

Matrix3r decomp_error(Matrix3r const &_A, Matrix3r const &_R, Matrix3r const &_U, Matrix3r const &_D)
{
    Matrix3r UT = _U.transpose();
    return _A - (_U * _D * UT) * _R;
}

void test_jacobi_rotation()
{
    real A[3][3], R[3][3];
    Matrix3r eA, eR, totalDiffR;
    totalDiffR = Matrix3r::Zero();
    int p = 0, q = 0;
    for (int i = 0; i < 20; ++i) {
        Matrix3r diffR = Matrix3r::Zero();
        identity(R);
        eR.setIdentity();
        rand_mat3(A);
        eA = eigen_mat3(A);
        std::cout << "eA:\n" << eA << "\n\n" << std::endl;
        for (p = 0; p < 3; ++p) {
            for (q = 0; q < 3; ++q) {
                std::cout << "p, q = " << p << ", " << q << std::endl;
                if (A[p][q] == 0.f) continue;
                std::cout << "A[" << p << ", " << q << "] = " << eA(p, q) << "\n" << std::endl;
                MathFunctions::jacobiRotate(eA, eR, p, q);
                jacobi_rotation(A, R, p, q);
                diffR += eR - eigen_mat3(R);
                std::cout << "A[" << p << ", " << q << "] = " << eA(p, q) << "\n"
                    << std::fixed << std::setprecision(8)
                    << "eA:\n" << eA << "\n\n"
                    << "A:\n" << eigen_mat3(A) << "\n\n"
                    << "eR:\n" << eR << "\n\n"
                    << "R:\n" << eigen_mat3(R) << "\n\n"
                    << "single jacobi rotation error: \n"
                    << eR - eigen_mat3(R) << "\n\n"
                    << std::endl;
            }
        }
        std::cout << "one jacobi error:\n"
            << std::fixed << std::setprecision(8)
            << diffR
            << "\n\n" << std::endl;
        totalDiffR += diffR;
    }
    std::cout << "total jacobi error:\n"
        << std::fixed << std::setprecision(8)
        << totalDiffR
        << "\n\n" << std::endl;
}

void test_eigen_decomp()
{
    real A[3][3], E[3][3], e[3];
    Matrix3r eA, eE, diffE;
    diffE = Matrix3r::Zero();
    Vector3r ee, diffe;
    diffe = Vector3r::Zero();
    for (int i = 0; i < 20; ++i) {
        rand_mat3(A);
        eA = eigen_mat3(A);
        MathFunctions::eigenDecomposition(eA, eE, ee);
        eigen_decomposition(A, E, e, 10);
        std::cout << std::fixed << std::setprecision(8)
            << "eA:\n" << eA << "\n\n"
            << "eE:\n" << eE << "\n\n"
            << "E:\n" << eigen_mat3(E) << "\n\n"
            << "diffE\n" << eE - eigen_mat3(E) << "\n\n"
            << "ee:\n" << ee << "\n\n"
            << "e:\n" << eigen_vec3(e) << "\n\n"
            << "diffe\n" << ee - eigen_vec3(e) << "\n\n" 
            << std::endl;
        diffE += eE - eigen_mat3(E);
        diffe += ee - eigen_vec3(e);
    }
    std::cout << std::fixed << std::setprecision(8)
        << "total diffE:\n" << diffE << "\n\n"
        << "total diffe:\n" << diffe << "\n\n"
        << std::endl;
}

void test_polar_decomp()
{
    int const num_jacobi_iter = 10;
    real A[3][3], R[3][3], U[3][3], D[3][3];
    Matrix3r eA, eR, eU, eD, error_plain, error_eigen;
    Matrix3r diffR = Matrix3r::Zero();
    for (int i = 0; i < 20; ++i) {
        rand_mat3(A);
        eA = eigen_mat3(A);
        polar_decomposition(A, R, U, D, num_jacobi_iter);
        MathFunctions::polarDecomposition(eA, eR, eU, eD);

        std::cout << std::fixed << std::setprecision(8) // << std::setw(18)
            << "\n===================== test " << i << " ===================== \n\n"
            << "\nA:\n" << eigen_mat3(A) << "\n\n"
            //<< "--------------------- plain version --------------------- \n" 
            //<< "\nR:\n" << eigen_mat3(R) << "\nU:\n" << eigen_mat3(U) << "\nD:\n" << eigen_mat3(D) << "\n"
            //<< "\ndecomp error: \n" << decomp_error(eigen_mat3(A), eigen_mat3(R), eigen_mat3(U), eigen_mat3(D)) << "\n\n"
            //<< "--------------------- eigen version --------------------- \n" 
            //<< "\n\nR:\n" << eR << "\n\nU:\n" << eU << "\n\nD:\n" << eD << "\n\n"
            << "\ndecomp error: \n" << decomp_error(eA, eR, eU, eD) << "\n\n"
            << "--------------------- difference --------------------- " << std::endl
            // << "diff A:\n" << eA - eigen_mat3(A) << "\n\n" 
            << "diff R:\n" << eR - eigen_mat3(R) << "\n\n"
            << "diff U:\n" << eU - eigen_mat3(U) << "\n\n"
            << "diff D:\n" << eD - eigen_mat3(D) << "\n\n"
            << std::endl;
        diffR += eR - eigen_mat3(R);
    }
    std::cout << std::fixed << std::setprecision(8)
        << "diff R:\n" << diffR << "\n\n"
        << std::endl;
}

int main(int argc, char **argv)
{
    std::srand(static_cast<unsigned>(std::time(0)));
    //test_jacobi_rotation();
    //test_eigen_decomp();
    test_polar_decomp();
    return 0;
}
