#pragma once
#include <glm/glm.hpp>
#include <algorithm>
#include <vector>

const float kInfF = std::numeric_limits<float>::infinity();
const float kMaxF = std::numeric_limits<float>::max();
const float kEpsilonF = std::numeric_limits<float>::epsilon();

template<typename T>
T max3(T const &a, T const &b, T const &c)
{
    return std::max(std::max(a, b), c);
}

glm::vec3 rotateVec3(glm::vec3 const &_v, float _angle, glm::vec3 const &_axis);

// 0.f * inf = 0.f
glm::vec3 multiplyByVec3Componentwise(glm::vec3 const &_v1, glm::vec3 const &_v2);

// dot(a, cross(b, c)) = det(a, b, c)
float ScalarTriple(glm::vec3 const & _a, glm::vec3 const & _b, glm::vec3 const & _c);


// ...NVIDIA Confidential Information begins...
// _winding positive ==> ccw
bool 
intersectRayTriangleTwoSided(
    const glm::vec3 & _o, 
    const glm::vec3 & _dir,
    const glm::vec3 & _a, 
    const glm::vec3 & _b, 
    const glm::vec3 & _c,
    float & _t, glm::vec3 & _bary, 
    float & _winding);

// divide box[_lower, _upper] into _dim.x * _dim.y * dim.z _voxels, 
// fill the voxels occupied by _mesh
struct Mesh;

struct Volume
{
    std::vector<int> voxels_;
    glm::ivec3 dim_;
    glm::vec3 voxel_dim_;
    glm::vec3 lower_;
    glm::vec3 upper_;
    Volume(
        Mesh const &_mesh,
        glm::ivec3 const & _dim,
        glm::vec3 const & _lower,
        glm::vec3 const & _upper);

    int getIdx(glm::ivec3 const &_pos) const;
    int& operator[](glm::ivec3 const & _pos);
    int const & operator[](glm::ivec3 const & _pos) const;
    bool isOnSurface(glm::ivec3 const & _pos, float &_dist) const;
};

// SDF: negative inside _vol, positive outside _vol
// 3D fast marching method 
// (FMM J. Sethian. A fast marching level set method for monotonically advancing fronts.
// Proc. Natl. Acad. Sci., 93:1591–1595, 1996.)

struct SDF {
    std::vector<float> dists_;
    glm::ivec3 dim_;
    // float max_dist_;
    SDF(Volume const &_vol);
    int getIdx(glm::ivec3 const &_pos) const;
    float& operator[] (glm::ivec3 const & _pos);
    float const & operator[] (glm::ivec3 const & _pos) const;
    glm::vec3 getNormal(glm::ivec3 const & _pos) const; // return normalized normal when possible
};

// ...NVIDIA Confidential Information ends...