#include <iostream>
#include <glm/ext.hpp>
#include "cg_math.h"
#include "misc.h"
#include "mesh.h"
#include "bvh.h"

glm::vec3
rotateVec3(glm::vec3 const &_v, float _angle, glm::vec3 const &_axis)
{
    glm::quat rotation;
    rotation = rotate(rotation, _angle, glm::normalize(_axis));
    return (glm::mat3_cast(rotation) * _v);
}

glm::vec3 
multiplyByVec3Componentwise(glm::vec3 const &_v1, glm::vec3 const &_v2)
{
    glm::vec3 prod;
    for (int i = 0; i < 3; ++i) {
        if (_v1[i] == 0.f) {
            prod[i] = 0.f;
        }
        else {
            prod[i] = _v1[i] * _v2[i];
        }
    }
    return prod;
}

float 
ScalarTriple(glm::vec3 const & _a, glm::vec3 const & _b, glm::vec3 const & _c)
{
    return glm::dot(_a, glm::cross(_b, _c));
}

bool
intersectRayTriangleTwoSided(const glm::vec3 & _o, const glm::vec3 & _dir,
    const glm::vec3 & _a, const glm::vec3 & _b, const glm::vec3 & _c,
    float & _t, glm::vec3 & _bary, float & _winding)
{
    glm::vec3 ab = _b - _a;
    glm::vec3 ac = _c - _a;
    glm::vec3 n = glm::cross(ab, ac);

    float det = dot(-_dir, n);
    float ood = 1.f / det;
    glm::vec3 ao = _o - _a;

    _t = ood * glm::dot(ao, n);
    if (_t < 0.f) {
        return false;
    }
    glm::vec3 e = glm::cross(-_dir, ao);
    _bary.y = glm::dot(ac, e) * ood;
    if (_bary.y < 0.f || _bary.y > 1.f) {
        return false;
    }
    _bary.z = - glm::dot(ab, e) * ood;
    if (_bary.z < 0.f || _bary.y + _bary.z > 1.f) {
        return false;
    }

    _bary.x = 1 - _bary.y - _bary.z;
    _winding = det;
    return true;
}

int 
Volume::getIdx(glm::ivec3 const &_pos) const
{
    assert(glm::max(_pos, dim_ - 1) == dim_ - 1);
    assert(glm::min(_pos, glm::ivec3(0)) == glm::ivec3(0));
    return _pos.z * dim_.x * dim_.y + _pos.y * dim_.x + _pos.x;
}

int&
Volume::operator[](glm::ivec3 const &_pos)
{
    return voxels_[getIdx(_pos)];
}

int const &
Volume::operator[](glm::ivec3 const &_pos) const
{
    return voxels_[getIdx(_pos)];
}

Volume::Volume(
    Mesh const & _mesh, 
    glm::ivec3 const & _dim,
    glm::vec3 const & _lower, 
    glm::vec3 const & _upper)
{
    dim_ = _dim;
    lower_ = _lower;
    upper_ = _upper;
    voxels_.assign(dim_.x * dim_.y * dim_.z, 0x00);
    BVH bvh(_mesh);
    const glm::vec3 diag{ _upper - _lower };
    const glm::vec3 delta{ diag / static_cast<glm::vec3>(dim_) };
    voxel_dim_ = delta;

    // Ray advances a little step further after hit a triangle. Not very robust.
    const float eps = 0.00001f * diag.z;

    for (int x = 0; x < dim_.x; ++x) {
        for (int y = 0; y < dim_.y; ++y) {
            bool inside = false;
            glm::vec3 origin{ _lower + delta * glm::vec3(x + 0.5f, y + 0.5f, 0.f) };
            glm::vec3 dir{ 0.f, 0.f, 1.f };
            Ray ray{origin, dir};
            int last_tri_idx = -1;
            bool hit = true;
            while (hit) {
                float t, winding;
                glm::vec3 bary;
                int tri_idx;
                hit = bvh.intersect(ray, t, bary, tri_idx, winding);
                if (hit) {
                    if (inside) {
                        const float z_hit = (ray.endAt(t).z - _lower.z) / delta.z;
                        int z_begin = int(std::floor((ray.origin_.z - _lower.z) / delta.z + 0.5f));
                        int z_end = std::min(int(std::floor(z_hit + 0.5f)), dim_.z - 1);
                        for (int z = z_begin; z < z_end; ++z) {
                            (*this)[glm::ivec3(x, y, z)] = 0x01;
                        }
                    }
                    inside = !inside;
                    if (tri_idx == last_tri_idx) {
                        std::cerr << "Error: self intersection "
                            << "(probably epsilon is too small)"
                            << std::endl;
                    }
                    last_tri_idx = tri_idx;
                    ray.advance(t + eps);
                } // endif: hit
            } // end while: ray reaches the end.
        } // loop over _dim.y
    } // loop over _dim.x
}

bool 
Volume::isOnSurface(glm::ivec3 const & _pos, float &_dist) const
{

    bool location = (*this)[_pos] != 0; // true ==> inside, false ==> outside
    float min_dist = kInfF;

    glm::ivec3 i_min = glm::max(_pos - 1, glm::ivec3(0));
    glm::ivec3 i_max = glm::min(_pos + 1, dim_ - 1);
    
    for (int z = i_min.z; z <= i_max.z; ++z) {
        for (int y = i_min.y; y <= i_max.y; ++y) {
            for (int x = i_min.x; x <= i_max.x; ++x) {
                bool location_neighbor = (*this)[glm::ivec3(x,y,z)] != 0;
                if (location != location_neighbor) {
                    glm::vec3 d = static_cast<glm::vec3>(_pos - glm::ivec3(x,y,z));
                    min_dist = std::min(glm::length(d) * 0.5f, min_dist); // distance to the "real" surface
                }
            }
        }
    }
    _dist = min_dist;
    return _dist < kInfF;
}

struct FMMNode
{
    glm::ivec3 p_;
    float d_;
    glm::ivec3 sp_; // source point
    bool operator < (const FMMNode& _fn) const 
    {
        return d_ > _fn.d_;
    }
};

int
SDF::getIdx(glm::ivec3 const &_pos) const
{
    assert(glm::max(_pos, dim_ - 1) == dim_ - 1);
    assert(glm::min(_pos, glm::ivec3(0)) == glm::ivec3(0));
    return _pos.z * dim_.x * dim_.y + _pos.y * dim_.x + _pos.x;
}

float& 
SDF::operator[] (glm::ivec3 const & _pos)
{
    return dists_[getIdx(_pos)];
}

float const & 
SDF::operator[] (glm::ivec3 const & _pos) const
{
    return dists_[getIdx(_pos)];
}

glm::vec3
SDF::getNormal(glm::ivec3 const & _pos) const
{
    glm::ivec3 left  = glm::max(_pos - glm::ivec3(1, 0, 0), glm::ivec3(0));
    glm::ivec3 right = glm::min(_pos + glm::ivec3(1, 0, 0), dim_ - 1);
    glm::ivec3 down  = glm::max(_pos - glm::ivec3(0, 1, 0), glm::ivec3(0));
    glm::ivec3 up    = glm::min(_pos + glm::ivec3(0, 1, 0), dim_ - 1);
    glm::ivec3 back  = glm::max(_pos - glm::ivec3(0, 0, 1), glm::ivec3(0));
    glm::ivec3 front = glm::min(_pos + glm::ivec3(0, 0, 1), dim_ - 1);

    glm::vec3 n{ (*this)[right] - (*this)[left], (*this)[up] - (*this)[down], (*this)[front] - (*this)[back] };
    return glm::length(n) == 0.f ? n : glm::normalize(n);
}

SDF::SDF(Volume const &_vol)
{
    dim_ = _vol.dim_;
    const float scale = 1.f / max3(dim_.x, dim_.y, dim_.z);
    dists_.resize(dim_.x * dim_.y * dim_.z);
    std::vector<FMMNode> queue;

    // find surface points 
    for (int z = 0; z < dim_.z; ++z) {
        for (int y = 0; y < dim_.y; ++y) {
            for (int x = 0; x < dim_.x; ++x) {
                glm::ivec3 pos(x, y, z);
                float dist = kInfF;
                if (_vol.isOnSurface(pos, dist)) {
                    FMMNode fn = {glm::ivec3(x,y,z), dist, glm::ivec3(x,y,z)};
                    queue.push_back(fn);
                }
                (*this)[pos] = kInfF;
            }
        }
    }

    if (queue.empty()) return;

    std::make_heap(queue.begin(), queue.end());
    while (!queue.empty()) {
        std::pop_heap(queue.begin(), queue.end());
        FMMNode fn = queue.back();
        queue.pop_back();
        // freeze node if not yet frozen
        if ((*this)[fn.p_] == kInfF) {
            (*this)[fn.p_] = fn.d_;

            glm::ivec3 i_min = glm::max(fn.p_ - 1, glm::ivec3(0));
            glm::ivec3 i_max = glm::min(fn.p_ + 1, dim_ - 1);

            for (int z = i_min.z; z <= i_max.z; ++z) {
                for (int y = i_min.y; y <= i_max.y; ++y) {
                    for (int x = i_min.x; x <= i_max.x; ++x) {
                        glm::ivec3 pos(x, y, z);
                        if (fn.p_ != pos && (*this)[pos] == kInfF) {
                            glm::vec3 dp = pos - fn.sp_;
                            float d = glm::length(dp) + (*this)[fn.sp_];
                            
                            assert(d > 0.f);
                            
                            FMMNode neighbor = { pos, d, fn.sp_ };
                            queue.push_back(neighbor);
                            std::push_heap(queue.begin(), queue.end());
                        } // endif: fronzen neighbor
                    } // end for: x 
                } // end for: y
            } // end for: z
        } // end if: node frozen
    } // end while: queue.empty()

    for (int z = 0; z < dim_.z; ++z) {
        for (int y = 0; y < dim_.y; ++y) {
            for (int x = 0; x < dim_.x; ++x) {
                glm::ivec3 pos(x, y, z);
                (*this)[pos] *= (_vol[pos] == 1 ? -1.f : 1.f) * scale;
            } // end for: x
        } // end for: y
    } // end for: z

    // auto result = std::max_element(dists_.begin(), dists_.end());
    // max_dist_ = dists_.at(result - dists_.begin());
    // std::cout << "max dist = " << max_dist_ << std::endl;
}
