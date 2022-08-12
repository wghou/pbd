#include <algorithm>

#include <iostream>
#include "misc.h"

#include "mesh.h"
#include "bvh.h"


// The maximum number of triangles allowed in one leaf node.
const int kMaxBVHLeafSize = 6;

BVH::BVH(Mesh const & _mesh) : 
    mesh_(_mesh), 
    nodes_(1), 
    triangle_indices_(_mesh.getNumTriangles())
{
    nodes_.reserve(_mesh.getNumTriangles() * 2);
    for (int i = 0; i < _mesh.getNumTriangles(); ++i) {
        triangle_indices_[i] = i;
    }
    build(0, 0, _mesh.getNumTriangles(), 0);
}

void
BVH::build(int _node_idx, int _first_triangle_idx, int _num_triangles, int _depth)
{
    assert(_node_idx >= 0);
    assert(_node_idx < static_cast<int>(nodes_.size()));
    if (_node_idx == 0 && _num_triangles == 0) return;
    assert(_num_triangles > 0);
    assert(_first_triangle_idx + _num_triangles <= mesh_.getNumTriangles());

    BVHNode &node = nodes_[_node_idx];
    int axis = _depth % 3;

    if (_num_triangles <= kMaxBVHLeafSize) {
        node.left_ = -1;
        node.right_ = -1;
        node.triangle_idx_ = _first_triangle_idx;
        node.num_triangles_ = _num_triangles;
        for (int i = 0; i < _num_triangles; ++i) {
            int tri_idx = triangle_indices_[_first_triangle_idx + i];
            for (int j = 0; j < 3; ++j) {
                int vert_idx = mesh_.indices_[tri_idx * 3 + j];
                node.aabb_.extend(mesh_.positions_[vert_idx]);
            }
        }
    }
    else {
        std::nth_element(
            triangle_indices_.begin() + _first_triangle_idx,
            triangle_indices_.begin() + _first_triangle_idx + _num_triangles / 2,
            triangle_indices_.begin() + _first_triangle_idx + _num_triangles,
            [&](int l, int r) -> bool {
                auto &v = mesh_.positions_;
                float min_l, max_l, min_r, max_r;
                min_l = min_r =  kInfF;
                max_l = max_r = -kInfF;
                for (int i = 0; i < 3; ++i) {
                    int idx_l = mesh_.indices_[l * 3 + i];
                    int idx_r = mesh_.indices_[r * 3 + i];
                    min_l = std::min(min_l, v[idx_l][axis]);
                    max_l = std::max(max_l, v[idx_l][axis]);
                    min_r = std::min(min_r, v[idx_r][axis]);
                    max_r = std::max(max_r, v[idx_r][axis]);
                }
                return min_l + max_l < min_r + max_r;
            }
        );
        int const num_nodes = static_cast<int>(nodes_.size());
        node.left_ = num_nodes + 0;
        node.right_ = num_nodes + 1;
        nodes_.push_back(BVHNode());
        nodes_.push_back(BVHNode());
        node.triangle_idx_ = _first_triangle_idx;
        node.num_triangles_ = _num_triangles;
        
        int n_tri = _num_triangles / 2;
        build(node.left_, _first_triangle_idx, n_tri, _depth + 1);
        build(node.right_, _first_triangle_idx + n_tri, _num_triangles - n_tri, _depth + 1);

        BVHNode &nl = nodes_[node.left_];
        BVHNode &nr = nodes_[node.right_];
        node.aabb_.min_ = glm::min(nl.aabb_.min_, nr.aabb_.min_);
        node.aabb_.max_ = glm::max(nl.aabb_.max_, nr.aabb_.max_);
    }
}

bool
BVH::intersect(
    Ray const& _ray, 
    float &_t, 
    glm::vec3 & _bary, 
    int & _tri_idx, 
    float & _winding) const
{
    bool hit = false;
    int stack[64]; // worst case: depth of the BVH ==> largest number of triangles: pow(2, 64)
    int stack_size = 0;

    _t = kInfF;
    _bary = glm::vec3(0.f);
    _tri_idx = -1;

    {
        float t_min = 0.f; // will only be used once
        if (nodes_[0].aabb_.intersect(_ray, t_min, _t)) {
            stack[stack_size++] = 0; // root index = 0
        }
    }

    while (stack_size > 0) {
        const BVHNode & node = nodes_[stack[--stack_size]];
        if (node.left_ < 0) { // node is a leaf,
            for (int i = 0; i < node.num_triangles_; ++i) {
                int idx = triangle_indices_[node.triangle_idx_ + i];
                assert(idx >= 0);
                float dist;
                int vert_idx_0 = mesh_.indices_[idx * 3 + 0];
                int vert_idx_1 = mesh_.indices_[idx * 3 + 1];
                int vert_idx_2 = mesh_.indices_[idx * 3 + 2];
                glm::vec3 const & a = mesh_.positions_[vert_idx_0];
                glm::vec3 const & b = mesh_.positions_[vert_idx_1];
                glm::vec3 const & c = mesh_.positions_[vert_idx_2];
                glm::vec3 bary;
                float winding;
                if (intersectRayTriangleTwoSided(_ray.origin_, _ray.dir_,
                        a, b, c,
                        dist, bary, winding)) {
                    hit = true;
                    if (dist < _t || _tri_idx == -1) {
                        _t = dist;
                        _bary = bary;
                        _tri_idx = idx;
                        _winding = winding;
                    } // endif: nearest hit found.
                } // endif: ray-triangle intersection
            } // end for: process all the triangles in the leaf 
        } // endif: leaf node is reached. 
        else {
            float t_min_l = 0.f;
            float t_max_l = _t;
            float t_min_r = 0.f;
            float t_max_r = _t;

            bool il = nodes_[node.left_].aabb_.intersect(_ray, t_min_l, t_max_l);
            bool ir = nodes_[node.right_].aabb_.intersect(_ray, t_min_r, t_max_r);

            if (!il && !ir) {}
            else if (il ^ ir) {
                stack[stack_size++] = il ? node.left_ : node.right_;
            }
            else {
                // put the nearest node on the top of the stack
                if (t_min_l < t_min_r) { 
                    stack[stack_size++] = node.right_;
                    stack[stack_size++] = node.left_;
                }
                else {
                    stack[stack_size++] = node.left_;
                    stack[stack_size++] = node.right_;
                }
            }
        } // end else: ray children intersection 
    } // end while
    return hit;
}