#pragma once
#include <vector>
#include "aabb.h"

struct Mesh;

/*
* A BVH node.
*
* At any time, the following must hold:
* - left, right    must either both be -1, or must both point to valid child node indices.
* - triangle_idx   must always point to a valid index into triangle_soup.
* - num_triangles  must always be greater than 0.
*/

struct BVHNode
{
    AABB aabb_;
    int left_ = -1;
    int right_ = -1;
    int triangle_idx_ = -1;
    int num_triangles_ = 0;
};

struct BVH
{
    BVH(Mesh const & _mesh);
    void build(int _node_idx, int _first_triangle_idx, int _num_triangles, int _depth);
    
    // _ray and BVH must be in the same coordinate system
    // _t: travel distance of the ray to the hit point
    // _bary: barycentric coordinates of the hit point
    // _tri_idx: index of the nearest triangle to the hit point
    //_ccw: winding of the triangle, true if counter clockwise
    bool intersect(
        Ray const& _ray, 
        float &_t, 
        glm::vec3 & _bary, 
        int & _tri_idx, 
        float & _winding) const;

    const Mesh & mesh_;
    std::vector<BVHNode> nodes_;
    std::vector<int> triangle_indices_;

};
