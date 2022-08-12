#pragma once
#include "cg_math.h"

struct Ray
{
    glm::vec3 origin_;
    glm::vec3 dir_;
    glm::vec3 dir_inv_; // 1.f / dir_

    Ray() : origin_(kInfF), dir_(glm::vec3(0.f, 1.f, 0.f)) { dir_inv_ = 1.f / dir_; }
    Ray(glm::vec3 const & _origin, glm::vec3 const & _dir);
    void advance(float _t);
    glm::vec3 endAt(float _t) const;
};

struct AABB
{
    glm::vec3 min_ = glm::vec3(kInfF);
    glm::vec3 max_ = glm::vec3(-kInfF);

    void extend(glm::vec3 const & _vertex);

    // _t_min and t_max specify a segment of the ray to be tested, 
    // and will return the segment inside AABB if there is an intersection.
    bool intersect(const Ray &_ray, float &_t_min, float &_t_max) const;

    bool contains(AABB const& _aabb) const;
    bool isEmpty() const;
};

