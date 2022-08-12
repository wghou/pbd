#include "aabb.h"

Ray::Ray(glm::vec3 const & _origin, glm::vec3 const & _dir) : 
    origin_(_origin), dir_(_dir)
{
    dir_ = glm::normalize(dir_);
    dir_inv_ = 1.f / dir_;
}

void
Ray::advance(float _t)
{
    origin_ += _t * dir_;
}

glm::vec3 
Ray::endAt(float _t) const
{
    return origin_ + _t * dir_;
}

void
AABB::extend(glm::vec3 const & _vertex)
{
    min_ = glm::min(min_, _vertex - glm::vec3(kEpsilonF));
    max_ = glm::max(max_, _vertex + glm::vec3(kEpsilonF));
}

bool
AABB::intersect(const Ray &_ray, float &_t_min, float &_t_max) const
{
    glm::vec3 t_1 = multiplyByVec3Componentwise(min_ - _ray.origin_, _ray.dir_inv_);
    glm::vec3 t_2 = multiplyByVec3Componentwise(max_ - _ray.origin_, _ray.dir_inv_);

    glm::vec3 t_min2 = glm::min(t_1, t_2);
    glm::vec3 t_max2 = glm::max(t_1, t_2);

    _t_min = glm::max(glm::max(t_min2.x, t_min2.y), glm::max(t_min2.z, _t_min));
    _t_max = glm::min(glm::min(t_max2.x, t_max2.y), glm::min(t_max2.z, _t_max));

    return _t_min <= _t_max;
}

bool
AABB::contains(AABB const& _aabb) const
{
    glm::vec3 diff_min = _aabb.min_ - min_;
    glm::vec3 diff_max = max_ - _aabb.max_;
    return (glm::abs(diff_min) == diff_min) && (glm::abs(diff_max) == diff_max);
}

bool
AABB::isEmpty() const
{
    glm::vec3 diff = max_ - min_;
    // max_[i] >= min_[i], \forall i \in {0, 1, 2} ==> true
    return glm::abs(diff) == diff;
}