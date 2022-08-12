#pragma once

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <vector>

struct BVH;

struct Mesh {
    Mesh(const std::string &);
    Mesh(const Mesh &_m);
    void calcNormals();
    void normalize(); // center the mesh, and scale it into the unit cube
    void translate(const glm::vec3&);
    void scale(glm::vec3 const &_s);
    void setColor(const glm::vec3&);

    GLsizei getNumVert() const { return static_cast<GLsizei>(positions_.size()); }
    GLsizei getNumIndices() const { return static_cast<GLsizei>(indices_.size()); }
    GLsizei getNumTriangles() const { return getNumIndices() / 3; }
    
    GLsizei getSizePositions() const { return sizeof(glm::vec3) * getNumVert(); }
    GLsizei getSizeNormals() const { return sizeof(glm::vec3) * getNumVert(); }
    GLsizei getSizeIndices() const { return sizeof(GLuint) * getNumIndices(); }

    static GLsizei getAttrSizePosition() { return 3; }
    static GLsizei getAttrSizeNormal() { return 3; }
    static GLsizei getAttrSizeColor() { return 3; }

    void printTriPos(int _tri_idx) const;
    void printTriIdx(int _tri_idx) const;

    std::vector<glm::vec3> positions_;
    std::vector<glm::vec3> normals_;
    std::vector<GLuint> indices_;
    std::vector<glm::vec2> uvs_;
    std::vector<glm::vec3> colors_;
    glm::vec3 lower_;
    glm::vec3 upper_;
};

struct BVHMesh {
    BVHMesh(BVH & _bvh);
    GLsizei getNumVert() const { return static_cast<GLsizei>(positions_.size()); }
    GLsizei getNumIndices() const { return static_cast<GLsizei>(indices_.size()); }
    GLsizei getSizePositions() const { return sizeof(glm::vec3) * getNumVert(); }
    GLsizei getSizeDepths() const { return sizeof(GLubyte) * getNumVert(); }
    GLsizei getSizeIndices() const { return sizeof(GLuint) * getNumIndices(); }
    GLsizei getAttrSizePosition() { return 3; }
    GLsizei getAttrSizeDepth() { return 1; }

    std::vector<glm::vec3> positions_;
    std::vector<GLubyte> depths_;
    std::vector<GLuint> indices_;
};

// plane equation: ax + by + cz + d = 0 ==> dot(n, p) + d = 0
class PlaneMesh
{
public:
    PlaneMesh() {}
    PlaneMesh(glm::vec4 _v, float _tex_size = 2.f) { init(_v.x, _v.y, _v.z, _v.w, _tex_size); }
    PlaneMesh(float _a, float _b, float _c, float _d, float _tex_size = 2.f) { init(_a, _b, _c, _d, _tex_size); }
    void init(float _a, float _b, float _c, float _d, float _tex_size = 2.f);

    static GLsizei getNumVert() { return (plane_dim_ + 1) * (plane_dim_ + 1); }
    static GLsizei getNumIndices() { return plane_dim_ * plane_dim_ * 6; }

    static GLsizei getSizePositions() { return getNumVert() * sizeof(glm::vec3); }
    static GLsizei getSizeNormals() { return getNumVert() * sizeof(glm::vec3); }
    static GLsizei getSizeTexUVs() { return getNumVert() * sizeof(glm::vec2); }
    static GLsizei getSizeIndices() { return getNumIndices() * sizeof(GLuint); }

    static GLsizei getAttrSizePosition() { return 3; }
    static GLsizei getAttrSizeNormal() { return 3; }
    static GLsizei getAttrSizeTexUV() { return 2; }

    ~PlaneMesh() {}

    std::vector<glm::vec3> positions_;
    std::vector<glm::vec3> normals_;
    std::vector<glm::vec2> tex_uvs_;
    std::vector<GLuint> indices_;

    static const int plane_dim_ = 8; // must be even
};

