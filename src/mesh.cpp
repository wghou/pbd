#include <iostream>
#include "misc.h"
#include <string>
#include <glm/ext.hpp>
#include <rply.h>
#include <libobj.hpp>

#include "mesh.h"
#include "bvh.h"
#include "aabb.h"

// temp memory for parsing ply files
static bool sNoError;
static std::vector<GLuint> sFaces;
static std::vector<glm::vec3> sPositions;
static std::vector<glm::vec3> sNormals;
static std::vector<glm::vec3> sObjNormals;

static void 
initTempMem()
{
    sNoError = true;
    sFaces.clear();
    sPositions.clear();
    sNormals.clear();
    sObjNormals.clear();
}

static void
ply_error_callback(p_ply _ply, const char *_message)
{
    std::cerr << _message << std::endl;
    sNoError = false;
}

static int 
position_callback(p_ply_argument _arg)
{
    long dim_index;
    ply_get_argument_user_data(_arg, NULL, &dim_index);
    float value = static_cast<float>(ply_get_argument_value(_arg));
    if (dim_index == 0) {
        sPositions.push_back(glm::vec3(value));
    }
    else {
        sPositions.back()[dim_index] = value;
    }
    return 1;
}

static int
normal_callback(p_ply_argument _arg)
{
    long dim_index;
    ply_get_argument_user_data(_arg, NULL, &dim_index);
    float value = static_cast<float>(ply_get_argument_value(_arg));
    if (dim_index == 0) {
        sNormals.push_back(glm::vec3(value));
    }
    else {
        sNormals.back()[dim_index] = value;
    }
    return 1;
}

static int
face_callback(p_ply_argument _arg)
{
    long length, value_index;
    ply_get_argument_property(_arg, NULL, &length, &value_index);
    if (value_index >= 0) {
        sFaces.push_back(static_cast<GLuint>(ply_get_argument_value(_arg)));
    }
    long last_index = length - 1;
    if (value_index == last_index) {
        size_t size = sFaces.size();
        switch (length) {
        case 3:
            break;
        case 4:
            // 0-1-2-3 (ccw) ==> 0-1-2--3-0-2
            sFaces.push_back(sFaces[size - 4]);
            sFaces.push_back(sFaces[size - 2]);
            break;
        default:
            std::cerr << "unrecognized polygon" << std::endl;
            sNoError = false;
            return 0;
        }
    }
    return 1;
}

static bool 
loadMeshFromPly(const std::string &_path)
{
    initTempMem();
    std::cout << "loading ply file: " << _path << std::endl;
    p_ply ply = ply_open(_path.c_str(), ply_error_callback, 0, NULL);
    if (!ply) return false;
    if (!ply_read_header(ply)) {
        ply_close(ply);
        sNoError = false;
        return false;
    }
    ply_set_read_cb(ply, "vertex", "x", position_callback, NULL, 0);
    ply_set_read_cb(ply, "vertex", "y", position_callback, NULL, 1);
    ply_set_read_cb(ply, "vertex", "z", position_callback, NULL, 2);
    ply_set_read_cb(ply, "vertex", "nx", normal_callback, NULL, 0);
    ply_set_read_cb(ply, "vertex", "ny", normal_callback, NULL, 1);
    ply_set_read_cb(ply, "vertex", "nz", normal_callback, NULL, 2);
    ply_set_read_cb(ply, "face", "vertex_indices", face_callback, NULL, 0);
    if (!ply_read(ply)) {
        sNoError = false;
    }
    ply_close(ply);
    if (sPositions.size() == 0 || sFaces.size() == 0) {
        std::cerr << "empty mesh. " << std::endl;
        sNoError = false;
    }
    if (sFaces.size() % 3 != 0) {
        std::cerr << "number of faces is abnormal. " << std::endl;
        sNoError = false;
    }
    if (sNormals.size() != 0 && sNormals.size() != sPositions.size()) {
        std::cerr << "number of normals is abnormal." << std::endl;
        sNoError = false;
    }
    return sNoError;
}

static void 
obj_vertex_callback(obj::float_type x, obj::float_type y, obj::float_type z)
{
    sPositions.push_back(glm::vec3(x, y, z));
}

static void
obj_normal_callback(obj::float_type x, obj::float_type y, obj::float_type z)
{
    sObjNormals.push_back(glm::vec3(x, y, z));
}

static void
obj_tri_face_vert_idx_normal_idx_callback(const obj::index_2_tuple_type &_v1, const obj::index_2_tuple_type &_v2, const obj::index_2_tuple_type &_v3)
{
    sFaces.push_back(static_cast<GLuint>(std::get<0>(_v1)) - 1);
    sNormals.push_back(sObjNormals[std::get<1>(_v1) - 1]);

    sFaces.push_back(static_cast<GLuint>(std::get<0>(_v2)) - 1);
    sNormals.push_back(sObjNormals[std::get<1>(_v2) - 1]);
    
    sFaces.push_back(static_cast<GLuint>(std::get<0>(_v3)) - 1);
    sNormals.push_back(sObjNormals[std::get<1>(_v3) - 1]);
}

static void
obj_tri_face_vert_idx_tex_uv_normal_idx_callback(const obj::index_3_tuple_type &_v1, const obj::index_3_tuple_type &_v2, const obj::index_3_tuple_type &_v3)
{
    sFaces.push_back(static_cast<GLuint>(std::get<0>(_v1)) - 1);
    sNormals.push_back(sObjNormals[std::get<2>(_v1) - 1]);

    sFaces.push_back(static_cast<GLuint>(std::get<0>(_v2)) - 1);
    sNormals.push_back(sObjNormals[std::get<2>(_v2) - 1]);

    sFaces.push_back(static_cast<GLuint>(std::get<0>(_v3)) - 1);
    sNormals.push_back(sObjNormals[std::get<2>(_v3) - 1]);
}


static void
obj_error_callback(size_t _line_num, std::string const& _error_msg)
{
    sNoError = false;
    std::cout << "line " << _line_num << " " << _error_msg << std::endl;
}

static bool
loadMeshFromObj(const std::string &_path)
{
    initTempMem();
    std::cout << "loading obj file: " << _path << std::endl;
    obj::obj_parser obj_parser{ obj::obj_parser::triangulate_faces };
    obj_parser.geometric_vertex_callback(obj_vertex_callback);
    obj_parser.vertex_normal_callback(obj_normal_callback);
    obj_parser.face_callbacks(
        NULL, NULL, obj_tri_face_vert_idx_normal_idx_callback, obj_tri_face_vert_idx_tex_uv_normal_idx_callback,
        NULL, NULL, NULL, NULL,
        NULL, NULL, NULL, 
        NULL, NULL, NULL, 
        NULL, NULL, NULL, 
        NULL, NULL, NULL);
    obj_parser.error_callback(obj_error_callback);
    bool parser_ret = obj_parser.parse(_path);
    std::cout << "parser ret = " << parser_ret << ", sNoError = " << sNoError << std::endl;
    sNoError = parser_ret && sNoError;
    return sNoError;
}

Mesh::Mesh(const std::string &_path)
{
    std::string suffix = _path.substr(_path.rfind(".") + 1);
    if (suffix == "ply") {
        if (!loadMeshFromPly(_path)) {
            std::cerr << "load ply file " << _path << " failed." << std::endl;
        }
    }
    else if (suffix == "obj") {
        if (!loadMeshFromObj(_path)) {
            std::cerr << "load obj file " << _path << " failed." << std::endl;
        }
    }
    else {
        sNoError = false;
        std::cerr << suffix << " format is not supported." << std::endl;
    }

    if (!sNoError) {
        exit(1);
    }

    positions_ = std::move(sPositions);
    std::cout << "num vert: " << positions_.size() << std::endl;
    normals_ = std::move(sNormals);
    indices_ = std::move(sFaces);
    std::cout << "num tri: " << indices_.size() / 3 << std::endl;
    initTempMem(); // reset temp memory
    colors_.resize(getNumVert(), glm::vec3(0.f, 0.f, 1.f));
    calcNormals();
    normalize();
}

Mesh::Mesh(const Mesh &_m)
{
    positions_ = _m.positions_;
    normals_ = _m.normals_;
    indices_ = _m.indices_;
    uvs_ = _m.uvs_;
    colors_ = _m.colors_;
    lower_ = _m.lower_;
    upper_ = _m.upper_;
}

void
Mesh::calcNormals()
{
    // if (positions_.size() == normals_.size()) return;
    normals_.resize(positions_.size(), glm::vec3(0.f));
    for (size_t i = 0; i < indices_.size(); i += 3) {
        GLuint idx0 = indices_[i + 0];
        GLuint idx1 = indices_[i + 1];
        GLuint idx2 = indices_[i + 2];
        glm::vec3 v01 = positions_[idx1] - positions_[idx0];
        glm::vec3 v02 = positions_[idx2] - positions_[idx0];
        glm::vec3 normal = glm::normalize(glm::cross(v01, v02));
        normals_[idx0] += normal;
        normals_[idx1] += normal;
        normals_[idx2] += normal;
    }
    for (auto& normal : normals_) {
        normal = glm::normalize(normal);
    }
}

void
Mesh::normalize()
{
    lower_ = positions_.front();
    upper_ = positions_.front();
    for (auto &pos : positions_) {
        lower_ = glm::min(lower_, pos);
        upper_ = glm::max(upper_, pos);
    }
    glm::vec3 diag = upper_ - lower_;
    float scale = 1.f / glm::max(glm::max(diag.x, diag.y), diag.z);

    lower_ *= scale;
    upper_ *= scale;
    glm::vec3 translation = -lower_; //-(lower_ + upper_) / 2.f;
    for (auto &pos : positions_) {
        pos *= scale;
        pos += translation;
    }
    lower_ += translation;
    upper_ += translation;
}

void
Mesh::translate(const glm::vec3& _translation)
{
    for (auto &pos : positions_) {
        pos += _translation;
    }
    lower_ += _translation;
    upper_ += _translation;
}

void
Mesh::scale(glm::vec3 const& _s)
{
    glm::vec3 t = -lower_;
    upper_ = (upper_ + t) * _s - t;
    for (auto &pos : positions_) {
        pos = (pos + t) * _s - t;
    }
    
}

void
Mesh::setColor(const glm::vec3& _color)
{
    for (glm::vec3 & c : colors_) {
        c = _color;
    }
}

void
Mesh::printTriIdx(int _tri_idx) const
{
    int vert_idx_0 = indices_[_tri_idx * 3 + 0];
    int vert_idx_1 = indices_[_tri_idx * 3 + 1];
    int vert_idx_2 = indices_[_tri_idx * 3 + 2];
    std::cout << "tri " << _tri_idx << " [" << glm::vec3(vert_idx_0, vert_idx_1, vert_idx_2) << "] " << std::flush;
}

void
Mesh::printTriPos(int _tri_idx) const
{
    int vert_idx_0 = indices_[_tri_idx * 3 + 0];
    int vert_idx_1 = indices_[_tri_idx * 3 + 1];
    int vert_idx_2 = indices_[_tri_idx * 3 + 2];
    glm::vec3 const & a = positions_[vert_idx_0];
    glm::vec3 const & b = positions_[vert_idx_1];
    glm::vec3 const & c = positions_[vert_idx_2];
    std::cout << "tri " << _tri_idx << " [" << a << ", " << b << ", " << c << "] " << std::flush;
}


BVHMesh::BVHMesh(BVH & _bvh)
{
    positions_.reserve(_bvh.nodes_.size() * 8);
    depths_.reserve(_bvh.nodes_.size() * 8);
    indices_.reserve(_bvh.nodes_.size() * 24);
    std::vector<int> buf[2];
    // over reserved, can be smaller if necessary
    buf[0].reserve(_bvh.nodes_.size());
    buf[1].reserve(_bvh.nodes_.size());
    GLuint base_indices[24] = {
        0, 1, 2, 3, 4, 5, 6, 7,
        0, 5, 1, 4, 2, 7, 3, 6,
        0, 3, 1, 2, 4, 7, 5, 6
    };
    GLubyte depth = 0;
    int ping_idx = depth % 2;
    int pong_idx = (ping_idx + 1) % 2;
    buf[ping_idx].push_back(0);

    while (buf[ping_idx].size() > 0) {
        for (int i : buf[ping_idx]) {
            BVHNode const & bvh_node = _bvh.nodes_[i];
            int vert_offset = static_cast<int>(positions_.size());
            glm::vec3 min = bvh_node.aabb_.min_;
            glm::vec3 max = bvh_node.aabb_.max_;
            positions_.push_back(min); 
            positions_.push_back(glm::vec3(min.x, max.y, min.z));
            positions_.push_back(glm::vec3(max.x, max.y, min.z));
            positions_.push_back(glm::vec3(max.x, min.y, min.z));
            positions_.push_back(glm::vec3(min.x, max.y, max.z));
            positions_.push_back(glm::vec3(min.x, min.y, max.z));
            positions_.push_back(glm::vec3(max.x, min.y, max.z));
            positions_.push_back(max);
            depths_.insert(depths_.end(), 8, depth);
            for (GLuint i : base_indices) {
                indices_.push_back(vert_offset + i);
            }
            if (bvh_node.left_  != -1) buf[pong_idx].push_back(bvh_node.left_);
            if (bvh_node.right_ != -1) buf[pong_idx].push_back(bvh_node.right_);
        }
        buf[ping_idx].clear();
        ++depth;
        ping_idx = depth % 2;
        pong_idx = (ping_idx + 1) % 2;
    }
}

void
PlaneMesh::init(float _a, float _b, float _c, float _d, float _tex_size)
{
    glm::vec3 normal_dir = glm::vec3(_a, _b, _c);
    glm::vec3 n = glm::normalize(normal_dir);
    float d = _d / glm::length(normal_dir);
    glm::vec3 n_xz = glm::vec3(0.f, 1.f, 0.f);
    float angle = glm::acos(glm::dot(n_xz, n));
    glm::vec3 axis = glm::vec3(1.0, 0.0, 0.0);
    if (glm::sin(angle) != 0.f) {
        axis = glm::normalize(glm::cross(n_xz, n));
    }
    glm::mat3 rotation_mat = glm::mat3_cast(glm::angleAxis(angle, axis));
    glm::vec3 translation = -d * n;

    //glm::vec3 basis[2], origin;
    //origin = translation;
    //basis[0] = rotation_mat * glm::vec3(1.f, 0.f, 0.f);
    //basis[1] = rotation_mat * glm::vec3(0.f, 0.f, 1.f);

    const int idx_max = plane_dim_ / 2;
    positions_.resize((plane_dim_ + 1) * (plane_dim_ + 1));
    tex_uvs_.resize((plane_dim_ + 1) * (plane_dim_ + 1));
    normals_.resize((plane_dim_ + 1) * (plane_dim_ + 1));

    float edge_len = 1024.f;
    for (int z = -idx_max; z <= idx_max; ++z) {
        for (int x = -idx_max; x <= idx_max; ++x) {
            int p_idx = (z + idx_max) * (plane_dim_ + 1) + (x + idx_max);
            glm::vec3 p (x * edge_len, 0.f, z * edge_len);
            
            positions_[p_idx] = rotation_mat * p + translation;
            tex_uvs_[p_idx] = glm::vec2(p.x / _tex_size, p.z / _tex_size);
            // tex_uvs_[p_idx] = glm::vec2(p.x, p.z);
            normals_[p_idx] = n;
        }
    }

    indices_.resize(plane_dim_ * plane_dim_ * 6);
    for (int z = 0; z < plane_dim_; ++z) {
        for (int x = 0; x < plane_dim_; ++x) {
            // ccw, start from top-left (x: horizontal, z: vertical)
            int idx0 = z * (plane_dim_ + 1) + x;
            int idx1 = (z + 1) * (plane_dim_ + 1) + x;
            int idx2 = (z + 1) * (plane_dim_ + 1) + (x + 1);
            int idx3 = z * (plane_dim_ + 1) + (x + 1);

            int start = (z * plane_dim_ + x) * 6;
            indices_[start + 0] = idx0;
            indices_[start + 1] = idx1;
            indices_[start + 2] = idx2;
            indices_[start + 3] = idx0;
            indices_[start + 4] = idx2;
            indices_[start + 5] = idx3;
        }
    }
}

