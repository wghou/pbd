#include <iostream>
#include <fstream>
#include "gl_helper.h"
#include "cg_math.h"


Camera::Camera()
{
}

void
Camera::init(glm::vec3 const &_e, glm::vec3 const &_lookAt, glm::vec3 const &_up)
{
    e_ = _e;
    w_ = glm::normalize(-_lookAt);
    u_ = glm::normalize(glm::cross(_up, w_));
    v_ = glm::normalize(glm::cross(w_, u_));
    velocity_ = glm::vec3(0.f);
    updateViewMat();
}

void 
Camera::setVelocity(char _dir, float _velocity)
{
    int component = _dir - 'u';
    if (component > 2 || component < 0) {
        std::cerr << "set camera velocity failed: "
            << _dir << " is invalid." 
            << std::endl;
        return;
    }
    velocity_[component] = _velocity;
}

void
Camera::rotate(float _angle, glm::vec3 const &_axis)
{
    u_ = rotateVec3(u_, _angle, _axis);
    v_ = rotateVec3(v_, _angle, _axis);
    w_ = rotateVec3(w_, _angle, _axis);
}

void 
Camera::updateViewMat(float dT)
{
    e_ += dT * glm::vec3(view_mat_inv_ * glm::vec4(velocity_, 0.f));
    view_mat_[0] = glm::vec4(u_, -glm::dot(u_, e_));
    view_mat_[1] = glm::vec4(v_, -glm::dot(v_, e_));
    view_mat_[2] = glm::vec4(w_, -glm::dot(w_, e_));
    view_mat_[3] = glm::vec4(0.f, 0.f, 0.f, 1.f);
    view_mat_ = glm::transpose(view_mat_);
    view_mat_inv_ = glm::inverse(view_mat_);
}


GLProgram::GLProgram()
{
}

void
GLProgram::init(
    std::unordered_map<std::string, GLenum> _shader_path_types,
    std::vector<std::string> _uniform_names, 
    std::string _prog_name)
{
    prog_obj_ = glCreateProgram();
    prog_name_ = _prog_name;   
    shader_path_types_ = _shader_path_types;
    for (const auto &name : _uniform_names) {
        uniform_name_locs_[name] = -1;
    }
    
    refresh();
}

GLProgram::~GLProgram()
{
    glDeleteProgram(prog_obj_);
}

void
GLProgram::refresh()
{
    for (auto &shader : shaders_) {
        glDetachShader(prog_obj_, shader);
        glDeleteShader(shader);
    }
    shaders_.clear();
    for (auto &pair : shader_path_types_) {
        attachShader(pair.first, pair.second);
    }
    link();
}

void 
GLProgram::use()
{
    glUseProgram(prog_obj_);
}

void 
GLProgram::attachShader(std::string const &_file_path, GLenum _shader_type)
{
    std::ifstream is;
    is.open(_file_path, std::ifstream::binary | std::ios::ate);
    if (!is.is_open()) {
        std::cerr << _file_path << "not found." << std::endl;
        return;
    }
    size_t size = static_cast<size_t>(is.tellg());
    char *src_str = new char[size + 1];
    is.seekg(0, std::ios::beg);
    is.read(src_str, size);
    src_str[is.gcount()] = '\0';
    is.close();
    GLuint shader_obj = glCreateShader(_shader_type);
    glShaderSource(shader_obj, 1, &src_str, NULL);
    delete[] src_str;
    
    glCompileShader(shader_obj);
    GLint compile_result = 1;
    glGetShaderiv(shader_obj, GL_COMPILE_STATUS, &compile_result);
    
    if (compile_result == GL_FALSE) {
        GLint log_length;
        glGetShaderiv(shader_obj, GL_INFO_LOG_LENGTH, &log_length);
        GLchar* info_log_buffer = new GLchar[log_length];
        glGetShaderInfoLog(shader_obj, log_length, NULL, info_log_buffer);
        std::cerr << "shader compile failed (" << _file_path << "): \n"
                  << info_log_buffer << std::endl;
        delete [] info_log_buffer;
        glDeleteShader(shader_obj);
        return;
    }
    glAttachShader(prog_obj_, shader_obj);
    shaders_.push_back(shader_obj);
}

void
GLProgram::link()
{
    glLinkProgram(prog_obj_);
    GLint link_result = GL_TRUE;
    glGetProgramiv(prog_obj_, GL_LINK_STATUS, &link_result);
    if (link_result == GL_FALSE) {
        GLint log_length;
        glGetProgramiv(prog_obj_, GL_INFO_LOG_LENGTH, &log_length);
        GLchar* info_log_buffer = new GLchar[log_length];
        glGetProgramInfoLog(prog_obj_, log_length, NULL, info_log_buffer);
        std::cerr << "program link failed (" << prog_name_ << "): \n"
                  << info_log_buffer << std::endl;
        delete [] info_log_buffer;
    } else {
        for (auto &pair : uniform_name_locs_) {
            const GLchar* name = pair.first.c_str();
            uniform_name_locs_[pair.first] = glGetUniformLocation(prog_obj_, name);
        }
    }
}

GLuint 
GLProgram::operator[] (const GLchar *_uniform_name)
{
    GLuint location;
    try {
        location = uniform_name_locs_.at(std::string(_uniform_name));
    } catch (std::out_of_range) {
        location = -1;
        std::cerr << "uniform " << _uniform_name << " not found." << std::endl;
    }
    return location;
}


#define GL_ERROR(x) case (x): return std::string(#x);

std::string
getGLErrorStr(GLenum _error)
{
    switch (_error) {
        GL_ERROR(GL_INVALID_OPERATION);
        GL_ERROR(GL_INVALID_ENUM);
        GL_ERROR(GL_INVALID_VALUE);
        GL_ERROR(GL_OUT_OF_MEMORY);
        GL_ERROR(GL_FRAMEBUFFER_COMPLETE);   // return by glCheckFramebufferStatus()
        GL_ERROR(GL_INVALID_FRAMEBUFFER_OPERATION);
        GL_ERROR(GL_FRAMEBUFFER_UNDEFINED);
        GL_ERROR(GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT);
        GL_ERROR(GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT);
        GL_ERROR(GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER);
        GL_ERROR(GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER);
        GL_ERROR(GL_FRAMEBUFFER_UNSUPPORTED);
        GL_ERROR(GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE);
    default:
        return std::string("UNKOWN ERROR: ") + std::to_string(_error);
    }
}

void
checkGLError(const std::string &_location)
{
    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        std::cerr << getGLErrorStr(error) << " at " << _location << std::endl;
    }
    else {
        // std::cout << "No error detected at " << _location << std::endl;
    }
}

void
checkFBO(const std::string &_fbo_name)
{
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "Framebuffer " << _fbo_name << " is NOT complete" << std::endl;
    }
    else {
        //std::cout << "Framebuffer " << _fbo_name << " is complete" << std::endl;
    }
}











