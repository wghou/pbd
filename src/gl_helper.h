#pragma once
#include <GL/glew.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <glm/glm.hpp>

#define TIMING_GL(expr, event_name) do {glFinish(); Timer t; t.start(); expr; glFinish(); t.stop(); std::cout << event_name << " took " << t.getElapsedMilliseconds() << " ms." << std::endl; } while (0)

#define RELEASE_GL_VAO(vao) do {if (glIsVertexArray(vao)) {glDeleteVertexArrays(1, &vao);} } while(0)
#define RELEASE_GL_BOs(bo, n) do {if (glIsBuffer(bo[0])) {glDeleteBuffers(n, bo);}} while(0)
#define RELEASE_GL_BO(bo) do {if (glIsBuffer(bo)) {glDeleteBuffers(1, &bo);}} while(0)
#define RELEASE_GL_FBOs(fbo, n) do {if (glIsFramebuffer(fbo[0])) {glDeleteFramebuffers(n, fbo);}} while(0)
#define RELEASE_GL_FBO(fbo) do {if (glIsFramebuffer(fbo)) {glDeleteFramebuffers(1, &fbo);}} while(0)
#define RELEASE_GL_TEX(tex) do {if (glIsTexture(tex)) {glDeleteTextures(1, &tex);}} while(0)
#define RELEASE_GL_TEXs(tex, n) do {if (glIsTexture(tex[0])) {glDeleteTextures(n, tex);}} while(0)
#define RELEASE_GL_Sampler(sampler) do {if (glIsSampler(sampler)) {glDeleteSamplers(1, &sampler);}} while(0)
#define RELEASE_GL_Samplers(sampler, n) do {if (glIsSampler(sampler[0])) {glDeleteSamplers(n, sampler);}} while(0)
#define BUF_OFFSET_GL(idx) (static_cast<char*>(0) + (idx))

struct Mesh;

std::string getGLErrorStr(GLenum _error);
void checkGLError(const std::string &_location);
void checkFBO(const std::string &_fbo_name);

class Camera
{
public:
    Camera();
    ~Camera() {}
    void init(glm::vec3 const &_e, glm::vec3 const &_lookAt, glm::vec3 const &_up);
    glm::vec3 const & getPosition() { return e_; }
    glm::vec3 const & getU() { return u_; }
    glm::vec3 const & getW() { return w_; }
    const glm::mat4& getViewMat() { return view_mat_; }
    const glm::mat4& getViewInv() { return view_mat_inv_; }
    // _dir is one of the axes {'u', 'v', 'w'} in camera space
    void setVelocity(char _dir, float _velocity);
    // _axis in world space
    void rotate(float _angle, glm::vec3 const &_axis);
    void updateViewMat(float _dT = 0.f);
private:
    glm::vec3 e_; // position
    glm::vec3 u_; // horizontal
    glm::vec3 v_; // vertical
    glm::vec3 w_; // negative lookAt
    glm::vec3 velocity_; // velocity in camera space
    glm::mat4 view_mat_;
    glm::mat4 view_mat_inv_; // inverse view matrix
};

// OpenGL program object
class GLProgram 
{
public:
    GLProgram();
    ~GLProgram();
    
    void init(
        std::unordered_map<std::string, GLenum> _shader_path_types, 
        std::vector<std::string> _uniform_names,
        std::string _prog_name = "");

    void refresh();
    void use();
    
    GLuint operator[] (const GLchar *_uniform);
    
private:
    GLuint prog_obj_;
    std::string prog_name_;
    std::unordered_map<std::string, GLenum> shader_path_types_;
    std::vector<GLuint> shaders_;
    std::unordered_map<std::string, GLuint> uniform_name_locs_;
    
    void attachShader(std::string const &_file_path, GLenum _shader_type);
    void link();
};

