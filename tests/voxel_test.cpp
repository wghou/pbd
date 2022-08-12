#include <GL/glew.h>
#include <iostream>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <glm/ext.hpp>

#include <cuda_helper.h>
#include <gui.h>
#include "voxelization/voxel_render.h"

static GLFWwindow* gWindow = nullptr;
int gWindowWidth, gWindowHeight;
double gCursorPosX, gCursorPosY;
const int gNumSamples = 4;
bool gMouseRightDown = false;
bool gPause = false;
bool gToggleGui = true;

float gRealDT;

static void error_callback(int error, const char* description)
{
    std::cerr << "Error: " << description << std::endl;
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
    if (action == GLFW_PRESS) {
        float cam_speed = kCameraSpeed;
        switch (mods) {
        case GLFW_MOD_SHIFT:
            cam_speed = kCameraHighSpeed;
            break;
        case GLFW_MOD_ALT:
            cam_speed = kCameraLowSpeed;
            break;
        default:
            break;
        }
        switch (key)
        {
        case GLFW_KEY_W:
            gCamera.setVelocity('w', -cam_speed);
            break;
        case GLFW_KEY_S:
            gCamera.setVelocity('w', cam_speed);
            break;
        case GLFW_KEY_A:
            gCamera.setVelocity('u', -cam_speed);
            break;
        case GLFW_KEY_D:
            gCamera.setVelocity('u', cam_speed);
            break;
        case GLFW_KEY_E:
            gCamera.setVelocity('v', cam_speed);
            break;
        case GLFW_KEY_Q:
            gCamera.setVelocity('v', -cam_speed);
            break;
        case GLFW_KEY_R: // Ctrl + R: reset camera
            if (mods == GLFW_MOD_CONTROL) gCamera.init(kCameraPos, kLookAt, kUp);
            break;
        case GLFW_KEY_H:
            gToggleGui = !gToggleGui;
            break;
        case GLFW_KEY_M:
            gToggleDrawMesh = !gToggleDrawMesh;
            break;
        case GLFW_KEY_LEFT:
            gSelectedVoxel[0] += 1;
            break;
        case GLFW_KEY_UP:
            gSelectedVoxel[1] += 1;
            break;
        case GLFW_KEY_PAGE_UP:
            gSelectedVoxel[2] += 1;
            break;
        case GLFW_KEY_RIGHT:
            gSelectedVoxel[0] -= 1;
            break;
        case GLFW_KEY_DOWN:
            gSelectedVoxel[1] -= 1;
            break;
        case GLFW_KEY_PAGE_DOWN:
            gSelectedVoxel[2] -= 1;
            break;
        case GLFW_KEY_HOME:
            gSelectedVoxel = glm::uvec3(0);
            break;
        case GLFW_KEY_N:
            ++gSelectedLevel;
            gSelectedVoxel = glm::uvec3(0);
            break;
        case GLFW_KEY_P:
            --gSelectedLevel;
            gSelectedVoxel = glm::uvec3(0);
            break;
        case GLFW_KEY_9:
            gSelectedLevel = 9;
            gSelectedVoxel = glm::uvec3(0);
            break;
        case GLFW_KEY_V:
            gToggleDrawVolume = !gToggleDrawVolume;
            break;
        case GLFW_KEY_F5:
            reloadShaders();
            break;
        case GLFW_KEY_TAB:
            if (mods == GLFW_MOD_SHIFT) {
                gToggleVolumeWireframe = !gToggleVolumeWireframe;
            }
            else {
                gToggleWireframe = !gToggleWireframe;
            }
            break;
        case GLFW_KEY_SPACE:
            gPause = !gPause;
            break;
        default:
            break;
        }
    }

    if (action == GLFW_RELEASE) {
        switch (key)
        {
        case GLFW_KEY_W:
        case GLFW_KEY_S:
            gCamera.setVelocity('w', 0.f);
            break;
        case GLFW_KEY_A:
        case GLFW_KEY_D:
            gCamera.setVelocity('u', 0.f);
            break;
        case GLFW_KEY_E:
        case GLFW_KEY_Q:
            gCamera.setVelocity('v', 0.f);
            break;
        default:
            break;
        }
    }
}

static void window_size_callback(GLFWwindow* window, int width, int height)
{
    updateProjMat(width, height);
    glViewport(0, 0, width, height);
    gWindowWidth = width;
    gWindowHeight = height;
}

static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (gMouseRightDown) {
        float delta_x = static_cast<float>(xpos - gCursorPosX);
        float delta_y = static_cast<float>(ypos - gCursorPosY);
        float angle_x = -delta_x * kCameraRotationSpeed * glm::pi<float>() / static_cast<float>(gWindowWidth);
        float angle_y = -delta_y * kCameraRotationSpeed * glm::pi<float>() / static_cast<float>(gWindowHeight);
        gCamera.rotate(angle_x, glm::vec3(0.f, 1.f, 0.f));
        gCamera.rotate(angle_y, gCamera.getU());
        gCursorPosX = xpos;
        gCursorPosY = ypos;
    }
}

static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        switch (action) {
        case GLFW_PRESS:
            glfwGetCursorPos(window, &gCursorPosX, &gCursorPosY);
            gMouseRightDown = true;
            break;
        case GLFW_RELEASE:
            gMouseRightDown = false;
            break;
        default:
            break;
        }

    }
}


int main(int argc, char **argv)
{
    glfwSetErrorCallback(error_callback);

    if (!glfwInit()) {
        exit(EXIT_FAILURE);
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_SAMPLES, gNumSamples);
    gWindow = glfwCreateWindow(1600, 900, "Voxelization Test", NULL, NULL);
    glfwSetWindowPos(gWindow, 160, 75);
    if (!gWindow) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwSetKeyCallback(gWindow, key_callback);
    glfwSetWindowSizeCallback(gWindow, window_size_callback);
    glfwSetCursorPosCallback(gWindow, cursor_position_callback);
    glfwSetMouseButtonCallback(gWindow, mouse_button_callback);
    glfwMakeContextCurrent(gWindow);
    glfwSwapInterval(1);
    glfwGetFramebufferSize(gWindow, &gWindowWidth, &gWindowHeight);

    GLenum err = glewInit();
    if (GLEW_OK != err) {
        std::cerr << "Error: " << glewGetErrorString(err) << std::endl;
    }

    findCudaGLDevice();

    ImGui_ImplGlfwGL3_Init(gWindow, false);
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->AddFontFromFileTTF("../data/fonts/DroidSans.ttf", 18.0f);

    init();

    glViewport(0, 0, gWindowWidth, gWindowHeight);

    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    
    Timer timer;
    timer.start();

    while (!glfwWindowShouldClose(gWindow)) {
        timer.stop();
        gRealDT = static_cast<float>(timer.getElapsedMilliseconds());
        timer.start();
        gCamera.updateViewMat(gRealDT / 1000.f);

        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        render();
        glfwSwapBuffers(gWindow);
        glfwPollEvents();
    }
    clear();
    ImGui_ImplGlfwGL3_Shutdown();
    glfwDestroyWindow(gWindow);
    gWindow = nullptr;
    glfwTerminate();
    exit(EXIT_SUCCESS);

    return 0;
}