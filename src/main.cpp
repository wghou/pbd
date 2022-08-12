#include <GL/glew.h>
#include <iostream>
#include <stdio.h>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <glm/ext.hpp>
#include "misc.h"
#include "cuda_helper.h"
#include "scene.h"
#include "gui.h"

static GLFWwindow* gWindow = nullptr;
int gWindowWidth, gWindowHeight;
double gCursorPosX, gCursorPosY;
int gNumSamples = 4;
bool gToggleMSAA = true;
bool gMouseRightDown = false;
bool gMouseLeftDown = false;
bool gPause = false;
bool gToggleGui = true;
bool gToggleCapturePrev = false;
bool gToggleCapture = false;
bool gToggleWall = false;
float gRealDT;

void drawGui();
void captureVideo();

int gCurrSceneIdx = 0;
int gImguiSelectedSceneIdx = gCurrSceneIdx;
std::vector<std::unique_ptr<Scene>> gScenes;
char ** gSceneNames;

static void error_callback(int error, const char* description)
{
    std::cerr << "Error: " << description << std::endl;
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    int num_scenes = static_cast<int>(gScenes.size());

    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
    if (action == GLFW_PRESS) {
        float cam_speed = kCameraSpeed;
        switch(mods) {
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
            else gScenes[gCurrSceneIdx]->init(false);
            break;
        case GLFW_KEY_C:
            gToggleCapturePrev = gToggleCapture;
            gToggleCapture = !gToggleCapture;
            break;
        case GLFW_KEY_F:
            gToggleDrawFluid = !gToggleDrawFluid;
            break;
        case GLFW_KEY_H:
            gToggleGui = !gToggleGui;
            break;
        case GLFW_KEY_M:
            gToggleDrawMesh = !gToggleDrawMesh;
            break;
        case GLFW_KEY_O:
            gToggleStepMode = !gToggleStepMode;
            if (!gToggleStepMode) gPause = false;
            else gPause = true;
            break;
        case GLFW_KEY_T:
            gToggleWall = !gToggleWall;
            break;
        case GLFW_KEY_V:
            gToggleDrawParticle = !gToggleDrawParticle;
            break;
        case GLFW_KEY_F5:
            reloadShaders();
            break;
        case GLFW_KEY_TAB:
            gToggleWireframe = !gToggleWireframe;
            break;
        case GLFW_KEY_SPACE:
            gPause = !gPause;
            break;
        case GLFW_KEY_LEFT:
            if (gCurrSceneIdx > 0) {
                gScenes[gCurrSceneIdx]->release();
                gCurrSceneIdx -= 1;
                gImguiSelectedSceneIdx = gCurrSceneIdx;
                gScenes[gCurrSceneIdx]->init();
            }
            break;
        case GLFW_KEY_RIGHT:
            if (gCurrSceneIdx < num_scenes - 1) {
                gScenes[gCurrSceneIdx]->release();
                gCurrSceneIdx += 1;
                gImguiSelectedSceneIdx = gCurrSceneIdx;
                gScenes[gCurrSceneIdx]->init();
            }
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
        case GLFW_KEY_Q:
        case GLFW_KEY_E:
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
    gScenes[gCurrSceneIdx]->resizeFramebuffers(width, height);
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
        //gCursorPosX = xpos;
        //gCursorPosY = ypos;
    }
    else if (gMouseLeftDown) {

    }
    gCursorPosX = xpos;
    gCursorPosY = ypos;
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
    else if (button == GLFW_MOUSE_BUTTON_LEFT) {
        switch (action) {
        case GLFW_PRESS:
            glfwGetCursorPos(window, &gCursorPosX, &gCursorPosY);
            gMouseLeftDown = true;
            gScenes[gCurrSceneIdx]->pickParticle((int)gCursorPosX, (int)(gWindowHeight - gCursorPosY));
            break;
        case GLFW_RELEASE:
            gMouseLeftDown = false;
            gScenes[gCurrSceneIdx]->pickParticle(-1, -1);
            break;
        default:
            break;
        }

    }
}

std::string gCaptureCommand;

// open pipe to ffmpeg's stdin in binary write mode
FILE* gFFmpegPipe = NULL;

int* gFFmpegBuffer = NULL;

void captureVideo()
{
    if (!gToggleCapturePrev && gToggleCapture) { // start capture
        std::time_t now = std::time(nullptr);
        std::string now_str(std::ctime(&now));
        for (auto &c : now_str) {
            if (c == ' ' || c == ':' || c == '\n' || c== '\r') c = '_';
        }
        gCaptureCommand = 
              std::string("ffmpeg -r 60 -f rawvideo -pix_fmt bgra -s " 
            + std::to_string(gWindowWidth) + "x" +std::to_string(gWindowHeight) 
            + " -i - -threads 0") 
            + std::string(" -preset ultrafast -y -pix_fmt yuv420p -crf 24 -vf vflip ") 
            + std::string("../data/log/pbd_")
            + now_str
            + std::string(".mp4");
        gFFmpegPipe = _popen(gCaptureCommand.data(), "wb");
        gFFmpegBuffer = new int[gWindowWidth * gWindowHeight];
        gToggleCapturePrev = gToggleCapture;
        std::cout << "Capture started..." << std::endl;
    }

    else if (gToggleCapturePrev && gToggleCapture) { // during capture
        glReadPixels(0, 0, gWindowWidth, gWindowHeight, GL_BGRA, GL_UNSIGNED_BYTE, gFFmpegBuffer);
        fwrite(gFFmpegBuffer, sizeof(int)* gWindowWidth * gWindowHeight, 1, gFFmpegPipe);
    }

    else if (gToggleCapturePrev && !gToggleCapture) { // stop capture
        _pclose(gFFmpegPipe);
        gFFmpegPipe = NULL;
        delete gFFmpegBuffer;
        gFFmpegBuffer = NULL;
        gToggleCapturePrev = gToggleCapture;
        std::cout << "Capture stopped..." << std::endl;
    }
}

void setupScenes()
{
    gScenes.push_back( std::unique_ptr<Scene>(
        new RigidPile{
            "box (7) pile (6x6x6)",
            "../data/assets/box.ply",  
            0.075f,
            make_uint3(7),
            make_int3(6, 6, 6),
            64000,
            -1.f}
    ));

    gScenes.push_back(std::unique_ptr<Scene>(
        new TwoRigidBodies{
        "bunny(60) vs dragon(84)",
        "../data/assets/bunny.ply",
        "../data/assets/dragon.obj",
        0.075f,
        make_uint3(60),
        make_uint3(84),
        (60000) }
    ));

    gScenes.push_back(std::unique_ptr<Scene>(
        new RigidPile{
        "bunny (20) pile (2x20x2)",
        "../data/assets/bunny.ply",
        0.075f,
        make_uint3(20),
        make_int3(2, 20, 2),
        105000,
        -1.f }
    ));

    gScenes.push_back(std::unique_ptr<Scene>(
        new GranularPile{
        "granular sandcastle (62, =)",
        "../data/assets/sandcastle.obj",
        0.075f,
        make_uint3(62),
        65000}
    ));

    gScenes.push_back(std::unique_ptr<Scene>(
        new GranularPile{
        "granular sandcastle (62)",
        "../data/assets/sandcastle.obj",
        0.075f,
        make_uint3(62),
        20000,
        -1.f}
    ));

    gScenes.push_back(std::unique_ptr<Scene>(
        new BreakingDam{
        "breaking dam",
        0.05f,
        make_int3(25, 60, 25),
        40000,
        1000.f }
    ));

    gScenes.push_back(std::unique_ptr<Scene>(
        new RigidFluid{
        "rigid fluid(0.02 rho)",
        "../data/assets/armadillo.ply",
        0.05f,
        0.02f,
        make_int3(40),
        0.05f,
        make_int3(25, 60, 25),
        45000,
        1000.f, -1.f}
    ));


    gScenes.push_back(std::unique_ptr<Scene>(
        new RigidFluid{
        "rigid fluid(2 rho)",
        "../data/assets/torus.obj",
        0.05f,
        2.f,
        make_int3(28),
        0.05f,
        make_int3(25, 60, 25),
        42000,
        1000.f, -1.f }
    ));

    gScenes.push_back(std::unique_ptr<Scene>(
        new RigidFluid{
        "rigid fluid(16 rho)",
        "../data/assets/torus.obj",
        0.05f,
        16.f,
        make_int3(28),
        0.05f,
        make_int3(25, 60, 25),
        42000,
        1000.f, -1.f }
    ));

    gScenes.push_back(std::unique_ptr<Scene>(
        new RigidFluid{
        "rigid fluid(4 rho)",
        "../data/assets/sandcastle.obj",
        0.05f,
        4.f,
        make_int3(32),
        0.05f,
        make_int3(25, 60, 25),
        45000,
        1000.f, -2.f }
    ));

    gScenes.push_back(std::unique_ptr<Scene>(
        new RigidPile{
        "box (7, =) pile (6x6x6)",
        "../data/assets/box.ply",
        0.075f,
        make_uint3(7),
        make_int3(6, 6, 6),
        75000 }
    ));

    gScenes.push_back(std::unique_ptr<Scene>(
        new RigidPile{
        "bunny (20, =) pile (2x20x2)",
        "../data/assets/bunny.ply",
        0.075f,
        make_uint3(20),
        make_int3(2, 20, 2),
        130000 }
    ));

    gScenes.push_back(std::unique_ptr<Scene>(
        new RigidFluid{
        "rigid fluid (0.02 rho, =)",
        "../data/assets/armadillo.ply",
        0.05f,
        0.02f,
        make_int3(40),
        0.05f,
        make_int3(25, 60, 25),
        45000,
        1000.f }
    ));

    gScenes.push_back(std::unique_ptr<Scene>(
        new RigidFluid{
        "rigid fluid (2 rho, =)",
        "../data/assets/torus.obj",
        0.05f,
        2.f,
        make_int3(28),
        0.05f,
        make_int3(25, 60, 25),
        42000,
        1000.f }
    ));

    gScenes.push_back(std::unique_ptr<Scene>(
        new RigidFluid{
        "rigid fluid (4 rho, =)",
        "../data/assets/sandcastle.obj",
        0.05f,
        4.f,
        make_int3(32),
        0.05f,
        make_int3(25, 60, 25),
        47000,
        1000.f }
    ));

    gScenes.push_back(std::unique_ptr<Scene>(
        new RigidPile{
        "box (2, =) stack (10)",
        "../data/assets/box.ply",
        0.075f,
        make_uint3(2),
        make_int3(1, 10, 1),
        400,
        -1.f }
    ));

    gScenes.push_back(std::unique_ptr<Scene>(
        new RigidPile{
        "box (2, =) pile (10x10x10)",
        "../data/assets/box.ply",
        0.075f,
        make_uint3(2),
        make_int3(10, 10, 10),
        9000,
        -1.f }
    ));

    // post set-up
    int num_scenes = static_cast<int>(gScenes.size());
    gSceneNames = new char *[num_scenes];
    for (int i = 0; i < num_scenes; ++i) {
        gSceneNames[i] = new char[gScenes[i]->getName().size() + 1];
        strcpy(gSceneNames[i], gScenes[i]->getName().c_str());
    }
}

int main(int argc, char **argv)
{
    glfwSetErrorCallback(error_callback);
    
    if (!glfwInit()) {
        exit(EXIT_FAILURE);
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    //glfwWindowHint(GLFW_SAMPLES, gNumSamples);
    gWindow = glfwCreateWindow(1600, 900, "Position Based Dynamics", NULL, NULL);
    glfwSetWindowPos(gWindow, 160, 75);
    if (!gWindow) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwSetKeyCallback(gWindow, key_callback);
    glfwSetFramebufferSizeCallback(gWindow, window_size_callback);
    glfwSetCursorPosCallback(gWindow, cursor_position_callback);
    glfwSetMouseButtonCallback(gWindow, mouse_button_callback);
    glfwMakeContextCurrent(gWindow);
    //glfwSwapInterval(1);
    
    GLenum err = glewInit();
    if (GLEW_OK != err) {
        std::cerr << "Error: " << glewGetErrorString(err) << std::endl;
    }

    findCudaGLDevice();

    ImGui_ImplGlfwGL3_Init(gWindow, false);
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->AddFontFromFileTTF("../data/fonts/DroidSans.ttf", 18.0f);

    GLint msaa_enabled;
    glGetIntegerv(GL_SAMPLE_BUFFERS, &msaa_enabled);
    if (msaa_enabled != 1) {
        std::cerr << "Multisampling not enabled." << std::endl;
    }
    
    GLint max_color_attach = 0;
    glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &max_color_attach);
    GLint max_draw_buf = 0;
    glGetIntegerv(GL_MAX_DRAW_BUFFERS, &max_draw_buf);
    std::cout << "GL_MAX_COLOR_ATTACHMENTS = " << max_color_attach
              << ", GL_MAX_DRAW_BUFFERS = " << max_draw_buf
              << std::endl;

    glfwGetFramebufferSize(gWindow, &gWindowWidth, &gWindowHeight);
    
    updateProjMat(gWindowWidth, gWindowHeight);
    initPrograms();
    gCamera.init(kCameraPos, kLookAt, kUp);
    setupLight();
    setupScenes();
    gScenes[gCurrSceneIdx]->init();
    glViewport(0, 0, gWindowWidth, gWindowHeight);
    
    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    //glEnable(GL_MULTISAMPLE);
    
    Timer frame_timer;
    frame_timer.start();
    while (!glfwWindowShouldClose(gWindow)) {
        frame_timer.stop();
        gRealDT = 0.8f * static_cast<float>(frame_timer.getElapsedMilliseconds()) + 0.2f * gRealDT;
        gCamera.updateViewMat(gRealDT * 0.001f);
        if (gImguiSelectedSceneIdx != gCurrSceneIdx) {
            gScenes[gCurrSceneIdx]->release();
            gCurrSceneIdx = gImguiSelectedSceneIdx;
            gScenes[gCurrSceneIdx]->init();
        }
        frame_timer.start();
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        gScenes[gCurrSceneIdx]->update();
        gScenes[gCurrSceneIdx]->render();
        if (gToggleGui) {
            drawGui();
        }
        checkGLError("main:503");

        glfwSwapBuffers(gWindow);
        captureVideo();
        glfwPollEvents();
    }
    ImGui_ImplGlfwGL3_Shutdown();

    for (size_t i = 0; i < gScenes.size(); ++i) {
        delete[] gSceneNames[i];
    }
    delete[] gSceneNames;
    gScenes[gCurrSceneIdx]->release();
    
    glfwDestroyWindow(gWindow);
    gWindow = nullptr;
    glfwTerminate();
    exit(EXIT_SUCCESS);
}

void
beginSimInfoWindow()
{
    ImGuiWindowFlags window_flags = 0;
    window_flags |= ImGuiWindowFlags_NoTitleBar;
    window_flags |= ImGuiWindowFlags_NoMove;
    window_flags |= ImGuiWindowFlags_NoCollapse;
    window_flags |= ImGuiWindowFlags_NoResize;
    //ImGui::SetNextWindowSize(ImVec2(225, 300), ImGuiSetCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2((float)gWindowWidth - 235.f, 0.f), ImGuiSetCond_Always);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.f, 0.f, 0.f, 0.3f));
    ImGui::Begin("Sim Info", nullptr, window_flags);
}

void
endSimInfoWindow()
{
    ImGui::End();
    ImGui::PopStyleColor();
}

void
beginCheckBox()
{
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(-0.1f, -0.1f)); // checkbox frame
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 9.f);
}

void
endCheckBox()
{
    ImGui::PopStyleVar(2);
}

void
beginButton()
{
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(45.f, 0.f)); // checkbox frame
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 9.f);
}

void
endButton()
{
    ImGui::PopStyleVar(2);
}

void
beginSlider()
{
    ImGui::PushItemWidth(-0.1f); // hide slider label
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(-0.5f, 0.5f)); // slider frame
}

void
endSlider()
{
    ImGui::PopStyleVar();
    ImGui::PopItemWidth();
}

void
beginSceneListWindow()
{
    ImGuiWindowFlags window_flags = 0;
    window_flags |= ImGuiWindowFlags_NoMove;
    window_flags |= ImGuiWindowFlags_NoCollapse;
    window_flags |= ImGuiWindowFlags_NoResize;
    ImGui::SetNextWindowPos(ImVec2(2.f, 2.f), ImGuiSetCond_Always);
    ImGui::SetNextWindowSize(ImVec2(200.f, gWindowHeight * 0.32f), ImGuiSetCond_Always);

    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.f, 0.f, 0.f, 0.7f));
    ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.f, 0.f, 0.f, 0.7f));
    ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.f, 0.f, 0.f, 0.7f));
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.4f, 0.4f, 0.4f, 0.4f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.4f, 0.4f, 0.4f, 0.8f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.7f, 0.7f, 0.7f, 0.5f));

    ImGui::Begin("Scenes", nullptr, window_flags);
    ImGui::PushItemWidth(198.f);
}

void
EndSceneListWindow()
{
    ImGui::PopItemWidth();
    ImGui::End();
    ImGui::PopStyleColor(6);
}

void
beginOptionWindow()
{
    ImGuiWindowFlags window_flags = 0;
    window_flags |= ImGuiWindowFlags_NoMove;
    window_flags |= ImGuiWindowFlags_NoCollapse;
    window_flags |= ImGuiWindowFlags_NoResize;
    ImGui::SetNextWindowPos(ImVec2(2.f, gWindowHeight * 0.5f), ImGuiSetCond_Always);
    ImGui::SetNextWindowSize(ImVec2(200.f, gWindowHeight * 0.49f), ImGuiSetCond_Always);

    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.f, 0.f, 0.f, 0.7f));
    ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.f, 0.f, 0.f, 0.7f));
    ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.f, 0.f, 0.f, 0.7f));
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.4f, 0.4f, 0.4f, 0.4f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.4f, 0.4f, 0.4f, 0.8f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.7f, 0.7f, 0.7f, 0.5f));

    //ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(3.f, 3.f));
    ImGui::Begin("Options", nullptr, window_flags);
}

void
endOptionWindow()
{
    ImGui::End();
    ImGui::PopStyleColor(6);
}

void
drawGui()
{
    ImGuiWindowFlags window_flags = 0;
    Params& solver_params = gScenes[gCurrSceneIdx]->getSolverParams();
    ImGuiStyle&  im_style = ImGui::GetStyle();
    im_style.WindowTitleAlign = ImVec2(0.1f, 0.5f);
    im_style.ItemSpacing = ImVec2(10.f, 10.f);

    ImGui_ImplGlfwGL3_NewFrame();
    
    beginSimInfoWindow();
    ImGui::Text("Window: %d x %d", gWindowWidth, gWindowHeight);
    ImGui::Text("Frame count: %d", solver_params.num_frames_);
    ImGui::Text("Frame Time Total: %.3f ms", gRealDT, 1000.f / gRealDT);
    ImGui::Text("Frame Per Second: %.3f", 1000.f / gRealDT);
    ImGui::Text("Particle Count: %d", solver_params.num_particles_);
    ImGui::Text("Rigid Count: %d", solver_params.num_rigids_);
    ImGui::Text("Num Substeps: %d", solver_params.num_substeps_);
    ImGui::Text("Num Iterations: %d", solver_params.num_iterations_);
    endSimInfoWindow();

    beginSceneListWindow();
    int num_scenes = static_cast<int>(gScenes.size());
    ImGui::ListBox(" scene list", &gImguiSelectedSceneIdx, gSceneNames, num_scenes);
    EndSceneListWindow();

    beginOptionWindow();
    ImGui::Separator();
    ImGui::Text("Global");
    
    beginCheckBox();
    bool toggle_msaa = gToggleMSAA;
    ImGui::Checkbox("MSAA", &gToggleMSAA);
    
    if (toggle_msaa != gToggleMSAA) {
        gScenes[gCurrSceneIdx]->resizeFramebuffers(gWindowWidth, gWindowHeight);
    }

    ImGui::Checkbox("Draw Points", &gToggleDrawParticle);
    ImGui::Checkbox("Draw Fluid", &gToggleDrawFluid);
    ImGui::Checkbox("Draw Mesh", &gToggleDrawMesh);
    endCheckBox();

    ImGui::Separator();
    beginButton();
    if (ImGui::Button("Reset Scene")) {
        gScenes[gCurrSceneIdx]->init(true);
    }
    endButton();
    beginSlider();
    ImGui::SliderFloat(" fluid opacity", &gFluidParticleAlpha, 0.f, 1.f, "fluid opacity: %.3f", 1.f);
    ImGui::Separator();
    ImGui::SliderInt(" iterations", &solver_params.num_iterations_, 1, 20, "iterations:\t %.0f");
    ImGui::SliderInt(" substeps", &solver_params.num_substeps_, 1, 10, "substeps:\t %.0f");
    ImGui::SliderInt(" pre-stabilize", &solver_params.num_pre_stabilize_iterations_, 0, 40, "pre-stabilize:\t %.0f");
    ImGui::SliderFloat(" gravity x", &solver_params.gravity_.x, -50.0f, 50.0f, "gravity x:\t %.0f", 1.0f);
    ImGui::SliderFloat(" gravity y", &solver_params.gravity_.y, -50.0f, 50.0f, "gravity y:\t %.0f", 1.0f);
    ImGui::SliderFloat(" gravity z", &solver_params.gravity_.z, -50.0f, 50.0f, "gravity z:\t %.0f", 1.0f);
    ImGui::SliderFloat(" shock propagation", &solver_params.shock_propagation_, 0.0f, 15.f, "shock propagate: %.3f", 1.f);
    ImGui::SliderFloat(" damping", &solver_params.damping_force_factor_, 0.0f, 2.f, "damping: %.3f", 1.f);
    ImGui::SliderFloat(" dynamic friction", &solver_params.dynamic_friction_, 0.0f, 1.f, "dynamic friction: %.3f", 1.f);
    ImGui::SliderFloat(" static friction", &solver_params.static_friction_, 0.0f, 1.f, "static friction: %.3f", 1.f);
    ImGui::SliderFloat(" particle friction", &solver_params.particle_friction_, 0.0f, 1.0f, "particle friction: %.3f", 1.f);
    ImGui::SliderFloat(" sleeping threshold", &solver_params.sleeping_threshold_quad_, 0.0f, 0.5f, "sleep threshold: %.3f", 1.f);
    ImGui::SliderFloat(" particle margin", &solver_params.particle_collision_margin_, 0.f, 0.4f, "particle margin: %.3f", 1.f);
    ImGui::SliderFloat(" shape rest extent", &solver_params.shape_rest_extent_, 0.f, 2.f, "shape overlap: %.3f", 1.f);
    ImGui::Separator();
    ImGui::SliderFloat(" fluid CFM eps", &solver_params.fluid_cfm_eps_, 0.f, 2000.f, "fluid CFM eps: %.3f", 3.f);
    ImGui::SliderFloat(" fluid corr: K", &solver_params.fluid_corr_k_, 0.f, 0.1f, "fluid corr K: %.3f", 1.f);
    ImGui::SliderFloat(" fluid corr: n", &solver_params.fluid_corr_n_, 1.f, 10.f, "fluid corr n: %.3f", 1.f);
    ImGui::SliderFloat(" fluid corr: dq", &solver_params.fluid_corr_refer_q_, 0.f, 0.5f, "fluid corr dq: %.3f", 1.f);
    endSlider();
    endOptionWindow();
    
    glViewport(0, 0, gWindowWidth, gWindowHeight);
    ImGui::Render();
}

