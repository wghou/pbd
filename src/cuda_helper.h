#pragma once
#include <GL/glew.h>
#include "cuda_math_helper.h"
#include "misc.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <glm/glm.hpp>
#include <sstream>
#include <iomanip>
#include <string>
#include <unordered_map>

#ifdef _PROFILE_SOLVER_
#define BEGIN_TIMER(func_name) beginTimer(func_name)
#define STOP_TIMER(func_name) stopTimer(func_name)
#else
#define BEGIN_TIMER(func_name) 
#define STOP_TIMER(func_name)
#endif



// phase
#define kPhaseFluidFlag 1 << 25
#define kPhaseSelfCollideFlag 1 << 24
#define kPhaseGroupMask 0x00ffffff

const uint kCudaBlockSize = 256;
const uint kCuda2DBlockSize = 16;

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

#define FREE_CUDA_ARRAY(ptr) do { freeArray(ptr); ptr = 0; } while(0)
#define UNREGISTER_GL_BO(gl_res) do { if (gl_res != 0) {unregisterGLBufferObject(gl_res); gl_res = 0;} } while(0)

// conversion between glm::vec and cuda vector type
// glm::vec --> cuda vector type
inline float3
make_float3(glm::vec3 const& v)
{
    return make_float3(v.x, v.y, v.z);
}

inline float3
make_float3(glm::vec4 const& v)
{
    return make_float3(v.x, v.y, v.z);
}

inline float4
make_float4(glm::vec3 const& v)
{
    return make_float4(v.x, v.y, v.z, 0.f);
}

inline float4
make_float4(glm::vec4 const& v)
{
    return make_float4(v.x, v.y, v.z, v.w);
}

inline float4
make_float4(glm::vec3 const& v, float s)
{
    return make_float4(v.x, v.y, v.z, s);
}

inline int4
make_int4(glm::ivec4 const &v)
{
    return make_int4(v.x, v.y, v.z, v.w);
}

inline uint3
make_uint3(glm::uvec3 const &v)
{
    return make_uint3(v.x, v.y, v.z);
}

// cuda vector type --> glm::vec
inline glm::uvec3
make_uvec3(uint3 const& v)
{
    return glm::uvec3(v.x, v.y, v.z);
}

inline glm::ivec3
make_ivec3(int3 const& v)
{
    return glm::ivec3(v.x, v.y, v.z);
}

inline glm::vec3
make_vec3(float3 const& v)
{
    return glm::vec3(v.x, v.y, v.z);
}

inline glm::vec3
make_vec3(uint3 const& v)
{
    return glm::vec3(v.x, v.y, v.z);
}

inline glm::vec3
make_vec3(int3 const& v)
{
    return glm::vec3(v.x, v.y, v.z);
}

inline glm::vec3
make_vec3(float4 const& v)
{
    return glm::vec3(v.x, v.y, v.z);
}

inline glm::vec4
make_vec4(float4 const& v)
{
    return glm::vec4(v.x, v.y, v.z, v.w);
}

inline glm::uvec4
make_uvec4(uint4 const& v)
{
    return glm::uvec4(v.x, v.y, v.z, v.w);
}

inline bool
operator == (float4 const &v1, float4 const &v2)
{
    return v1.x == v2.x &&
        v1.y == v2.y &&
        v1.z == v2.z &&
        v1.w == v2.w;
}

static const char *_cudaGetErrorEnum(cudaError_t error)
{
    switch (error)
    {
    case cudaSuccess:
        return "cudaSuccess";

    case cudaErrorMissingConfiguration:
        return "cudaErrorMissingConfiguration";

    case cudaErrorMemoryAllocation:
        return "cudaErrorMemoryAllocation";

    case cudaErrorInitializationError:
        return "cudaErrorInitializationError";

    case cudaErrorLaunchFailure:
        return "cudaErrorLaunchFailure";

    case cudaErrorPriorLaunchFailure:
        return "cudaErrorPriorLaunchFailure";

    case cudaErrorLaunchTimeout:
        return "cudaErrorLaunchTimeout";

    case cudaErrorLaunchOutOfResources:
        return "cudaErrorLaunchOutOfResources";

    case cudaErrorInvalidDeviceFunction:
        return "cudaErrorInvalidDeviceFunction";

    case cudaErrorInvalidConfiguration:
        return "cudaErrorInvalidConfiguration";

    case cudaErrorInvalidDevice:
        return "cudaErrorInvalidDevice";

    case cudaErrorInvalidValue:
        return "cudaErrorInvalidValue";

    case cudaErrorInvalidPitchValue:
        return "cudaErrorInvalidPitchValue";

    case cudaErrorInvalidSymbol:
        return "cudaErrorInvalidSymbol";

    case cudaErrorMapBufferObjectFailed:
        return "cudaErrorMapBufferObjectFailed";

    case cudaErrorUnmapBufferObjectFailed:
        return "cudaErrorUnmapBufferObjectFailed";

    case cudaErrorInvalidHostPointer:
        return "cudaErrorInvalidHostPointer";

    case cudaErrorInvalidDevicePointer:
        return "cudaErrorInvalidDevicePointer";

    case cudaErrorInvalidTexture:
        return "cudaErrorInvalidTexture";

    case cudaErrorInvalidTextureBinding:
        return "cudaErrorInvalidTextureBinding";

    case cudaErrorInvalidChannelDescriptor:
        return "cudaErrorInvalidChannelDescriptor";

    case cudaErrorInvalidMemcpyDirection:
        return "cudaErrorInvalidMemcpyDirection";

    case cudaErrorAddressOfConstant:
        return "cudaErrorAddressOfConstant";

    case cudaErrorTextureFetchFailed:
        return "cudaErrorTextureFetchFailed";

    case cudaErrorTextureNotBound:
        return "cudaErrorTextureNotBound";

    case cudaErrorSynchronizationError:
        return "cudaErrorSynchronizationError";

    case cudaErrorInvalidFilterSetting:
        return "cudaErrorInvalidFilterSetting";

    case cudaErrorInvalidNormSetting:
        return "cudaErrorInvalidNormSetting";

    case cudaErrorMixedDeviceExecution:
        return "cudaErrorMixedDeviceExecution";

    case cudaErrorCudartUnloading:
        return "cudaErrorCudartUnloading";

    case cudaErrorUnknown:
        return "cudaErrorUnknown";

    case cudaErrorNotYetImplemented:
        return "cudaErrorNotYetImplemented";

    case cudaErrorMemoryValueTooLarge:
        return "cudaErrorMemoryValueTooLarge";

    case cudaErrorInvalidResourceHandle:
        return "cudaErrorInvalidResourceHandle";

    case cudaErrorNotReady:
        return "cudaErrorNotReady";

    case cudaErrorInsufficientDriver:
        return "cudaErrorInsufficientDriver";

    case cudaErrorSetOnActiveProcess:
        return "cudaErrorSetOnActiveProcess";

    case cudaErrorInvalidSurface:
        return "cudaErrorInvalidSurface";

    case cudaErrorNoDevice:
        return "cudaErrorNoDevice";

    case cudaErrorECCUncorrectable:
        return "cudaErrorECCUncorrectable";

    case cudaErrorSharedObjectSymbolNotFound:
        return "cudaErrorSharedObjectSymbolNotFound";

    case cudaErrorSharedObjectInitFailed:
        return "cudaErrorSharedObjectInitFailed";

    case cudaErrorUnsupportedLimit:
        return "cudaErrorUnsupportedLimit";

    case cudaErrorDuplicateVariableName:
        return "cudaErrorDuplicateVariableName";

    case cudaErrorDuplicateTextureName:
        return "cudaErrorDuplicateTextureName";

    case cudaErrorDuplicateSurfaceName:
        return "cudaErrorDuplicateSurfaceName";

    case cudaErrorDevicesUnavailable:
        return "cudaErrorDevicesUnavailable";

    case cudaErrorInvalidKernelImage:
        return "cudaErrorInvalidKernelImage";

    case cudaErrorNoKernelImageForDevice:
        return "cudaErrorNoKernelImageForDevice";

    case cudaErrorIncompatibleDriverContext:
        return "cudaErrorIncompatibleDriverContext";

    case cudaErrorPeerAccessAlreadyEnabled:
        return "cudaErrorPeerAccessAlreadyEnabled";

    case cudaErrorPeerAccessNotEnabled:
        return "cudaErrorPeerAccessNotEnabled";

    case cudaErrorDeviceAlreadyInUse:
        return "cudaErrorDeviceAlreadyInUse";

    case cudaErrorProfilerDisabled:
        return "cudaErrorProfilerDisabled";

    case cudaErrorProfilerNotInitialized:
        return "cudaErrorProfilerNotInitialized";

    case cudaErrorProfilerAlreadyStarted:
        return "cudaErrorProfilerAlreadyStarted";

    case cudaErrorProfilerAlreadyStopped:
        return "cudaErrorProfilerAlreadyStopped";

        /* Since CUDA 4.0*/
    case cudaErrorAssert:
        return "cudaErrorAssert";

    case cudaErrorTooManyPeers:
        return "cudaErrorTooManyPeers";

    case cudaErrorHostMemoryAlreadyRegistered:
        return "cudaErrorHostMemoryAlreadyRegistered";

    case cudaErrorHostMemoryNotRegistered:
        return "cudaErrorHostMemoryNotRegistered";

        /* Since CUDA 5.0 */
    case cudaErrorOperatingSystem:
        return "cudaErrorOperatingSystem";

    case cudaErrorPeerAccessUnsupported:
        return "cudaErrorPeerAccessUnsupported";

    case cudaErrorLaunchMaxDepthExceeded:
        return "cudaErrorLaunchMaxDepthExceeded";

    case cudaErrorLaunchFileScopedTex:
        return "cudaErrorLaunchFileScopedTex";

    case cudaErrorLaunchFileScopedSurf:
        return "cudaErrorLaunchFileScopedSurf";

    case cudaErrorSyncDepthExceeded:
        return "cudaErrorSyncDepthExceeded";

    case cudaErrorLaunchPendingCountExceeded:
        return "cudaErrorLaunchPendingCountExceeded";

    case cudaErrorNotPermitted:
        return "cudaErrorNotPermitted";

    case cudaErrorNotSupported:
        return "cudaErrorNotSupported";

        /* Since CUDA 6.0 */
    case cudaErrorHardwareStackError:
        return "cudaErrorHardwareStackError";

    case cudaErrorIllegalInstruction:
        return "cudaErrorIllegalInstruction";

    case cudaErrorMisalignedAddress:
        return "cudaErrorMisalignedAddress";

    case cudaErrorInvalidAddressSpace:
        return "cudaErrorInvalidAddressSpace";

    case cudaErrorInvalidPc:
        return "cudaErrorInvalidPc";

    case cudaErrorIllegalAddress:
        return "cudaErrorIllegalAddress";

        /* Since CUDA 6.5*/
    case cudaErrorInvalidPtx:
        return "cudaErrorInvalidPtx";

    case cudaErrorInvalidGraphicsContext:
        return "cudaErrorInvalidGraphicsContext";

    case cudaErrorStartupFailure:
        return "cudaErrorStartupFailure";

    case cudaErrorApiFailureBase:
        return "cudaErrorApiFailureBase";

        /* Since CUDA 8.0*/
    case cudaErrorNvlinkUncorrectable:
        return "cudaErrorNvlinkUncorrectable";
    }

    return "<unknown>";
}

template< typename T >
void check(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
        cudaDeviceReset();
        // Make sure we call CUDA Device Reset before exiting
        exit(EXIT_FAILURE);
    }
}


inline void __getLastCudaError(const char *errorMessage, const char *file, const int line)
{
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
                file, line, errorMessage, (int)err, cudaGetErrorString(err));
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

// Float To Int conversion
inline int ftoi(float value)
{
    return (value >= 0 ? (int)(value + 0.5) : (int)(value - 0.5));
}

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        { 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
        { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
        { 0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
        { 0x53, 128}, // Maxwell Generation (SM 5.3) GM20x class
        { 0x60, 64 }, // Pascal Generation (SM 6.0) GP100 class
        { 0x61, 128}, // Pascal Generation (SM 6.1) GP10x class
        { 0x62, 128}, // Pascal Generation (SM 6.2) GP10x class
        {   -1, -1 }
    };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one to run properly
    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index-1].Cores);
    return nGpuArchCoresPerSM[index-1].Cores;
}
// end of GPU Architecture definitions

// This function returns the best GPU (with maximum GFLOPS)
inline int gpuGetMaxGflopsDeviceId()
{
    int current_device     = 0, sm_per_multiproc  = 0;
    int max_perf_device    = 0;
    int device_count       = 0, best_SM_arch      = 0;
    int devices_prohibited = 0;
    
    unsigned long long max_compute_perf = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceCount(&device_count);
    
    checkCudaErrors(cudaGetDeviceCount(&device_count));

    if (device_count == 0)
    {
        fprintf(stderr, "gpuGetMaxGflopsDeviceId() CUDA error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }

    // Find the best major SM Architecture GPU device
    while (current_device < device_count)
    {
        cudaGetDeviceProperties(&deviceProp, current_device);

        // If this GPU is not running on Compute Mode prohibited, then we can add it to the list
        if (deviceProp.computeMode != cudaComputeModeProhibited)
        {
            if (deviceProp.major > 0 && deviceProp.major < 9999)
            {
                best_SM_arch = best_SM_arch > deviceProp.major ? best_SM_arch : deviceProp.major;
            }
        }
        else
        {
            devices_prohibited++;
        }

        current_device++;
    }

    if (devices_prohibited == device_count)
    {
    	fprintf(stderr, "gpuGetMaxGflopsDeviceId() CUDA error: all devices have compute mode prohibited.\n");
    	exit(EXIT_FAILURE);
    }

    // Find the best CUDA capable GPU device
    current_device = 0;

    while (current_device < device_count)
    {
        cudaGetDeviceProperties(&deviceProp, current_device);

        // If this GPU is not running on Compute Mode prohibited, then we can add it to the list
        if (deviceProp.computeMode != cudaComputeModeProhibited)
        {
            if (deviceProp.major == 9999 && deviceProp.minor == 9999)
            {
                sm_per_multiproc = 1;
            }
            else
            {
                sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
            }

            unsigned long long compute_perf  = (unsigned long long) deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;

            if (compute_perf  > max_compute_perf)
            {
                // If we find GPU with SM major > 2, search only these
                if (best_SM_arch > 2)
                {
                    // If our device==dest_SM_arch, choose this, or else pass
                    if (deviceProp.major == best_SM_arch)
                    {
                        max_compute_perf  = compute_perf;
                        max_perf_device   = current_device;
                    }
                }
                else
                {
                    max_compute_perf  = compute_perf;
                    max_perf_device   = current_device;
                }
            }
        }

        ++current_device;
    }

    return max_perf_device;
}


// Initialization code to find the best CUDA Device
inline int findCudaDevice()
{
    cudaDeviceProp deviceProp;
    int devID = 0;

    // Pick the device with highest Gflops/s
    devID = gpuGetMaxGflopsDeviceId();
    checkCudaErrors(cudaSetDevice(devID));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);

    return devID;
}

// General check for CUDA GPU SM Capabilities
inline bool checkCudaCapabilities(int major_version, int minor_version)
{
    cudaDeviceProp deviceProp;
    deviceProp.major = 0;
    deviceProp.minor = 0;
    int dev;

    checkCudaErrors(cudaGetDevice(&dev));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

    if ((deviceProp.major > major_version) ||
        (deviceProp.major == major_version && deviceProp.minor >= minor_version))
    {
        printf("  Device %d: <%16s >, Compute SM %d.%d detected\n", dev, deviceProp.name, deviceProp.major, deviceProp.minor);
        return true;
    }
    else
    {
        printf("  No GPU device was found that can support CUDA compute capability %d.%d.\n", major_version, minor_version);
        return false;
    }
}

// These are CUDA OpenGL Helper functions
#include <cuda_gl_interop.h>

// This function will pick the best CUDA device available with OpenGL interop
inline int findCudaGLDevice()
{
    int devID = 0;
    // Pick the device with highest Gflops/s
    devID = gpuGetMaxGflopsDeviceId();
    cudaGLSetGLDevice(devID);

    return devID;
}

inline void allocateArray(void **devPtr, size_t size)
{
    checkCudaErrors(cudaMalloc(devPtr, size));
}

inline void freeArray(void *devPtr)
{
    checkCudaErrors(cudaFree(devPtr));
}

inline void deviceSync()
{
    checkCudaErrors(cudaDeviceSynchronize());
}

inline void copyArrayToDevice(void *device, const void *host, int offset, int size)
{
    checkCudaErrors(cudaMemcpy((char *)device + offset, host, size, cudaMemcpyHostToDevice));
}

inline void createArray(void **devPtr, const void *host, size_t offset, size_t size)
{
    allocateArray(devPtr, size);
    copyArrayToDevice(*devPtr, host, (int)offset, (int)size);
}

inline void registerGLBufferObject(GLuint vbo, struct cudaGraphicsResource **cuda_vbo_resource, unsigned int flags = cudaGraphicsMapFlagsNone)
{
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(cuda_vbo_resource, vbo, flags));
}

inline void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
{
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));
}

inline void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource)
{
    void *ptr;
    checkCudaErrors(cudaGraphicsMapResources(1, cuda_vbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&ptr, &num_bytes,
        *cuda_vbo_resource));
    return ptr;
}

inline void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
{
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
}

inline void copyArrayFromDevice(void *host, const void *device,
    struct cudaGraphicsResource **cuda_vbo_resource, int size)
{
    if (cuda_vbo_resource)
    {
        device = mapGLBufferObject(cuda_vbo_resource);
    }

    checkCudaErrors(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));

    if (cuda_vbo_resource)
    {
        unmapGLBufferObject(*cuda_vbo_resource);
    }
}

//Round a / b to nearest higher integer value
inline uint iDivUp(uint a, uint b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

// compute grid and thread block size for a given number of elements
inline void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
{
    numThreads = min(blockSize, n);
    numBlocks = iDivUp(n, numThreads);
}


// Timing
extern const int INTERVAL;
struct Timing {
    Timer t;
    double total_time;
    int num_iter;

    void start() {
        t.start();
    }

    double stop() {
        t.stop();
        ++num_iter;
        total_time += t.getElapsedMilliseconds();
        double average_time = -1.0;
        if (num_iter == INTERVAL) {
            average_time = total_time / (double)INTERVAL;
            total_time = 0.0;
            num_iter = 0;
        }
        return average_time;
    }
};
static std::unordered_map<std::string, Timing> sTimers;

inline void beginTimer(std::string const &_func_name) {
    deviceSync();
    sTimers[_func_name].start();
}

inline void stopTimer(std::string const &_func_name) {
    deviceSync();
    double average_time = sTimers.at(_func_name).stop();
    if (average_time > -1.0) {
        std::cout << _func_name << " average time: " << average_time << " ms" << std::endl;
    }
}


// debug print
inline 
std::ostream&
operator << (std::ostream& os, float3 const& v)
{
    os << v.x << ", " << v.y << ", " << v.z << std::flush;
    return os;
}

inline
std::ostream&
operator << (std::ostream& os, float4 const& v)
{
    os << v.x << ", " << v.y << ", " << v.z << ", " << v.w << std::flush;
    return os;
}

template <typename T, int n>
std::string
readbackVecNArray(int _num_elements, T *_device_mem, std::string const &_device_mem_name)
{
    std::ostringstream ss;
    std::vector <T> host_mem(_num_elements);
    deviceSync();
    copyArrayFromDevice(host_mem.data(), _device_mem, 0, _num_elements * sizeof(T));
    ss << _device_mem_name << std::endl;
    for (int i = 0; i < _num_elements; ++i) {
        ss << std::noshowpos << i << ": " << std::fixed << std::showpos << std::setprecision(5) << host_mem[i] << std::endl;
    }
    ss << std::endl;
    return ss.str();
}

template <typename T>
std::string
readbackScalarArray(int _num_elements, T *_device_mem, std::string const &_device_mem_name, int line_length = 3, int unit_length = 9)
{
    std::ostringstream ss;
    std::vector <T> host_mem(_num_elements);
    deviceSync();
    copyArrayFromDevice(host_mem.data(), _device_mem, 0, _num_elements * sizeof(T));
    ss << _device_mem_name << std::endl;
    for (int i = 0; i < _num_elements; ++i) {
        ss << std::fixed << std::showpos << std::setprecision(5) << host_mem[i] << std::flush;
        if (i % line_length == line_length - 1) {
            ss << std::endl;
        }
        else {
            ss << " " << std::flush;
        }
        if (i % unit_length == unit_length - 1) {
            ss << std::endl;
        }
    }
    ss << std::endl;
    return ss.str();
}