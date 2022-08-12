#include <stdio.h>
#include <cuda_runtime_api.h>

int main(int argc, char **argv){
    struct cudaDeviceProp dev_prop;

    int rc = cudaGetDeviceProperties(&dev_prop, 0);
    if(rc != cudaSuccess) {
        cudaError_t error = cudaGetLastError();
        printf("CUDA error: %s", cudaGetErrorString(error));
        return rc; /* Failure */
    }
    else {
        printf("%d%d", dev_prop.major, dev_prop.minor);
        return 0; /* Success */
    }
}