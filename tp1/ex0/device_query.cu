#include <cuda_runtime.h>
#include <iostream>

int main(int argc, char** argv)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::cerr << "Getting GPU Data." << std::endl;

    for (int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        if (dev == 0)
        {
            if (deviceProp.major == 9999 && deviceProp.minor == 9999)
            {
                std::cerr << "No CUDA GPU has been detected" << std::endl;
                return -1;
            }
            else if (deviceCount == 1)
            {
                std::cerr << "There is 1 device supporting CUDA" << std::endl;
            }
            else
            {
                std::cerr << "There are " << deviceCount
                          << " devices supporting CUDA" << std::endl;
            }
        }

        std::cerr << "Device " << dev << " name: " << deviceProp.name << "."
                  << std::endl;

        std::cerr << " Computational Capabilities: " << deviceProp.major << "."
                  << deviceProp.minor << std::endl;

        std::cerr << " Maximum global memory size: "
                  << deviceProp.totalGlobalMem << std::endl;

        std::cerr << " Maximum constant memory size: "
                  << deviceProp.totalConstMem << std::endl;

        std::cerr << " Maximum shared memory size per block: "
                  << deviceProp.sharedMemPerBlock << std::endl;

        std::cerr << " Maximum block dimensions: "
                  << deviceProp.maxThreadsDim[0] << " x "
                  << deviceProp.maxThreadsDim[1] << " x "
                  << deviceProp.maxThreadsDim[2] << std::endl;

        std::cerr << " Maximum grid dimensions: " << deviceProp.maxGridSize[0]
                  << " x " << deviceProp.maxGridSize[1] << " x "
                  << deviceProp.maxGridSize[2] << std::endl;

        std::cerr << " Warp size: " << deviceProp.warpSize << std::endl;
    }

    std::cerr << "End of GPU data gathering." << std::endl;
    return 0;
}