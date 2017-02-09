################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/reduction.cpp 

CU_SRCS += \
../src/reduction_kernel.cu 

CU_DEPS += \
./src/reduction_kernel.d 

OBJS += \
./src/reduction.o \
./src/reduction_kernel.o 

CPP_DEPS += \
./src/reduction.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I"/nfs/ug/homes-1/z/zhangw56/Desktop/cuda/samples/NVIDIA_CUDA-8.0_Samples/6_Advanced" -I"/nfs/ug/homes-1/z/zhangw56/Desktop/cuda/samples/NVIDIA_CUDA-8.0_Samples/common/inc" -I"/nfs/ug/homes-1/z/zhangw56/Desktop/cuda/samples/NVIDIA_CUDA-8.0_Samples/6_Advanced/reduction" -G -g -O0 -gencode arch=compute_52,code=sm_52 -m64 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I"/nfs/ug/homes-1/z/zhangw56/Desktop/cuda/samples/NVIDIA_CUDA-8.0_Samples/6_Advanced" -I"/nfs/ug/homes-1/z/zhangw56/Desktop/cuda/samples/NVIDIA_CUDA-8.0_Samples/common/inc" -I"/nfs/ug/homes-1/z/zhangw56/Desktop/cuda/samples/NVIDIA_CUDA-8.0_Samples/6_Advanced/reduction" -G -g -O0 --compile -m64  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I"/nfs/ug/homes-1/z/zhangw56/Desktop/cuda/samples/NVIDIA_CUDA-8.0_Samples/6_Advanced" -I"/nfs/ug/homes-1/z/zhangw56/Desktop/cuda/samples/NVIDIA_CUDA-8.0_Samples/common/inc" -I"/nfs/ug/homes-1/z/zhangw56/Desktop/cuda/samples/NVIDIA_CUDA-8.0_Samples/6_Advanced/reduction" -G -g -O0 -gencode arch=compute_52,code=sm_52 -m64 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I"/nfs/ug/homes-1/z/zhangw56/Desktop/cuda/samples/NVIDIA_CUDA-8.0_Samples/6_Advanced" -I"/nfs/ug/homes-1/z/zhangw56/Desktop/cuda/samples/NVIDIA_CUDA-8.0_Samples/common/inc" -I"/nfs/ug/homes-1/z/zhangw56/Desktop/cuda/samples/NVIDIA_CUDA-8.0_Samples/6_Advanced/reduction" -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_52,code=compute_52 -gencode arch=compute_52,code=sm_52 -m64  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


