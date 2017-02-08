#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#define NUMBERS 512*1024*1024 
int* generateRandomArray(int num){
	int *result;
	result = (int*)malloc(sizeof(int) * num);
	for (int i = 0; i < num; i++){
		result[num] = rand() % 20 - 10;
	}
	return result;

}


double CPUtime(){
       struct timeval tp;
       gettimeofday (&tp, NULL);
       return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

int main(){

	int *h_array = generateRandomArray(NUMBERS); //512M numbers
	int bytes = sizeof(int) * NUMBERS;

	//alloc mem on GPU
	int *d_array;
	cudaMalloc((void *)d_array, bytes);

	//copy data to GPU
	cudaMemcpy(d_array, h_array, bytes, cudaMemcpyHostToDevice);

	//do the work
	//define struct
	dim3 block(%d, %d);
	dim3 grid(%d, %d, %d);
	
	//copy back the result




}


