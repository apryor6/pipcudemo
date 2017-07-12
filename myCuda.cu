#include "myCuda.cuh"
#include <cstdio>


// define a helper function for checking CUDA errors. See this thread: https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define cudaErrchk(ans) { GPUAssert((ans), __FILE__, __LINE__); }
inline void GPUAssert(cudaError_t code, const char *file, int line, bool abort=true){
	if (code != cudaSuccess)
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

namespace pipcudemo{

	
	// implement the CUDA kernels

	__global__ void add_d(int *a_d, int *b_d, int *c_d){
		if (threadIdx.x == 0){
			*c_d = *a_d + *b_d;
		}
	}

	__global__ void subtract_d(int *a_d, int *b_d, int *c_d){
		if (threadIdx.x == 0){
			*c_d = *a_d - *b_d;
		}
	}


	__global__ void multiply_d(int *a_d, int *b_d, int *c_d){
		if (threadIdx.x == 0){
			*c_d = *a_d * *b_d;
		}
	}

	__global__ void divide_d(int *a_d, int *b_d, int *c_d){
		if (threadIdx.x == 0){
			*c_d = *a_d / *b_d;
		}
	}

	
	// implement the wrappers that copy memory and invoke the kernels
	__host__ int add(int a, int b){
		int *a_d, *b_d, *c_d, result;

		// allocate memory on device
		cudaErrchk(cudaMalloc(&a_d, sizeof(int)));
		cudaErrchk(cudaMalloc(&b_d, sizeof(int)));
		cudaErrchk(cudaMalloc(&c_d, sizeof(int)));

		// copy memory to device
		cudaErrchk(cudaMemcpy(a_d, &a, sizeof(int), cudaMemcpyHostToDevice));
		cudaErrchk(cudaMemcpy(b_d, &b, sizeof(int), cudaMemcpyHostToDevice));

		// do the calculation
		add_d<<<1,1>>>(a_d, b_d, c_d);

		// copy result back
		cudaErrchk(cudaMemcpy(&result, c_d, sizeof(int), cudaMemcpyDeviceToHost));

		return result;
	}

	
	__host__ int subtract(int a, int b){
		int *a_d, *b_d, *c_d, result;

		// allocate memory on device
		cudaErrchk(cudaMalloc(&a_d, sizeof(int)));
		cudaErrchk(cudaMalloc(&b_d, sizeof(int)));
		cudaErrchk(cudaMalloc(&c_d, sizeof(int)));

		// copy memory to device
		cudaErrchk(cudaMemcpy(a_d, &a, sizeof(int), cudaMemcpyHostToDevice));
		cudaErrchk(cudaMemcpy(b_d, &b, sizeof(int), cudaMemcpyHostToDevice));

		// do the calculation
		subtract_d<<<1,1>>>(a_d, b_d, c_d);

		// copy result back
		cudaErrchk(cudaMemcpy(&result, c_d, sizeof(int), cudaMemcpyDeviceToHost));

		return result;
	}

	
	__host__ int multiply(int a, int b){
		int *a_d, *b_d, *c_d, result;

		// allocate memory on device
		cudaErrchk(cudaMalloc(&a_d, sizeof(int)));
		cudaErrchk(cudaMalloc(&b_d, sizeof(int)));
		cudaErrchk(cudaMalloc(&c_d, sizeof(int)));

		// copy memory to device
		cudaErrchk(cudaMemcpy(a_d, &a, sizeof(int), cudaMemcpyHostToDevice));
		cudaErrchk(cudaMemcpy(b_d, &b, sizeof(int), cudaMemcpyHostToDevice));

		// do the calculation
		multiply_d<<<1,1>>>(a_d, b_d, c_d);

		// copy result back
		cudaErrchk(cudaMemcpy(&result, c_d, sizeof(int), cudaMemcpyDeviceToHost));

		return result;
	}

	
	__host__ int divide(int a, int b){
		int *a_d, *b_d, *c_d, result;

		// allocate memory on device
		cudaErrchk(cudaMalloc(&a_d, sizeof(int)));
		cudaErrchk(cudaMalloc(&b_d, sizeof(int)));
		cudaErrchk(cudaMalloc(&c_d, sizeof(int)));

		// copy memory to device
		cudaErrchk(cudaMemcpy(a_d, &a, sizeof(int), cudaMemcpyHostToDevice));
		cudaErrchk(cudaMemcpy(b_d, &b, sizeof(int), cudaMemcpyHostToDevice));

		// do the calculation
		divide_d<<<1,1>>>(a_d, b_d, c_d);

		// copy result back
		cudaErrchk(cudaMemcpy(&result, c_d, sizeof(int), cudaMemcpyDeviceToHost));

		return result;
	}

}