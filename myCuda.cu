#include "myCuda.cuh"
#include <cstdio>

#define cudaErrchk(ans) { GPUAssert((ans), __FILE__, __LINE__); }
inline void GPUAssert(cudaError_t code, const char *file, int line, bool abort=true){
	if (code != cudaSuccess)
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

namespace myCuda{

	 
	__global__ void add_d(int *a_d, int *b_d, int *c_d){
		if (threadIdx.x == 0){
			*c_d = *a_d + *b_d;
		}
	}

	
	int add(int a, int b){
		int *a_d, *b_d, *c_d, result;
		cudaErrchk(cudaMalloc(&a_d, sizeof(int)));
		cudaErrchk(cudaMalloc(&b_d, sizeof(int)));
		cudaErrchk(cudaMalloc(&c_d, sizeof(int)));
		cudaErrchk(cudaMemcpy(a_d, &a, sizeof(int), cudaMemcpyHostToDevice));
		cudaErrchk(cudaMemcpy(b_d, &b, sizeof(int), cudaMemcpyHostToDevice));
		add_d<<<1,1>>>(a_d, b_d, c_d);
		cudaErrchk(cudaMemcpy(&result, c_d, sizeof(int), cudaMemcpyDeviceToHost));

		return result;
	}

	
	T subtract(T a, T b){
	}

	
	T multiply(T a, T b){
	}

	
	T divide(T a, T b){
	}


}