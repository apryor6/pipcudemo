# Building a pip-installable Python package that invokes custom CUDA code  

*The code for this post is located [here](https://github.com/apryor6/pipcudemo). To run the code you will need a CUDA-enabled GPU and an installation of the [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) *

Python is easy. GPUs are fast. Combining Python and CUDA is a fantastic way to leverage the performance benefits and parallelism inherent in GPUs with all of the syntactic niceties of Python. However, there is a potential snag. If the package is written purely in Python, C, or C++, distribution is easily accomplished using `setuptools` and users can install your package and its dependencies with either `pip install` or through the `setup.py` script. Unfortunately, `setuptools` is not currently compatible with NVIDIA's compiler, `nvcc`, which is necessary for compiling CUDA code. So how can we make a Python package that involves CUDA? One solution to this problem is to compile the CUDA code into a shared library which is then linked against by the Python package just like any other C/C++ library. In this tutorial I will walk through all of the steps necessary to write a C++/CUDA library and then to create a Python extension package that can be uploaded to PyPi and subsequently installed by users with `pip`. I'm a big fan of [CMake](https://cmake.org/), and will use it to build the shared library.   

The functions implemented in this example will be simple to illustrate the methodology, but if you are interested in "real-world" example, I used this technique when developing `PyPrismatic`, a GPU-accelerated package for image simulation in electron microscopy as part of my Ph.D. Full details of this can be found at [www.prism-em.com](www.prism-em.com), which includes a walkthrough of the source code (you can also read a free-print version of the associated academic paper [here](https://arxiv.org/abs/1706.08563)).

## The CUDA/C++ Code

Our library is going to be a simple calculator that can add, subtract, multiply, and divide two integers. Be aware that this is actually a bad application of GPUs, but it is intended to be a simple example. If you were to instead add two arrays of integers, GPUs are great.  

What we will do is expose some C++ functions that our Python package will be able to invoke, and the implementation of those functions will invoke CUDA code. The key is that Python never directly sees CUDA. It sees a C++ function signature (which it knows how to work with), but the actual implementation of that function is delayed until the linking process, and by that point the CUDA code will have been compiled.  

First, let's prototype our C++/CUDA functions

~~~ c++
// "myCuda.cuh"

#ifndef MYCUDA_H
#define MYCUDA_H

namespace pipcudemo{

    int add(int a, int b);	
	int subtract(int a, int b);
	int multiply(int a, int b);
	int divide(int a, int b);

}

#endif //MYCUDA_H

~~~

Super simple. I wrapped it in a namespace `pipcudemo` to prevent any name conflicts with the standard library and because I think it is clearer.   

The implementation is just a wrapper which invokes a `__host__` CUDA function.  

### Error checking

CUDA API calls return error codes, and it is the responsibility of the programmer to monitor them. A very helpful macro is discussed [in this StackOverflow thread](https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api). You just wrap CUDA API calls in this macro and if a failure occurs at runtime it will gracefully exit and print a useful error message.

~~~ c++
// define a helper function for checking CUDA errors.
#define cudaErrchk(ans) { GPUAssert((ans), __FILE__, __LINE__); }
inline void GPUAssert(cudaError_t code, const char *file, int line, bool abort=true){
	if (code != cudaSuccess)
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
~~~

Next let's focus on implementing one of our functions, `add`.  In the `__host__` function, we must allocate memory on the GPU, copy our inputs, launch a CUDA kernel to perform the calculation, and then copy the result back. Our function takes in two arguments, and we must also allocate a third for the result. The kernel, which is indicated by the `__global__` keyword, is trivial -- we just make thread 0 perform the arithmetic. In a proper GPU application, you would normally specify many thousands or more threads in the launch configuration that would do work in parallel, but here we just create one thread.

~~~ c++
// within myCuda.cu

	__global__ void add_d(int *a_d, int *b_d, int *c_d){
		if (threadIdx.x == 0){
			*c_d = *a_d + *b_d;
		}
	}

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

~~~

The "_d" suffix is commonly used to indicate names of objects associated with the device. The rest of the implementations for our calculator are virtually identical, and here is the full implementation.

~~~ c++
//myCuda.cu

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

~~~

## Creating the shared library with CMake

We will next create a shared library called `mylib` that exposes these 4 functions `add`, `subtract`, `multiply`, and `divide`. Good practice is to have a single header file for the library. In our case it is just a single `#include` statement

~~~ c++
// mylib.h

#ifndef MYLIB_H
#define MYLIB_H

#include "myCuda.cuh"

#endif //MYLIB_H
~~~

For the CMake infrastructure, we make use of [FindCUDA](https://cmake.org/cmake/help/v3.0/module/FindCUDA.html) to create a minimalistic `CMakeLists.txt`:

~~~
# CMakeLists.txt

cmake_minimum_required(VERSION 3.0)
project(pipcudemo)

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_30)

cuda_add_library(mylib SHARED myCuda.cu)

~~~

The line `find_package(CUDA REQUIRED)` attempts to locate a CUDA installation on the user's system and populates several variables containing include/library paths. I add `-arch=sm_30` to `CUDA_NVCC_FLAGS` to avoid a current warning about future deprecation of older architectures. Last, we create the shared library with the call to `cuda_add_library`.   

At this point, it would be smart to make a test program in C++ to make sure everything works. Here is a small driver program that will test our calculator functionality with a few `assert` statements.

~~~ c++
// testDriver.cpp

#include <cassert>
#include <iostream>
#include "myLib.h"

int main(){

assert(pipcudemo::add(3, 3) == 6);
std::cout << "3 + 3 = 6" << std::endl;

assert(pipcudemo::subtract(3, 3) == 0);
std::cout << "3 - 3 = 0" << std::endl;

assert(pipcudemo::multiply(3, 3) == 9);
std::cout << "3 * 3 = 9" << std::endl;

assert(pipcudemo::divide(3, 3) == 1);
std::cout << "3 / 3 = 1" << std::endl;

std::cout << "Test passed." << std::endl;

}

~~~

We can also compile and link this executable by adding two more lines to our `CMakeLists.txt`

~~~
# CMakeLists.txt

cmake_minimum_required(VERSION 3.0)
project(pipcudemo)

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_30)

cuda_add_library(mylib SHARED myCuda.cu)

cuda_add_executable(test_mylib testDriver.cpp)
target_link_libraries(test_mylib mylib)

~~~

The call to `cuda_add_executable` will create an executable called `test_mylib` using the driver program we just wrote. We must remember to link with our shared library using `target_link_libraries`, otherwise the driver won't have access to the calculator implementations.   

A little more detail on what is going on during compilation might be helpful. The `#include` statements are only pulling in the function prototypes, which is telling our program "these are functions that I promise will be implemented". The compiler is going to then demand that it finds implementations of the function prototypes it was promised. Otherwise, a compilation error will be thrown. These implementations can either be found within source files that are compiled into the executable or can exist in libraries against which the executable is linked. If the library is linked statically, then the function definitions are "hard-coded" into the executable just as if you had provided the function definitions as source files. Linking instead against a shared library (called "dynamic" linking) will tell our program "find me at runtime, and I promise I will have these function definitions ready for you", and that is enough to satisfy the compiler. Static linking results in fewer files, but the executables are larger and recompilation of the entire program is required if any part of it changes. Dynamic linking is more modular, but requires that the shared libraries be found at runtime. There are advantages and disadvantages to both. I've always run into issues when attempting to statically link CUDA libraries, so I chose the dynamic approach here.  

To build everything, we follow normal CMake practice and make a build directory.

~~~
mkdir build
cd build
~~~

Then invoke `cmake`

~~~
cmake ../
make 
~~~

If your CUDA installation is in a standard place like /usr/local/ on *nix systems then this may be all that is necessary. Otherwise, you may need to manually specify the location of CUDA and reconfigure.

If everything went smoothly, you can run the test program with

~~~
./test_mylib
~~~

and should see some text print including "Test passed." Shared libraries have different names depending on your operating system, but will be something like "libmylib.so", "libmylib.dylib", "mylib.dll", etc. The executable needs to find the shared library, and thus it may be necessary to add the library to the search path. The variable containing paths to shared libraries is `LD_LIBRARY_PATH` on Linux, `DYLD_LIBRARY_PATH` on Mac, and `PATH`.  

Now that we have a working CUDA library, we can move on to creating a Python wrapper for it.

## Creating the Python package

There are a few necessary ingredients for making a Python package

1. A `setup.py` script
2. A folder of the same name as the package
3. An `__init__.py` file within the package folder

And a few ingredients necessary for a Python C++ extension

1. C/C++ function(s) that parse arguments passed from Python, invoke the desired C/C++ code, and construct Python return values.
2. A PyInit_modulename method that initializes the module name "modulename"
3. A method definition table that maps the desired name of the function in Python to the actual C/C++ methods and defines the calling signature
4. A module definition structure that connects the method definition table to the module initialization function