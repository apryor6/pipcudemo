#include "myLib.h"
#include "myCuda.cuh"

namespace pipcudemo{

	int add(int a, int b){
		return myCuda::add(a, b);
	}

	int subtract(int a, int b){
		return myCuda::subtract(a, b);
	}

	int multiply(int a, int b){
		return myCuda::multiply(a, b);
	}

	int divide(int a, int b){
		return myCuda::divide(a, b);
	}

}