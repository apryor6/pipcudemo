#include <cassert>
#include <iostream>
#include "mylib.h"

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