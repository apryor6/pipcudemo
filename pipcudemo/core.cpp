#include "Python.h"
#include "myLib.h"

static PyObject* pipcudemo_core_add(PyObject *self, PyObject *args){
	int a,b;
		if (!PyArg_ParseTuple(args, "ii", &a, &b))return NULL;
		return Py_BuildValue("i", pipcudemo::add(a, b));
	}

static PyMethodDef pipcudemo_core_methods[] = {
	{"add",(PyCFunction)pipcudemo_core_add, METH_VARARGS, "Execute Prismatic calculation"},
   	{NULL, NULL, 0, NULL}
};


static struct PyModuleDef module_def = {
	PyModuleDef_HEAD_INIT,"pipcudemo.core","An example project showing how to build a pip-installable Python package that invokes custom CUDA/C++ code.",-1,pipcudemo_core_methods
};

PyMODINIT_FUNC PyInit_core(){
	return PyModule_Create(&module_def);
}
