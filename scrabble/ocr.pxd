cimport cython

from scrabble cimport *


@cython.final(True)
cdef class Dirs:
    cdef object this_board_dir, this_board, this_letters  # Path


ctypedef packed struct seen_tup:
    FLO_t conf
    Py_UCS4 l



#ctypedef int (*f_type)(str, bint)

# cdef extern from "Python.h":
#     ctypedef cnparr (*ftyp)(object, bint)
#     ftyp cv_readf

# cdef extern from "funcobject.h":
#     #PyAPI_DATA(PyTypeObject) PyFunction_Type
#     ctypedef struct PyTypeObject
#
# cdef extern from "Python.h":
#     #cdef extern PyFunction_Type
#     ctypedef PyTypeObject asd

# from cpython.object cimport PyObject_Call
# cdef inline cnparr pc(object o, object a):
#     return PyObject_Call(o, a, NULL)

##cdef inline PyObject* cca(PyObject *func, PyObject *arg, PyObject *kw);
