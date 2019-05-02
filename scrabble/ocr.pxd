cimport cython

from scrabble cimport *

# cimport numpy as cnp
#
# ctypedef cnp.uint8_t BOOL_t
# ctypedef cnp.uint16_t UINT16_t
# ctypedef cnp.int_t INTC_t
# ctypedef cnp.npy_intp INTP_t
# ctypedef cnp.float32_t FLO_t
# ctypedef cnp.ndarray cnparr
# # ctypedef cnp.int32_t STR_t

#cdef object cv2inrange

#cdef int _img_cut_range[2][2][2][2]
#cdef list IMG_CUT_RANGE

@cython.final(True)
cdef class Dirs:
    cdef object this_board_dir, this_board, this_letters  # Path


cdef BOOL_t[:, :, :] get_img(str img_name)
cdef cnparr[BOOL_t, ndim=3] cut_img(BOOL_t[:, :, :] img, bint is_big, bint is_lets)

#cdef dict letter_templates
#cdef void create_letter_templates(bint is_big) except *


ctypedef packed struct seen_tup:
    FLO_t conf
    Py_UCS4 l


# cdef void find_letter_match(
#     cnparr[BOOL_t, ndim=2] gimg, bint is_rack, float spacing, char[:, ::1] dest
# ) except *

cpdef void show_img(cnparr img_array)

#cdef char[:, ::1] create_board(cnparr[BOOL_t, ndim=3] board, bint is_big)

#cpdef list get_rack(BOOL_t[:, :, :] img)


#cdef void cmain(str filename, bint overwrite, str log_level) except *


#cdef cnparr cv_read3(str f, bint fl)

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
