ctypedef const char cchr
ctypedef const char* cchrp
ctypedef unsigned char uchr
ctypedef unsigned char* uchrp
ctypedef const unsigned char cuchr
ctypedef const unsigned char* cuchrp

#from cpython.getargs cimport

cdef bint can_log(Py_UNICODE lvl) nogil
cdef int lvl_alias[127]

cdef extern from "stdarg.h":
    ctypedef struct va_list:
        pass
    #ctypedef struct fake_type:
    #    pass
    void va_start(va_list, void* arg) nogil
    #void* va_arg(va_list, fake_type)
    void va_end(va_list) nogil
    #fake_type int_type "int"
    #fake_type char_type "char*"

cdef extern from 'stdio.h':
    void vprintf(const char* f, va_list arg) nogil


cdef void clog(cuchr[:] ctxt, Py_ssize_t ts, int c, bint bold=*) nogil
cdef cuchr[:] chklog(s, int lvl)

cdef void lox(s)
cdef void lod(s)
cdef void lov(s)
cdef void loi(s)
cdef void lon(s)
cdef void low(s)
cdef void los(s)
cdef void loe(s)
cdef void loc(s)

cdef void clos(cchrp s, ...) nogil

cdef object lo
cdef int lo_lvl

cdef cchrp KS_RES, KS_BLK, KS_RED, KS_GRN, KS_YEL, KS_BLU, KS_MAG, KS_CYN, KS_WHT, KS_BLK_L, KS_RED_L, KS_GRN_L, KS_YEL_L, KS_BLU_L, KS_MAG_L, KS_CYN_L, KS_WHT_L

ctypedef enum LogLvl:
    NOTSET = 0
    SPAM = 5
    DEBUG = 10
    VERBOSE = 15
    INFO = 20
    NOTICE = 25
    WARNING = 30
    SUCCESS = 35
    ERROR = 40
    CRITICAL = 50
    ALWAYS = 60

# ctypedef enum LvlAlias:
#     _ = 0
#     x = 5
#     d = 10
#     v = 15
#     i = 20
#     n = 25
#     w = 30
#     s = 35
#     e = 40
#     c = 50
#     a = 60
