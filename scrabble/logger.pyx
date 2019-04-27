# cython: warn.maybe_uninitialized=True, warn.undeclared=True, warn.unused=True, warn.unused_arg=True, warn.unused_result=True, infer_types.verbose=True

from libc.stdio cimport printf, puts

cdef object log_init
cdef str DEFAULT_LOGLEVEL

from .logs import log_init
from .settings import DEFAULT_LOGLEVEL


DEF NUL = b'\0'

DEF l__ = b'(_) '
DEF l_x = b'(x) '
DEF l_d = b'(d) '
DEF l_v = b'(v) '
DEF l_i = b'(i) '
DEF l_n = b'(n) '
DEF l_w = b'(w) '
DEF l_s = b'(s) '
DEF l_e = b'(e) '
DEF l_c = b'(c) '
DEF l_a = b'(a) '

DEF _ks = b'\x1B['
#DEF _ke = b'm'
DEF K_RES = 0
DEF K_BLK = 30
DEF K_RED = 31
DEF K_GRN = 32
DEF K_YEL = 33
DEF K_BLU = 34
DEF K_MAG = 35
DEF K_CYN = 36
DEF K_WHT = 37
DEF K_BLK_L = 90
DEF K_RED_L = 91
DEF K_GRN_L = 92
DEF K_YEL_L = 93
DEF K_BLU_L = 94
DEF K_MAG_L = 95
DEF K_CYN_L = 96
DEF K_WHT_L = 97

cdef bytes _ks_res = _ks + b'0m'
cdef bytes _ks_blk = _ks + b'30m'
cdef bytes _ks_red = _ks + b'31m'
cdef bytes _ks_grn = _ks + b'32m'
cdef bytes _ks_yel = _ks + b'33m'
cdef bytes _ks_blu = _ks + b'34m'
cdef bytes _ks_mag = _ks + b'35m'
cdef bytes _ks_cyn = _ks + b'36m'
cdef bytes _ks_wht = _ks + b'37m'
cdef bytes _ks_blk_l = _ks + b'90m'
cdef bytes _ks_red_l = _ks + b'91m'
cdef bytes _ks_grn_l = _ks + b'92m'
cdef bytes _ks_yel_l = _ks + b'93m'
cdef bytes _ks_blu_l = _ks + b'94m'
cdef bytes _ks_mag_l = _ks + b'95m'
cdef bytes _ks_cyn_l = _ks + b'96m'
cdef bytes _ks_wht_l = _ks + b'97m'

cdef cchrp KS_RES = _ks_res
cdef cchrp KS_BLK = _ks_blk
cdef cchrp KS_RED = _ks_red
cdef cchrp KS_GRN = _ks_grn
cdef cchrp KS_YEL = _ks_yel
cdef cchrp KS_BLU = _ks_blu
cdef cchrp KS_MAG = _ks_mag
cdef cchrp KS_CYN = _ks_cyn
cdef cchrp KS_WHT = _ks_wht
cdef cchrp KS_BLK_L = _ks_blk_l
cdef cchrp KS_RED_L = _ks_red_l
cdef cchrp KS_GRN_L = _ks_grn_l
cdef cchrp KS_YEL_L = _ks_yel_l
cdef cchrp KS_BLU_L = _ks_blu_l
cdef cchrp KS_MAG_L = _ks_mag_l
cdef cchrp KS_CYN_L = _ks_cyn_l
cdef cchrp KS_WHT_L = _ks_wht_l


cdef object lo = log_init(DEFAULT_LOGLEVEL)
cdef int lo_lvl = lo.get_level(DEFAULT_LOGLEVEL)


#todo add bold and stuff
cdef void clog(cuchr[:] ctxt, Py_ssize_t ts, int c, bint bold = False) nogil:
    cdef Py_ssize_t i = 0

    printf('%s', KS_BLK_L)
    while i < 3:
        printf('%c', ctxt[i])
        i += 1

    printf('%s%s%d;%dm', KS_RES, _ks, bold, c)
    while i < ts:
        printf('%c', ctxt[i])
        i += 1
    puts(KS_RES)

# todo check if this is actually faster

#def chklog(s, int lvl, *ar) -> cchr[:]:
cdef cuchr[:] chklog(s, int lvl):
    if s is None: return NUL
    if lo_lvl > lvl: return NUL
    if type(s) is not unicode: return NUL

    cdef bytes s_st
    if lvl == 0:    s_st = l__
    elif lvl == 5:  s_st = l_x
    elif lvl == 10: s_st = l_d
    elif lvl == 15: s_st = l_v
    elif lvl == 20: s_st = l_i
    elif lvl == 25: s_st = l_n
    elif lvl == 30: s_st = l_w
    elif lvl == 35: s_st = l_s
    elif lvl == 40: s_st = l_e
    elif lvl == 50: s_st = l_c
    elif lvl == 60: s_st = l_a
    else: s_st = bytes(lvl)

    #cdef str sp = '(' + s_st + ') ' + s
    #s = '(' + s_st + ') ' + s
    #cdef bytes sb = <bytes>(<unicode>s).encode('utf-8')
    #sb = (<unicode>s).encode('utf8')
    #return sb

    #s = s % ar

    #return <bytes>s_st + <bytes>((<unicode>s).encode('utf8'))
    return <bytes>s_st + (<unicode>s).encode('utf8')


cdef void clos(cchrp s, ...) nogil:
    cdef va_list args
    printf('%s%s%s %s', KS_BLK_L, l_s, KS_RES, KS_GRN_L)
    va_start(args, <void*>s)
    vprintf(s, args)
    va_end(args)
    puts(KS_RES)


cdef void los(s):
    cdef LogLvl lvl = SUCCESS ##
    cdef Py_ssize_t color = K_GRN_L ##

    cdef cuchr[:] txt = chklog(s, lvl)
    cdef Py_ssize_t ts = len(txt)

    if ts > 3: clog(txt, ts, color) #bold

cdef void loe(s):
    cdef LogLvl lvl = ERROR
    cdef Py_ssize_t color = K_RED
    cdef cuchr[:] txt = chklog(s, lvl)
    cdef Py_ssize_t ts = len(txt)
    if ts > 3: clog(txt, ts, color)

cdef void loi(s):
    cdef LogLvl lvl = INFO
    cdef Py_ssize_t color = K_CYN_L
    cdef cuchr[:] txt = chklog(s, lvl)
    cdef Py_ssize_t ts = len(txt)
    if ts > 3: clog(txt, ts, color)
