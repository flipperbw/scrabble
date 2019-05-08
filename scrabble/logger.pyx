# cython: warn.maybe_uninitialized=True, warn.undeclared=True, warn.unused=True, warn.unused_arg=True, warn.unused_result=True, infer_types.verbose=True

from libc.stdio cimport printf, puts

cdef str DEFAULT_LOGLEVEL
from scrabble.settings import DEFAULT_LOGLEVEL
#cdef object log_init
#from scrabble.logs import log_init


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

#cdef cchrp _ks = b'\x1B['
DEF _ks = b'\x1B['
#DEF _ke = b'm'

#cdef bytes _ks_res = _ks + b'0m'
DEF _ks_res = _ks + b'0m'
DEF _ks_blk = _ks + b'30m'
DEF _ks_red = _ks + b'31m'
DEF _ks_grn = _ks + b'32m'
DEF _ks_yel = _ks + b'33m'
DEF _ks_blu = _ks + b'34m'
DEF _ks_mag = _ks + b'35m'
DEF _ks_cyn = _ks + b'36m'
DEF _ks_wht = _ks + b'37m'
DEF _ks_blk_l = _ks + b'90m'
DEF _ks_red_l = _ks + b'91m'
DEF _ks_grn_l = _ks + b'92m'
DEF _ks_yel_l = _ks + b'93m'
DEF _ks_blu_l = _ks + b'94m'
DEF _ks_mag_l = _ks + b'95m'
DEF _ks_cyn_l = _ks + b'96m'
DEF _ks_wht_l = _ks + b'97m'

KS_RES = _ks_res
KS_BLK = _ks_blk
KS_RED = _ks_red
KS_GRN = _ks_grn
KS_YEL = _ks_yel
KS_BLU = _ks_blu
KS_MAG = _ks_mag
KS_CYN = _ks_cyn
KS_WHT = _ks_wht
KS_BLK_L = _ks_blk_l
KS_RED_L = _ks_red_l
KS_GRN_L = _ks_grn_l
KS_YEL_L = _ks_yel_l
KS_BLU_L = _ks_blu_l
KS_MAG_L = _ks_mag_l
KS_CYN_L = _ks_cyn_l
KS_WHT_L = _ks_wht_l

cdef int lvl_alias[127]
lvl_alias[ord('x')] = 5
lvl_alias[ord('d')] = 10
lvl_alias[ord('v')] = 15
lvl_alias[ord('i')] = 20
lvl_alias[ord('n')] = 25
lvl_alias[ord('w')] = 30
lvl_alias[ord('s')] = 35
lvl_alias[ord('e')] = 40
lvl_alias[ord('c')] = 50
lvl_alias[ord('a')] = 60


cdef int def_lvl = (<str>DEFAULT_LOGLEVEL.lower())[0]
cdef int lo_lvl = lvl_alias[def_lvl]

cdef bint can_log(Py_UNICODE lvl) nogil:
    cdef int lal = lvl_alias[lvl]
    return lo_lvl <= lal


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
    if lo_lvl > lvl: return NUL
    if s is None: return NUL
    #if type(s) is not unicode: return NUL  # todo

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
    else: s_st = b'(' + bytes(lvl) + b')'

    #cdef str sp = '(' + s_st + ') ' + s
    #s = '(' + s_st + ') ' + s
    #cdef bytes sb = <bytes>(<unicode>s).encode('utf-8')
    #sb = (<unicode>s).encode('utf8')
    #return sb

    #s = s % ar

    #return <bytes>s_st + <bytes>((<unicode>s).encode('utf8'))

    if type(s) is not unicode:
        return <bytes>s_st + str(s).encode('utf8')  # todo
    return <bytes>s_st + (<unicode>s).encode('utf8')


cdef void clos(cchrp s, ...) nogil:
    cdef va_list args
    printf('%s%s%s%s', KS_BLK_L, l_s, KS_RES, KS_GRN_L)
    va_start(args, <void*>s)
    vprintf(s, args)
    va_end(args)
    puts(KS_RES)


cdef void lox(s):
    cdef LogLvl lvl = SPAM
    cdef Py_ssize_t color = K_WHT
    cdef cuchr[:] txt = chklog(s, lvl)
    cdef Py_ssize_t ts = len(txt)
    if ts > 3: clog(txt, ts, color)

cdef void lod(s):
    cdef LogLvl lvl = DEBUG
    cdef Py_ssize_t color = K_CYN
    cdef cuchr[:] txt = chklog(s, lvl)
    cdef Py_ssize_t ts = len(txt)
    if ts > 3: clog(txt, ts, color)

cdef void lov(s):
    cdef LogLvl lvl = VERBOSE
    cdef Py_ssize_t color = K_BLU
    cdef cuchr[:] txt = chklog(s, lvl)
    cdef Py_ssize_t ts = len(txt)
    if ts > 3: clog(txt, ts, color)

cdef void loi(s):
    cdef LogLvl lvl = INFO
    cdef Py_ssize_t color = K_RES
    cdef cuchr[:] txt = chklog(s, lvl)
    cdef Py_ssize_t ts = len(txt)
    if ts > 3: clog(txt, ts, color)

cdef void lon(s):
    cdef LogLvl lvl = NOTICE
    cdef Py_ssize_t color = K_MAG
    cdef cuchr[:] txt = chklog(s, lvl)
    cdef Py_ssize_t ts = len(txt)
    if ts > 3: clog(txt, ts, color)

cdef void low(s):
    cdef LogLvl lvl = WARNING
    cdef Py_ssize_t color = K_YEL
    cdef cuchr[:] txt = chklog(s, lvl)
    cdef Py_ssize_t ts = len(txt)
    if ts > 3: clog(txt, ts, color)

cdef void los(s):
    cdef LogLvl lvl = SUCCESS ##
    cdef Py_ssize_t color = K_GRN_L ##

    cdef cuchr[:] txt = chklog(s, lvl)
    cdef Py_ssize_t ts = len(txt)

    if ts > 3: clog(txt, ts, color) #bold

cdef void loe(s):
    cdef LogLvl lvl = ERROR
    cdef Py_ssize_t color = K_RED_L
    cdef cuchr[:] txt = chklog(s, lvl)
    cdef Py_ssize_t ts = len(txt)
    if ts > 3: clog(txt, ts, color)

cdef void loc(s):
    cdef LogLvl lvl = CRITICAL
    cdef Py_ssize_t color = K_RED
    cdef cuchr[:] txt = chklog(s, lvl)
    cdef Py_ssize_t ts = len(txt)
    if ts > 3: clog(txt, ts, color)
