# cython: warn.maybe_uninitialized=True, warn.undeclared=True, warn.unused=True, warn.unused_arg=True, warn.unused_result=True, infer_types.verbose=True

from libc.stdio cimport printf, puts

from scrabble.logger cimport KS_RES


u_dash = '\u2500'
u_bx_ul = '\u250c'
u_bx_ur = '\u2510'
u_bx_bl = '\u2514'
u_bx_br = '\u2518'
u_sep_hor_le = '\u2524'
u_sep_hor_ri = '\u251C'
u_sep_ver_up = '\u2534'
u_sep_ver_dn = '\u252C'


cdef void print_board_top(Py_ssize_t iy) nogil:
    # todo fix small 10.
    cdef int smalltens = 9361
    cdef Py_UNICODE smallten_char
    cdef Py_ssize_t i = 0, j = 0

    # - col nums
    printf('      ')
    while i < iy:
        if i < 10 or i > 20:
            printf('%zu', i)
        else:
            smallten_char = smalltens + (i - 10)
            printf('%lc', smallten_char)
        if i != iy - 1:
            printf(' ')
        i += 1
    printf('\n')

    # - col seps top
    printf('   %lc%lc', u_bx_ul, u_dash)
    while j < iy:
        printf('%lc%lc', u_dash, u_sep_ver_up)
        j += 1
    printf('%lc%lc%lc\n', u_dash, u_dash, u_bx_ur)


cdef void print_board_row(uchr[:, ::1] board) nogil:
    cdef Py_ssize_t i, j
    #cdef Py_ssize_t ix = board.shape[0]
    #cdef Py_ssize_t *iy = &board.shape[1]  # todo diff?
    cdef uchr nval

    # - rows
    for i in range(board.shape[0]):
        printf('%2zu %lc  ', i, u_sep_hor_le)
        for j in range(board.shape[1]):
            nval = board[i, j]
            if not nval:
                printf(' ')
            else:
                printf('%c', nval)

            if j != board.shape[1] - 1:
                printf(' ')

        printf('  %lc\n', u_sep_hor_ri)


cdef void print_board_btm(Py_ssize_t iy) nogil:
    cdef Py_ssize_t i = 0

    # - col seps bottom
    printf('   %lc%lc', u_bx_bl, u_dash)
    while i < iy:
        printf('%lc%lc', u_dash, u_sep_ver_dn)
        i += 1
    printf('%lc%lc%lc\n', u_dash, u_dash, u_bx_br)


cdef void print_board(uchr[:, ::1] board) nogil:
    print_board_top(board.shape[1])
    print_board_row(board)
    print_board_btm(board.shape[1])


cdef void print_board_clr(uchr[:, ::1] board, cchrp c) nogil:
    printf('%s', c)
    print_board(board)
    puts(KS_RES)
