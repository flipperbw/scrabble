ctypedef unsigned char uchr
ctypedef const char* cchrp

cdef Py_UNICODE u_dash, u_bx_ul, u_bx_ur, u_bx_bl, u_bx_br, u_sep_hor_le, u_sep_hor_ri, u_sep_ver_up, u_sep_ver_dn

cdef void print_board_top(Py_ssize_t iy) nogil
cdef void print_board_row(uchr[:, ::1] board) nogil
cdef void print_board_btm(Py_ssize_t iy) nogil

cdef void print_board(uchr[:, ::1] board) nogil
cdef void print_board_clr(uchr[:, ::1] board, cchrp c) nogil
