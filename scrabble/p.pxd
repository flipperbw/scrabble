cimport cython
cimport numpy as cnp

DEF MAX_NODES = 15
DEF MAX_ORD = 127  # todo replace 127

ctypedef cnp.uint32_t INT_32
ctypedef unsigned short STRU_t
ctypedef cnp.int32_t STR_t
ctypedef cnp.uint8_t BOOL_t
ctypedef cnp.intp_t SIZE_t

ctypedef const void * c_void

ctypedef unsigned char uchr
ctypedef unsigned char* uchrp
ctypedef const unsigned char cuchr
ctypedef const unsigned char* cuchrp
ctypedef const char cchr
ctypedef const char* cchrp

#ctypedef unsigned short us
#ctypedef (us, us) dual
#ctypedef us dual[2]

# cdef packed struct lpts_t:
#     uchr amt
#     uchr pts

cdef object npz
cdef object npe

ctypedef packed struct Letter:
    BOOL_t is_blank
    BOOL_t from_rack
    BOOL_t pts
    BOOL_t x
    BOOL_t y
    #STR_t value
    uchr value
    # todo define getter


ctypedef packed struct Letter_List:
    Letter l[MAX_NODES]
    Py_ssize_t len


ctypedef packed struct WordDict:
    char word[64]
    #char* word
    #Letter_List* letters
    Letter_List letters
    BOOL_t is_col
    STRU_t pts

ctypedef packed struct WordDict_List:
    WordDict l[10000]  # todo catch 10000
    Py_ssize_t len


cdef void sol(WordDict wd, char* buff) nogil

#ctypedef long valid_let_t[MAX_ORD][2]

# cdef packed struct multiplier_t:
#     uchr amt
#     uchrp typ


ctypedef packed struct N:
    # - x: r/c, y: lval
    long pts_lets[2][MAX_ORD]

    # - x: r/c, y: lval
    BOOL_t valid_lets[2][MAX_ORD]
    #valid_let_t valid_lets[2]

    # - x: r/c, y: lens
    BOOL_t valid_lengths[2][MAX_ORD]  # technically only 15

    Letter letter

    BOOL_t mult_a
    BOOL_t mult_w

    BOOL_t pts

    bint is_start
    bint has_edge
    bint has_val

#
# ctypedef packed struct NList:
#     N l[MAX_NODES]
#     Py_ssize_t len


@cython.final(True)
cdef class Node:
    cdef N n

    cdef public long[:, :] plet_view
    cdef public BOOL_t[:, :] vlet_view
    cdef public BOOL_t[:, :] vlen_view

    cdef Node up, down, left, right
    #cnp.ndarray[:] up_lets, down_lets, left_lets, right_lets
    #Letter[:] up_lets, down_lets, left_lets, right_lets

    cdef str up_word, down_word, left_word, right_word
    cdef STR_t up_pts, down_pts, left_pts, right_pts

    #def __cinit__?


# todo do I need final?
@cython.final(True)
cdef class Board:
    cdef:
        #object[:, :] board, default_board

        #nodelist_t nodes
        Node[:, ::1] nodes
        N nodesn[MAX_NODES][MAX_NODES]
        N[:, :] nodesnv

        WordDict_List words

        bint new_game

    cdef void _set_edge(self, Py_ssize_t r, Py_ssize_t c)
    cdef void _set_adj_words(self, Node n, Py_ssize_t d)
    cdef void _set_lets(self, Node n)
    cdef bint _check_adj_words(self, BOOL_t i, Node bef, Node aft, str bef_w, str aft_w)
    cdef void _set_map(self, Node[:] nodes, bint is_col)


@cython.final(True)
cdef class CSettings:
    cdef:
        #object[:, ::1] board, default_board
        char[:, ::1] board
        object[:, :] default_board
        #Node[:, :] board, default_board  # todo

        # ord = uint8
        BOOL_t[::1] points
        BOOL_t[::1] amts

        # ord = uint8
        int rack[MAX_ORD]
        int[:] rack_v

        #(Py_ssize_t, Py_ssize_t) shape
        Py_ssize_t shape[2]

        list rack_l
        Py_ssize_t rack_s
        BOOL_t blanks

        #set words
        frozenset words
        Board node_board
        int num_results


cdef STRU_t calc_pts(Letter_List lets_info, N[:] nodes, bint is_col, Py_ssize_t start) nogil

#cdef bint lets_match(STR_t[::1] word, Py_ssize_t wl, valid_let_t[:] vl_list, Py_ssize_t start) nogil
cdef bint lets_match(STR_t[::1] word, Py_ssize_t wl, N[:] nodes, Py_ssize_t start, bint is_col) nogil

cdef bint rack_check(STR_t[::1] word, Py_ssize_t wl, bint nvals[MAX_NODES], Py_ssize_t start, BOOL_t blanks, int[:] base_rack) nogil

cdef Letter_List rack_match(STR_t[::1] word, Py_ssize_t wl, N[:] nodes, Py_ssize_t start, int[:] base_rack) nogil

cdef void parse_nodes(N[:] nodes, STR_t[:, ::1] sw, SIZE_t[::1] swlens, bint is_col) nogil


cdef void print_board(uchr[:, ::1] nodes, Letter_List lets) nogil
cdef int mycmp(c_void pa, c_void pb) nogil
cdef void show_solution(uchr[:, ::1] nodes, WordDict_List words, bint no_words) nogil


cpdef object checkfile(tuple paths, bint is_file=*)

cdef void solve(str dictionary) except *

cdef void cmain(str filename, str dictionary, bint no_words, list exclude_letters, int num_results, str log_level) except *

# todo check if need func sig
