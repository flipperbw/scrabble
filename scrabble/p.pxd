cimport cython

cimport numpy as cnp

DEF MAX_NODES = 15
DEF MAX_ORD = 127  # todo replace 127

ctypedef cnp.uint32_t STRU_t
ctypedef cnp.int32_t STR_t
ctypedef cnp.uint8_t BOOL_t
ctypedef cnp.intp_t SIZE_t

ctypedef packed struct Letter:
    BOOL_t is_blank
    BOOL_t from_rack
    BOOL_t pts
    BOOL_t x
    BOOL_t y
    STR_t value
    # todo define getter


@cython.final(True)
cdef class CSettings:
    cdef:
        #(Py_ssize_t, Py_ssize_t) shape
        Py_ssize_t shape[2]

        #object[:, ::1] board, default_board
        object[:, ::1] board
        object[:, :] default_board
        #Node[:, :] board, default_board  # todo

        # ord = uint8
        BOOL_t[::1] rack
        list rack_l

        Py_ssize_t rack_s

        BOOL_t blanks

        # ord = uint8
        BOOL_t[::1] points
        BOOL_t[::1] amts

        #set words
        frozenset words
        Board node_board
        int num_results


@cython.final(True)
cdef class WordDict:
    cdef:
        #str word
        STR_t[::1] word
        #Py_ssize_t wl, s
        BOOL_t wl
        STR_t s
        BOOL_t is_col
        readonly STR_t pts
        list letters  # of Letter
        #Letter[:] letters
        #Letter letters[100]

    cdef str sol(self)


#ctypedef long valid_let_t[MAX_ORD][2]


ctypedef packed struct N:
    Letter letter

    BOOL_t mult_a
    BOOL_t mult_w

    bint is_start
    bint has_edge
    bint has_val
    BOOL_t pts

    # x: r/c, y: lval
    long pts_lets[2][MAX_ORD]

    # x: r/c, y: lval
    bint valid_lets[2][MAX_ORD]
    #valid_let_t valid_lets[2]

    # x: r/c, y: lens
    bint valid_lengths[2][MAX_ORD]  # technically only 15


@cython.final(True)
cdef class Node:
    cdef N n

    cdef public long[:, :] plet_view
    cdef public bint[:, :] vlet_view
    cdef public bint[:, :] vlen_view

    cdef readonly str display

    cdef Node up, down, left, right
    #cnp.ndarray[:] up_lets, down_lets, left_lets, right_lets
    #Letter[:] up_lets, down_lets, left_lets, right_lets

    cdef str up_word, down_word, left_word, right_word
    cdef STR_t up_pts, down_pts, left_pts, right_pts


@cython.final(True)
cdef class Board:
    cdef:
        #object[:, :] board, default_board

        #nodelist_t nodes
        #cnp.ndarray nodes
        #cnp.ndarray[:, :] nodes
        Node[:, ::1] nodes

        Py_ssize_t nodes_rl, nodes_cl
        bint new_game

        list words  # todo of word_dicts, what about an object? _pyx_v_self->words = ((PyObject*)__pyx_t_4);
        #np.ndarray[dtype=word_dict] words
        #cnp.ndarray words
        #WordDict[:] words

    cdef Node get(self, int x, int y)
    cdef Node get_by_attr(self, str attr, v)
    cdef void _set_edge(self, Py_ssize_t r, Py_ssize_t c)
    cdef void _set_adj_words(self, Node n, str d)
    cdef void _set_lets(self, Node n)
    cdef bint _check_adj_words(self, BOOL_t i, Node bef, Node aft, str bef_w, str aft_w)
    cdef void _set_map(self, Node[:] nodes, bint is_col)


@cython.final(True)
cdef SIZE_t set_word_dict(STR_t[::1] ww, Py_ssize_t wl, N nodes[MAX_NODES], Letter[::1] lets_info, bint is_col, Py_ssize_t start) nogil

@cython.final(True)
#cdef bint lets_match(STR_t[::1] word, Py_ssize_t wl, valid_let_t[:] vl_list, Py_ssize_t start) nogil
cdef bint lets_match(STR_t[::1] word, Py_ssize_t wl, N nodes[MAX_NODES], Py_ssize_t start, bint is_col) nogil

@cython.final(True)
cdef bint rack_check(STR_t[::1] word, Py_ssize_t wl, bint nvals[MAX_NODES], Py_ssize_t start, BOOL_t blanks) nogil

@cython.final(True)
cdef Letter[::1] rack_match(STR_t[::1] word, Py_ssize_t wl, N nodes[MAX_NODES], Py_ssize_t start) nogil

@cython.final(True)
cdef void parse_nodes(N nodes[MAX_NODES], STR_t[:, ::1] sw, SIZE_t[::1] swlens, bint is_col) # nogil

@cython.final(True)
cdef void solve(str dictionary)

# todo check if need func sig
