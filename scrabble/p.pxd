cimport numpy as cnp

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


cdef class CSettings:
    cdef:
        #(Py_ssize_t, Py_ssize_t) shape
        Py_ssize_t shape[2]

        object[:, ::1] board
        object[::1, :] default_board
        #Node[:, :] board, default_board  # todo

        #list letters
        # ord = uint8
        BOOL_t[::1] rack

        Py_ssize_t rack_s

        BOOL_t blanks

        # ord = uint8
        BOOL_t[::1] points
        BOOL_t[::1] amts

        #set words
        frozenset words
        Board node_board
        int num_results


cdef class WordDict:
    cdef:
        #str word
        STR_t[::1] word
        #Py_ssize_t wl, s
        BOOL_t wl
        int s
        bint is_col
        readonly STR_t pts
        list letters  # of Letter
        #Letter[:] letters
        #Letter letters[100]


cdef class Node:
    #uchr x, y

    cdef Letter letter

    cdef BOOL_t mult_a
    cdef BOOL_t mult_w

    cdef bint is_start
    cdef bint has_val
    cdef BOOL_t pts

    cdef readonly str display

    cdef Node up, down, left, right
    #cnp.ndarray[:] up_lets, down_lets, left_lets, right_lets
    #Letter[:] up_lets, down_lets, left_lets, right_lets

    cdef str up_word, down_word, left_word, right_word
    cdef STR_t up_pts, down_pts, left_pts, right_pts

    cdef bint has_edge

    # x: r/c, y: lval, z: ok/pts
    cdef STR_t[:, :, ::1] valid_lets

    # x: r/c, y: lens
    cdef BOOL_t[:, ::1] valid_lengths


cdef class Board:
    cdef:
        #object[:, :] board, default_board
        bint new_game
        #nodelist_t nodes
        #cnp.ndarray nodes
        #cnp.ndarray[:, :] nodes
        Node[:, ::1] nodes
        Py_ssize_t nodes_rl, nodes_cl

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

cdef bint lets_match(STR_t[::1] word, Py_ssize_t wl, STR_t[:, :, :, ::1] vl_list, bint is_col, Py_ssize_t start) nogil
cdef bint rack_check(STR_t[::1] word, Py_ssize_t wl, BOOL_t[::1] nvals, Py_ssize_t start) nogil
# todo check if need func sig
