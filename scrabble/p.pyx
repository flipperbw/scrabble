# cython: warn.maybe_uninitialized=True, warn.undeclared=True, warn.unused=True, warn.unused_arg=True, warn.unused_result=True, infer_types.verbose=True

"""Parse and solve a scrabble board."""

cimport cython
#from cython.parallel import prange

from libc.stdio cimport printf, snprintf, puts
from libc.stdlib cimport qsort, malloc, free

#from scrabble import logger as log
from scrabble cimport logger as log
from scrabble.logger cimport can_log, los, low, lov, loe, loc, clos
from scrabble.utils cimport print_board_clr, print_board_top, print_board_btm

cdef object json, pickle, sys, np, Path, _s
#cdef module np

import json
import pickle
import sys

import numpy as np
from pathlib import Path

import scrabble.settings as _s

#import signal
#from functools import lru_cache


#cnp.import_array()
#cnp.import_umath()


cdef type STR = <type>np.int32
cdef type BOOL = <type>np.uint8
cdef type SIZE = <type>np.intp

cdef object npz = np.zeros
cdef object npe = np.empty

#DTYPE = np.intc

# todo no numpy array

#ctypedef np.ndarray[str, ndim=2] board_t  # str 32?
#ctypedef np.ndarray[object, ndim=2] nb_t
#ctypedef np.ndarray[object, ndim=1] nodelist_t

#todo more of these
#cdef bytes NUL = b'\0'
DEF NUL = b'\0'
DEF BL = ord('?')
DEF MAX_BL = 2
DEF L_ST = 65
DEF L_EN = 91
DEF MAX_NODES = 15
DEF MAX_ORD = 127  # todo replace 127

#cdef object lo = log.lo
#los = log.los


@cython.final(True)
cdef class CSettings:
    def __cinit__(self):
        #self.shape = (0, 0)
        self.shape[0] = 0
        self.shape[1] = 0

        #self.board = None
        # self.board = np.zeros(self.shape, np.object_)
        # self.default_board = np.zeros(self.shape, np.object_)
        self.board = None
        self.default_board = None

        cdef Py_ssize_t i = 0
        while i < MAX_ORD:
            self.rack[i] = 0
            i += 1

        #self.rack_v = self.rack

        self.rack_l = []
        self.rack_s = 0
        self.blanks = 0

        self.include_lets = []
        for i in range(7):
            self.include_lets_c[i] = 0

        self.words = frozenset()  ## type: Set[bytes]
        self.node_board = None
        self.num_results = _s.NUM_RESULTS


cdef CSettings Settings = CSettings()


#ctypedef class WordDict:

#@cython.freelist(10000)

# *
@cython.wraparound(False)
cdef void sol(WordDict wd, char* buff) except *: # nogil:
    cdef Letter lf = wd.letters.l[0]
    cdef Letter ll = wd.letters.l[wd.letters.len - 1]

    cdef char direc
    cdef BOOL_t p1, p2, p3

    if wd.is_col:
        direc = b'c'
        p1 = lf.y
        p2 = lf.x
        p3 = ll.x
    else:
        direc = b'r'
        p1 = lf.x
        p2 = lf.y
        p3 = ll.y

    cdef Py_ssize_t i = 0
    cdef uchr word[MAX_NODES]

    while i < MAX_NODES:
        if i < wd.letters.len:
            word[i] = wd.letters.l[i].value
        else:
            word[i] = NUL
        i += 1

    #cdef char* buffz = wd.word

    snprintf(buff, 64, 'pts: %3i | dir: %c | pos: %2u x %2u,%2u | w: %s',
        wd.pts, direc, p1, p2, p3, word
    )


@cython.final(True)
cdef class Node:
    # todo test no types in cinit
    def __cinit__(self, BOOL_t x, BOOL_t y, uchr val, BOOL_t mult_a, BOOL_t mult_w):
        self.n.letter.x = x
        self.n.letter.y = y

        self.n.letter.is_blank = False
        self.n.letter.from_rack = False
        #self.letter.pts = 0

        self.n.mult_a = mult_a
        self.n.mult_w = mult_w

        #self.n.is_start = is_start
        self.n.has_edge = False

        #cdef lpts_t lpt

        if not val:
            self.n.has_val = False
            self.n.letter.value = 0
            self.n.pts = 0
            #self.display = ' '  # todo display method
        else:
            self.n.has_val = True
            self.n.letter.value = val
            self.n.pts = Settings.points[self.n.letter.value]
            #self.display = val.upper()

            # try:
            #     lpt = Settings.points[self.value]
            #     self.points = lpt.pts
            # except (KeyError, IndexError):
            #     lo.e('Could not get point value of "{}"'.format(val))
            #     sys.exit(1)

        self.n.letter.pts = self.n.pts

        self.up = None
        self.down = None
        self.left = None
        self.right = None

        # self.up_lets = None
        # self.down_lets = None
        # self.left_lets = None
        # self.right_lets = None

        self.up_word = ''
        self.down_word = ''
        self.left_word = ''
        self.right_word = ''

        self.up_pts = 0
        self.down_pts = 0
        self.left_pts = 0
        self.right_pts = 0

        # cdef cnp.npy_intp dims_vlets[3]
        # dims_vlets[0] = 2
        # dims_vlets[1] = MAX_ORD
        # dims_vlets[2] = 2
        # self.n.valid_lets = cnp.PyArray_ZEROS(3, dims_vlets, cnp.NPY_INT32, 0)

        # cdef cnp.npy_intp dims_vlens[2]
        # dims_vlens[0] = 2
        # dims_vlens[1] = Settings.shape[0] if Settings.shape[0] > Settings.shape[1] else Settings.shape[1]
        # self.valid_lengths = cnp.PyArray_ZEROS(2, dims_vlens, cnp.NPY_UINT8, 0)

        self.plet_view = self.n.pts_lets
        self.plet_view[:, :] = 0

        self.vlet_view = self.n.valid_lets
        self.vlet_view[:, :] = False

        self.vlen_view = self.n.valid_lengths
        self.vlen_view[:, :] = False


    def str_pos(self) -> str:
        return f'[{self.n.letter.x:2d},{self.n.letter.y:2d}]'

    def __str__(self) -> str:
        cdef str s
        if self.n.has_val:
            s = chr(self.n.letter.value)  # todo switch to c stuff
        else:
            s = '_'
        return '<Node: {} v: {}>'.format(self.str_pos(), s)

    def __repr__(self) -> str:
        return self.__str__()


@cython.final(True)
cdef class Board:
    def __cinit__(self, uchr[:, ::1] board, object[:, :] default_board):  # todo fix c contig
        cdef:
            Py_ssize_t r, c, du
            str dbval, x #, d, bval
            uchr bval
            BOOL_t mult_a, mult_w
            Node node
            #cnp.ndarray board

        #self.board = board
        #self.default_board = default_board

        self.nodes = np.zeros_like(default_board, Node)
        #self.nodes = np.empty_like(default_board, Node)

        #self.nodes_rl = Settings.shape[0] # default board shape instead
        #self.nodes_cl = Settings.shape[1]

        self.new_game = False

        self.words.len = 0
        #self.words_view = self.words.l

        for r in range(default_board.shape[0]):
            for c in range(default_board.shape[1]):
                bval = board[r, c]
                dbval = default_board[r, c]

                mult_a = 1
                mult_w = 0

                if dbval:
                    dbval = dbval.strip()
                    if dbval == 'x':
                        if not bval:
                            self.new_game = True
                    else:
                        x = (<str>dbval)[0]
                        mult_a = <BOOL_t>int(x)
                        if (<str>dbval)[1] == 'w':
                            mult_w = 1

                node = Node(<BOOL_t>r, <BOOL_t>c, bval, mult_a, mult_w)
                self.nodes[r, c] = node

        cdef Py_ssize_t midx, st #midy

        if not self.new_game:
            for r in range(default_board.shape[0]):
                for c in range(default_board.shape[1]):
                    node = self.nodes[r, c]

                    self._set_edge(r, c)

                    for du in range(4): # u, d, l, r
                        self._set_adj_words(node, du)

                    self._set_lets(node)

            for r in range(default_board.shape[0]):
                self._set_map(self.nodes[r], 0)
            for r in range(default_board.shape[1]):
                self._set_map(self.nodes[:, r], 1)

        else:
            midx = default_board.shape[0] // 2
            for r in range(midx + 1):
                node = self.nodes[midx, r]
                st = 1 if r == midx else (midx - r)
                node.vlen_view[0, st:(default_board.shape[0] - r)] = True

            # midy = default_board.shape[1] // 2
            # for c in range(midy + 1):
            #     node = self.nodes[c, midy]
            #     st = 1 if c == midy else (midy - c)
            #     node.vlen_view[1, st:(default_board.shape[1] - c)] = True

        self.nodesnv = <N[:self.nodes.shape[0], :self.nodes.shape[1]]>self.nodesn
        #self.nodesnv = self.nodesn[:self.nodes_rl, :self.nodes_cl]

        for r in range(self.nodes.shape[0]):
            for c in range(self.nodes.shape[1]):
                self.nodesnv[r, c] = self.nodes[r, c].n


    # #@lru_cache(1023)  # todo reenable?
    # cdef Node get(self, int x, int y):
    #     return <Node>self.nodes[x, y]

    # cdef Node get_by_attr(self, str attr, v):
    #     return filter(lambda obj: getattr(obj, attr) == v, self.nodes)


    # cdef str _lets_as_str(self, list letters):
    #     #cdef DTYPE_t v
    #     return ''.join([chr(l['value']) for l in letters]) # todo make func


    @cython.wraparound(False)
    cdef void _set_edge(self, Py_ssize_t r, Py_ssize_t c) except *:
        cdef Node node = self.nodes[r, c]

        if r > 0:
            node.up = self.nodes[r-1, c]
            if node.up.n.has_val:
                node.n.has_edge = True

        if r < self.nodes.shape[0] - 1:
            node.down = self.nodes[r+1, c]
            if node.down.n.has_val:
                node.n.has_edge = True

        if c > 0:
            node.left = self.nodes[r, c-1]
            if node.left.n.has_val:
                node.n.has_edge = True

        if c < self.nodes.shape[1] - 1:
            node.right = self.nodes[r, c+1]
            if node.right.n.has_val:
                node.n.has_edge = True


    cdef void _set_adj_words(self, Node n, Py_ssize_t d) except *:
        cdef:
            Node[:] loop_nodes
            #cnp.ndarray loop_nodes
            bint rev = False
            int xx = n.n.letter.x
            int yy = n.n.letter.y


        if d == 0:
            loop_nodes = self.nodes[:xx, yy][::-1]
            rev = True
        elif d == 1:
            loop_nodes = self.nodes[xx+1:, yy]
        elif d == 2:
            loop_nodes = self.nodes[xx, :yy][::-1]
            rev = True
        elif d == 3:
            loop_nodes = self.nodes[xx, yy+1:]
        else:
            return

        cdef:
            Py_ssize_t nl = loop_nodes.shape[0]
            Py_ssize_t ni
            N* p
            #str l_s
            Py_UCS4 lls
            str lets_str = ''
            STR_t lets_pts = 0
            #uchr nv

        for ni in range(nl):
            p = &loop_nodes[ni].n
            if not p.has_val:
                break
            else:
                lets_pts += p.pts

                #nv = p.letter.value
                #l_s = <str>chr(<int>nv)
                lls = p.letter.value
                if rev:
                    #lets_str = <str>lls + lets_str
                    lets_str = lls + lets_str
                else:
                    lets_str += lls

        if lets_str:
            if d == 0:
                n.up_word = lets_str
                n.up_pts = lets_pts
            elif d == 1:
                n.down_word = lets_str
                n.down_pts = lets_pts
            elif d == 2:
                n.left_word = lets_str
                n.left_pts = lets_pts
            elif d == 3:
                n.right_word = lets_str
                n.right_pts = lets_pts


    # move to checknodes?
    cdef void _set_lets(self, Node n) except *:
        if n.n.has_val:
            n.vlet_view[:, n.n.letter.value] = True
            return

        if not n.n.has_edge:
            n.vlet_view[:, L_ST:L_EN] = True
            return

        #cdef Py_ssize_t i_s = L_ST
        cdef uchr i_s = L_ST

        while i_s < L_EN:
            # - rows
            if self._check_adj_words(i_s, n.up, n.down, n.up_word, n.down_word):
                n.vlet_view[0, i_s] = True
                n.plet_view[0, i_s] = n.up_pts + n.down_pts

            # - cols
            if self._check_adj_words(i_s, n.left, n.right, n.left_word, n.right_word):
                n.vlet_view[1, i_s] = True
                n.plet_view[1, i_s] = n.left_pts + n.right_pts

            i_s += 1


    cdef bint _check_adj_words(self, uchr i, Node bef, Node aft, str bef_w, str aft_w) except *:
        cdef str new_word

        if (bef is None or not bef.n.has_val) and (aft is None or not aft.n.has_val):
            return True

        #new_word = <str>bef_w + <str>chr(i) + aft_w
        new_word = <str>bef_w + <str>chr(i) + <str>aft_w
        #new_word = ''.join(chr(ns.value) for ns in new_let_list)

        #lo.x(new_word)

        return new_word in Settings.words

    @cython.wraparound(False)
    cdef void _set_map(self, Node[:] nodes, bint is_col) except *:
        cdef:
            Py_ssize_t t, l, e, ai1
            Py_ssize_t nlen = nodes.shape[0]
            Py_ssize_t max_swl = Settings.shape[is_col]

            # todo this vs numpy bint?
            bint has_edge, has_blanks

            N* no

            BOOL_t valid_lengths[MAX_NODES][MAX_NODES]
            BOOL_t[:, :] vlen_view = valid_lengths

        #vlen_view = valid_lengths

        vlen_view[:, :] = True

        # disable one letter words
        vlen_view[:, 0] = False

        # disable last node
        vlen_view[nlen - 1, :] = False

        # disable < max word length
        if max_swl < nlen:
            vlen_view[:, max_swl:] = False

        #lo.w(f'\n{pd.DataFrame(np.asarray(nodes))}')

        # - iterate through each node
        for t in range(nlen - 1):
            #lo.i(f'=={t}')

            if t != 0:
                # disable for nodes that are too long
                vlen_view[t, nlen-t:] = False

                # if prev node has a val, disable for all lengths
                no = &nodes[t - 1].n
                if no.has_val:
                    vlen_view[t, :] = False
                    continue

            # for all possible wls...
            for l in range(max_swl - t):
                # for each valid wl, if last node has a val, disable for that wl
                #lo.v(l)
                ai1 = t + l + 1
                if ai1 < nlen:
                    no = &nodes[ai1].n
                    #lo.d(no)
                    if no.has_val:
                        #lo.d(f'810: {t} {l}')
                        vlen_view[t, l] = False
                        #lo.e(f'{vlen_view.base[t]}')
                        continue

                has_edge = False
                has_blanks = False

                #lo.v(f'{t}:{ai1}')

                # ..this should go in reverse, set all, then stop (and set all other nodes too)
                # if no edges or blanks, disable for that wl

                #for nc in nodes[]
                #for e in range(t, ai1):
                e = t
                while e < ai1:
                    no = &nodes[e].n
                    if no.has_edge:
                        has_edge = True
                    if not no.has_val:
                        has_blanks = True

                    if has_edge and has_blanks:
                        #lo.w('break')
                        break

                    e += 1

                #lo.w(has_blanks)
                #lo.w(has_edge)
                if not has_edge or not has_blanks:
                    #lo.v('set f')
                    vlen_view[t, l] = False

                #lo.s(f'{vlen_view.base[t]}')

        #lo.s(f'\n{vlen_view.base}')

        for t in range(nlen):
            nodes[t].vlen_view[is_col, :nlen] = vlen_view[t, :nlen]


    def __str__(self) -> str:
        return '<Board: size {:d}x{:d} | words: {}>'.format(self.nodes.shape[0], self.nodes.shape[1], self.words.len)

    # def __reduce__(self):
    #     return rebuild, (Settings.board, Settings.default_board)


#cdef npss = np.searchsorted


#todo what is the difference in not setting cdef? infer types?
#   why was i doing SIZE_t below?

# @cython.binding(True)
# @cython.linetrace(True)
# @cython.wraparound(False)
# cpdef void set_word_dict(STR_t[:] ww, Py_ssize_t wl, Node[:] nodes, Letter[:] lets_info, bint is_col, Py_ssize_t start):
@cython.wraparound(False)
cdef STRU_t calc_pts(Letter_List lets_info, N[:] nodes, bint is_col, Py_ssize_t start) except *: # nogil:
    cdef:
        Py_ssize_t i
        Py_ssize_t lcnt = 0
        N nd
        Letter le
        uchr nv
        BOOL_t lpts
        #BOOL_t bl = Settings.blanks
        STRU_t pts = 0  # todo: is sizet needed or better?
        STRU_t extra_pts
        STRU_t tot_extra_pts = 0
        STRU_t tot_pts
        uchr word_mult = 1

    # TODO HANDLE BLANKS

    for i in range(lets_info.len):
        nd = nodes[i + start]
        le = lets_info.l[i]
        nv = le.value

        if le.from_rack:
            lcnt += 1

        # le.from_rack
        if not nd.has_val and nd.mult_w:
            word_mult *= nd.mult_a

        lpts = le.pts
        pts += lpts

        # make sure this isnt counting has_val and upper words
        extra_pts = nd.pts_lets[is_col][nv]
        if extra_pts > 0:
            extra_pts += lpts
            if nd.mult_w:
                extra_pts *= nd.mult_a
            tot_extra_pts += extra_pts

    pts *= word_mult

    if lcnt == 7:
        pts += 35

    tot_pts = pts + tot_extra_pts

    return tot_pts


#todo should this be pyssize for word?
#todo test nogil more

@cython.wraparound(False)
cdef bint lets_match(cchrp word, Py_ssize_t wl, N[:] nodes, Py_ssize_t start, bint is_col) except *: # nogil:
#cdef bint lets_match(STR_t[::1] word, Py_ssize_t wl, valid_let_t[:] vl_list, Py_ssize_t start) nogil:
    cdef Py_ssize_t i
    cdef STR_t nv

    for i in range(wl):
        nv = word[i]
        if not nodes[i + start].valid_lets[is_col][nv]:
            return False

    return True


# todo put pointer on wl?
# todo indicate error
# return c type or pointer?

# ctypedef struct Letter_list:
#     Letter

#cdef object nz = np.zeros  # todo

#ctypedef (bint, Letter[:]) lets_ret
# ctypedef packed struct lets_ret:
#     bint success
#     Letter[:] letters

#DEF nu = NULL

# todo combine this with below?
@cython.wraparound(False)
cdef bint rack_check(cchrp word, Py_ssize_t wl, bint *nvals, Py_ssize_t start, BOOL_t blanks, int* base_rack) except *: # nogil:  # or memview for nvals?
    # todo this vs numpy ssize?
    cdef:
        BOOL_t nval
        Py_ssize_t i
        Py_ssize_t r = 0
        STR_t let
        BOOL_t num
        int rack[MAX_ORD]


    while r < MAX_ORD:
        rack[r] = base_rack[r]
        r += 1

    for i in range(wl):
        nval = nvals[i + start]

        # todo remove
        if nval:
            continue

        let = word[i]
        num = rack[let]

        if not num:
            if blanks > 0:
                blanks -= 1
            else:
                return False
        else:
            rack[let] -= 1

    # todo need?
    # if np.sum(rack) == Settings.rack_s:
    #     lo.e('no letters used')
    #     return

    return True


@cython.wraparound(False)
cdef Letter_List rack_match(cchrp word, Py_ssize_t wl, N[:] nodes, Py_ssize_t start, int* base_rack, BOOL_t* spts) except *: # nogil:
    cdef:
        Py_ssize_t i
        Py_ssize_t r = 0
        STR_t let
        int rack[MAX_ORD]
        BOOL_t num

        Letter_List lets_info
        #Letter[:] lets_view = <Letter[:MAX_NODES]>lets_info

        Letter le
        BOOL_t lepts
        N n

    lets_info.len = wl

    while r < MAX_ORD:
        rack[r] = base_rack[r]
        r += 1

    for i in range(MAX_NODES):
        if i >= wl:
            le.x = le.y = le.pts = le.is_blank = le.from_rack = le.value = 0
            lets_info.l[i] = le
        else:
            n = nodes[i + start]
            le = n.letter

            if n.has_val:
                lets_info.l[i] = le
                continue

            le.from_rack = True

            let = word[i]
            le.value = let
            num = rack[let]

            if not num:
                le.is_blank = True
            else:
                rack[let] -= 1
                lepts = spts[let]
                if not n.mult_w:
                    lepts *= n.mult_a

                le.pts = lepts

            lets_info.l[i] = le

    return lets_info


#todo combine
@cython.wraparound(False)
cdef void parse_new(N[:] nodes, cchr **sw, int* swlens, UINT32_t swl) except *: # nogil:
    cdef:
        bint is_col = False
        Py_ssize_t nlen = Settings.shape[is_col]
        Py_ssize_t s, wl, wl1, sn
        bint nvals[MAX_NODES]

        cchrp ww

        BOOL_t blanks = Settings.blanks
        int* base_rack = Settings.rack
        BOOL_t* base_pts = Settings.points

        Letter_List lets_info
        STRU_t tot_pts

        WordDict wd
        SIZE_t orig_len = Settings.node_board.words.len

    for s in range(MAX_NODES):
        nvals[s] = 0

    for s in range(swl):
        wl = swlens[s]
        if wl > nlen:
            continue
        wl1 = wl - 1

        ww = sw[s]

        for sn in range(nlen):
            # - is the word a valid length?
            if nodes[sn].valid_lengths[is_col][wl1] is False:
                #lo.e('not valid')
                continue

            # - do we have enough in the rack?
            if rack_check(ww, wl, nvals, sn, blanks, base_rack) is False:
                #lo.e('not enough rack')
                continue

            lets_info = rack_match(ww, wl, nodes, sn, base_rack, base_pts)
            tot_pts = calc_pts(lets_info, nodes, is_col, sn)

            wd.is_col = is_col
            wd.pts = tot_pts
            wd.letters = lets_info
            sol(wd, wd.word)

            Settings.node_board.words.l[orig_len] = wd
            orig_len += 1

    Settings.node_board.words.len = orig_len



#cdef loi


@cython.binding(True)
@cython.linetrace(True)
@cython.wraparound(False)
cdef void parse_nodes(N[:] nodes, cchr **sw, int* swlens, UINT32_t swl, bint is_col) except *:
# @cython.wraparound(False)
# cdef void parse_nodes(N[:] nodes, cchr **sw, int* swlens, UINT32_t swl, bint is_col): # nogil:
    cdef:
        Py_ssize_t t
        Py_ssize_t nlen = Settings.shape[is_col]
        N n
        bint bhas_edge = False

    # - check if empty
    for t in range(nlen):
        n = nodes[t]
        if n.has_edge:
            bhas_edge = True
            break

    if not bhas_edge:
        #if lo.is_enabled('i'):
        #loi('-> [empty]')
        return

    cdef:
        Py_ssize_t i
        bint nvals[MAX_NODES]
        Py_ssize_t s, wl, wl1, sn
        #SIZE_t *wl

        #STR_t[::1] ww
        cchrp ww

        BOOL_t blanks = Settings.blanks
        int* base_rack = Settings.rack
        BOOL_t* base_pts = Settings.points

        #Letter_List* lets_info
        Letter_List lets_info
        STRU_t tot_pts

        WordDict wd
        SIZE_t orig_len = Settings.node_board.words.len


    for i in range(nlen):
        n = nodes[i]
        nvals[i] = n.has_val

    for s in range(swl):
        # is there a problem with the conversion here
        wl = swlens[s]
        if wl > nlen:
            continue

        wl1 = wl - 1

        ww = sw[s]  # put limiter here?
        #lo.d(sw.base.view(f'<U{sw.shape[1]}')[s, 0])

        #for i in prange(nlen - wl + 1, nogil=True):
        for sn in range(nlen):
            # - is the word a valid length?
            if nodes[sn].valid_lengths[is_col][wl1] is False:
                #lo.e('not valid')
                continue

            # - do the letters match the board?
            if lets_match(ww, wl, nodes, sn, is_col) is False:
                #lo.e('dont match')
                continue

            # - do we have enough in the rack?
            if rack_check(ww, wl, nvals, sn, blanks, base_rack) is False:
                #lo.e('not enough rack')
                continue

            lets_info = rack_match(ww, wl, nodes, sn, base_rack, base_pts)
            tot_pts = calc_pts(lets_info, nodes, is_col, sn)

            wd.is_col = is_col
            wd.pts = tot_pts
            wd.letters = lets_info
            sol(wd, wd.word)

            Settings.node_board.words.l[orig_len] = wd
            orig_len += 1
            #loe(Settings.node_board.words.l[orig_len].word)

    Settings.node_board.words.len = orig_len


# this is for profiling, bug in cProfile
def _unused(): pass


"""
cdef str _print_node_range(
    vector[pair[int, int]] n  # list[tuple[int,int]]
    #list n  # list[tuple[int,int]]
    #np.ndarray[np.int, ndim=2] n  # list[tuple[int,int]]
):
    cdef:
        pair[int, int] s = n.front()
        pair[int, int] e = n.back()
        (int, int, int, int) x = (s.first, s.second, e.first, e.second)

    return '[%2i,%2i] : [%2i,%2i]' % x
"""


cdef void print_board(uchr[:, ::1] nodes, Letter_List lets): # nogil:
    cdef:
        Py_ssize_t r, c, l, rown, coln

        Py_UNICODE u_sep_hor_le = '\u2524'
        Py_UNICODE u_sep_hor_ri = '\u251C'

        char* board_hl = b'\x1b[33m'  # yellow
        char* board_cl = b'\x1b[0m'

        uchr best_map[MAX_NODES][MAX_NODES]
        #uchr[:, ::1] best_map_v = best_map
        uchr nval, bval
        Letter letter
        BOOL_t x, y


    print_board_top(nodes.shape[1])

    # best_map_v[:, :] = 0

    # while r < MAX_NODES:
    #     c = 0
    #     while c < MAX_NODES:
    #         best_map[r][c] = 0
    #         c += 1
    #     r += 1

    # - set best map
    for r in range(MAX_NODES):
        for c in range(MAX_NODES):
            best_map[r][c] = 0

    for l in range(lets.len):
        letter = lets.l[l]
        x = letter.x
        y = letter.y
        nval = nodes[x, y]
        if not nval:
            best_map[x][y] = letter.value

    # - rows
    for rown in range(nodes.shape[0]):
        printf('%2zu %lc  ', rown, u_sep_hor_le)
        for coln in range(nodes.shape[1]):
            nval = nodes[rown, coln]
            if not nval:
                bval = best_map[rown][coln]
                if bval:
                    printf('%s%c%s', board_hl, bval, board_cl)
                else:
                    printf(' ')
            else:
                printf('%c', nval)

            if coln != nodes.shape[1] - 1:
                printf(' ')

        printf('  %lc\n', u_sep_hor_ri)

    print_board_btm(nodes.shape[1])


cdef int mycmp(c_void pa, c_void pb) nogil:
    cdef STRU_t a = (<WordDict *>pa).pts
    cdef STRU_t b = (<WordDict *>pb).pts
    if a < b: return 1
    if a > b: return -1
    return 0


# TODO put found words into dict if not exist

#[:]
cdef void show_solution(uchr[:, ::1] nodes, WordDict_List words, bint no_words): # nogil:
    # todo mark blanks

    if words.len == 0:
        printf('\nNo solution.\n')
        return

    cdef:
        WordDict* best
        Py_ssize_t cut_num
        WordDict* word_list = words.l
   
    qsort(word_list, words.len, sizeof(WordDict), &mycmp)
    best = &word_list[0]
    
    if no_words:  # todo: print xs instead?
        printf('\n<solution hidden> (len: %zu, pts: %i)\n', best.letters.len, best.pts)
        return

    if can_log('s'):
        if Settings.num_results == 0:
            cut_num = words.len
        else:
            cut_num = Settings.num_results if Settings.num_results < words.len else words.len

        printf('\n')
        #los('-- Results (%i / %i) --\n', cut_num, words.len)
        #los('-- Results ({} / {}) --\n'.format(cut_num, words.len))
        clos('-- Results (%zu / %zu) --\n', cut_num, words.len)
        cut_num -= 1
        while cut_num >= 0:
            #w = &word_list[cut_num]
            #los(word_list[cut_num].word)
            # printf('%s(s)%s %s%s%s\n',
            #    log.KS_BLK_L, log.KS_RES, log.KS_GRN_L, word_list[cut_num].word, log.KS_RES
            # )
            clos('%s', word_list[cut_num].word)
            cut_num -= 1

    printf('\n')
    print_board(nodes, best.letters)
    printf('\nPoints: %i\n', best.pts)


@cython.final(True)
cpdef object checkfile(tuple paths, bint is_file = True):
    cdef object filepath = Path(*paths)
    if not filepath.exists():
        loc('Could not find file: {}'.format(filepath.absolute()))
        sys.exit(1)
    if is_file:
        if not filepath.is_file():
            loc('Path exists but is not a file: {}'.format(filepath.absolute()))
            sys.exit(1)
    else:
        if not filepath.is_dir():
            loc('Path exists but is not a directory: {}'.format(filepath.absolute()))
            sys.exit(1)

    return filepath


#def?
cdef list get_sw(list wordlist, BOOL_t blanks):
    cdef list incl
    cdef object s_sw = _s.SEARCH_WORDS
    #cdef int* swlens = <int*>malloc(swl * sizeof(int*))
    #cdef Py_ssize_t i

    if s_sw is None:
        incl = [l for l in Settings.include_lets if l != '?']
        # incl = []
        # for i in range(7):
        #     incl

        #s_search_words = set()

        # for w in words:
        #     for sl in range(Settings.rack_s):
        #         l = Settings.rack_l[sl]
        #         if l in w:
        #             s_search_words.add(w)
        #             break

        #search_words = frozenset(s_search_words)

        if not incl:
            if blanks > 0:
                return wordlist
            else:
                #return wordlist
                return [w for w in wordlist if any([l in w for l in Settings.rack_l])]

        else:
            return [w for w in wordlist if all([l in w for l in incl])]  # technically wrong for >1
            # print(len(wordlist))
            # search_words = [w for w in wordlist if all(l in w for l in incl)]
            # print(len(search_words))
            # return search_words

    elif isinstance(s_sw, tuple):
        return wordlist[s_sw[0]: s_sw[1]]

    elif isinstance(s_sw, list):
        return s_sw

    loc('Incompatible search words type: {}'.format(type(s_sw)))
    sys.exit(1)


# todo: set default for dict? put in settings? move all settings out to options?
#cpdef, def

# @cython.binding(True)
# @cython.linetrace(True)
#cpdef void solve(str dictionary):
@cython.wraparound(False)
cdef void solve(str dictionary) except *:
    cdef:
        str wordlist_load
        list _wordlist, wordlist # = []
        #str w
        Py_ssize_t mnlen #, wl  #, sl
        #set s_words
        frozenset words
        #frozenset search_words
        list search_words
        #object l

    wordlist_load = checkfile((_s.WORDS_DIR, <str>dictionary + '.txt')).read_text()
    if not wordlist_load:
        loc('No words in wordlist')
        sys.exit(1)

    mnlen = Settings.shape[0] if Settings.shape[0] > Settings.shape[1] else Settings.shape[1]

    #s_words = {w for w in wordlist_load.splitlines() if len(w) <= mnlen}
    #s_words = set()

    # todo why does this check for none?
    _wordlist = wordlist_load.splitlines()
    # for w in wordlist:
    #     wl = len(<str>w)
    #     if wl <= mnlen:
    #         wordlist.remove(w)
    wordlist = [w for w in _wordlist if len(w) <= mnlen]

    if not wordlist:
        loc('No words loaded from wordlist')
        sys.exit(1)

    words = frozenset(wordlist)

    cdef BOOL_t blanks = Settings.rack[BL]

    if blanks > MAX_BL:
        loc(f'Blanks greater than allowed (found {blanks}, max {MAX_BL})')
        sys.exit(1)

    search_words = get_sw(wordlist, blanks)

    cdef UINT32_t swl = len(search_words)
    if swl == 0:
        loc('No search words')
        sys.exit(1)

    Settings.blanks = blanks

    # todo allow extra mods
    # print(board)
    # print(board[10])
    # board[8][10] = 'Y'
    # letters = [b'F', b'G']

    #cdef set bwords = {s for s in words}
    Settings.words = words

    # -- TODO CHECK CONTIGUOUS --

    cdef:
        str sws
        Py_ssize_t swi

    # todo as one?
    #cdef uchrp ss
    #cdef list sw = [(s.encode('utf8'), len(s)) for s in search_words]
    #cdef cnp.ndarray sw = np.zeros(swl, f'|S{mnlen}')
    #cdef cnp.ndarray[:] _sw = np.array([s for s in search_words])
    # const

    # xx = np.array(list(search_words))
    # print(xx.shape)
    # print(xx.dtype)

    #cdef STR_t[:, ::1] sw = np.array(list(search_words)).view(STR).reshape(swl, -1)

    cdef cchr** sw = <cchr**>malloc(swl * sizeof(cchrp))
    cdef int* swlens = <int*>malloc(swl * sizeof(int*))

    for swi in range(swl):
        sws = (<list>search_words)[swi]
        sw[swi] = PyUnicode_AsUTF8(sws)
        swlens[swi] = len(sws)

    #Settings.search_words = sw
    #Settings.search_words_l = swl

    cdef Board full = Board(Settings.board, Settings.default_board)
    Settings.node_board = full

    #cdef Node[::1] sing_nodes
    #cdef N[:] ns_r = npe(Settings.shape[0], N)
    #cdef N[:] ns_c = npe(Settings.shape[1], N)
    # cdef N ns_r[MAX_NODES]
    # cdef N ns_c[MAX_NODES]
    # cdef N[::1] ns_rv = <N[:Settings.shape[0]]>ns_r
    # cdef N[::1] ns_cv = <N[:Settings.shape[1]]>ns_c

    cdef list search_rows, search_cols
    cdef Py_ssize_t ir, ic
    cdef bint is_col

    low('Solving...\n')

    if full.new_game:
        low(' = Fresh game = ')
        parse_new(full.nodesnv[(Settings.shape[0] // 2)], sw, swlens, swl)
        #parse_new(full.nodesnv[:, (Settings.shape[1] // 2)], sw, swlens, swl, True)

    else:
        if _s.SEARCH_NODES is None:
            search_rows = list(range(Settings.shape[0]))
            search_cols = list(range(Settings.shape[1]))
            if can_log('s'):
                low('Checking all lines (%2i x %2i)...' % (Settings.shape[0], Settings.shape[1]))

        else:
            search_rows = _s.SEARCH_NODES[0]
            search_cols = _s.SEARCH_NODES[1]
            if can_log('s'):
                low(f'Checking custom lines ({search_rows}, {search_cols})...')

        is_col = False
        for ir in search_rows:
            parse_nodes(full.nodesnv[ir], sw, swlens, swl, is_col)

        is_col = True
        for ic in search_cols:
            #sing_nodes = nodes.T[i]
            #sing_nodes = nodes[i]
            #sing_nodes = np.asarray(nodes.T[i], Node, 'C')
            #
            # for ni in range(Settings.shape[1]):
            #     no = full.nodes[ni, ic]
            #     ns_c[ni] = no.n

            #parse_nodes(ns_rv[:Settings.shape[0]], sw, swlens, is_col)
            parse_nodes(full.nodesnv[:, ic], sw, swlens, swl, is_col)

    free(sw)
    free(swlens)


# except *

# @cython.binding(True)
# @cython.linetrace(True)
# cpdef void cmain(
#     str filename, str dictionary, bint no_words, list exclude_letters, bint overwrite, int num_results, str log_level
# ):
@cython.wraparound(False)
cdef void cmain(
    str filename, str dictionary, bint no_words, list exclude_letters, list include_letters, int num_results, str log_level
) except *:
    log_level = log_level.lower()
    if log_level == 'spam': log_level = 'x'
    cdef Py_UCS4 log_lvl_ch = (<str>log_level)[0]
    log.lo_lvl = log.lvl_alias[log_lvl_ch]

    cdef object this_board_dir # type: Path
    cdef uchr[:, ::1] board  # todo switch to Py_UCS4
    cdef list rack

    # todo individ check so can just pass rack
    if filename is not None:
        this_board_dir = checkfile((_s.BOARD_DIR, filename), is_file=False)
        #with checkfile((this_board_dir, _s.BOARD_FILENAME)).open('rb') as f:
        board = pickle.load(checkfile((this_board_dir, _s.BOARD_FILENAME)).open('rb')).astype(BOOL)
        rack = pickle.load(checkfile((this_board_dir, _s.LETTERS_FILENAME)).open('rb'))

    else:
        board = np.array(_s.BOARD, dtype='|S1')  # s1?
        rack = _s.LETTERS

    if not rack:
        loc('Rack is empty')
        return

    cdef str el

    cdef uchr incl_c[7]
    incl_c[:] = [0, 0, 0, 0, 0, 0, 0]
    cdef Py_ssize_t eli

    for eli in range(len(include_letters)):
        el = (<list>include_letters)[eli]
        if el not in rack:
            loc(f'Include letter "{el}" not in rack')
            sys.exit(1)
        incl_c[eli] = ord(el)

    for el in exclude_letters:
        if el not in rack:
            loc(f'Exclude letter "{el}" not in rack')
            sys.exit(1)
        rack.remove(el)

    Settings.include_lets = include_letters
    Settings.include_lets_c = incl_c

    Settings.rack_l = rack
    Settings.rack_s = len(rack)

    cdef Py_UCS4 l
    cdef Py_ssize_t li
    for li in range(Settings.rack_s):
        #todo check less than 7 letters
        l = (<list>rack)[li]
        if len(l) == 1:
            Settings.rack[l] += 1
        else:
            loe('Rack letter is not valid: "%s"' % l)

    cdef int board_size
    cdef object board_name

    board_size = board.size
    if board_size == MAX_NODES * MAX_NODES:
        board_name = _s.DEF_BOARD_BIG
    elif board_size == 11 * 11:
        board_name = _s.DEF_BOARD_SMALL
    else:
        loc('Board size ({}) has no match'.format(board_size))
        return

    cdef object[:, :] default_board
    cdef dict points

    default_board = pickle.load(checkfile((board_name,)).open('rb'))  # type: np.ndarray  # .to_numpy('|S2')

    points = json.load(checkfile((_s.POINTS_DIR, <str>dictionary + '.json')).open())  # Dict[str, List[int]]

    cdef Py_ssize_t dx, dy, ds
    cdef Py_UNICODE* do
    cdef char du[2]
    if can_log('w'):
        low('Game Board:')
        print_board_clr(board.base, log.KS_GRN_L)
        if can_log('v'):
            # todo make a func
            lov('Default:')
            printf('%s', log.KS_BLU)
            for dx in range(default_board.shape[0]):
                for dy in range(default_board.shape[1]):
                    do = default_board[dx, dy]
                    ds = len(do)
                    if ds == 0: du = [NUL, NUL]
                    elif ds == 1: du = [do[0], NUL]
                    else: du = [do[0], do[1]]
                    printf('%-2s', du)
                puts('')
            puts(log.KS_RES)

        los('Rack:\n{}\n'.format(rack))
    else:
        printf('Running...\n')

    Settings.default_board = default_board  # todo put back?
    Settings.board = board
    Settings.shape = board.shape

    #cdef lpts_t lpt
    cdef str k
    cdef list v
    cdef BOOL_t vv[2]
    cdef SIZE_t ok

    for k, v in points.items():
        ok = ord(k)
        vv = v  # type: list
        Settings.amts[ok] = vv[0]
        Settings.points[ok] = vv[1]

    Settings.num_results = num_results

    # todo add search words and nodes

    solve(dictionary)

    cdef uchr solved_board[MAX_NODES][MAX_NODES]
    cdef uchr[:, ::1] solved_board_v = solved_board
    cdef Py_ssize_t r, c

    for r in range(board.shape[0]):
        for c in range(board.shape[1]):
            solved_board[r][c] = Settings.node_board.nodes[r, c].n.letter.value

    show_solution(
        solved_board_v[:board.shape[0], :board.shape[1]], Settings.node_board.words, no_words
    )


# def
def main(
    filename: str = None, dictionary: str = _s.DICTIONARY, no_words: bool = False, exclude_letters: list = None, include_letters: list = None, num_results: int = _s.NUM_RESULTS,
    log_level: str = _s.DEFAULT_LOGLEVEL, **_kw: dict
) -> None:

    if exclude_letters is None:
        exclude_letters = []
    if include_letters is None:
        include_letters = []

    cmain(filename, dictionary, no_words, exclude_letters, include_letters, num_results, log_level)
