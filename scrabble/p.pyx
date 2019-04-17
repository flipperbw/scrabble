"""Parse and solve a scrabble board."""

cimport cython
#from cython.parallel import prange

cdef object json, md5, Path, np, pd, log_init, _s
#cdef module np
#sys
#import sys

import json

from hashlib import md5
from pathlib import Path

#import signal
#from functools import lru_cache

import numpy as np
cimport numpy as cnp

import pandas as pd  # remove

from logs import log_init

import settings as _s


cnp.import_array()
cnp.import_umath()

#ctypedef


cdef object lo = log_init(_s.DEFAULT_LOGLEVEL)


#ctypedef unsigned short us
ctypedef unsigned char uchr

#ctypedef unsigned char* uchrp
#ctypedef (us, us) dual
#ctypedef us dual[2]

#ctypedef cnp.uint32_t DTYPE_t
#ctypedef cnp.int32_t STR_t
#ctypedef cnp.uint8_t BOOL_t
#ctypedef cnp.intp_t SIZE_t
cdef type STR = np.int32
cdef type BOOL = np.uint8
cdef type SIZE = np.intp

cdef object npz = np.zeros
cdef object npe = np.empty

#DTYPE = np.intc

# todo no numpy array

#ctypedef np.ndarray[str, ndim=2] board_t  # str 32?
#ctypedef np.ndarray[object, ndim=2] nb_t
#ctypedef np.ndarray[object, ndim=1] nodelist_t

#todo more of these
#cdef bytes NUL = b'\0'
#DEF NUL = b'\0'
DEF bl = ord('?')
DEF MAX_ORD = 127  # todo replace 127
DEF L_ST = 65
DEF L_EN = 91

# cdef packed struct multiplier_t:
#     uchr amt
#     uchrp typ

# cdef packed struct lpts_t:
#     uchr amt
#     uchr pts


cdef Letter[::1] lets_empty = npe(15, [('is_blank', BOOL), ('from_rack', BOOL), ('pts', BOOL), ('x', BOOL), ('y', BOOL), ('value', STR)])


# todo : do comments like these causes c stuff?
"""
@cython.freelist(50000)  # todo check
    def get_pos(self) -> tuple:
        return self.x, self.y

    def get_pos_str(self) -> str:
        return f'{self.x:2d},{self.y:2d}'

    def get_val(self) -> str:
        if self.value:
            #return self.value.view('U1')  # pointer?
            return chr(self.value)
        else:
            return ''

    def __str__(self) -> str:
        return '<'\
            f'v: {self.get_val()}, '\
            f'pos: [{self.get_pos_str()}], '\
            f'bl: {"T" if self.is_blank is True else "f"}, '\
            f'rk: {"T" if self.from_rack is True else "f"}, '\
            f'pts: {"_" if self.pts == 0 and self.is_blank is False else self.pts}, '\
            f'mult: "{"_" if self.mult.amt == 1 else str(self.mult.amt) + self.mult.typ.decode("utf8").upper()}"'\
            '>'

    def __repr__(self) -> str:
        return self.__str__()

    def __reduce__(self):
        return rebuild_letter, (self.is_blank, self.from_rack, self.pts, self.mult, self.x, self.y, self.value)


def rebuild_letter(is_blank, from_rack, pts, mult, x, y, value):
    l = Letter()
    l.is_blank = is_blank
    l.from_rack = from_rack
    l.pts = pts
    l.mult = mult
    l.x = x
    l.y = y
    l.value = value
    return l
"""


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

        self.rack = npz(MAX_ORD, BOOL)
        self.rack_l = []

        self.rack_s = 0
        self.blanks = 0

        self.points = None
        self.amts = None

        self.words = frozenset()  ## type: Set[bytes]
        self.node_board = None
        self.num_results = _s.NUM_RESULTS


cdef CSettings Settings = CSettings()


#ctypedef class WordDict:

#@cython.freelist(10000)

@cython.final(True)
cdef class WordDict:
    #Letter[:] letters
    def __cinit__(
        #self, STR_t[::1] word, SIZE wl, Py_ssize_t s, bint is_col, SIZE_t pts, list letters not None  # todo check not none
        self, word, int wl, int s, bint is_col, int pts, list letters
    ):
        # lo.e(letters)
        # lo.e(letters.shape)
        # lo.e(letters[0])
        #self._word = word
        #self.word = ''.join([chr(l) for l in word])

        self.word = word
        self.wl = wl
        self.s = s
        self.is_col = is_col
        self.pts = pts
        self.letters = letters
        #self.letters = list(letters)

    #def str sol(self):
    cdef str sol(self):
        cdef str direc = 'c' if self.is_col is True else 'r'

        cdef Letter lf = self.letters[0]
        cdef Letter ll = self.letters[-1]

        cdef str pos
        if self.is_col is True:
            pos = f'{lf.y:2d} x {lf.x:2d},{ll.x:2d}'
        else:
            pos = f'{lf.x:2d} x {lf.y:2d},{ll.y:2d}'

        # todo fix
        # lo.e(self.word.base.view(f'<U{self.word.base.shape[1]}')[self.s, 0])
        # lo.e(self.word.base.view(f'<U{self.word.base.shape[1]}')[self.s])
        #lo.e(self.word.base[self.s].view(f'<U{self.word.base.shape[1]}')[0])
        #lo.e(self.wl)

        #cdef str word = ''.join([chr(l) for l in self.word])

        cdef str word = ''
        cdef STRU_t wln
        cdef str wll
        cdef Py_ssize_t i
        for i in range(self.wl):
            wln = self.word[i]
            wll = <str>chr(wln)
            word += wll

        return 'pts: {0:3d} | dir: {1} | pos: {2} | w: {3}'.format(
            <STR_t>self.pts, direc, pos, word
        )

        #cdef str word = self.word.base[self.s].view(f'<U{self.word.base.shape[1]}')[0]

        #return f'pts: {<STR_t>self.pts:3d} | dir: {<str>direc} | pos: {<str>pos} | w: {<str>word}'
        #return f'pts: {<STR_t>self.pts:3d} | dir: {<str>direc} | pos: {<str>pos} | w: {self.word.base[self.s, :self.wl]}'

    # def __str__(self) -> str:
    #     # cdef Letter l
    #     # cdef list ll = []
    #     # cdef str li
    #     # for l in self.letters:
    #     #     li = str(l)
    #     #     ll.append(li)
    #     # cdef str ls = '\n  '.join(ll)
    #     return '<w: {} | pts: {} | dir: {} | letts: [\n  {}\n]>'.format(
    #         self.word, self.pts, self.direc, '\n  '.join([str(l) for l in self.letters])
    #     )
    #
    # def __repr__(self) -> str:
    #     return self.__str__()

    def __reduce__(self):
        return rebuild_worddict, (
            self.word.base[self.s, :self.wl],
            self.wl,
            self.s,
            self.is_col,
            self.pts,
            # np.array(self.letters, [
            #     ('is_blank', BOOL), ('from_rack', BOOL), ('pts', BOOL), ('x', BOOL), ('y', BOOL), ('value', STR)
            # ])
            list(self.letters)
        )

cpdef rebuild_worddict(word, wl, s, is_col, pts, letters):
    #cdef STR_t[:] word = np.array([ord(w) for w in _word], STR)
    return WordDict(word, wl, s, is_col, pts, letters)


@cython.final(True)
cdef class Node:
    # todo test no types in cinit
    def __cinit__(self, BOOL_t x, BOOL_t y, str val, BOOL_t mult_a, BOOL_t mult_w, bint is_start):
        self.letter.x = x
        self.letter.y = y

        self.letter.is_blank = False
        self.letter.from_rack = False
        #self.letter.pts = 0

        self.mult_a = mult_a
        self.mult_w = mult_w

        self.is_start = is_start

        #cdef lpts_t lpt

        if not val:
            self.has_val = False
            self.letter.value = 0
            self.pts = 0
            self.display = ' '  # todo display method
        else:
            self.has_val = True
            self.letter.value = ord(val)
            self.pts = Settings.points[self.letter.value]
            self.display = val.upper()

            # try:
            #     lpt = Settings.points[self.value]
            #     self.points = lpt.pts
            # except (KeyError, IndexError):
            #     lo.e('Could not get point value of "{}"'.format(val))
            #     sys.exit(1)

        self.letter.pts = self.pts

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

        self.has_edge = False

        self.valid_lets = npz((2, MAX_ORD, 2), STR)

        self.valid_lengths = npz((2, max(Settings.shape)), BOOL)


    def str_pos(self) -> str:
        return f'[{self.letter.x:2d},{self.letter.y:2d}]'

    def __str__(self) -> str:
        cdef str s
        if self.has_val:
            s = chr(self.letter.value)
        else:
            s = '_'
        return '<Node: {} v: {}>'.format(self.str_pos(), s)

    def __repr__(self) -> str:
        return self.__str__()


@cython.final(True)
cdef class Board:
    def __cinit__(self, object[:, ::1] board, object[:, :] default_board):  # todo fix c contig
        cdef:
            Py_ssize_t r, c
            str dbval, bval, x, d
            BOOL_t mult_a, mult_w
            Node node
            #cnp.ndarray board

        #self.board = board
        #self.default_board = default_board
        self.nodes = np.zeros_like(default_board, Node)

        self.nodes_rl = Settings.shape[0]
        self.nodes_cl = Settings.shape[1]

        #self.words = np.empty(0, WordDict)
        self.words = []

        self.new_game = False

        for r in range(self.nodes_rl):
            for c in range(self.nodes_cl):
                bval = board[r, c]

                if not bval:
                    bval = None
                else:
                    bval = bval.strip()
                    if not bval:
                        bval = None
                    else:
                        bval = bval.upper()

                dbval = default_board[r, c]

                mult_a = 1
                mult_w = 0

                if dbval:
                    dbval = dbval.strip()
                    if dbval == 'x':
                        if bval is None:
                            self.new_game = True
                    else:
                        x = dbval[0]
                        mult_a = <BOOL_t>int(x)
                        if dbval[1] == 'w':
                            mult_w = 1

                if bval is None:
                    bval = ''

                node = Node(r, c, bval, mult_a, mult_w, self.new_game)
                self.nodes[r, c] = node

        for r in range(self.nodes_rl):
            for c in range(self.nodes_cl):
                node = self.nodes[r, c]

                self._set_edge(r, c)

                for d in ('up', 'down', 'left', 'right'):
                    self._set_adj_words(node, d)

                self._set_lets(node)

        for r in range(Settings.shape[0]):
            self._set_map(self.nodes[r], 0)
        for r in range(Settings.shape[1]):
            self._set_map(self.nodes[:, r], 1)


    #@lru_cache(1023)  # todo reenable?
    cdef Node get(self, int x, int y):
        return <Node>self.nodes[x, y]

    cdef Node get_by_attr(self, str attr, v):
        return filter(lambda obj: getattr(obj, attr) == v, self.nodes)


    # cdef str _lets_as_str(self, list letters):
    #     #cdef DTYPE_t v
    #     return ''.join([chr(l['value']) for l in letters]) # todo make func


    @cython.wraparound(False)
    cdef void _set_edge(self, Py_ssize_t r, Py_ssize_t c):
        cdef Node node = self.nodes[r, c]

        if r > 0:
            node.up = self.nodes[r-1, c]
            if node.up.has_val:
                node.has_edge = True

        if r < self.nodes_rl - 1:
            node.down = self.nodes[r+1, c]
            if node.down.has_val:
                node.has_edge = True

        if c > 0:
            node.left = self.nodes[r, c-1]
            if node.left.has_val:
                node.has_edge = True

        if c < self.nodes_cl - 1:
            node.right = self.nodes[r, c+1]
            if node.right.has_val:
                node.has_edge = True


    cdef void _set_adj_words(self, Node n, str d):
        cdef:
            Node[:] loop_nodes
            #cnp.ndarray loop_nodes
            bint rev = False
            int xx = n.letter.x
            int yy = n.letter.y


        if d == 'up':
            loop_nodes = self.nodes[:xx, yy][::-1]
            rev = True
        elif d == 'down':
            loop_nodes = self.nodes[xx+1:, yy]
        elif d == 'left':
            loop_nodes = self.nodes[xx, :yy][::-1]
            rev = True
        elif d == 'right':
            loop_nodes = self.nodes[xx, yy+1:]
        else:
            return

        cdef:
            Py_ssize_t nl = loop_nodes.shape[0]
            Py_ssize_t ni
            Node p
            str l_s
            str lets_str = ''
            STR_t lets_pts = 0
            STR_t nv

        for ni in range(nl):
            p = loop_nodes[ni]
            if not p.has_val:
                break
            else:
                lets_pts += p.pts  # <STR_t>

                nv = p.letter.value
                l_s = chr(nv)
                if rev:
                    lets_str = l_s + lets_str
                else:
                    lets_str += l_s

        if lets_str:
            if d == 'up':
                n.up_word = lets_str
                n.up_pts = lets_pts
            elif d == 'down':
                n.down_word = lets_str
                n.down_pts = lets_pts
            elif d == 'left':
                n.left_word = lets_str
                n.left_pts = lets_pts
            elif d == 'right':
                n.right_word = lets_str
                n.right_pts = lets_pts


    # move to checknodes?
    cdef void _set_lets(self, Node n):
        if n.has_val:
            n.valid_lets[:, n.letter.value, 0] = 1
            return

        if not n.has_edge:
            n.valid_lets[:, L_ST:L_EN, 0] = 1
            return

        cdef Py_ssize_t i_s = L_ST

        while i_s < L_EN:
            # - rows
            if self._check_adj_words(i_s, n.up, n.down, n.up_word, n.down_word):
                n.valid_lets[0, i_s, 0] = 1
                n.valid_lets[0, i_s, 1] = n.up_pts + n.down_pts

            # - cols
            if self._check_adj_words(i_s, n.left, n.right, n.left_word, n.right_word):
                n.valid_lets[1, i_s, 0] = 1
                n.valid_lets[1, i_s, 1] = n.left_pts + n.right_pts

            i_s += 1


    cdef bint _check_adj_words(self, BOOL_t i, Node bef, Node aft, str bef_w, str aft_w):
        if (bef is None or not bef.has_val) and (aft is None or not aft.has_val):
            return True

        cdef str new_word = bef_w + <str>chr(i) + aft_w
        #new_word = ''.join(chr(ns.value) for ns in new_let_list)

        #lo.x(new_word)

        return new_word in Settings.words


    cdef void _set_map(self, Node[:] nodes, bint is_col):
        cdef:
            Py_ssize_t t, l, e, ai1
            Py_ssize_t nlen = nodes.shape[0]
            Py_ssize_t max_swl = Settings.shape[is_col]

            # todo this vs numpy bint?
            cdef bint has_edge, has_blanks

            cdef Node no

            #bint?
            # x = Node, y = lens
            BOOL_t[:, ::1] valid_lengths = np.ones((nlen, nlen), BOOL)

        # disable one letter words
        valid_lengths[:, 0] = 0

        # disable last node
        valid_lengths[nlen - 1, :] = 0

        # disable < max word length
        if max_swl < nlen:
            valid_lengths[:, max_swl:] = 0

        #lo.w(f'\n{pd.DataFrame(np.asarray(nodes))}')

        # - iterate through each node
        for t in range(nlen - 1):
            #lo.i(f'=={t}')

            if t != 0:
                # disable for nodes that are too long
                valid_lengths[t, nlen-t:] = False

                # if prev node has a val, disable for all lengths
                no = nodes[t - 1]
                if no.has_val:
                    valid_lengths[t, :] = False
                    continue

            # for all possible wls...
            for l in range(max_swl - t):
                # for each valid wl, if last node has a val, disable for that wl
                #lo.v(l)
                ai1 = t + l + 1
                if ai1 < nlen:
                    no = nodes[ai1]
                    #lo.d(no)
                    if no.has_val:
                        #lo.d(f'810: {t} {l}')
                        valid_lengths[t, l] = False
                        #lo.e(f'{valid_lengths.base[t]}')
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
                    no = nodes[e]
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
                    valid_lengths[t, l] = False

                #lo.s(f'{valid_lengths.base[t]}')

        #lo.s(f'\n{valid_lengths.base}')


        for t in range(nlen):
            nodes[t].valid_lengths[is_col] = valid_lengths[t]
            #lo.e(nodes[t].valid_lengths.base[is_col])


    def __str__(self) -> str:
        return '<Board: size {:d}x{:d} | words: {}>'.format(self.nodes_rl, self.nodes_cl, len(self.words))

    # def __reduce__(self):
    #     return rebuild, (Settings.board, Settings.default_board)


#cdef npss = np.searchsorted


#todo what is the difference in not setting cdef? infer types?


# @cython.binding(True)
# @cython.linetrace(True)
# @cython.wraparound(False)
# cpdef void set_word_dict(STR_t[:] ww, Py_ssize_t wl, Node[:] nodes, Letter[:] lets_info, bint is_col, Py_ssize_t start):
@cython.wraparound(False)
cdef SIZE_t set_word_dict(STR_t[::1] ww, Py_ssize_t wl, Node[::1] nodes, Letter[::1] lets_info, bint is_col, Py_ssize_t start):
    cdef:
        Py_ssize_t i
        Py_ssize_t lcnt = 0
        Node nd
        STR_t nv
        Letter le
        BOOL_t lpts
        #BOOL_t bl = Settings.blanks
        SIZE_t pts = 0  # todo: is sizet needed or better?
        SIZE_t extra_pts
        SIZE_t tot_extra_pts = 0
        SIZE_t tot_pts
        uchr word_mult = 1
        #WordDict w

    # TODO HANDLE BLANKS

    # todo fix this letter stuff

    #lo.c(lets_info.base.flags)
    #lo.c(nodes.base.flags)
    #lo.c(ww.base.flags)

    for i in range(wl):
        nv = ww[i]
        nd = nodes[i + start]
        le = lets_info[i]

        #lo.d(type(le))  # todo: is this actually a dict?

        if le.from_rack:
            lcnt += 1

        # le.from_rack
        if not nd.has_val and nd.mult_w:
            word_mult *= nd.mult_a

        lpts = le.pts
        pts += lpts

        # make sure this isnt counting has_val and upper words
        extra_pts = nd.valid_lets[is_col, nv, 1]
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

    # w = WordDict(
    #     ww,
    #     is_col,
    #     tot_pts,
    #     list(lets_info[:wl])
    # )
    #
    # # if w not in Settings.node_board.words:
    # #     Settings.node_board.words.append(w)
    # # else:
    # #     lo.e('already saw %s', w)
    #
    # Settings.node_board.words.append(w)

def _u2(): pass


#todo should this be pyssize for word?
#todo test nogil more

@cython.wraparound(False)
cdef bint lets_match(STR_t[::1] word, Py_ssize_t wl, STR_t[:, :, ::1] vl_list, Py_ssize_t start) nogil:
    cdef Py_ssize_t i
    cdef STR_t nv

    for i in range(wl):
        nv = word[i]
        # mismatch in nv?
        if not vl_list[i + start, nv, 0]:
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


@cython.wraparound(False)
cdef bint rack_check(STR_t[::1] word, Py_ssize_t wl, BOOL_t[::1] nvals, Py_ssize_t start) nogil:
    # todo this vs numpy ssize?
    cdef BOOL_t nval
    cdef Py_ssize_t i
    cdef STR_t let
    cdef BOOL_t num
    cdef BOOL_t blanks

    cdef BOOL_t[::1] rack
    #cdef BOOL_t[:] rack = Settings.rack.copy()
    #cdef BOOL_t[:] rack = rack_empty.copy()
    #cdef BOOL_t[:] rack = np.zeros(MAX_ORD, BOOL)
    #cdef uchr[:] rack = np.zeros(MAX_ORD, BOOL)

    with gil:
        rack = Settings.rack.copy()
        blanks = Settings.blanks

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
cdef Letter[::1] rack_match(STR_t[::1] word, Py_ssize_t wl, Node[::1] nodes, Py_ssize_t start):
    cdef:
        Py_ssize_t i
        STR_t let

        BOOL_t[::1] rack = Settings.rack.copy()
        #BOOL_t[:] rack = rack_empty.copy()
        #BOOL_t[:] rack = np.zeros(MAX_ORD, BOOL)
        #uchr[:] rack = np.zeros(MAX_ORD, BOOL)

        BOOL_t num

        #int?
        Letter[::1] lets_info = lets_empty.copy()

        #list npa = [('is_blank', BOOL), ('from_rack', BOOL), ('pts', BOOL), ('x', BOOL), ('y', BOOL), ('value', STR)]
        #Letter[:] lets_info = np.empty(wl, [('is_blank', BOOL), ('from_rack', BOOL), ('pts', BOOL), ('x', BOOL), ('y', BOOL), ('value', STR)])
        #Letter[:] lets_info = np.zeros(wl, [('is_blank', BOOL), ('from_rack', BOOL), ('pts', BOOL), ('x', BOOL), ('y', BOOL), ('value', STR)])
        #Letter lets_info[wl]

        Letter le

        BOOL_t[::1] spts = Settings.points
        BOOL_t lepts

        Node n


    for i in range(wl):
        n = nodes[i + start]
        le = n.letter

        if n.has_val:
            lets_info[i] = le
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

        lets_info[i] = le

    return lets_info[:wl]


cdef void add_word(STR_t[::1] ww, Py_ssize_t wl, Py_ssize_t s, bint is_col, SIZE_t tot_pts, Letter[::1] lets_info):
    cdef list letslist
    cdef WordDict w

    letslist = list(lets_info)

    # lo.e(letslist)
    # lo.e(lets_info)
    # lo.e(lets_info.shape)
    # lo.e(lets_info[0])

    w = WordDict(
        ww,
        <int>wl,
        <int>s,
        is_col,
        tot_pts,
        letslist
        #lets_info
    )

    Settings.node_board.words.append(w)


# nvs = np.searchsorted(ls, nv)
# nvs = npss(ls, nv)
# befls = ls[:nvs]
# aftls = ls[nvs+1:]
# ls = np.concatenate([
#     zl,
#     befls,
#     aftls
# ])  # out=ls?

# lsl = np.count_nonzero(ls)


# @cython.wraparound(False)
# cdef bint check_nodes(STR_t[::1] ww, Py_ssize_t wl, Py_ssize_t sn, BOOL_t[:, ::1] vlens, STR_t[:, :, ::1] vlets, BOOL_t[::1] nvals) nogil:
#
#     # - is the word a valid length?
#     if vlens[sn, wl - 1] == 0:
#         #lo.e('not valid')
#         return False
#
#     # - do the letters match the board?
#     if lets_match(ww, wl, vlets, sn) is False:
#         #lo.e('dont match')
#         return False
#
#     # - do we have enough in the rack?
#     if rack_check(ww, wl, nvals, sn) is False:
#         return False
#
#     return True


#cdef loi


@cython.binding(True)
@cython.linetrace(True)
@cython.wraparound(False)
cpdef void parse_nodes(Node[::1] nodes, STR_t[:, ::1] sw, SIZE_t[::1] swlens, bint is_col):
# @cython.wraparound(False)
# cdef void parse_nodes(Node[::1] nodes, STR_t[:, ::1] sw, SIZE_t[::1] swlens, bint is_col):
    cdef:
        #(uchrp, Py_ssize_t) w
        #cnp.ndarray[:] sw = Settings.search_words
        #Py_ssize_t swl = Settings.search_words_l

        Py_ssize_t t
        Py_ssize_t nlen = nodes.shape[0]

        Node n

        cnp.npy_intp dims_vlens[2]
        cnp.npy_intp dims_vlets[3]
        cnp.npy_intp dims_nvals[1]  # todo without array?

        #BOOL_t[:, :, ::1] vlens = npe((nlen, 2, max(Settings.shape)), BOOL)
        BOOL_t[:, ::1] vlens
        STR_t[:, :, ::1] vlets
        BOOL_t[::1] nvals

        bint bhas_edge = False


    # - check if empty
    for t in range(nlen):
        n = nodes[t]
        if n.has_edge:
            bhas_edge = True
            break

    if not bhas_edge:
        if lo.is_enabled('i'):
            lo.i('-> [empty]')
        return


    dims_vlens[0] = nlen
    dims_vlens[1] = Settings.shape[is_col]

    dims_vlets[0] = nlen
    dims_vlets[1] = MAX_ORD
    dims_vlets[2] = 2

    dims_nvals[0] = nlen

    vlens = cnp.PyArray_EMPTY(2, dims_vlens, cnp.NPY_UINT8, 0)
    vlets = cnp.PyArray_EMPTY(3, dims_vlets, cnp.NPY_INT32, 0)
    nvals = cnp.PyArray_EMPTY(1, dims_nvals, cnp.NPY_UINT8, 0)

    for t in range(nlen):
        n = nodes[t]
        vlens[t] = n.valid_lengths[is_col]
        vlets[t] = n.valid_lets[is_col]
        nvals[t] = n.has_val


    cdef:
        Py_ssize_t s, wl, wl1, sn
        STR_t[::1] ww
        Letter[::1] lets_info
        #bint matches
        SIZE_t tot_pts
        #WordDict w
        #list letslist


    #Settings.search_words_l  # todo swap test

    for s in range(sw.shape[0]):
        #w = sw[s]
        # is there a problem with the conversion here
        wl = swlens[s]
        if wl > nlen:
            continue

        wl1 =  wl - 1

        ww = sw[s]
        # vs ww[:] = ...
        #lo.d(sw.base.view(f'<U{sw.shape[1]}')[s, 0])

        #for i in prange(nlen - wl + 1, nogil=True):

        for sn in range(nlen):
            # - is the word a valid length?
            if vlens[sn, wl1] == 0:
                #lo.e('not valid')
                continue

            # - do the letters match the board?
            if lets_match(ww, wl, vlets, sn) is False:
                #lo.e('dont match')
                continue

            # - do we have enough in the rack?
            if rack_check(ww, wl, nvals, sn) is False:
                #lo.e('not enough rack')
                continue

            #with gil:
            lets_info = rack_match(ww, wl, nodes, sn)

            tot_pts = set_word_dict(ww, wl, nodes, lets_info, is_col, sn)

            add_word(ww, wl, s, is_col, tot_pts, lets_info)

            # letslist = list(lets_info)
            #
            # # lo.e(letslist)
            # # lo.e(lets_info)
            # # lo.e(lets_info.shape)
            # # lo.e(lets_info[0])
            #
            # w = WordDict(
            #     ww,
            #     <int>wl,
            #     <int>s,
            #     is_col,
            #     tot_pts,
            #     letslist
            #     #lets_info
            # )
            #
            # Settings.node_board.words.append(w)


def _unused(): pass


def loadfile(*paths: tuple, is_file: bool = True) -> Path:
    cdef object filepath = Path(*paths)
    if not filepath.exists():
        lo.c('Could not find file: {}'.format(filepath.absolute()))
        exit(1)
    if is_file:
        if not filepath.is_file():
            lo.c('Path exists but is not a file: {}'.format(filepath.absolute()))
            exit(1)
    else:
        if not filepath.is_dir():
            lo.c('Path exists but is not a directory: {}'.format(filepath.absolute()))
            exit(1)

    return filepath


# todo: set default for dict? put in settings? move all settings out to options?
#cpdef, def


# @cython.binding(True)
# @cython.linetrace(True)
# cpdef void solve(str dictionary):
cdef void solve(str dictionary):
    cdef:
        #object wl_path = loadfile(_s.WORDS_DIR, dictionary + '.txt')
        #list wordlist = wl_path.read_text().splitlines()
        #list wordlist = []
        list wordlist = loadfile(_s.WORDS_DIR, dictionary + '.txt').read_text().splitlines()

        BOOL_t blanks
        Py_ssize_t wl_len, mnlen, wi, wl #, sl
        set s_words
        frozenset words
        #set s_search_words
        frozenset search_words
        str w
        #object l

    if not wordlist:
        return

    blanks = Settings.rack[bl]
    mnlen = max(Settings.shape)

    #s_words = {w for w in wordlist if len(w) <= mnlen}
    s_words = set()

    # todo why does this check for none?
    wl_len = len(wordlist)
    for wi in range(wl_len):
        w = <str>(wordlist[wi])
        wl = len(w)
        if wl <= mnlen:
            s_words.add(w)

    words = frozenset(s_words)

    if _s.SEARCH_WORDS is None:
        if blanks > 0:
            search_words = words
        else:
            # TODO FIX THIS
            #lo.e(len(words))

            # todo: faster if str? why gen slower?
            #search_words = frozenset({w for w in words if any([l in w for l in Settings.rack_l])})

            #s_search_words = set()

            # for w in words:
            #     for sl in range(Settings.rack_s):
            #         l = Settings.rack_l[sl]
            #         if l in w:
            #             s_search_words.add(w)
            #             break

            #search_words = frozenset(s_search_words)

            search_words = words

            #lo.e(len(search_words))

    elif isinstance(_s.SEARCH_WORDS, tuple):
        search_words = frozenset(wordlist[_s.SEARCH_WORDS[0]: _s.SEARCH_WORDS[1]])
    elif isinstance(_s.SEARCH_WORDS, set):
        search_words = frozenset(_s.SEARCH_WORDS)
    else:
        search_words = frozenset()  # why necessary?
        lo.c('Incompatible search words type: {}'.format(type(_s.SEARCH_WORDS)))
        #sys.exit(1)
        exit(1)

    cdef Py_ssize_t swl = len(search_words)
    if swl == 0:
        lo.c('No search words')
        exit(1)

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

    cdef STR_t[:, ::1] sw = np.array(list(search_words)).view(STR).reshape(swl, -1)

    #cdef cnp.ndarray[cnp.uint8_t, ndim=1] swlens = np.empty(swl, np.uint8)
    cdef SIZE_t[::1] swlens = npe(swl, SIZE)

    for swi, sws in enumerate(search_words):
        #sw[swi] = sws.encode('utf8')
        swlens[swi] = len(sws)

    #Settings.search_words = sw
    #Settings.search_words_l = swl

    cdef Board full = Board(Settings.board, Settings.default_board)
    Settings.node_board = full

    cdef Node[:, ::1] nodes = full.nodes # need?
    cdef Node[::1] sing_nodes
    cdef Py_ssize_t i, tot
    cdef int ic
    cdef list search_rows, search_cols
    cdef bint is_col

    lo.s('Solving...\n')

    if full.new_game:
        lo.s(' = Fresh game = ')
        #todo fix
        #no = next(full.get_by_attr('is_start', True), None)
        # if no:
        #     check_node(no)

    else:
        if _s.SEARCH_NODES is None:
            search_rows = list(range(Settings.shape[0]))
            search_cols = list(range(Settings.shape[1]))

        else:
            search_rows = _s.SEARCH_NODES[0]
            search_cols = _s.SEARCH_NODES[1]

        ic = 0
        tot = len(search_rows) + len(search_cols)

        is_col = False
        for i in search_rows:
            if lo.is_enabled('s'):
                ic += 1
                lo.s('Checking row %2i  (%2i / %i)', i, ic, tot)
            sing_nodes = nodes[i]
            parse_nodes(sing_nodes, sw, swlens, is_col)

        is_col = True
        for i in search_cols:
            if lo.is_enabled('s'):
                ic += 1
                lo.s('Checking col %2i  (%2i / %i)', i, ic, tot)

            #sing_nodes = nodes.T[i]
            #sing_nodes = nodes[i]
            sing_nodes = np.asarray(nodes.T[i], Node, 'C')
            parse_nodes(sing_nodes, sw, swlens, is_col)


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

#convert to python?
cdef void show_solution(
    #WordDict[:]
    object[:, ::1] nodes,
    #str[:, ::1] nodes,
    list words, bint no_words
):
    # todo mark blanks

    cdef:
        WordDict w, best
        list newlist, cutlist, js
        Py_ssize_t rown, i
        BOOL_t x, y
        str horiz
        Letter letter
        object[:, ::1] nodes_copy
        #str[:, ::1] nodes_copy

    if not words:
        print('\nNo solution.')
        return

    newlist = sorted(words, key=lambda k: k.pts, reverse=True)
    if lo.is_enabled('s'):
        if Settings.num_results == 0:
            cutlist = newlist
        else:
            cutlist = newlist[:Settings.num_results]

        print()
        lo.s(f'-- Results ({len(cutlist)} / {len(newlist)}) --\n')
        for w in reversed(cutlist):
            lo.s(w.sol())

    best = newlist[0]

    # todo fix combining chars
    # todo fix small 10.

    cdef str u_dash = '\u2500'

    cdef str u_bx_ul = '\u250c'
    cdef str u_bx_ur = '\u2510'
    cdef str u_bx_bl = '\u2514'
    cdef str u_bx_br = '\u2518'

    cdef str u_cm_u = '\u0305'
    cdef str u_cm_b = '\u0332'
    cdef str u_cm = '|' + u_cm_u + u_cm_b

    cdef str board_hl = '\x1b[33m'  # yellow
    cdef str board_cl = '\x1b[0m'

    cdef int smalltens = 9361

    if no_words:  # todo: print xs instead?
        print(f'\n<solution hidden> (len: {len(best.word)})')

    else:
        nodes_copy = nodes[:, :]

        for letter in best.letters:
            x = letter.x
            y = letter.y
            if not nodes_copy[x, y] or nodes_copy[x, y] == ' ':
                nodes_copy[x, y] = f'{board_hl}{chr(letter.value)}{board_cl}'

        js = []
        for i in range(nodes_copy.shape[0]):
            if i < 10 or i > 20:
                js.append(str(i))
            else:
                js.append(chr(smalltens + (i - 10)))

        horiz =  u_dash + (f'{u_dash}+' * nodes_copy.shape[1]) + u_dash

        print(f'\n     {" ".join(js)}')

        print('   ' + u_bx_ul + horiz + u_bx_ur)

        for rown in range(nodes_copy.shape[0]):
            print(f'{rown:2d} {u_cm} {" ".join(nodes_copy[rown])}  {u_cm}')

        print('   ' + u_bx_bl + horiz + u_bx_br)

    print(f'\nPoints: {best.pts}')


# except *

# @cython.binding(True)
# @cython.linetrace(True)
# cpdef void cmain(
#     str filename, str dictionary, bint no_words, list exclude_letters, bint overwrite, int num_results, str log_level
# ):
cdef void cmain(
    str filename, str dictionary, bint no_words, list exclude_letters, bint overwrite, int num_results, str log_level
):
    cdef:
        dict points
        object pdboard
        #object[:, ::1] board, default_board
        object[:, ::1] board
        object[:, :] default_board
        #cnp.ndarray letters
        list rack
        #todo declare object rather than list?
        int board_size
        object board_name
        bint has_cache

    log_level = log_level.upper()
    if log_level != lo.logger.getEffectiveLevel():
        lo.set_level(log_level)

    cdef object this_board_dir
    if filename is not None:
        this_board_dir = loadfile(_s.BOARD_DIR, filename, is_file=False)

        pdboard = pd.read_pickle(loadfile(this_board_dir, _s.BOARD_FILENAME))
        rack = pd.read_pickle(loadfile(this_board_dir, _s.LETTERS_FILENAME))

    else:
        filename = '_manual'

        pdboard = pd.DataFrame(_s.BOARD)
        rack = _s.LETTERS

    cdef str el
    if exclude_letters:
        for el in exclude_letters:
            #letters = letters[letters != el]
            rack.remove(el)

    #cdef list rack_b = [ord(l) for l in letters]
    cdef BOOL_t[::1] rack_b = npz(MAX_ORD, BOOL)
    cdef object l
    for l in rack:
        # actually a long... TODO check ords
        rack_b[<Py_ssize_t>ord(l)] += 1

    #rack_b.sort()

    Settings.rack_l = rack
    Settings.rack[::1] = rack_b
    Settings.rack_s = len(rack)

    board = pdboard.to_numpy(np.object_)  # type: np.ndarray

    board_size = board.size
    if board_size == 15 * 15:
        board_name = _s.DEF_BOARD_BIG
    elif board_size == 11 * 11:
        board_name = _s.DEF_BOARD_SMALL
    else:
        board_name = None
        lo.c('Board size ({}) has no match'.format(board_size))
        #sys.exit(1)
        exit(1)

    default_board = pd.read_pickle(loadfile(board_name)).to_numpy(np.object_)

    points = json.load(loadfile(_s.POINTS_DIR, dictionary + '.json').open())  # Dict[str, List[int]]

    if lo.is_enabled('s'):
        lo.s('Game Board:\n{}'.format(pdboard))
        lo.v('Default:\n{}'.format(pd.read_pickle(board_name)))
        lo.s('Rack:\n{}'.format(rack))
        print()
    else:
        print('Running...')

    Settings.default_board = default_board  # todo put back?
    Settings.board = board
    Settings.shape = board.shape

    #cdef dict cpoints = {}
    cdef BOOL_t[::1] cl_points = npz(MAX_ORD, BOOL)
    cdef BOOL_t[::1] cl_amts = npz(MAX_ORD, BOOL)
    #cdef lpts_t lpt
    cdef str k
    cdef list v
    cdef SIZE_t ok
    for k, v in points.items():
        ok = ord(k)
        cl_amts[ok] = (<int>v[0])
        cl_points[ok] = (<int>v[1])
    Settings.points = cl_points
    Settings.amts = cl_amts

    Settings.num_results = num_results

    # todo add search words and nodes

    cdef str md5_rack = md5(''.join(sorted(rack)).encode()).hexdigest()[:9:1]
    cdef object solution_filename = Path(_s.SOLUTIONS_DIR, '{}_{}.npz'.format(filename, md5_rack))

    if overwrite is True:
        has_cache = False
    else:
        has_cache = solution_filename.exists()

    cdef Py_ssize_t r, c
    cdef object[:, ::1] solved_board, s_nodes
    #cdef str[:, ::1] solved_board, s_nodes

    cdef object solution_data  # <class 'numpy.lib.npyio.NpzFile'>
    cdef list s_words

    if has_cache is False:
        solve(dictionary)

        solved_board = npz(Settings.shape, np.object_)
        #solved_board = npz(Settings.shape, np.str_)
        for r in range(solved_board.shape[0]):
            for c in range(solved_board.shape[1]):
                solved_board[r, c] = Settings.node_board.nodes[r, c].display

        show_solution(nodes=solved_board, words=Settings.node_board.words, no_words=no_words)

        np.savez(solution_filename, nodes=solved_board, words=Settings.node_board.words)

    else:
        lo.s('Found existing solution')
        solution_data = np.load(solution_filename)

        s_nodes = solution_data['nodes']
        s_words = solution_data['words'].tolist()

        show_solution(nodes=s_nodes, words=s_words, no_words=no_words)



# def
def main(
    filename: str = None, dictionary: str = _s.DICTIONARY, no_words: bool = False, exclude_letters: list = None, overwrite: bool = False, num_results: int = _s.NUM_RESULTS,
    log_level: str = _s.DEFAULT_LOGLEVEL, **_kw: dict
) -> None:
    cmain(filename, dictionary, no_words, exclude_letters, overwrite, num_results, log_level)
