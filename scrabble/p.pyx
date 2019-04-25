# cython: warn.maybe_uninitialized=True, warn.undeclared=True, warn.unused=True, warn.unused_arg=True, warn.unused_result=True, infer_types.verbose=True

"""Parse and solve a scrabble board."""

cimport cython
#from cython.parallel import prange

from libc.stdio cimport printf, snprintf


cdef object json, np, pd, md5, Path, _s, log_init
#cdef module np
#import sys

import json
import numpy as np
import pandas as pd  # remove

from hashlib import md5
from pathlib import Path

import settings as _s
from logs import log_init

#import signal
#from functools import lru_cache


#cnp.import_array()
#cnp.import_umath()


cdef object lo = log_init(_s.DEFAULT_LOGLEVEL)


#ctypedef unsigned short us
ctypedef unsigned char uchr

#ctypedef unsigned char* uchrp
#ctypedef (us, us) dual
#ctypedef us dual[2]

#ctypedef cnp.uint32_t DTYPE_t
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
DEF bl = ord('?')
DEF L_ST = 65
DEF L_EN = 91
DEF MAX_NODES = 15
DEF MAX_ORD = 127  # todo replace 127


# cdef packed struct lpts_t:
#     uchr amt
#     uchr pts


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
    def __cinit__(self, int pts):
        #self.w = w
        self.pts = pts

    #def str sol(self):
    @cython.wraparound(False)
    cdef void sol(self, char* buffer):
        cdef char direc = b'c' if self.w.is_col is True else b'r'

        cdef Letter lf = self.w.letters.l[0]
        cdef Letter ll = self.w.letters.l[self.w.letters.len - 1]

        cdef BOOL_t p1, p2, p3
        if self.w.is_col is True:
            p1 = lf.y
            p2 = lf.x
            p3 = ll.x
        else:
            p1 = lf.x
            p2 = lf.y
            p3 = ll.y

        cdef char word[MAX_NODES + 1]
        cdef Py_ssize_t i
        for i in range(MAX_NODES + 1):
            if i < self.w.letters.len:
                word[i] = self.w.letters.l[i].value
            else:
                word[i] = NUL

        snprintf(buffer, 64, 'pts: %3i | dir: %c | pos: %2u x %2u,%2u | w: %s',
            self.pts, direc, p1, p2, p3, word
        )

    def __reduce__(self):
        return rebuild_worddict, (self.w, self.pts)


@cython.final(True)
cpdef WordDict rebuild_worddict(WordDict_Struct ws, int pts):
    cdef WordDict wd = WordDict(pts)
    wd.w = ws
    return wd


@cython.final(True)
cdef class Node:
    # todo test no types in cinit
    def __cinit__(self, BOOL_t x, BOOL_t y, str val, BOOL_t mult_a, BOOL_t mult_w, bint is_start):
        self.n.letter.x = x
        self.n.letter.y = y

        self.n.letter.is_blank = False
        self.n.letter.from_rack = False
        #self.letter.pts = 0

        self.n.mult_a = mult_a
        self.n.mult_w = mult_w

        self.n.is_start = is_start
        self.n.has_edge = False

        #cdef lpts_t lpt

        if not val:
            self.n.has_val = False
            self.n.letter.value = 0
            self.n.pts = 0
            #self.display = ' '  # todo display method
        else:
            self.n.has_val = True
            self.n.letter.value = ord(val)
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
            s = chr(self.n.letter.value)
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
        #self.nodes = np.empty_like(default_board, Node)

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
            if node.up.n.has_val:
                node.n.has_edge = True

        if r < self.nodes_rl - 1:
            node.down = self.nodes[r+1, c]
            if node.down.n.has_val:
                node.n.has_edge = True

        if c > 0:
            node.left = self.nodes[r, c-1]
            if node.left.n.has_val:
                node.n.has_edge = True

        if c < self.nodes_cl - 1:
            node.right = self.nodes[r, c+1]
            if node.right.n.has_val:
                node.n.has_edge = True


    cdef void _set_adj_words(self, Node n, str d):
        cdef:
            Node[:] loop_nodes
            #cnp.ndarray loop_nodes
            bint rev = False
            int xx = n.n.letter.x
            int yy = n.n.letter.y


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
            if not p.n.has_val:
                break
            else:
                lets_pts += p.n.pts  # <STR_t>

                nv = p.n.letter.value
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
        if n.n.has_val:
            n.vlet_view[:, n.n.letter.value] = True
            return

        if not n.n.has_edge:
            n.vlet_view[:, L_ST:L_EN] = True
            return

        cdef Py_ssize_t i_s = L_ST

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


    cdef bint _check_adj_words(self, BOOL_t i, Node bef, Node aft, str bef_w, str aft_w):
        cdef str new_word

        if (bef is None or not bef.n.has_val) and (aft is None or not aft.n.has_val):
            return True

        new_word = bef_w + <str>chr(i) + aft_w
        #new_word = ''.join(chr(ns.value) for ns in new_let_list)

        #lo.x(new_word)

        return new_word in Settings.words


    cdef void _set_map(self, Node[:] nodes, bint is_col):
        cdef:
            Py_ssize_t t, l, e, ai1
            Py_ssize_t nlen = nodes.shape[0]
            Py_ssize_t max_swl = Settings.shape[is_col]

            # todo this vs numpy bint?
            bint has_edge, has_blanks

            Node no

            bint valid_lengths[MAX_NODES][MAX_NODES]
            bint[:, :] vlen_view

        vlen_view = valid_lengths

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
                no = nodes[t - 1]
                if no.n.has_val:
                    vlen_view[t, :] = False
                    continue

            # for all possible wls...
            for l in range(max_swl - t):
                # for each valid wl, if last node has a val, disable for that wl
                #lo.v(l)
                ai1 = t + l + 1
                if ai1 < nlen:
                    no = nodes[ai1]
                    #lo.d(no)
                    if no.n.has_val:
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
                    no = nodes[e]
                    if no.n.has_edge:
                        has_edge = True
                    if not no.n.has_val:
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
cdef SIZE_t calc_pts(Letter_List lets_info, N nodes[MAX_NODES], bint is_col, Py_ssize_t start) nogil:
    cdef:
        Py_ssize_t i
        Py_ssize_t lcnt = 0
        N nd
        Letter le
        STR_t nv
        BOOL_t lpts
        #BOOL_t bl = Settings.blanks
        SIZE_t pts = 0  # todo: is sizet needed or better?
        SIZE_t extra_pts
        SIZE_t tot_extra_pts = 0
        SIZE_t tot_pts
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
cdef bint lets_match(STR_t[::1] word, Py_ssize_t wl, N nodes[MAX_NODES], Py_ssize_t start, bint is_col) nogil:
#cdef bint lets_match(STR_t[::1] word, Py_ssize_t wl, valid_let_t[:] vl_list, Py_ssize_t start) nogil:
    cdef Py_ssize_t i
    cdef STR_t nv

    for i in range(wl):
        nv = word[i]
        # mismatch in nv?
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
cdef bint rack_check(STR_t[::1] word, Py_ssize_t wl, bint nvals[MAX_NODES], Py_ssize_t start, BOOL_t blanks, int[:] base_rack) nogil:  # or memview for nvals?
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
cdef Letter_List rack_match(STR_t[::1] word, Py_ssize_t wl, N nodes[MAX_NODES], Py_ssize_t start, int[:] base_rack) nogil:
    cdef:
        Py_ssize_t i
        Py_ssize_t r = 0
        STR_t let
        int rack[MAX_ORD]
        BOOL_t num

        Letter_List lets_info
        #Letter[:] lets_view = <Letter[:MAX_NODES]>lets_info

        Letter le
        BOOL_t[::1] spts = Settings.points
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


cdef void add_word(Letter_List lets_info, bint is_col, SIZE_t tot_pts):
    cdef WordDict w
    cdef WordDict_Struct ws

    w = WordDict(tot_pts)

    ws.is_col = is_col
    ws.letters = lets_info

    w.w = ws

    Settings.node_board.words.append(w)


#cdef loi


# @cython.binding(True)
# @cython.linetrace(True)
# @cython.wraparound(False)
# cpdef void parse_nodes(N nodes[MAX_NODES], STR_t[:, ::1] sw, SIZE_t[::1] swlens, bint is_col) except *:
@cython.wraparound(False)
cdef void parse_nodes(N nodes[MAX_NODES], STR_t[:, ::1] sw, SIZE_t[::1] swlens, bint is_col): # nogil:
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
        #with gil:
        if lo.is_enabled('i'):
            lo.i('-> [empty]')
        return


    cdef:
        Py_ssize_t i
        bint nvals[MAX_NODES]
        Py_ssize_t s, wl, wl1, sn
        STR_t[::1] ww
        BOOL_t blanks = Settings.blanks
        int[:] base_rack = Settings.rack
        Letter_List lets_info
        SIZE_t tot_pts

    for i in range(nlen):
        n = nodes[i]
        nvals[i] = n.has_val

    for s in range(sw.shape[0]):
        # is there a problem with the conversion here
        wl = swlens[s]
        if wl > nlen:
            continue

        wl1 =  wl - 1

        ww = sw[s]
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

            lets_info = rack_match(ww, wl, nodes, sn, base_rack)

            tot_pts = calc_pts(lets_info, nodes, is_col, sn)

            add_word(lets_info, is_col, tot_pts)


def _unused(): pass


@cython.final(True)
cpdef object loadfile(tuple paths, bint is_file = True):
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
#cpdef void solve(str dictionary):
cdef void solve(str dictionary):
    cdef:
        list wordlist
        BOOL_t blanks
        Py_ssize_t wl_len, mnlen, wi, wl #, sl
        set s_words
        frozenset words
        #set s_search_words
        frozenset search_words
        str w
        #object l

    wordlist = loadfile((_s.WORDS_DIR, dictionary + '.txt')).read_text().splitlines()
    if not wordlist:
        return

    blanks = Settings.rack[bl]
    mnlen = Settings.shape[0] if Settings.shape[0] > Settings.shape[1] else Settings.shape[1]

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
    #cdef Node[::1] sing_nodes
    #cdef N[:] ns_r = npe(Settings.shape[0], N)
    #cdef N[:] ns_c = npe(Settings.shape[1], N)
    cdef N ns_r[MAX_NODES]
    cdef N ns_c[MAX_NODES]
    #cdef N[:] ns_rv
    #cdef N[:] ns_cv

    cdef Py_ssize_t ni
    cdef Node no

    cdef Py_ssize_t ir, ic
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
            if lo.is_enabled('s'):
                lo.s('Checking all lines (%2i x %2i)...', Settings.shape[0], Settings.shape[1])

        else:
            search_rows = _s.SEARCH_NODES[0]
            search_cols = _s.SEARCH_NODES[1]
            if lo.is_enabled('s'):
                lo.s(f'Checking custom lines ({search_rows}, {search_cols})...')


        is_col = False
        for ir in search_rows:
            #sing_nodes = nodes[i]

            for ni in range(Settings.shape[0]):
                no = nodes[ir, ni]
                ns_r[ni] = no.n

            #ns_rv = ns_r

            #parse_nodes(ns_rv[:Settings.shape[0]], sw, swlens, is_col)
            parse_nodes(ns_r, sw, swlens, is_col)

        is_col = True
        for ic in search_cols:
            #sing_nodes = nodes.T[i]
            #sing_nodes = nodes[i]
            #sing_nodes = np.asarray(nodes.T[i], Node, 'C')

            for ni in range(Settings.shape[1]):
                no = nodes[ni, ic]
                ns_c[ni] = no.n

            #ns_cv = ns_c

            parse_nodes(ns_c, sw, swlens, is_col)


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


cdef void print_board(STR_t[:, ::1] nodes, Letter_List lets):
    cdef:
        Py_ssize_t i
        Py_ssize_t rown, colt = 0, colb = 0

        # todo fix small 10.
        int smalltens = 9361
        Py_UNICODE smallten_char

        Py_UNICODE u_dash = '\u2500'

        Py_UNICODE u_bx_ul = '\u250c'
        Py_UNICODE u_bx_ur = '\u2510'
        Py_UNICODE u_bx_bl = '\u2514'
        Py_UNICODE u_bx_br = '\u2518'

        Py_UNICODE u_sep_hor_le = '\u2524'
        Py_UNICODE u_sep_hor_ri = '\u251C'
        Py_UNICODE u_sep_ver_up = '\u2534'
        Py_UNICODE u_sep_ver_dn = '\u252C'

        char* board_hl = b'\x1b[33m'  # yellow
        char* board_cl = b'\x1b[0m'


    # - col nums
    printf('\n      ')
    for i in range(nodes.shape[0]):
        if i < 10 or i > 20:
            printf('%zu', i)
        else:
            smallten_char = smalltens + (i - 10)
            printf('%lc', smallten_char)
        if i != nodes.shape[0] - 1:
            printf(' ')
    printf('\n')

    # - col seps top
    printf('   %lc%lc', u_bx_ul, u_dash)
    while colt < nodes.shape[1]:
        printf('%lc%lc', u_dash, u_sep_ver_up)
        colt += 1
    printf('%lc%lc%lc\n', u_dash, u_dash, u_bx_ur)


    cdef:
        STR_t best_map[MAX_NODES][MAX_NODES]
        STR_t[:, ::1] best_map_v = best_map
        STR_t nval, bval
        Py_ssize_t l
        Letter letter
        BOOL_t x, y

    best_map_v[:, :] = 0

    for l in range(lets.len):
        letter = lets.l[l]
        x = letter.x
        y = letter.y
        nval = nodes[x, y]
        if not nval:
            best_map_v[x, y] = letter.value

    # - rows
    for rown in range(nodes.shape[0]):
        printf('%2zu %lc  ', rown, u_sep_hor_le)
        for i in range(nodes.shape[1]):
            nval = nodes[rown, i]
            if not nval:
                bval = best_map_v[rown, i]
                if bval:
                    printf('%s%c%s', board_hl, bval, board_cl)
                else:
                    printf(' ')
            else:
                printf('%c', nval)

            if i != nodes.shape[1] - 1:
                printf(' ')

        printf('  %lc\n', u_sep_hor_ri)

    # - col seps bottom
    printf('   %lc%lc', u_bx_bl, u_dash)
    while colb < nodes.shape[1]:
        printf('%lc%lc', u_dash, u_sep_ver_dn)
        colb += 1
    printf('%lc%lc%lc\n', u_dash, u_dash, u_bx_br)


#WordDict[:]
cdef void show_solution(STR_t[:, ::1] nodes, list words, bint no_words):
    # todo mark blanks

    if not words:
        printf('\nNo solution.\n')
        return

    cdef list newlist = sorted(words, key=lambda k: k.pts, reverse=True)
    # if not newlist:
    #     lo.e('Error: solution list is empty')
    #     return

    cdef:
        WordDict w, best
        list cutlist
        char buffer[64]

    if lo.is_enabled('s'):
        if Settings.num_results == 0:
            cutlist = newlist
        else:
            cutlist = newlist[:Settings.num_results]

        printf('\n')
        lo.s(f'-- Results ({len(cutlist)} / {len(newlist)}) --\n')
        for w in reversed(cutlist):
            w.sol(buffer)
            lo.s(buffer.decode())

    best = newlist[0]

    if no_words:  # todo: print xs instead?
        printf('\n<solution hidden> (len: %zu)\n', best.w.letters.len)

    else:
        print_board(nodes, best.w.letters)


    printf('\nPoints: %i\n', best.pts)


# except *

# @cython.binding(True)
# @cython.linetrace(True)
# cpdef void cmain(
#     str filename, str dictionary, bint no_words, list exclude_letters, bint overwrite, int num_results, str log_level
# ):
@cython.wraparound(False)
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
        this_board_dir = loadfile((_s.BOARD_DIR, filename), is_file=False)

        pdboard = pd.read_pickle(loadfile((this_board_dir, _s.BOARD_FILENAME)))
        rack = pd.read_pickle(loadfile((this_board_dir, _s.LETTERS_FILENAME)))

    else:
        filename = '_manual'

        pdboard = pd.DataFrame(_s.BOARD)
        rack = _s.LETTERS

    cdef str el
    if exclude_letters:
        for el in exclude_letters:
            #letters = letters[letters != el]
            rack.remove(el)

    cdef object l
    for l in rack:
        # actually a long... TODO check ords
        Settings.rack[<Py_ssize_t>ord(l)] += 1

    #rack_b.sort()

    Settings.rack_l = rack
    Settings.rack_s = len(rack)

    board = pdboard.to_numpy(np.object_)  # type: np.ndarray

    board_size = board.size
    if board_size == MAX_NODES * MAX_NODES:
        board_name = _s.DEF_BOARD_BIG
    elif board_size == 11 * 11:
        board_name = _s.DEF_BOARD_SMALL
    else:
        board_name = None
        lo.c('Board size ({}) has no match'.format(board_size))
        #sys.exit(1)
        exit(1)

    default_board = pd.read_pickle(loadfile((board_name,))).to_numpy(np.object_)

    points = json.load(loadfile((_s.POINTS_DIR, dictionary + '.json')).open())  # Dict[str, List[int]]

    if lo.is_enabled('s'):
        lo.s('Game Board:\n{}'.format(pdboard))
        lo.v('Default:\n{}'.format(pd.read_pickle(board_name)))
        lo.s('Rack:\n{}'.format(rack))
        printf('\n')
    else:
        printf('Running...\n')

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
    #cdef object[:, ::1] solved_board, s_nodes
    cdef STR_t[:, ::1] solved_board, s_nodes

    cdef object solution_data  # <class 'numpy.lib.npyio.NpzFile'>
    cdef list s_words

    if has_cache is False:
        solve(dictionary)

        #solved_board = npz(Settings.shape, np.object_)
        solved_board = npz((Settings.shape[0], Settings.shape[1]), STR)
        for r in range(solved_board.shape[0]):
            for c in range(solved_board.shape[1]):
                solved_board[r, c] = Settings.node_board.nodes[r, c].n.letter.value

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
