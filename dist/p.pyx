"""Parse and solve a scrabble board."""

cimport cython
#from cython.parallel import prange

cdef object json, sys, md5, Path, np, pd, log_init, _s

import json
import sys
from hashlib import md5
from pathlib import Path

#import signal
#from functools import lru_cache

cimport numpy as cnp
import numpy as np

import pandas as pd  # remove

from logs import log_init
import settings as _s

cnp.import_array()


cdef str DEFAULT_LOGLEVEL = 'SUCCESS'
cdef str DICTIONARY = 'wwf'
cdef int NUM_RESULTS = 15


cdef object lo = log_init(DEFAULT_LOGLEVEL)

#cdef object exc

ctypedef unsigned short us
#ctypedef (us, us) dual
#ctypedef us dual[2]
#ctypedef unsigned char* uchr
ctypedef unsigned char uchrn

#ctypedef cnp.uint32_t DTYPE_t
ctypedef cnp.int32_t DTYPE_t
DTYPE = np.int32
D8 = np.uint8

# no numpy array .flags to see contiguous
#DTYPE = np.intc

#ctypedef np.ndarray[str, ndim=2] board_t  # str 32?
#ctypedef np.ndarray[object, ndim=2] nb_t
#ctypedef np.ndarray[object, ndim=1] nodelist_t

#cdef bytes NUL = b'\0'
DEF NUL = b'\0'
DEF bl = ord('?')
DEF o_l = ord('l')
DEF o_w = ord('w')
DEF o_x = ord('x')
#more of these

# cdef packed struct multiplier_t:
#     uchrn amt
#     uchr typ

# cdef packed struct lpts_t:
#     uchrn amt
#     uchrn pts


ctypedef packed struct Letter:
    bint is_blank
    bint from_rack
    uchrn pts
    uchrn mult_a
    DTYPE_t mult_t
    uchrn x
    uchrn y
    DTYPE_t value


# @cython.freelist(50000)  # todo check
# cdef class Letter:
#     cdef:
#         bint is_blank
#         bint from_rack
#         uchrn pts
#         multiplier_t mult
#         uchrn x
#         uchrn y
#         #const unsigned char* value
#         #DTYPE_t* value  # todo
#         DTYPE_t value
#         # todo define getter
#
#     def __cinit__(self):
#         self.is_blank = False
#         self.from_rack = False
#         self.pts = 0
#
#         self.mult.amt = 1
#         self.mult.typ = NUL
#         self.x = 0
#         self.y = 0
#         self.value = 0
#
#     def get_pos(self) -> tuple:
#         return self.x, self.y
#
#     def get_pos_str(self) -> str:
#         return f'{self.x:2d},{self.y:2d}'
#
#     def get_val(self) -> str:
#         if self.value:
#             #return self.value.view('U1')  # pointer?
#             return chr(self.value)
#         else:
#             return ''
#
#     def __str__(self) -> str:
#         return '<'\
#             f'v: {self.get_val()}, '\
#             f'pos: [{self.get_pos_str()}], '\
#             f'bl: {"T" if self.is_blank is True else "f"}, '\
#             f'rk: {"T" if self.from_rack is True else "f"}, '\
#             f'pts: {"_" if self.pts == 0 and self.is_blank is False else self.pts}, '\
#             f'mult: "{"_" if self.mult.amt == 1 else str(self.mult.amt) + self.mult.typ.decode("utf8").upper()}"'\
#             '>'
#
#     def __repr__(self) -> str:
#         return self.__str__()
#
#     def __reduce__(self):
#         return rebuild_letter, (self.is_blank, self.from_rack, self.pts, self.mult, self.x, self.y, self.value)
#
#
# def rebuild_letter(is_blank, from_rack, pts, mult, x, y, value):
#     l = Letter()
#     l.is_blank = is_blank
#     l.from_rack = from_rack
#     l.pts = pts
#     l.mult = mult
#     l.x = x
#     l.y = y
#     l.value = value
#     return l


#ctypedef class WordDict:
cdef class WordDict:
    cdef:
        #DTYPE_t[:] _word
        str word
        #uchr word
        bint _is_row
        str direc
        readonly us pts
        list letters  # of Letter
        #Letter letters[100]

    def __cinit__(self, DTYPE_t[:] word, bint is_row, us pts, list letters):
        #self._word = word
        self.word = ''.join([chr(l) for l in word])
        self._is_row = is_row
        self.direc = 'row' if self._is_row is True else 'col'
        self.pts = pts
        self.letters = letters

    cdef str sol(self):
        cdef Letter lf = self.letters[0]
        cdef Letter ll = self.letters[-1]
        return 'pts: {:3d} | dir: "{}" | pos: [{:2d}{:2d}] - [{:2d}{:2d}] | w: {}'.format(
            self.pts, self.direc, lf.x, lf.y, ll.x, ll.y, self.word
        )

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
        return rebuild_worddict, (self.word, self._is_row, self.pts, self.letters)

def rebuild_worddict(_word, _is_row, pts, letters):
    cdef cnp.ndarray word = np.array([ord(w) for w in _word], dtype=DTYPE)
    return WordDict(word, _is_row, pts, letters)


cdef class CSettings:
    cdef:
        #cnp.ndarray board, default_board
        #cnp.ndarray[str, ndim=2] board, default_board

        list letters
        #cnp.ndarray[DTYPE_t, ndim=1] letters
        #DTYPE_t[:] letters
        Py_ssize_t letters_s

        uchrn blanks
        #dict shape

        uchrn[:] points
        uchrn[:] amts

        set words
        Board node_board
        int num_results

    def __cinit__(self):
        #self.board = None
        #self.default_board = None #np.empty((0,0), dtype=np.str_)

        self.letters = list()
        #self.letters = None
        #self.letters = np.empty(0) # dtype=bytes?

        self.letters_s = 0
        self.blanks = 0

        #self.shape = {'row': 0, 'col': 0}  # type: Dict[str, int]

        self.points = None
        self.amts = None

        self.words = set()  ## type: Set[bytes]
        self.node_board = None
        self.num_results = NUM_RESULTS


cdef CSettings Settings = CSettings()


cdef class Node:
    cdef:
        #uchrn x, y

        Letter letter

        # uchrn points
        # uchrn mult_a
        # DTYPE_t mult_t
        # DTYPE_t value

        bint is_start
        bint has_val

        Node up, down, left, right
        #cnp.ndarray[:] up_lets, down_lets, left_lets, right_lets
        #Letter[:] up_lets, down_lets, left_lets, right_lets
        str up_word, down_word, left_word, right_word
        bint has_edge

        public str display

    def __cinit__(self, uchrn x, uchrn y, str val, uchrn mult_a, DTYPE_t mult_t):  # todo test no types in cinit
        self.letter.x = x
        self.letter.y = y

        self.letter.is_blank = False
        self.letter.from_rack = False

        self.letter.mult_a = mult_a

        if mult_t == o_x:
            self.letter.mult_t = 0
            self.is_start = True
        else:
            self.letter.mult_t = mult_t
            self.is_start = False

        #cdef lpts_t lpt

        if not val:
            self.has_val = False
            self.display = ' '  # todo display method
            self.letter.value = 0
            self.letter.pts = 0
        else:
            self.has_val = True
            self.display = val.upper()
            self.letter.value = ord(val)
            self.letter.pts = Settings.points[self.letter.value]

            # try:
            #     lpt = Settings.points[self.value]
            #     self.points = lpt.pts
            # except (KeyError, IndexError):
            #     lo.e('Could not get point value of "{}"'.format(val))
            #     sys.exit(1)

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

        self.has_edge = False

    cdef str str_pos(self):
        return f'[{self.letter.x:2d},{self.letter.y:2d}]'

    def __str__(self) -> str:
        cdef str s = '_'
        #todo != 0
        if self.has_val:
            s = chr(self.letter.value)
        return '<Node: {} v: {}>'.format(self.str_pos(), s)

    def __repr__(self) -> str:
        return self.__str__()



cdef class Board:
    cdef:
        cnp.ndarray[:, :] board
        cnp.ndarray[:, :] default_board
        #cnp.ndarray default_board
        #np.ndarray[str, ndim=2] default_board
        bint new_game
        #nodelist_t nodes
        cnp.ndarray nodes
        #cnp.ndarray[:, :] nodes
        Py_ssize_t nodes_rl, nodes_cl

        list words  # todo of word_dicts, what about an object? _pyx_v_self->words = ((PyObject*)__pyx_t_4);
        #np.ndarray[dtype=word_dict] words
        #cnp.ndarray words
        #WordDict[:] words


    #def __cinit__(self, board=Settings.board, default_board=Settings.default_board):
    def __cinit__(self, cnp.ndarray[:, :] board, cnp.ndarray default_board):
        cdef:
            Py_ssize_t r, c
            uchrn mamt
            str dbval, bval, x, d
            uchrn mult_a
            DTYPE_t mult_t
            Node node
            #cnp.ndarray board

        self.board = board
        self.default_board = default_board
        self.nodes = np.zeros_like(default_board, dtype=Node)

        self.nodes_rl = board.shape[0]
        self.nodes_cl = board.shape[1]

        #self.words = np.empty(0, dtype=WordDict)
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

                #todo move mult up to no bval
                mult_a = 1
                mult_t = 0
                if dbval:
                    dbval = dbval.strip()
                    if dbval == 'x':
                        if bval is None:
                            self.new_game = True
                            mult_t = o_x
                    else:
                        x = dbval[0]
                        mamt = int(x)
                        mult_a = mamt
                        mult_t = ord(dbval[1])

                node = Node(r, c, bval, mult_a, mult_t)
                self.nodes[r, c] = node

        for r in range(self.nodes_rl):
            for c in range(self.nodes_cl):
                node = self.nodes[r, c]
                self._set_node(r, c)
                for d in ('up', 'down', 'left', 'right'):
                    self._set_adj_nodes(node, d)


    #@lru_cache(1023)  # todo reenable?
    cdef Node get(self, int x, int y):
        return <Node>self.nodes[x, y]

    cdef Node get_by_attr(self, str attr, v):
        return filter(lambda obj: getattr(obj, attr) == v, self.nodes)

    # #@lru_cache()
    # cdef np.ndarray[Node, ndim=1] get_row(self, int x):
    #     return self.nodes[x]
    #
    # #@lru_cache()
    # cdef np.ndarray[Node, ndim=1] get_col(self, int y):
    #     return self.nodes[:,y]

    # cdef str _lets_as_str(self, list letters):
    #     #cdef DTYPE_t v
    #     return ''.join([chr(l['value']) for l in letters]) # todo make func


    cdef void _set_node(self, int r, int c):
        cdef Node node = self.nodes[r, c]

        if r > 0:
            node.up = self.nodes[r-1, c]
            if node.up.has_val:
                node.has_edge = True
                return

        if r < self.nodes_rl - 1:
            node.down = self.nodes[r+1, c]
            if node.down.has_val:
                node.has_edge = True
                return

        if c > 0:
            node.left = self.nodes[r, c-1]
            if node.left.has_val:
                node.has_edge = True
                return

        if c < self.nodes_cl - 1:
            node.right = self.nodes[r, c+1]
            if node.right.has_val:
                node.has_edge = True
                return

        # for d in ('up', 'down', 'left', 'right'):
        #     self._set_adj_nodes(d, node)


    cdef void _set_adj_nodes(self, Node n, str d):
        cdef:
            cnp.ndarray[:] loop_nodes
            #cnp.ndarray loop_nodes
            bint rev = False
            int xx = n.letter.x
            int yy = n.letter.y
            Py_ssize_t nl, ni
            Node p

        if d == 'up':
            loop_nodes = self.nodes[:xx, yy][::-1]
            rev = True
        elif d == 'down':
            loop_nodes = self.nodes[xx+1:, yy]
        elif d == 'left':
            loop_nodes = self.nodes[xx, :yy][::-1]
            rev = True
        else:
            loop_nodes = self.nodes[xx, yy+1:]

        cdef str l_s
        cdef str lets_str = ''
        nl = loop_nodes.shape[0]  # is this wrong?

        for ni in range(nl):
            p = loop_nodes[ni]
            if not p.has_val:
                break
            else:
                l_s = chr(p.letter.value)
                if rev is True:
                    lets_str = l_s + lets_str
                else:
                    lets_str += l_s

        if lets_str:
            if d == 'up':
                n.up_word = lets_str
            elif d == 'down':
                n.down_word = lets_str
            elif d == 'left':
                n.left_word = lets_str
            elif d == 'right':
                n.right_word = lets_str

    def __str__(self) -> str:
        return '<Board: size {:d}x{:d} | words: {}>'.format(self.nodes_rl, self.nodes_cl, len(self.words))

    # def __reduce__(self):
    #     return rebuild, (Settings.board, Settings.default_board)



#cdef npss = np.searchsorted

# def get_points(list word not None) -> int:
cdef us get_points(list word):
    cdef:
        #Py_ssize_t len_word = word.shape[0]
        Py_ssize_t len_word = len(word)
        us pts = 0
        uchrn word_mult = 1
        Py_ssize_t l
        uchrn p
        Letter letter

    for l in range(len_word):
        letter = word[l]

        if letter.from_rack is False:
            pts += letter.pts
        elif letter.is_blank is False:
            p = letter.pts

            if letter.mult_t == o_l:
                p *= letter.mult_a
            elif letter.mult_t == o_w:
                word_mult *= letter.mult_a

            pts += p

    #lo.x(pts)

    pts *= word_mult

    return pts

# except *

#@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void check_and_get(Node[:] node_list, DTYPE_t[:] word, Py_ssize_t word_len, bint is_row):
    cdef Node first, last

    first = node_list[0]
    if is_row is True:
        if first.left is not None and first.left.has_val: return
    else:
        if first.up is not None and first.up.has_val: return

    last = node_list[word_len - 1]
    if is_row is True:
        if last.right is not None and last.right.has_val: return
    else:
        if last.down is not None and last.down.has_val: return

    #todo what is the difference in not setting here? infer types?
    cdef:
        uchrn blanks = Settings.blanks

        Py_ssize_t lslen = Settings.letters_s
        list ls = Settings.letters[:lslen]
        #cnp.ndarray[DTYPE_t, ndim=1] ls = Settings.letters.base.copy()
        #list ls = ['a','b']

        #set aw = Settings.words
        Node no
        us pts
        us extra_pts = 0 #new_pts
        Py_ssize_t i, lsl
        #size_t lsl
        Letter le
        bint is_blank

        str new_word

        #letter_list new_lets
        Node bef_node, aft_node

        list word_lets = []
        list new_let_list

        WordDict w #, nw

        DTYPE_t nv, nov
        int nvi
        str cnv

        uchrn[:] spts = Settings.points


    #todo for nv in word[:word_len]:

    for i in range(word_len):
        no = node_list[i]
        nv = word[i]

        le = no.letter

        # le.is_blank = False
        # le.from_rack = False
        le.pts = 0
        le.value = nv

        nov = no.letter.value
        if nov:
            if nov != nv:
                #lo.x(f'mismatch {nov} v {nv}')
                return
            else:
                le.pts = no.letter.pts

        else:
            is_blank = False
            nvi = nv

            if nvi not in ls:
                if blanks > 0:
                    ls.remove(bl)
                    blanks -= 1
                    is_blank = True
                    #lo.d('removed blank: %s', blanks)
                else:
                    # todo skip forward if the next letter doesnt exist?
                    return
            else:
                ls.remove(nvi)

            le.from_rack = True

            if is_blank is False:
                le.pts = spts[nv]
            else:
                le.is_blank = True

            if is_row is True:
                bef_node = no.up
                aft_node = no.down
            else:
                bef_node = no.left
                aft_node = no.right

            if (bef_node is not None and bef_node.has_val) or (aft_node is not None and aft_node.has_val):
                #lo.d('bef: %s', bef_lets)
                #lo.d('aft: %s', aft_lets)

                #new_word = ''.join(chr(ns.value) for ns in new_let_list)
                cnv = (<str>chr(nvi))
                if is_row is True:
                    new_word = no.up_word + cnv + no.down_word
                else:
                    new_word = no.left_word + cnv + no.right_word

                #lo.x(new_word)

                if new_word not in Settings.words: # aw
                    return

                new_let_list = []

                while bef_node is not None and bef_node.has_val:
                    new_let_list.append(bef_node.letter)
                    if is_row is True:
                        bef_node = bef_node.up
                    else:
                        bef_node = bef_node.left

                new_let_list.append(le)

                while aft_node is not None and bef_node.has_val:
                    new_let_list.append(aft_node.letter)
                    if is_row is True:
                        aft_node = aft_node.down
                    else:
                        aft_node = aft_node.right

                extra_pts += get_points(new_let_list)
                #extra_pts += (<us>get_points(new_let_list))

                #new_lets = np.array(new_let_list, dtype=object)
                # for ii in range(len(new_lets)):
                #     nw.letters[ii] = new_lets[ii]

        word_lets.append(le)

    #lsl = np.count_nonzero(ls)
    lsl = len(ls)
    if lsl == lslen:
        return

    pts = get_points(word_lets) + extra_pts

    if lsl == 0:
        pts += 35

    w = WordDict(
        word,
        is_row,
        pts,
        word_lets
    )

    # if w not in Settings.node_board.words:
    #     Settings.node_board.words.append(w)
    # else:
    #     lo.e('already saw %s', w)

    Settings.node_board.words.append(w)

#@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void check_nodes(Node[:] nodes, DTYPE_t[:, :] sw, uchrn[:] swlens, bint is_row):
    cdef:
        #str w #, chk_dir
        #(uchr, Py_ssize_t) w
        Node x, xt

        #cnp.ndarray[:] sw = Settings.search_words
        #Py_ssize_t swl = Settings.search_words_l

        Py_ssize_t word_len, i, j, s, t
        Py_ssize_t nlen = nodes.shape[0]
        Node[:] subnodes
        DTYPE_t[:] ww
        bint has_edge = False

    for t in range(nlen):
        xt = nodes[t]
        if xt.has_edge is True:
            has_edge = True
            break
    if has_edge is False:
        return

    #for s in range(Settings.search_words_l): # todo swap test
    for s in range(sw.shape[0]):
        #w = sw[s]
        word_len = swlens[s]

        if word_len > nlen:
            continue

        ww = sw[s]
        #lo.d(sw.base.view(f'<U{sw.base.shape[1]}')[s, 0])

        #for i in prange(nlen - word_len + 1, nogil=True):
        for i in range(nlen - word_len + 1):
            has_edge = False
            for j in range(word_len):
                x = nodes[i + j]
                if x.has_edge is True:
                    has_edge = True
                    break
            if has_edge is True:
                subnodes = nodes[i:word_len + i]
                check_and_get(subnodes, ww, word_len, is_row)

    return


# todo: set default for dict? put in settings? move all settings out to options?
#cpdef, def

cdef void solve(str dictionary) except *:
    cdef list wordlist = []
    #todo make fnc exception
    try:
        wordlist = open(str(Path(_s.WORDS_DIR, dictionary + '.txt'))).read().splitlines()
    except FileNotFoundError as exc:
        lo.c('Could not find file: {}'.format(exc.filename))
        sys.exit(1)

    cdef uchrn blanks = Settings.letters.count(bl)

    cdef Board full = Settings.node_board

    cdef Py_ssize_t mnlen = max(full.nodes_cl, full.nodes_rl)
    cdef set words = {w for w in wordlist if len(w) <= mnlen}
    cdef set search_words = set()

    if _s.SEARCH_WORDS is None:
        if blanks > 0:
            search_words = words
        else:
            # todo: faster if str? why gen slower?
            search_words = {w for w in words if any([chr(l) in w for l in Settings.letters])}
    elif isinstance(_s.SEARCH_WORDS, tuple):
        search_words = set(wordlist[_s.SEARCH_WORDS[0]: _s.SEARCH_WORDS[1]])
    elif isinstance(_s.SEARCH_WORDS, set):
        search_words = _s.SEARCH_WORDS
    else:
        lo.c('Incompatible search words type: {}'.format(type(_s.SEARCH_WORDS)))
        sys.exit(1)

    Settings.blanks = blanks

    # todo allow extra mods
    # print(board)
    # print(board[10])
    # board[8][10] = 'Y'
    # letters = [b'F', b'G']

    #cdef set bwords = {s for s in words}
    Settings.words = words

    cdef str sws
    cdef Py_ssize_t swi, swl = len(search_words)

    # todo as one?
    #cdef uchr ss
    #cdef list sw = [(s.encode('utf8'), len(s)) for s in search_words]
    #cdef cnp.ndarray sw = np.zeros(swl, dtype=f'|S{mnlen}')
    #cdef cnp.ndarray[:] _sw = np.array([s for s in search_words])
    cdef DTYPE_t[:, :] sw = np.array([s for s in search_words])\
        .view(DTYPE).reshape(swl, -1)

    #cdef cnp.ndarray[cnp.uint8_t, ndim=1] swlens = np.empty(swl, dtype=np.uint8)
    cdef uchrn[:] swlens = np.zeros((swl,), dtype=D8)

    for swi, sws in enumerate(search_words):
        #sw[swi] = sws.encode('utf8')
        swlens[swi] = len(sws)

    #Settings.search_words = sw
    #Settings.search_words_l = swl

    #cdef cnp.ndarray[object, ndim=2] nodes = full.nodes
    cdef Py_ssize_t i, tot
    cdef int ic
    #cdef list search_rows, search_cols
    cdef bint is_row

    if full.new_game:
        lo.s(' = Fresh game = ')
        #todo fix
        #no = next(full.get_by_attr('is_start', True), None)
        # if no:
        #     check_node(no)

    else:
        if _s.SEARCH_NODES is None:
            search_rows = range(full.nodes_rl)
            search_cols = range(full.nodes_cl)

        else:
            search_rows = _s.SEARCH_NODES[0]
            search_cols = _s.SEARCH_NODES[1]

        ic = 0
        tot = len(search_rows) + len(search_cols)

        is_row = True
        for i in search_rows:
            if lo.is_enabled('s'):
                ic += 1
                lo.s('Checking row %2i  (%2i / %i)', i, ic, tot)
            check_nodes(full.nodes[i], sw, swlens, is_row)

        is_row = False
        for i in search_cols:
            if lo.is_enabled('s'):
                ic += 1
                lo.s('Checking col %2i  (%2i / %i)', i, ic, tot)
            check_nodes(full.nodes[:,i], sw, swlens, is_row)


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
    object[:, :] nodes, list words, bint no_words
)  except * :
    # todo mark blanks

    cdef:
        WordDict w, best
        list newlist, cutlist, js
        Py_ssize_t rown, i
        uchrn x, y
        str horiz
        Letter letter
        object[:, :] nodes_copy

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

    if no_words:  # todo: print xs instead?
        print(f'\n<solution hidden> (len: {len(best.word)})')

    else:
        nodes_copy = nodes[:, :]

        for letter in best.letters:
            x = letter.x
            y = letter.y
            if not nodes_copy[x, y] or nodes_copy[x, y] == ' ':
                nodes_copy[x, y] = f'\x1b[33m{chr(letter.value)}\x1b[0m'

        js = []
        for i in range(nodes_copy.shape[0]):
            if i < 10 or i > 20:
                js.append(str(i))
            else:
                js.append(chr(9361 + (i - 10)))

        horiz = '\u2500+' * nodes_copy.shape[1]

        print(f'\n     {" ".join(js)}')
        print('   \u250c\u2500' + horiz + '\u2500\u2510')

        for rown in range(nodes_copy.shape[0]):
            print('{:2d} |\u0332\u0305 {}  |\u0332\u0305'.format(rown, ' '.join([n for n in nodes_copy[rown]])))

        print('   \u2514\u2500' + horiz + '\u2500\u2518')

    print('\nPoints: {}'.format(best.pts))


cdef void cmain(
    str filename, str dictionary, bint no_words, list exclude_letters, bint overwrite, int num_results, str log_level
)  except * :
    cdef:
        dict points = {}
        object pdboard
        cnp.ndarray board, default_board
        #cnp.ndarray letters
        list letters = []
        #todo declare object rather than list?
        int board_size
        object board_name
        bint has_cache

    log_level = log_level.upper()
    if log_level != DEFAULT_LOGLEVEL:
        lo.set_level(log_level)

    cdef object this_board_dir
    if filename is not None:
        this_board_dir = Path(_s.BOARD_DIR, filename)

        try:
            pdboard = pd.read_pickle(Path(this_board_dir, _s.BOARD_FILENAME))
            #letters = np.asarray(pd.read_pickle(Path(this_board_dir, _s.LETTERS_FILENAME)), dtype=np.bytes_)
            letters = pd.read_pickle(Path(this_board_dir, _s.LETTERS_FILENAME))
            #lo.n(letters.dtype)
        except FileNotFoundError as exc:
            lo.c('Could not find file: {}'.format(exc.filename))
            sys.exit(1)

    else:
        filename = '_manual'
        pdboard = pd.DataFrame(_s.BOARD)
        #letters = np.asarray(_s.LETTERS, dtype=np.bytes_)
        letters = _s.LETTERS

    cdef str el
    if exclude_letters:
        for el in exclude_letters:
            #letters = letters[letters != el]
            letters.remove(el)

    #cdef bytes[:] bletters = [l.encode('utf8') for l in letters]
    cdef list bletters = [ord(l) for l in letters]
    #cdef cnp.ndarray bletters = np.array([ord(l) for l in letters], dtype=np.uint32)
    #bletters.sort()
    Settings.letters = bletters
    Settings.letters_s = len(bletters)

    board = pdboard.to_numpy()

    board_size = board.size
    if board_size == 15 * 15:
        board_name = _s.DEF_BOARD_BIG
    elif board_size == 11 * 11:
        board_name = _s.DEF_BOARD_SMALL
    else:
        lo.c('Board size ({}) has no match'.format(board_size))
        sys.exit(1)

    try:
        default_board = pd.read_pickle(board_name).to_numpy()
    except FileNotFoundError as exc:
        lo.c('Could not find file: {}'.format(exc.filename))
        sys.exit(1)

    try:
        points = json.load(open(str(Path(_s.POINTS_DIR, dictionary + '.json'))))  ## type: Dict[str, List[int]]
    except FileNotFoundError as exc:
        lo.c('Could not find file: {}'.format(exc.filename))
        sys.exit(1)

    #Settings.default_board = default_board
    #Settings.board = board
    #Settings.shape = shapez

    #cdef dict cpoints = {}
    cdef uchrn[:] cl_points = np.zeros(127, dtype=D8)
    cdef uchrn[:] cl_amts = np.zeros(127, dtype=D8)
    #cdef lpts_t lpt
    cdef str k
    cdef list v
    for k, v in points.items():
        cl_amts[ord(k)] = v[0]
        cl_points[ord(k)] = v[1]
    Settings.points = cl_points
    Settings.amts = cl_amts

    if lo.is_enabled('s'):
        lo.s('Game Board:\n{}'.format(pdboard))
        lo.s('Letters:\n{}'.format(letters))
        print()
    else:
        print('Running...')

    # cdef Board full = Board(board, default_board)
    # Settings.node_board = full

    Settings.num_results = num_results

    # todo add search words and nodes
    cdef str md5_letters = md5(''.join(sorted(letters)).encode()).hexdigest()[:9]
    cdef object solution_filename = Path(_s.SOLUTIONS_DIR, '{}_{}.npz'.format(filename, md5_letters))

    if overwrite is True:
        has_cache = False
    else:
        has_cache = solution_filename.exists()

    cdef Board full
    cdef object solution_data
    cdef Py_ssize_t r, c
    cdef object[:, :] solved_board, s_nodes
    cdef list s_words

    if has_cache is False:
        full = Board(board, default_board)
        Settings.node_board = full

        lo.s('Solving...\n')
        solve(dictionary)
        #solve(letters, dictionary)

        solved_board = np.zeros((Settings.node_board.nodes_rl, Settings.node_board.nodes_cl), dtype=np.object_)
        for r in range(solved_board.shape[0]):
            for c in range(solved_board.shape[1]):
                solved_board[r, c] = Settings.node_board.nodes[r, c].display

        show_solution(nodes=solved_board, words=Settings.node_board.words, no_words=no_words)

        np.savez(solution_filename, nodes=solved_board, words=Settings.node_board.words)
    else:
        lo.s('Found existing solution')
        solution_data = np.load(solution_filename)

        lo.e(type(solution_data))

        s_nodes = solution_data['nodes']
        s_words = solution_data['words'].tolist()

        show_solution(nodes=s_nodes, words=s_words, no_words=no_words)


#cpdef void main(
cpdef void main(
    filename: str = None, dictionary: str = DICTIONARY, no_words: bool = False, exclude_letters: list = None, overwrite: bool = False, num_results: int = NUM_RESULTS,
    log_level: str = DEFAULT_LOGLEVEL, profile: bool = False
):
    cmain(filename, dictionary, no_words, exclude_letters, overwrite, num_results, log_level)
