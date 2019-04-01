"""Parse and solve a scrabble board."""

cimport cython

cdef object json, sys, md5, Path, np, pd, log_init, _s

import json
import sys
from hashlib import md5
from pathlib import Path

#import signal
#from functools import lru_cache

import numpy as np
cimport numpy as cnp

import pandas as pd  # remove

from logs import log_init
import settings as _s


cdef str DEFAULT_LOGLEVEL = 'SUCCESS'
cdef str DICTIONARY = 'wwf'
cdef int NUM_RESULTS = 15


cdef object lo = log_init(DEFAULT_LOGLEVEL, skip_main=True)

#cdef object exc

ctypedef unsigned short us
#ctypedef (us, us) dual
#ctypedef us dual[2]
ctypedef unsigned char* uchr
ctypedef unsigned char uchrn

ctypedef cnp.uint32_t DTYPE_t

# no numpy array .flags to see contiguous
#DTYPE = np.uint16
#ctypedef cnp.uint16_t DTYPE_t
#DTYPE = np.intc
# ctypedef cnp.int32_t DTYPE_t
#ctypedef np.ndarray[str, ndim=2] board_t  # str 32?
#ctypedef np.ndarray[object, ndim=2] nb_t
#ctypedef np.ndarray[object, ndim=1] nodelist_t

#cdef bytes NUL = b'\0'
DEF NUL = b'\0'
DEF bl = ord('?')
#more of these

cdef packed struct multiplier_t:
    uchrn amt
    uchr typ

cdef packed struct lpts_t:
    uchrn amt
    uchrn pts

@cython.freelist(10000)  # todo check
cdef class Letter:
    cdef:
        bint is_blank
        bint from_rack
        uchrn pts
        multiplier_t mult
        uchrn x
        uchrn y
        #const unsigned char* value
        #DTYPE_t* value  # todo
        DTYPE_t value
        # todo define getter

    def __cinit__(self) -> None:
        self.is_blank = False
        self.from_rack = False
        self.pts = 0

        self.mult.amt = 1
        self.mult.typ = NUL
        self.x = 0
        self.y = 0
        self.value = 0

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


cdef class WordDict:
    cdef:
        str word
        #uchr word
        str direc
        public us pts
        list letters  # of Letter
        #Letter letters[100]

    def __cinit__(self, DTYPE_t[:] word, str direc, us pts, list letters):
        self.word = ''.join([chr(l) for l in word])
        self.direc = direc
        self.pts = pts
        self.letters = letters

    cdef str sol(self):
        cdef Letter lf = self.letters[0]
        cdef Letter ll = self.letters[-1]
        return f'pts: {self.pts:3d} | dir: "{self.direc}" | pos: [{lf.get_pos_str()}] - [{ll.get_pos_str()}] | w: {self.word}'

    def __str__(self) -> str:
        # cdef Letter l
        # cdef list ll = []
        # cdef str li
        # for l in self.letters:
        #     li = str(l)
        #     ll.append(li)
        # cdef str ls = '\n  '.join(ll)
        return '<w: {} | pts: {} | dir: {} | letts: [\n  {}\n]>'.format(
            self.word, self.pts, self.direc, '\n  '.join(str(l) for l in self.letters)
        )

    def __repr__(self) -> str:
        return self.__str__()

    def __reduce__(self):
        return rebuild_worddict, (self.word, self.direc, self.pts, self.letters)

def rebuild_worddict(word, direc, pts, letters):
    return WordDict(word.encode('utf8'), direc, pts, letters)


cdef class CSettings:
    cdef:
        cnp.ndarray board, default_board
        #cnp.ndarray[str, ndim=2] board, default_board

        #list letters
        #cnp.ndarray[DTYPE_t, ndim=1] letters
        DTYPE_t[:] letters

        uchrn blanks
        #dict shape

        dict points
        set words
        #list search_words
        #cnp.ndarray[:] search_words
        #Py_ssize_t search_words_l

        Board node_board
        int num_results

    def __cinit__(self):
        self.board = None
        self.default_board = None #np.empty((0,0), dtype=np.str_)

        self.letters = None
        #self.letters = np.empty(0) # dtype=bytes?
        self.blanks = 0
        #self.shape = {'row': 0, 'col': 0}  # type: Dict[str, int]
        self.points = {}  ## type: Dict[str, List[int]]
        self.words = set()  ## type: Set[bytes]
        #self.search_words = None
        #self.search_words_l = 0
        self.node_board = None
        self.num_results = NUM_RESULTS

cdef CSettings Settings = CSettings()


cdef class Node:
    cdef:
        uchrn x, y
        uchrn points
        multiplier_t multiplier
        bint is_start
        Node up, down, left, right
        #cnp.ndarray[:] up_lets, down_lets, left_lets, right_lets
        Letter[:] up_lets, down_lets, left_lets, right_lets
        str up_word, down_word, left_word, right_word
        bint has_edge
        DTYPE_t value
        public str display

    def __cinit__(self, uchrn x, uchrn y, str val, multiplier_t multiplier):  # todo test no types in cinit
        self.x = x
        self.y = y

        self.multiplier = multiplier

        if multiplier.typ == b'x':
            self.is_start = True
            self.multiplier.typ = NUL
        else:
            self.is_start = False

        cdef lpts_t lpt

        if not val:
            self.value = 0
            self.display = ' '  # todo display method
            self.points = 0
        else:
            self.value = ord(val)
            self.display = val.upper()

            try:
                lpt = Settings.points[self.value]
                self.points = lpt.pts
            except (KeyError, IndexError):
                lo.e('Could not get point value of "{}"'.format(val))
                sys.exit(1)

        self.up = None
        self.down = None
        self.left = None
        self.right = None

        self.up_lets = None
        self.down_lets = None
        self.left_lets = None
        self.right_lets = None

        self.up_word = None
        self.down_word = None
        self.left_word = None
        self.right_word = None

        self.has_edge = False

    cpdef str str_pos(self):
        return f'[{self.x:2d},{self.y:2d}]'

    def __str__(self) -> str:
        cdef str s = '_'
        #todo != 0
        if self.value:
            s = chr(self.value)
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
    def __cinit__(self, cnp.ndarray board, cnp.ndarray default_board):
        cdef:
            Py_ssize_t r, c
            uchrn mamt
            str dbval, bval, x
            multiplier_t mult
            Node node
            #cnp.ndarray board
            #uchr mtyp
            bytes mtyp

        self.board = board
        self.default_board = default_board
        self.nodes = np.empty_like(default_board, dtype=Node)

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
                mult.amt = 1
                #mult.typ = NULL
                mult.typ = NUL
                if dbval:
                    dbval = dbval.strip()
                    if dbval == 'x':
                        if bval is None:
                            self.new_game = True
                            mult.typ = b'x'
                    else:
                        x = dbval[0]
                        mamt = int(x)
                        mult.amt = mamt
                        mtyp = dbval[1].encode('utf8')
                        mult.typ = mtyp

                node = Node(r, c, bval, mult)
                self.nodes[r, c] = node

        for r in range(self.nodes_rl):
            for c in range(self.nodes_cl):
                self._set_node(r, c)



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

    cdef str _lets_as_str(self, Letter[:] letters):
        #cdef DTYPE_t v
        return ''.join([l.get_val() for l in letters]) # todo make func

    cdef void _set_adj_nodes(self, str d, Node n) except *:
        cdef:
            #cnp.ndarray[:] loop_nodes, setret
            cnp.ndarray loop_nodes, setret
            list ret = []
            bint rev = False
            Letter let
            int xx = n.x
            int yy = n.y
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

        nl = loop_nodes.shape[0]  # is this wrong?
        for ni in range(nl):
            p = loop_nodes[ni]
            if not p.value:
                break
            else:
                let = Letter()
                let.pts = p.points
                let.value=p.value
                let.x=p.x
                let.y=p.y
                let.mult=p.multiplier

                ret.append(let)

        cdef str lets_str
        if ret:
            if rev is True:
                ret.reverse()

            setret = np.asarray(ret, dtype=object)
            lets_str = self._lets_as_str(setret)

            if d == 'up':
                n.up_lets = setret
                n.up_word = lets_str
            elif d == 'down':
                n.down_lets = setret
                n.down_word = lets_str
            elif d == 'left':
                n.left_lets = setret
                n.left_word = lets_str
            elif d == 'right':
                n.right_lets = setret
                n.right_word = lets_str

    cdef void _set_node(self, int r, int c):
        cdef:
            Node node
            str d

        node = self.nodes[r, c]

        if r > 0:
            node.up = self.nodes[r-1, c]
            if node.up.value:
                node.has_edge = True

        if r < self.nodes_rl - 1:
            node.down = self.nodes[r+1, c]
            if node.down.value:
                node.has_edge = True

        if c > 0:
            node.left = self.nodes[r, c-1]
            if node.left.value:
                node.has_edge = True

        if c < self.nodes_cl - 1:
            node.right = self.nodes[r, c+1]
            if node.right.value:
                node.has_edge = True

        for d in ('up', 'down', 'left', 'right'):
            self._set_adj_nodes(d, node)


    # def get_points(list word not None) -> int:
    cdef us get_points(self, list word):
        cdef:
            #Py_ssize_t len_word = word.shape[0]
            Py_ssize_t len_word = len(word)
            us pts = 0
            uchrn word_mult = 1
            Py_ssize_t l
            uchrn p
            Letter letter
            lpts_t lpt
            #dict spts = Settings.points

        for l in range(len_word):
            letter = (<Letter> word[l])

            if letter.from_rack is False:
                pts += letter.pts
            elif letter.is_blank is False:
                # lpt = <lpts_t> (spts[letter.value])
                # p = lpt.pts  # todo exception handling and above

                p = letter.pts

                if letter.mult.typ == <uchr>b'l':
                    p *= letter.mult.amt
                elif letter.mult.typ == <uchr>b'w':
                    word_mult *= letter.mult.amt

                pts += p

        #lo.x(pts)

        pts *= word_mult

        return pts

    def __str__(self) -> str:
        return '<Board: size {:d}x{:d} | words: {}>'.format(self.nodes_rl, self.nodes_cl, len(self.words))

    # def __reduce__(self):
    #     return rebuild, (Settings.board, Settings.default_board)


#cdef npss = np.searchsorted


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef void check_and_get(Node[:] node_list, str direc, DTYPE_t[:] word, Py_ssize_t word_len) except *:
    cdef Node chk_first, chk_last
    cdef Node first = node_list[0]
    cdef Node last = node_list[word_len - 1]

    if direc == 'r':
        chk_first = first.left
        chk_last = last.right
    else:
        chk_first = first.up
        chk_last = last.down

    if chk_first is not None and chk_first.value: return
    if chk_last is not None and chk_last.value: return

    #todo what is the difference in not setting here? infer types?
    cdef:
        uchrn blanks = Settings.blanks

        DTYPE_t[:] ls = Settings.letters.copy()
        DTYPE_t[:] befls, aftls, newls

        set aw = Settings.words
        Node no
        us pts
        us extra_pts = 0 #new_pts
        Py_ssize_t i, bi, nvi
        DTYPE_t nvv
        Py_ssize_t lsl
        #size_t lsl
        Letter le
        Letter b
        bint is_blank
        bint has_let

        str new_word #, new_direc

        #letter_list new_lets
        #cnp.ndarray[:] bef_lets, aft_lets #, np_lets #, new_lets
        Letter[:] bef_lets, aft_lets
        str bef_word, aft_word

        list word_lets = []
        list new_let_list

        WordDict w #, nw

        #bytes nv
        DTYPE_t nv, nov
        uchrn nvs

        lpts_t lpt
        dict spts = Settings.points

    #todo for nv in word[:word_len]:
    for i in range(word_len):
        no = node_list[i]
        nv = word[i]

        le = Letter()
        le.mult = no.multiplier
        le.x = no.x
        le.y = no.y
        le.value = nv
        #le.value = word[i]

        nov = no.value
        if nov:
            #if nov != (<const unsigned char*>nv):
            if nov != nv:
                #lo.d(f'mismatch {no.value} v {le.value}')
                return
            else:
                le.pts = no.points

        else:
            is_blank = False

            has_let = False
            for nvi in range(ls.shape[0]):
                nvv = ls[nvi]
                if nvv == nv:
                    ls[nvi] = 0
                    has_let = True
                    break

            if has_let is False:
                #lo.d(f'{le.value} not in {ls}')
                if blanks > 0:
                    #ls.remove(bl)
                    blanks -= 1
                    is_blank = True
                    #lo.d('removed blank: %s', blanks)
                else:
                    # todo skip forward if the next letter doesnt exist?
                    return

            # if nv not in ls:
            #     #lo.d(f'{le.value} not in {ls}')
            #     if blanks > 0:
            #         #ls.remove(bl)
            #         blanks -= 1
            #         is_blank = True
            #         #lo.d('removed blank: %s', blanks)
            #     else:
            #         # todo skip forward if the next letter doesnt exist?
            #         return
            # else:
            #     # nvs = np.searchsorted(ls, nv)
            #     # nvs = npss(ls, nv)
            #     # befls = ls[:nvs]
            #     # aftls = ls[nvs+1:]
            #     # ls = np.concatenate([
            #     #     zl,
            #     #     befls,
            #     #     aftls
            #     # ])  # out=ls?
            #     for nvi in range(ls.shape[0]):
            #         nvv = ls[nvi]
            #         if nvv == nv:
            #             ls[nvi] = 0
            #             break

            le.from_rack = True

            if is_blank is False:
                lpt = spts[nv]  # todo why
                #le.pts = spts[nv]['pts']
                le.pts = lpt.pts
                #le.pts = (<uchrn>gg(nv))
                #le.pts = gg(nv)
            else:
                le.is_blank = True

            if direc == 'r':
                bef_lets = no.up_lets
                bef_word = no.up_word
                aft_lets = no.down_lets
                aft_word = no.down_word
            else:
                bef_lets = no.left_lets
                bef_word = no.left_word
                aft_lets = no.right_lets
                aft_word = no.right_word

            if bef_lets is not None or aft_lets is not None:
                #lo.d('bef: %s', bef_lets)
                #lo.d('aft: %s', aft_lets)

                new_let_list = []
                new_word = ''

                if bef_lets is not None:
                    new_word = bef_word
                    for bi in range(bef_lets.shape[0]):
                        b = bef_lets[bi]
                        new_let_list.append(b)

                new_let_list.append(le)
                new_word += chr(le.value)

                if aft_lets is not None:
                    new_word += aft_word
                    for bi in range(aft_lets.shape[0]):
                        b = aft_lets[bi]
                        new_let_list.append(b)

                #new_word = ''.join(chr(ns.value) for ns in new_let_list)

                if new_word not in aw:
                    return

                #lo.x(new_word)
                extra_pts += (<us>Settings.node_board.get_points(new_let_list))

                #new_lets = np.array(new_let_list, dtype=object)
                # for ii in range(len(new_lets)):
                #     nw.letters[ii] = new_lets[ii]

        word_lets.append(le)

    lsl = np.count_nonzero(ls)
    if lsl == 0:
        return

    #lo.i('GOOD: %s at %s : %s', word, first.str_pos(), last.str_pos())

    pts = Settings.node_board.get_points(word_lets) + extra_pts

    if lsl == Settings.letters.shape[0]:
        pts += 35

    w = WordDict(
        word,
        direc,
        pts,
        word_lets
    )

    # if w not in Settings.node_board.words:
    #     Settings.node_board.words.append(w)
    # else:
    #     lo.e('already saw %s', w)

    Settings.node_board.words.append(w)


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef void check_nodes(
    Node[:] nodes,
    #object[:] sw,
    DTYPE_t[:, :] sw,
    #uchr[:] sw,
    #cnp.ndarray sw,
    #bytes[:] sw,
    uchrn[:] swlens,
    str direc
) except *:
    cdef:
        #str w #, chk_dir
        #(uchr, Py_ssize_t) w
        Node x, xt

        #cnp.ndarray[:] sw = Settings.search_words
        #Py_ssize_t swl = Settings.search_words_l
        Py_ssize_t swl = sw.shape[0]

        Py_ssize_t word_len, i, j, s, t
        Py_ssize_t nlen = len(nodes)
        #bytes wc
        Node[:] subnodes
        bint has_edge = False
        #str ww
        #cnp.bytes_ ww
        DTYPE_t[:] ww

    for t in range(nlen):
        xt = nodes[t]
        if xt.has_edge is True:
            has_edge = True
            break
    if has_edge is False:
        return

    # if direc == 'r':
    #     chk_dir = 'c'
    # else:
    #     chk_dir = 'r'

    #for s in range(Settings.search_words_l): # todo swap test
    for s in range(swl):
        #w = sw[s]
        word_len = swlens[s]

        if word_len > nlen:
            continue

        ww = sw[s]
        #lo.d(ww)

        for i in range(nlen - word_len + 1):
            has_edge = False
            for j in range(word_len):
                x = nodes[i + j]
                if x.has_edge is True:
                    has_edge = True
                    break
            if has_edge is True:
                subnodes = nodes[i:word_len + i]
                check_and_get(subnodes, direc, ww, word_len)

    return


# todo: set default for dict? put in settings? move all settings out to options?
cdef void solve(
    #cnp.ndarray letters, # str
    str dictionary
) except *:
    cdef list wordlist = []
    #todo make fnc exception
    try:
        wordlist = open(str(Path(_s.WORDS_DIR, dictionary + '.txt'))).read().splitlines()
    except FileNotFoundError as exc:
        lo.c('Could not find file: {}'.format(exc.filename))
        sys.exit(1)

    cdef uchrn blanks = list(Settings.letters).count(bl)
    #cdef uchrn blanks = Settings.letters.tolist().count(bl)
    cdef Board full = Settings.node_board
    #cdef str w
    cdef Py_ssize_t mnlen = max(full.nodes_cl, full.nodes_rl)
    cdef set words = {w for w in wordlist if len(w) <= mnlen}
    cdef set search_words = set()

    if _s.SEARCH_WORDS is None:
        if blanks > 0:
            search_words = words
        else:
            # todo: faster if str? why gen slower?
            lo.e(len(words))
            search_words = {w for w in words if any([chr(l) in w for l in Settings.letters])}
            lo.e(len(search_words))
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

    cdef str s
    cdef set bwords = {s.encode('utf8') for s in words}

    Settings.words = bwords

    cdef str sws
    cdef Py_ssize_t swi, swl = len(search_words)

    # todo as one?
    #cdef uchr ss
    #cdef list sw = [(s.encode('utf8'), len(s)) for s in search_words]
    #cdef cnp.ndarray sw = np.zeros(swl, dtype=f'|S{mnlen}')
    sw_ = np.array([s for s in search_words])
    cdef cnp.ndarray sw = sw_.view('uint32').reshape(sw_.shape[0], -1)

    #cdef cnp.ndarray[cnp.uint8_t, ndim=1] swlens = np.empty(swl, dtype=np.uint8)
    cdef cnp.ndarray swlens = np.zeros((swl,), dtype=np.uint8)

    for swi, sws in enumerate(search_words):
        #sw[swi] = sws.encode('utf8')
        swlens[swi] = len(sws)

    #Settings.search_words = sw
    #Settings.search_words_l = swl

    cdef cnp.ndarray[object, ndim=2] nodes = full.nodes
    #cdef cnp.ndarray[:, :] nodes = full.nodes
    cdef int i, ic, tot
    #cdef list search_rows, search_cols

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

        for i in search_rows:
            if lo.is_enabled('s'):
                ic += 1
                lo.s('Checking row %2i  (%2i / %i)', i, ic, tot)
            check_nodes(nodes[i], sw, swlens[:], 'r')

        for i in search_cols:
            if lo.is_enabled('s'):
                ic += 1
                lo.s('Checking col %2i  (%2i / %i)', i, ic, tot)
            check_nodes(nodes[:,i], sw, swlens[:], 'c')


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
        cnp.ndarray nodes, list words, bint no_words
) except *:
    # todo mark blanks

    cdef:
        WordDict w, best
        list newlist, cutlist, js
        Py_ssize_t rown, i
        int x, y
        str horiz
        Letter letter
        cnp.ndarray nodes_copy

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
        nodes_copy = nodes.copy()

        for letter in best.letters:
            x, y = letter.get_pos()
            if not nodes_copy[x, y] or nodes_copy[x, y] == ' ':
                nodes_copy[x, y] = f'\x1b[33m{letter.get_val()}\x1b[0m'

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


cdef void cmain(str filename, str dictionary, bint no_words, list exclude_letters, bint overwrite, int num_results, str log_level) except *:
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
    #cdef DTYPE_t[:] bletters = [ord(l) for l in letters]
    cdef cnp.ndarray bletters = np.array([ord(l) for l in letters], dtype=np.uint32)
    bletters.sort()
    Settings.letters = bletters

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

    Settings.default_board = default_board
    Settings.board = board
    #Settings.shape = shapez

    cdef dict cpoints = {}  # type: dict
    cdef lpts_t lpt
    cdef str k
    cdef list v
    for k, v in points.items():
        lpt.amt = v[0]
        lpt.pts = v[1]
        cpoints[ord(k)] = lpt
    Settings.points = cpoints

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

    cdef Board full, solution

    if has_cache is False:
        full = Board(board, default_board)
        Settings.node_board = full

        lo.s('Solving...\n')
        solve(dictionary)
        #solve(letters, dictionary)

        solved_board = np.empty((Settings.node_board.nodes_rl, Settings.node_board.nodes_cl), dtype='<U32')
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


cpdef void main(
    filename: str = None, dictionary: str = DICTIONARY, no_words: bool = False, exclude_letters: list = None, overwrite: bool = False, num_results: int = NUM_RESULTS,
    log_level: str = DEFAULT_LOGLEVEL, profile: bool = False
):
    cmain(filename, dictionary, no_words, exclude_letters, overwrite, num_results, log_level)
