"""Parse and solve a scrabble board."""

cimport cython

import gzip
import json
import sys
#from hashlib import md5

import pickle
#import dill
#dill.detect.trace(True)

#import signal

#from functools import lru_cache

from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
cimport numpy as cnp

import pandas as pd

from logs import log_init

import settings as _


# cdef class Node
# cdef class Board

ctypedef unsigned short us
#ctypedef np.ndarray[str, ndim=2] board_t  # str 32?
#ctypedef np.ndarray[object, ndim=2] nb_t
#ctypedef np.ndarray[object, ndim=1] nodelist_t


cdef str DICTIONARY = 'wwf'
cdef str DEFAULT_LOGLEVEL = 'SUCCESS'

lo = log_init(DEFAULT_LOGLEVEL, skip_main=True)


cdef bytes NUL = b'\0'

ctypedef (int, int) dual

ctypedef struct multiplier_t:
    int amt
    char* typ

# cdef class Letter:
#     cdef:
#         bint is_blank
#         int pts
#         multiplier_t mult
#         dual pos
#         char* value
#
#     def __cinit__(self, bint is_blank, int pts, multiplier_t mult, dual pos, char* value):
#         self.is_blank = is_blank
#         self.pts = pts
#         self.mult = mult
#         self.pos = pos
#         self.value = value
#
#     def __str__(self):
#         #f'mm: {self.mult}, '\
#         return '<'\
#             f'v: {self.value.decode("utf8")}, '\
#             f'pos: {self.pos[0]}:{self.pos[1]}, '\
#             f'bl: {"T" if self.is_blank is True else "f"}, '\
#             f'pts: {"_" if self.pts == 0 and self.is_blank is False else self.pts}, '\
#             f'mult: "{"_" if self.mult.amt == 1 else str(self.mult.amt) + self.mult.typ.decode("utf8").upper()}"'\
#             '>'
#
#     def __repr__(self):
#         return self.__str__()

ctypedef struct Letter:
    bint is_blank
    int pts
    multiplier_t mult
    dual pos
    char* value


# just need to set max size on these lists of structs?

# cdef class WordDict:
#     cdef:
#         str word
#         str direc
#         int pts
#         cnp.ndarray letters
#
#     def __cinit__(self, str word, str direc, int pts, cnp.ndarray letters):
#         self.word = word
#         self.direc = direc
#         self.pts = pts
#         self.letters = letters
#
#     def __str__(self):
#         #return f'<w: {self.word} | pts: {self.pts} | dir: {self.direc} | letts: \n{"\n  ".join(str(l) for l in self.letters)}>'
#         return '<w: {} | pts: {} | dir: {} | letts: [\n  {}\n]>'.format(self.word, self.pts, self.direc, '\n  '.join(str(l) for l in self.letters))
#
#     def __repr__(self):
#         return self.__str__()

# not packed?
ctypedef struct WordDict:
    int pts

    Letter* letters
    #cnp.ndarray letters
    #letter_t letters[]

    char* word
    char* direc


cdef class CSettings:
    cdef:
        cnp.ndarray board, default_board
        #np.ndarray[str, ndim=2] board, default_board
        cnp.ndarray letters
        #np.ndarray[str, ndim=1] letters
        short blanks
        dict shape, points
        set words, search_words
        Board node_board

    def __cinit__(self):
        self.board = np.empty((0,0), dtype=np.str_)
        self.default_board = np.empty((0,0), dtype=np.str_)
        self.letters = np.empty(0) # dtype=bytes?
        self.blanks = 0
        self.shape = {'row': 0, 'col': 0}  # type: Dict[str, int]
        self.points = {}  # type: Dict[str, List[int]]
        self.words = set()  # type: Set[str]
        self.search_words = set()  # type: Set[str]
        self.node_board = None


cdef CSettings Settings = CSettings()


cdef class Node:
    cdef:
        int x, y, points
        dual pos
        multiplier_t multiplier
        bint is_start
        Node up, down, left, right
        cnp.ndarray up_word, down_word, left_word, right_word
        bint has_edge
        char* value

    def __cinit__(self, int x, int y, str val, multiplier_t multiplier):
        self.x = x
        self.y = y
        self.pos = (x, y)

        self.multiplier = multiplier

        if multiplier.typ == b'x':
            self.is_start = True
            #self.multiplier.typ = NULL
            self.multiplier.typ = NUL
        else:
            self.is_start = False

        if not val:
            #self.value = NULL
            self.value = NUL
        else:
            _str = val.encode('utf8')
            self.value = _str

        self.points = 0
        if val is not None:
            try:
                self.points = Settings.points[val][1]
            except (KeyError, IndexError):
                lo.e('Could not get point value of "{}"'.format(val))
                sys.exit(1)

        self.up = None
        self.down = None
        self.left = None
        self.right = None

        self.up_word = None
        self.down_word = None
        self.left_word = None
        self.right_word = None

        self.has_edge = False

    def str_pos(self):
        return '[{:2d},{:2d}]'.format(*self.pos)

    def __str__(self):
        s = '_'
        if self.value[0]:
            s = self.value.decode('utf8')
        return '<Node: {} v: {:1s}>'.format(self.str_pos(), s)

    def __repr__(self):
        return self.__str__()


cdef class Board:
    cdef:
        cnp.ndarray default_board
        #cnp.ndarray[:,:] default_board
        #np.ndarray[str, ndim=2] default_board
        bint new_game
        #nodelist_t nodes
        cnp.ndarray nodes
        #np.ndarray[Node, ndim=2] nodes
        int nodes_rl, nodes_cl

        list words  # todo of word_dicts
        #np.ndarray[dtype=word_dict] words
        #np.ndarray words


    #def __cinit__(self, board=Settings.board, default_board=Settings.default_board):
    def __cinit__(self, cnp.ndarray _board, cnp.ndarray default_board):
        cdef:
            int r, c, mamt
            str dbval, bval, x
            multiplier_t mult
            Node node
            cnp.ndarray board
            bytes mtyp

        self.default_board = default_board[:,:]
        self.nodes = np.empty_like(default_board, dtype=Node)

        board = _board[:]
        self.nodes_rl = board.shape[0]
        self.nodes_cl = board.shape[1]

        #self.words = np.ndarray([], dtype=word_dict)
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

                node = Node(x=r, y=c, val=bval, multiplier=mult)
                self.nodes[r, c] = node

        for r in range(self.nodes_rl):
            for c in range(self.nodes_cl):
                self._set_node(r, c)


    #@lru_cache(1023)  # todo reenable?
    cdef Node get(self, int x, int y):
        return self.nodes[x, y]

    cdef Node get_by_attr(self, str attr, v):
        return filter(lambda obj: getattr(obj, attr) == v, self.nodes)

    # #@lru_cache()
    # cdef np.ndarray[Node, ndim=1] get_row(self, int x):
    #     return self.nodes[x]
    #
    # #@lru_cache()
    # cdef np.ndarray[Node, ndim=1] get_col(self, int y):
    #     return self.nodes[:,y]

    cdef void _set_adj_nodes(self, str d, Node n):
        cdef:
            cnp.ndarray loop_nodes, setret
            Node p
            list ret = []
            bint rev = False
            Letter let

        if d == 'up':
            loop_nodes = self.nodes[:n.x, n.y][::-1]
            rev = True
        elif d == 'down':
            loop_nodes = self.nodes[n.x+1:, n.y]
        elif d == 'left':
            loop_nodes = self.nodes[n.x, :n.y][::-1]
            rev = True
        else:
            loop_nodes = self.nodes[n.x, n.y+1:]

        for p in loop_nodes:
            if not p.value[0]:
                break
            else:
                # let = Letter(
                #     value=p.value,
                #     pos=p.pos,
                #     is_blank=False,
                #     mult=p.multiplier,
                #     pts=p.points
                # )
                let.value=p.value
                let.pos=p.pos
                let.is_blank=False
                let.mult=p.multiplier
                let.pts=p.points

                ret.append(let)
        if ret:
            if rev is True:
                ret.reverse()
            setret = np.asarray(ret, dtype=object)

            if d == 'up': n.up_word = setret
            elif d == 'down': n.down_word = setret
            elif d == 'left': n.left_word = setret
            elif d == 'right': n.right_word = setret

    cdef void _set_node(self, int r, int c):
        cdef:
            Node node
            str d

        node = self.nodes[r, c]

        if r > 0:
            node.up = self.nodes[r-1, c]
            if node.up.value[0]:
                node.has_edge = True
        elif r < self.nodes_rl - 1:
            node.down = self.nodes[r+1, c]
            if node.down.value[0]:
                node.has_edge = True
        elif c > 0:
            node.left = self.nodes[r, c-1]
            if node.left.value[0]:
                node.has_edge = True
        elif c < self.nodes_cl - 1:
            node.right = self.nodes[r, c+1]
            if node.right.value[0]:
                node.has_edge = True

        for d in ('up', 'down', 'left', 'right'):
            self._set_adj_nodes(d, node)


    cdef get_points(self, cnp.ndarray word):
        cdef:
            int pts = 0
            int word_mult = 1
            Py_ssize_t len_word = word.shape[0]
            int l, p
            Letter letter

        for l in range(len_word):
            letter = word[l]
            if letter.pts:
                pts += letter.pts
            else:
                if letter.is_blank is True:
                    p = 0
                else:
                    p = Settings.points[letter.value.decode('utf8')][1]
                    if not p:
                        lo.e('Could not get point value of "{}". Using {}'.format(letter.value, p))
                    if letter.mult.typ == b'l':
                        p *= letter.mult.amt
                    elif letter.mult.typ == b'w':
                        word_mult *= letter.mult.amt

        pts *= word_mult

        if len_word == 7:  # todo: or is it all letters?
            pts += 35

        return pts


#@cython.profile(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void check_and_get(Node[:] node_list, str direc, str word, Py_ssize_t word_len) except *:
    cdef:
        Node no, first, last, chk_first, chk_last

        #cnp.ndarray ls
        list ls

        short blanks
        int i, pts, new_pts
        Letter le, b
        bint is_blank
        str new_word #, new_direc

        #letter_list new_lets
        cnp.ndarray bef_lets, aft_lets, new_lets, np_lets

        list word_lets = []
        WordDict w, nw
        bytes nv, nww, new_direc

    first = node_list[0]
    last = node_list[word_len - 1]

    if direc == 'r':
        chk_first = first.left
        chk_last = last.right
    else:
        chk_first = first.up
        chk_last = last.down

    if chk_first is not None and chk_first.value[0]: return
    if chk_last is not None and chk_last.value[0]: return

    ls = Settings.letters.tolist()  # todo fix
    blanks = Settings.blanks

    for i in range(word_len):
        no = node_list[i]
        nv = word[i].encode('utf8')
        # le = Letter(
        #     value = nv,
        #     pos = no.pos,
        #     is_blank = False,
        #     pts = 0,
        #     mult = no.multiplier
        # )

        le.value = nv
        le.pos = no.pos
        le.is_blank = False
        le.pts = 0
        le.mult = no.multiplier

        if no.value[0]:
            if no.value != le.value:
                #lo.d(f'mismatch {no.value} v {le.value}')
                return
            else:
                le.pts = 0

        else:
            is_blank = False

            if le.value not in ls:
                #lo.d(f'{le.value} not in {ls}')
                if blanks > 0:
                    ls.remove(b'?')
                    blanks -= 1
                    is_blank = True
                    #lo.d('removed blank: %s', blanks)
                else:
                    # todo skip forward if the next letter doesnt exist?
                    #lo.d('bad')
                    return
            else:
                #lo.d(f'good {le}')
                ls.remove(le.value)

            le.is_blank = is_blank
            if is_blank is False:
                le.pts = Settings.points[le.value.decode('utf8')][1]

            if direc == 'r':
                bef_lets = no.up_word
                aft_lets = no.down_word
            else:
                bef_lets = no.left_word
                aft_lets = no.right_word

            if bef_lets is not None or aft_lets is not None:
                #lo.n('bef: %s', bef_lets)
                #lo.n('aft: %s', aft_lets)

                new_word = ''
                if bef_lets is not None:
                    for b in bef_lets:
                        new_word += b.value.decode('utf8')
                else:
                    bef_lets = np.empty(0)

                new_word += le.value.decode('utf8')

                if aft_lets is not None:
                    for b in aft_lets:
                        new_word += b.value.decode('utf8')
                else:
                    aft_lets = np.empty(0)

                if new_word not in Settings.words:
                    #lo.i('no %s', new_word)
                    return

                new_lets = np.asarray(bef_lets.tolist() + [le] + aft_lets.tolist())
                new_pts = Settings.node_board.get_points(new_lets)

                if direc == 'r':
                    new_direc = b'c'
                else:
                    new_direc = b'r'
                #nw = WordDict(word=new_word, direc=new_direc, pts=new_pts, letters=new_lets)
                nww = new_word.encode('utf8')
                nw.word = nww
                nw.direc = new_direc
                nw.pts = new_pts
                nw.letters = new_lets

                if nw not in Settings.node_board.words:
                    Settings.node_board.words.append(nw)
                else: lo.e('no %s', nw)

        word_lets.append(le)

    lo.i('GOOD: %s at %s : %s', word, first.str_pos(), last.str_pos())

    np_lets = np.asarray(word_lets)  # dtype?
    pts = Settings.node_board.get_points(np_lets)
    #w = WordDict(word=word, direc=direc, pts=pts, letters=np_lets)
    w.word=word
    w.direc=direc
    w.pts=pts
    w.letters=np_lets

    if w not in Settings.node_board.words:
        Settings.node_board.words.append(w)
    else: lo.e('no %s', w)


#@cython.profile(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void check_nodes(Node[:] nodes, str direc):
    cdef:
        str w #, chk_dir
        Node x, j
        int i
        Py_ssize_t word_len, nlen  # todo add word length to dict
        bytes wc
        #Node[:] subnode

    nlen = len(nodes)
    # if direc == 'r':
    #     chk_dir = 'c'
    # else:
    #     chk_dir = 'r'

    for w in Settings.search_words:
        word_len = len(w)
        if word_len > nlen:
            continue

        #lo.v(w)
        for i in range(nlen - word_len + 1):
            if any([x.has_edge is True for x in nodes[i:word_len + i]]):
                check_and_get(nodes[i:word_len + i], direc, w, word_len)

#
# @cython.boundscheck(False)
# #@cython.wraparound(False)
# cdef str _print_node_range(
#     vector[pair[int, int]] n  # list[tuple[int,int]]
#     #list n  # list[tuple[int,int]]
#     #np.ndarray[np.int, ndim=2] n  # list[tuple[int,int]]
# ):
#     cdef:
#         pair[int, int] s = n.front()
#         pair[int, int] e = n.back()
#         (int, int, int, int) x = (s.first, s.second, e.first, e.second)
#
#     return '[%2i,%2i] : [%2i,%2i]' % x
#
#
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef str _print_result(
#     dict o,
#     bint no_words = False
# ):
#     cdef:
#         str ow
#         str pts = str(o['pts'])
#         #Py_ssize_t owl
#
#     if not no_words:
#         ow = o['word']
#     else:
#         ow = '<hidden> (' + str(len(o['word'])) + ')'
#
#     cdef str s = '\n'.join([
#         'pts  : ' + pts,
#         'word : ' + ow,  # todo: hide nodes? show length?
#         'nodes: ' + _print_node_range(o["nodes"])
#     ])
#
#     # todo why are these tabs like 6 or 8?
#     s += '\nnew_words:'
#
#     cdef:
#         list nw = o['new_words']  # list[dict]
#         Py_ssize_t n, nw_l = len(nw)
#         dict nw_d
#         list nw_nodes
#         str nw_nodes_str, nw_word
#
#     if not no_words:
#         for n in range(nw_l):
#             nw_d = nw[n]
#             nw_nodes = nw_d["nodes"]
#             nw_word = nw_d["word"]
#             nw_nodes_str = _print_node_range(nw_nodes)
#             s += f'\n\t{nw_nodes_str}  {nw_word}'
#     else:
#         s += ' <hidden>'
#
#     return s



cdef void show_solution(bint no_words=False):
    # todo mark blanks

    if not Settings.node_board.words:
        print('\nNo solution.')
    else:
        # newlist = sorted(Settings.word_info, key=lambda k: k['pts'], reverse=True)
        # best_data = newlist[:1][0]
        # if lo.is_enabled('s'):
        #     #todo fix this
        #     top10 = []  # type: List[dict]
        #     seen = [(best_data['pts'], best_data['word'])]
        #     for n in newlist:
        #         if len(top10) == 10:
        #             break
        #         seen_tup = (n['pts'], n['word'])
        #         if seen_tup in seen:
        #             continue
        #         seen.append(seen_tup)
        #         top10.append(n)
        #
        #     top10.reverse()
        #
        #     print()
        #     lo.s('-- Top 10 --\n')
        #     for sidx, s in enumerate(top10):
        #         lo.s('Choice #{}\n{}\n'.format(sidx + 1, _print_result(s, no_words)))
        #
        #     lo.s('-- Best --\n{}'.format(_print_result(best_data, no_words)))
        #
        # if not no_words:  # todo: print xs instead?
        #     solved_board = Settings.board.copy()
        #
        #     for ni, node_tup in enumerate(best_data.get('nodes', [])):
        #         node = Settings.node_board.get(*node_tup)
        #         if not node.value:
        #             solved_board[node.y][node.x] = '\x1b[33m' + best_data['word'][ni] + '\x1b[0m'
        #
        #     print('\n' + '-' * ((Settings.shape['row'] * 2) - 1 + 4))
        #
        #     for row in solved_board.iterrows():
        #         row_data = row[1].to_list()
        #         row_str = []
        #         for rl in row_data:
        #             rl_len = len(rl)
        #             rl_val = rl
        #
        #             if rl_len == 0:
        #                 rl_val = ' '
        #             elif rl_len == 1:
        #                 rl_val = rl.upper()
        #
        #             row_str.append(rl_val)
        #
        #         print('| ' + ' '.join(row_str) + ' |')
        #
        #     print('-' * ((Settings.shape['row'] * 2) - 1 + 4))
        #
        # else:
        #     print('\n<solution hidden> ({})'.format(len(best_data["word"])))
        #
        # print('\nPoints: {}'.format(best_data["pts"]))

        for i in Settings.node_board.words:
            lo.s(i)

# todo: set default for dict? put in settings? move all settings out to options?
cdef void solve(
    cnp.ndarray letters, # str
    str dictionary
):
    #todo make fnc exception
    try:
        wordlist = open(str(Path(_.WORDS_DIR, dictionary + '.txt'))).read().splitlines()
    except FileNotFoundError as exc:
        lo.c('Could not find file: {}'.format(exc.filename))
        sys.exit(1)

    words = set(wordlist)
    blanks = list(letters).count(b'?')

    if _.SEARCH_WORDS is None:
        if blanks:
            search_words = words
        else:
            # todo: faster if str? why gen slower?
            search_words = {w for w in words if any(l in w for l in letters)}
    elif isinstance(_.SEARCH_WORDS, tuple):
        search_words = set(wordlist[_.SEARCH_WORDS[0]: _.SEARCH_WORDS[1]])
    elif isinstance(_.SEARCH_WORDS, set):
        search_words = _.SEARCH_WORDS
    else:
        lo.c('Incompatible search words type: {}'.format(type(_.SEARCH_WORDS)))
        sys.exit(1)

    # --

    # todo allow extra mods
    # print(board)
    # print(board[10])
    # board[6][10] = 'R'
    # board[8][10] = 'Y'
    # letters = ['F', 'G']

    Settings.blanks = blanks
    Settings.words = words
    Settings.search_words = search_words

    cdef Board full = Settings.node_board
    cdef cnp.ndarray[object, ndim=2] nodes = full.nodes
    cdef int i, ic, tot

    if full.new_game:
        lo.s(' = Fresh game = ')
        no = next(full.get_by_attr('is_start', True), None)
        # if no:
        #     check_node(no)

    else:
        if _.SEARCH_NODES is None:
            search_rows = range(full.nodes_rl)
            search_cols = range(full.nodes_cl)

        else:
            search_rows = _.SEARCH_NODES[0]
            search_cols = _.SEARCH_NODES[1]

        ic = 0
        tot = len(search_rows) + len(search_cols)

        for i in search_rows:
            if lo.is_enabled('s'):
                ic += 1
                lo.s('Checking row %i (#%i / %i)', i, ic, tot)
            check_nodes(nodes[i], 'r')

        for i in search_cols:
            if lo.is_enabled('s'):
                ic += 1
                lo.s('Checking col %i (#%i / %i)', i, ic, tot)
            check_nodes(nodes[:,i], 'c')


cdef void cmain(str filename, str dictionary, bint no_words, list exclude_letters, bint overwrite, str log_level):
    cdef:
        dict shapez, points
        cnp.ndarray board, default_board
        cnp.ndarray letters
        int board_size
        object board_name
        str md5_board, md5_letters
        Board full
        bint has_cache

    log_level = log_level.upper()
    if log_level != DEFAULT_LOGLEVEL:
        lo.set_level(log_level)

    if filename is not None:
        this_board_dir = Path(_.BOARD_DIR, filename)

        try:
            board = pd.read_pickle(Path(this_board_dir, _.BOARD_FILENAME)).to_numpy()
            letters = np.asarray(pd.read_pickle(Path(this_board_dir, _.LETTERS_FILENAME)), dtype=np.bytes_)
            #lo.n(letters.dtype)
        except FileNotFoundError as exc:
            lo.c('Could not find file: {}'.format(exc.filename))
            sys.exit(1)

    else:
        board = pd.DataFrame(_.BOARD).to_numpy()
        letters = np.asarray(_.LETTERS, dtype=np.bytes_)

    if exclude_letters:
        for el in exclude_letters:
            letters = letters[letters != el]

    # need?
    shapez = {
        'row': board.shape[0],
        'col': board.shape[1]
    }

    board_size = board.size
    if board_size == 15 * 15:
        board_name = _.DEF_BOARD_BIG
    elif board_size == 11 * 11:
        board_name = _.DEF_BOARD_SMALL
    else:
        lo.c('Board size ({}) has no match'.format(board_size))
        sys.exit(1)

    try:
        default_board = pd.read_pickle(board_name).to_numpy()
    except FileNotFoundError as exc:
        lo.c('Could not find file: {}'.format(exc.filename))
        sys.exit(1)

    try:
        points = json.load(open(str(Path(_.POINTS_DIR, dictionary + '.json'))))  # type: Dict[str, List[int]]
    except FileNotFoundError as exc:
        lo.c('Could not find file: {}'.format(exc.filename))
        sys.exit(1)

    Settings.default_board = default_board
    Settings.board = board
    Settings.letters = letters
    Settings.shape = shapez
    Settings.points = points

    if lo.is_enabled('s'):
        lo.s('Game Board:\n{}'.format(board))
        lo.s('Letters:\n{}'.format(letters))
        print()
    else:
        print('Running...')

    # print(board)
    # print(board.shape)
    # print(type(board.shape))

    full = Board(_board=board, default_board=default_board)
    #sys.exit(1)
    Settings.node_board = full

    #todo fix
    #md5_board = md5(board.tolist().encode()).hexdigest()[:9]
    #md5_letters = md5(''.join(sorted(letters)).encode()).hexdigest()[:9]

    #solution_filename = Path(_.SOLUTIONS_DIR, '{}_{}.pkl.gz'.format(md5_board, md5_letters))
    solution_filename = Path('xxxxxxx')

    if overwrite:
        has_cache = False
    else:
        has_cache = solution_filename.exists()

    if has_cache is False:
        solve(letters, dictionary)
        #dill.dump(Settings.word_info, gzip.open(str(solution_filename), 'wb'))  # todo remove nodes?
        #pickle.dump(Settings.word_info, gzip.open(str(solution_filename), 'wb'))  # todo remove nodes?
    else:
        lo.s('Found existing solution')

        #solution = dill.load(gzip.open(str(solution_filename)))  # type: List[Dict[str, Any]]
        solution = pickle.load(gzip.open(str(solution_filename)))  # type: List[Dict[str, Any]]
        #Settings.word_info = solution # fix

    show_solution(no_words)


def main(
    filename: str = None, dictionary: str = DICTIONARY, no_words: bool = False, exclude_letters: List[str] = None, overwrite: bool = False, log_level: str = DEFAULT_LOGLEVEL
):
    return cmain(filename, dictionary, no_words, exclude_letters, overwrite, log_level)
