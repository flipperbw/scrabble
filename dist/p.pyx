"""Parse and solve a scrabble board."""

cimport cython

cdef object gzip, json, sys, pickle, Path, np, pd, _s, log_init

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

import numpy as np
cimport numpy as cnp

import pandas as pd

from logs import log_init

import settings as _s

from typing import Any, Dict, List


cdef str DICTIONARY = 'wwf'
cdef str DEFAULT_LOGLEVEL = 'SUCCESS'

cdef object lo = log_init(DEFAULT_LOGLEVEL, skip_main=True)

# no numpy array .flags to see contiguous
#DTYPE = np.uint16
#ctypedef cnp.uint16_t DTYPE_t
#DTYPE = np.intc
# ctypedef cnp.int32_t DTYPE_t


ctypedef unsigned short us
#ctypedef signed int us
ctypedef (us, us) dual
#ctypedef us dual[2]
ctypedef unsigned char* uchr
ctypedef unsigned char uchrn

#ctypedef np.ndarray[str, ndim=2] board_t  # str 32?
#ctypedef np.ndarray[object, ndim=2] nb_t
#ctypedef np.ndarray[object, ndim=1] nodelist_t


#cdef bytes NUL = b'\0'
DEF NUL = b'\0'
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
        public dual pos
        uchr value

    #def __cinit__(self, bint is_blank, uchrn pts, multiplier_t mult, dual pos, uchr value):
    def __cinit__(self, multiplier_t mult, dual pos, uchr value):
        self.mult = mult
        self.pos = pos
        self.value = value

        self.is_blank = False
        self.from_rack = False
        self.pts = 0

    def __str__(self) -> str:
        return '<'\
            f'v: {self.value.decode("utf8")}, '\
            f'pos: [{self.pos[0]:2d},{self.pos[1]:2d}], '\
            f'bl: {"T" if self.is_blank is True else "f"}, '\
            f'rk: {"T" if self.from_rack is True else "f"}, '\
            f'pts: {"_" if self.pts == 0 and self.is_blank is False else self.pts}, '\
            f'mult: "{"_" if self.mult.amt == 1 else str(self.mult.amt) + self.mult.typ.decode("utf8").upper()}"'\
            '>'

    def __repr__(self) -> str:
        return self.__str__()

# cdef packed struct Letter_t:
#     bint is_blank
#     int pts
#     multiplier_t mult
#     dual pos
#     uchr value
#     #char value
#
# ctypedef Letter_t Letter

#todo reimplement lrucache

# just need to set max size on these lists of structs?

cdef class WordDict:
    cdef:
        str word
        #uchr word
        str direc
        public us pts
        list letters  # of Letter
        #Letter letters[100]

    def __cinit__(self, uchr word, str direc, us pts, list letters):
        self.word = word.decode('utf8')
        #self.word = word
        self.direc = direc
        self.pts = pts
        self.letters = letters

    cdef str sol(self):
        cdef dual p_s = self.letters[0].pos
        cdef dual p_e = self.letters[-1].pos
        return f'pts: {self.pts:3d} | dir: "{self.direc}" | pos: [{p_s[0]:2d},{p_s[1]:2d}] - [{p_e[0]:2d},{p_e[1]:2d}] | w: {self.word}'

    def __str__(self) -> str:
        cdef Letter l
        return '<w: {} | pts: {} | dir: {} | letts: [\n  {}\n]>'.format(self.word, self.pts, self.direc, '\n  '.join(str(l) for l in self.letters))

    def __repr__(self) -> str:
        return self.__str__()

# # not packed?
# cdef struct WordDict_t:
#     int pts
#
#     Letter* letters
#     #cnp.ndarray letters
#     #letter_t letters[]
#
#     char* word
#     char* direc
#
# ctypedef WordDict_t WordDict

cdef class CSettings:
    cdef:
        #cnp.ndarray board, default_board
        #np.ndarray[str, ndim=2] board, default_board

        list letters
        #cnp.ndarray letters
        #np.ndarray[str, ndim=1] letters

        uchrn blanks
        #dict shape
        dict points
        set words
        list search_words
        Py_ssize_t search_words_l
        Board node_board

    def __cinit__(self):
        #self.board = np.empty((0,0), dtype=np.str_)
        #self.default_board = np.empty((0,0), dtype=np.str_)
        self.letters = []
        #self.letters = np.empty(0) # dtype=bytes?
        self.blanks = 0
        #self.shape = {'row': 0, 'col': 0}  # type: Dict[str, int]
        #todo do types do anything for cython? infer?
        self.points = {}  ## type: Dict[str, List[int]]
        self.words = set()  ## type: Set[bytes]
        self.search_words = []
        self.search_words_l = 0
        self.node_board = None


cdef CSettings Settings = CSettings()


cdef class Node:
    cdef:
        us x, y
        dual pos
        uchrn points
        multiplier_t multiplier
        bint is_start
        Node up, down, left, right
        #cnp.ndarray up_word, down_word, left_word, right_word
        cnp.ndarray[:] up_word, down_word, left_word, right_word
        bint has_edge
        uchr value

    def __cinit__(self, us x, us y, str val, multiplier_t multiplier):  # todo test no types in cinit
        self.x = x
        self.y = y
        self.pos = (x, y)
        # self.pos[0] = x
        # self.pos[1] = y

        self.multiplier = multiplier

        if multiplier.typ == b'x':
            self.is_start = True
            self.multiplier.typ = NUL
        else:
            self.is_start = False

        cdef lpts_t lpt
        #cdef bytes _str

        if not val:
            self.value = NUL
            self.points = 0
        else:
            _str = val.encode('utf8')  # todo, assign directly?
            self.value = _str

            try:
                lpt = Settings.points[_str]
                self.points = lpt.pts
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

    cpdef str str_pos(self):
        return '[{:2d},{:2d}]'.format(*self.pos)

    def __str__(self):
        cdef str s = '_'
        if self.value[0]:
            s = self.value.decode('utf8')
        return '<Node: {} v: {}>'.format(self.str_pos(), s)

    def __repr__(self):
        return self.__str__()


cdef class Board:
    cdef:
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
    def __cinit__(self, cnp.ndarray[:, :] board, cnp.ndarray[:, :] default_board):
        cdef:
            Py_ssize_t r, c
            uchrn mamt
            str dbval, bval, x
            multiplier_t mult
            Node node
            #cnp.ndarray board
            #uchr mtyp
            bytes mtyp

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

    cdef void _set_adj_nodes(self, str d, Node n):
        cdef:
            cnp.ndarray[:] loop_nodes, setret
            #cnp.ndarray loop_nodes, setret
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
            if not p.value[0]:
                break
            else:
                let = Letter(
                    p.multiplier,
                    p.pos,
                    p.value
                )
                let.pts = p.points
                # let.value=p.value
                # let.pos=p.pos
                # let.is_blank=False
                # let.mult=p.multiplier
                # let.pts=p.points

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

        if r < self.nodes_rl - 1:
            node.down = self.nodes[r+1, c]
            if node.down.value[0]:
                node.has_edge = True

        if c > 0:
            node.left = self.nodes[r, c-1]
            if node.left.value[0]:
                node.has_edge = True

        if c < self.nodes_cl - 1:
            node.right = self.nodes[r, c+1]
            if node.right.value[0]:
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
            dict spts = Settings.points

        for l in range(len_word):
            letter = (<Letter> word[l])

            if letter.from_rack is False:
                pts += letter.pts
            elif letter.is_blank is False:
                lpt = <lpts_t> (spts[letter.value])
                p = lpt.pts  # exception handling

                if letter.mult.typ == <uchr>b'l':
                    p *= letter.mult.amt
                elif letter.mult.typ == <uchr>b'w':
                    word_mult *= letter.mult.amt

                pts += p

        #lo.x(pts)

        pts *= word_mult

        return pts

#@cython.profile(True)
#@cython.nonecheck(False)

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef void check_and_get(Node[:] node_list, str direc, uchr word, Py_ssize_t word_len) except *:
    cdef Node chk_first, chk_last
    cdef Node first = node_list[0]
    cdef Node last = node_list[word_len - 1]

    if direc == 'r':
        chk_first = first.left
        chk_last = last.right
    else:
        chk_first = first.up
        chk_last = last.down

    if chk_first is not None and chk_first.value[0]: return
    if chk_last is not None and chk_last.value[0]: return

    cdef uchrn blanks = Settings.blanks
    #todo what is the difference in not setting here? infer types?

    #ls = list(Settings.letters) #.tolist()  # todo fix. string?
    #cnp.ndarray ls
    cdef list ls = Settings.letters[:]

    cdef Node no

    cdef:
        us pts
        us extra_pts = 0 #new_pts
        Py_ssize_t i, bi, lsl
        #int nli = 0
        Letter le
        Letter b
        bint is_blank = False
        #str new_word #, new_direc
        bytes new_word

        #letter_list new_lets
        cnp.ndarray[:] bef_lets, aft_lets, np_lets #, new_lets

        list word_lets = []
        list new_let_list = []

        WordDict w #, nw
        bytes nww, new_direc
        bytes nv
        #uchrn nv
        lpts_t lpt

        dict spts = Settings.points

        uchr nov

    #for nv in word[:word_len]:
    for i in range(word_len):
        no = node_list[i]
        nv = word[i]

        le = Letter(
            no.multiplier,
            no.pos,
            nv
        )

        nov = no.value
        if nov[0]:
            if nov != (<uchr>nv):
                #lo.d(f'mismatch {no.value} v {le.value}')
                return
            else:
                le.pts = no.points

        else:
            is_blank = False

            if nv not in ls:
                #lo.d(f'{le.value} not in {ls}')
                if blanks > 0:
                    ls.remove(b'?')
                    blanks -= 1
                    is_blank = True
                    #lo.d('removed blank: %s', blanks)
                else:
                    # todo skip forward if the next letter doesnt exist?
                    return
            else:
                #lo.d(f'good {le}')
                ls.remove(nv)

            le.from_rack = True

            if is_blank is False:
                lpt = spts[nv]
                le.pts = lpt.pts
            else:
                le.is_blank = True

            if direc == 'r':
                bef_lets = no.up_word
                aft_lets = no.down_word
            else:
                bef_lets = no.left_word
                aft_lets = no.right_word

            if bef_lets is not None or aft_lets is not None:
                #lo.d('bef: %s', bef_lets)
                #lo.d('aft: %s', aft_lets)

                new_word = b''
                new_let_list = []

                if bef_lets is not None:
                    for bi in range(bef_lets.shape[0]):
                        b = bef_lets[bi]
                        #new_word[nli] = b.value
                        new_word += b.value
                        new_let_list.append(b)

                new_word += nv
                new_let_list.append(le)

                if aft_lets is not None:
                    for bi in range(aft_lets.shape[0]):
                        b = aft_lets[bi]
                        new_word += b.value
                        new_let_list.append(b)

                if new_word not in Settings.words:
                    return

                #lo.x(new_word)
                extra_pts += (<us>Settings.node_board.get_points(new_let_list))

                #new_lets = np.array(new_let_list, dtype=object)
                #new_pts = Settings.node_board.get_points(new_lets)
                #
                # if direc == 'r':
                #     new_direc = b'c'
                # else:
                #     new_direc = b'r'
                # #nw = WordDict(word=new_word, direc=new_direc, pts=new_pts, letters=new_lets)
                # nww = new_word.encode('utf8')
                # nw.word = nww
                # nw.direc = new_direc
                # nw.pts = new_pts
                # for ii in range(len(new_lets)):
                #     nw.letters[ii] = new_lets[ii]
                #
                # print(nw.direc)
                #print(nw)
                #print(nw.letters)

                # if nw not in Settings.node_board.words:
                #     Settings.node_board.words.append(nw)
                # else: lo.e('no %s', nw)
                #Settings.node_board.words.append(nw)

                #lo.x(repr(nw))
                #lo.x(repr(nw) in Settings.node_board.words)

        word_lets.append(le)

    lsl = len(ls)

    if lsl == len(Settings.letters):
        return

    #lo.i('GOOD: %s at %s : %s', word, first.str_pos(), last.str_pos())

    #np_lets = np.array(word_lets, dtype=object)
    pts = Settings.node_board.get_points(word_lets) + extra_pts

    if lsl == 0:
        pts += 35

    w = WordDict(
        word,
        direc,
        pts,
        #np_lets
        word_lets
    )

    # if w not in Settings.node_board.words:
    #     Settings.node_board.words.append(w)
    # else:
    #     lo.e('already saw %s', w)

    Settings.node_board.words.append(w)


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef void check_nodes(Node[:] nodes, str direc):
    cdef:
        #str w #, chk_dir
        (uchr, Py_ssize_t) w
        #tuple w
        Node x, xt
        list sw = Settings.search_words
        Py_ssize_t swl = Settings.search_words_l
        Py_ssize_t word_len, i, j, s, t
        Py_ssize_t nlen = len(nodes)
        #bytes wc
        Node[:] subnodes
        bint has_edge = False
        uchr ww

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
        w = sw[s]

        word_len = w[1]
        if word_len > nlen:
            continue

        ww = w[0]

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

"""
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
"""


cdef void show_solution(bint no_words=False):
    # todo mark blanks

    cdef WordDict w
    cdef list newlist

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

        # print(Settings.node_board.words[0])
        # print(dir(Settings.node_board.words[0]))
        # print(vars(Settings.node_board.words[0]))

        newlist = sorted(Settings.node_board.words, key=lambda k: k.pts, reverse=True)
        for w in reversed(newlist[:15]):
            lo.s(w.sol())

        #for i in Settings.node_board.words:
        #    lo.s(i)

# todo: set default for dict? put in settings? move all settings out to options?
cdef void solve(
    #cnp.ndarray letters, # str
    str dictionary
):
    #todo make fnc exception
    try:
        wordlist = open(str(Path(_s.WORDS_DIR, dictionary + '.txt'))).read().splitlines()
    except FileNotFoundError as exc:
        lo.c('Could not find file: {}'.format(exc.filename))
        sys.exit(1)

    #blanks = list(Settings.letters).count(b'?')
    cdef uchrn blanks = Settings.letters.count(b'?')
    cdef Board full = Settings.node_board
    cdef str w
    cdef Py_ssize_t mnlen = max(full.nodes_cl, full.nodes_rl)
    cdef set words = {w for w in wordlist if len(w) <= mnlen}
    cdef set search_words = set()

    if _s.SEARCH_WORDS is None:
        if blanks > 0:
            search_words = words
        else:
            # todo: faster if str? why gen slower?
            search_words = {w for w in words if any(l.decode('utf8') in w for l in Settings.letters)}
    elif isinstance(_s.SEARCH_WORDS, tuple):
        search_words = set(wordlist[_s.SEARCH_WORDS[0]: _s.SEARCH_WORDS[1]])
    elif isinstance(_s.SEARCH_WORDS, set):
        search_words = _s.SEARCH_WORDS
    else:
        lo.c('Incompatible search words type: {}'.format(type(_s.SEARCH_WORDS)))
        sys.exit(1)

    # --

    # todo allow extra mods
    # print(board)
    # print(board[10])
    # board[8][10] = 'Y'
    # letters = [b'F', b'G']

    cdef str s
    cdef set bwords = {s.encode('utf8') for s in words}
    cdef list sw = [(s.encode('utf8'), len(s)) for s in search_words]

    Settings.blanks = blanks
    Settings.words = bwords
    Settings.search_words = sw
    Settings.search_words_l = len(sw)


    cdef cnp.ndarray[object, ndim=2] nodes = full.nodes
    #cdef cnp.ndarray[:, :] nodes = full.nodes
    cdef int i, ic, tot

    if full.new_game:
        lo.s(' = Fresh game = ')
        no = next(full.get_by_attr('is_start', True), None)
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
            check_nodes(nodes[i], 'r')

        for i in search_cols:
            if lo.is_enabled('s'):
                ic += 1
                lo.s('Checking col %2i  (%2i / %i)', i, ic, tot)
            check_nodes(nodes[:,i], 'c')


cdef void cmain(str filename, str dictionary, bint no_words, list exclude_letters, bint overwrite, str log_level):
    cdef:
        dict points
        object pdboard
        cnp.ndarray board, default_board
        #cnp.ndarray letters
        list letters
        #todo declare object rather than list?
        int board_size
        object board_name
        bint has_cache

    log_level = log_level.upper()
    if log_level != DEFAULT_LOGLEVEL:
        lo.set_level(log_level)

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
        pdboard = pd.DataFrame(_s.BOARD).to_numpy()
        #letters = np.asarray(_s.LETTERS, dtype=np.bytes_)
        letters = _s.LETTERS

    if exclude_letters:
        for el in exclude_letters:
            #letters = letters[letters != el]
            letters.remove(el)

    #cdef bytes[:] bletters = [l.encode('utf8') for l in letters]
    cdef list bletters = [l.encode('utf8') for l in letters]

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
        points = json.load(open(str(Path(_s.POINTS_DIR, dictionary + '.json'))))  # type: Dict[str, List[int]]
    except FileNotFoundError as exc:
        lo.c('Could not find file: {}'.format(exc.filename))
        sys.exit(1)

    #Settings.default_board = default_board
    #Settings.board = board
    #Settings.shape = shapez

    cpoints = {}  # type: dict
    cdef lpts_t lpt
    for k, v in points.items():
        lpt.amt = v[0]
        lpt.pts = v[1]
        cpoints[k.encode('utf8')] = lpt
    Settings.points = cpoints

    if lo.is_enabled('s'):
        lo.s('Game Board:\n{}'.format(pdboard))
        lo.s('Letters:\n{}'.format(letters))
        print()
    else:
        print('Running...')

    cdef Board full = Board(board, default_board)
    Settings.node_board = full
    #sys.exit(1)

    cdef str md5_board, md5_letters

    #todo fix
    #md5_board = md5(board.tolist().encode()).hexdigest()[:9]
    #md5_letters = md5(''.join(sorted(letters)).encode()).hexdigest()[:9]

    #solution_filename = Path(_s.SOLUTIONS_DIR, '{}_{}.pkl.gz'.format(md5_board, md5_letters))
    solution_filename = Path('xxxxxxx')

    if overwrite:
        has_cache = False
    else:
        has_cache = solution_filename.exists()

    if has_cache is False:
        solve(dictionary)
        #solve(letters, dictionary)
        #dill.dump(Settings.word_info, gzip.open(str(solution_filename), 'wb'))  # todo remove nodes?
        #pickle.dump(Settings.word_info, gzip.open(str(solution_filename), 'wb'))  # todo remove nodes?
    else:
        lo.s('Found existing solution')

        #solution = dill.load(gzip.open(str(solution_filename)))  # type: List[Dict[str, Any]]
        solution = pickle.load(gzip.open(str(solution_filename)))  # type: List[Dict[str, Any]]
        #Settings.word_info = solution # fix

    show_solution(no_words)


cpdef void main(
    filename: str = None, dictionary: str = DICTIONARY, no_words: bool = False, exclude_letters: List[str] = None, overwrite: bool = False, log_level: str = DEFAULT_LOGLEVEL, profile: bool = False
):
    cmain(filename, dictionary, no_words, exclude_letters, overwrite, log_level)
