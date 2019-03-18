# distutils: language = c++
# cython: infer_types=True
# cython: profile=True
## cython: linetrace=True
## cython: binding=True
## distutils: define_macros=CYTHON_TRACE=1

"""Parse and solve a scrabble board."""

cimport cython

from libcpp.vector cimport vector
from libcpp.utility cimport pair
#from libcpp.string cimport string

import gzip
import json
#import signal
import sys
from hashlib import md5

import pickle
#import dill
#dill.detect.trace(True)
#import multiprocessing as mp
#from multiprocessing import Pool
#from pathos.multiprocessing import ProcessPool as Pool

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple #, Union
from functools import lru_cache

import numpy as np
cimport numpy as np

import pandas as pd
from logs import log_init

import settings as _

#from cpython cimport array
#import array


# cdef class Node
# cdef class Board

ctypedef unsigned short us
ctypedef np.ndarray[str[2], ndim=2] board_t  # str 32?
ctypedef np.ndarray[object, ndim=2] nb_t
ctypedef np.ndarray[object, ndim=1] nodelist_t


DICTIONARY = 'wwf'
DEFAULT_LOGLEVEL = 'SUCCESS'

lo = log_init(DEFAULT_LOGLEVEL)

#sys.setrecursionlimit(20000)


cdef class CSettings:
    cdef:
        public list word_info, letters
        bint use_pool
        public short cpus, blanks
        public np.ndarray[np.str, ndim=2] board, default_board
        public Board node_board
        public dict shape, points
        public set words, search_words

    def __cinit__(self):
        self.word_info = []  # type: List[Dict[str, Any]]

        self.use_pool = False
        #self.cpus = 1  # type: int

        self.board = np.array([[]])
        self.default_board = np.array([[]])
        self.shape = {'row': 0, 'col': 0}  # type: Dict[str, int]
        self.letters = []  # type: List[str]
        self.blanks = 0  # type: int
        self.points = {}  # type: Dict[str, List[int]]
        self.words = set()  # type: Set[str]
        self.search_words = set()  # type: Set[str]

        self.node_board = None  # type: Board


cdef CSettings Settings = CSettings()


cdef class Node:
    ctypedef int dual[2]

    cdef:
        int x, y, points
        dual pos
        str multiplier, value
        bint is_start

        # not packed?
        packed struct word_dict:
            str word, direc
            int idx, pts
            bint is_blank
            #todo nodes?

        list words  # todo of word_dicts
        #np.ndarray[word_dict] words
        #np.ndarray words

        # struct cnode:
        #     int x, v
        #     str value
        #
        # cnode up,down,left,right  #todo pointer to node instead?

        #list up_vals, down_vals, left_vals, right_vals  # of duals
        Node *up, *down, *left, *right
        str up_word, down_word, left_word, right_word

        bint has_edge


    def __cinit__(self, int x, int y, str value=None, str multiplier=None):
        self.x = x
        self.y = y
        self.pos = (x, y)

        self.multiplier = None
        self.is_start = False

        if multiplier:
            if multiplier == 'x':
                self.is_start = True
            else:
                self.multiplier = (int(multiplier[0]), multiplier[1])

        self.value = value
        self.points = 0
        if self.value:
            try:
                self.points = Settings.points[self.value][1]
            except (KeyError, IndexError):
                lo.e('Could not get point value of "{}". Using 0.'.format(self.value))

        #self.words = np.ndarray([], dtype=word_dict)
        self.words = []

        # self.up = {self.x - 1, self.y} if self.x >= 0 else None
        # self.down = (self.x + 1, self.y) if self.x < Settings.shape['row'] else None
        # self.left = (self.x, self.y - 1) if self.y >= 0 else None
        # self.right = (self.x, self.y + 1) if self.y < Settings.shape['col'] else None

        # self.up_vals = []
        # self.down_vals = []
        # self.left_vals = []
        # self.right_vals = []
        self.up = None
        self.down = None
        self.left = None
        self.right = None

        self.up_word = ''
        self.down_word = ''
        self.left_word = ''
        self.right_word = ''

        self.has_edge = False


    # cdef Node get_up(self): return self.get_cell(self.up)
    # cdef get_down(self): return self.get_cell(self.down)
    # cdef get_left(self): return self.get_cell(self.left)
    # cdef get_right(self): return self.get_cell(self.right)

    # @lru_cache(255)
    # cdef get_row(self):
    #     return Settings.node_board.get_row(self.x)
    #
    # @lru_cache(255)
    # def get_col(self):
    #     return Settings.node_board.get_col(self.y)

    # cdef _get_row_val_nodes(self):
    #     cdef int rv
    #
    #     bef, aft = self.row_vals
    #     bef = rv[0]
    #     return (
    #         [Settings.node_board.get(*n) for n in self.row_vals[0]],
    #         [Settings.node_board.get(*n) for n in self.row_vals[1]]
    #     )
    #
    # cdef _get_col_val_nodes(self):
    #     return {
    #         'bef': [Settings.node_board.get(*n) for n in self.col_vals['bef']],
    #         'aft': [Settings.node_board.get(*n) for n in self.col_vals['aft']]
    #     }
    #
    # @lru_cache(1023)
    # def get_side_val_nodes(self,
    #     str direc
    # ):
    #     if direc == 'row':
    #         return self._get_row_val_nodes()
    #     else:
    #         return self._get_col_val_nodes()

    # def get_adj_words(self,
    #     str direc
    # ):
    #     if direc == 'row':
    #         return self.row_words
    #     else:
    #         return self.col_words
    #
    #
    # def get_cell(self,
    #     typ  # type: Optional[Union[str, Tuple[int, int]]]
    # ):
    #     if isinstance(typ, str):
    #         cell_tup = getattr(self, typ)
    #     else:
    #         cell_tup = typ
    #     if cell_tup:
    #         return Settings.node_board.get(x=cell_tup[0], y=cell_tup[1])


    # def has_edge(self):
    #     up = self.get_up()
    #     down = self.get_down()
    #     left = self.get_left()
    #     right = self.get_right()
    #
    #     return (isinstance(up, Node) and up.value is not None) or \
    #            (isinstance(down, Node) and down.value is not None) or \
    #            (isinstance(left, Node) and left.value is not None) or \
    #            (isinstance(right, Node) and right.value is not None)

    # @lru_cache(2047)
    # def _get_poss_word_dict(self,
    #         direc,  # type: str
    #         word  # type: str
    # ):
    #     return self.poss_values.get(direc, {}).get(word, {})
    #
    # def get_poss_val(self,
    #         direc,  # type: str
    #         word,  # type: str
    #         idx  # type: int
    # ):
    #     return self._get_poss_word_dict(direc, word).get(idx)  # ''?
    #
    # def set_poss_val(self,
    #         direc,  # type: str
    #         word,  # type: str
    #         idx,  # type: int
    #         is_blank=False  # type: bool
    # ):
    #     pv_info = self.poss_values[direc]
    #     if word not in pv_info:
    #         pv_info[word] = {idx: is_blank}
    #     else:
    #         pv_info_word = pv_info[word]
    #         if idx not in pv_info_word:
    #             pv_info_word[idx] = is_blank
    #         else:
    #             return False
    #     return True

    def get_points(self,
        word,  # type: str
        nodes,  # type: List['Node']
        direc,  # type: str
        new_words=None,  # type: Optional[List[Dict]]
        **_kw
    ):
        # todo combine this optional thing into one type

        if _kw: lo.e('Extra kw args: {}'.format(_kw))

        pts = self._points_from_nodes(word, nodes, direc)
        if new_words:
            for nw_dict in new_words:
                new_pts = self._points_from_nodes(**nw_dict)
                pts += new_pts

        if len([x for x in nodes if x.value is None]) == 7:  # todo: or is it all letters?
            pts += 35

        return pts

    @staticmethod
    def _points_from_nodes(
            word,  # type: str
            nodes,  # type: List['Node']
            direc,  # type: str
            **_kw
    ):
        if _kw: lo.e('Extra kw args: {}'.format(_kw))

        pts = 0
        word_mult = 1
        idx = 0
        for n in nodes:
            if n.value:
                pts += (n.points or 0)
            else:
                pts += n._letter_points(direc, word, idx)
                n_mult = n.multiplier
                if n_mult and n_mult[1] == 'w':
                    word_mult *= n_mult[0]
            idx += 1

        pts *= word_mult

        return pts


    @lru_cache(2047)
    def _letter_points(self,
            direc,  # type: str
            word,  # type: str
            idx  # type: int
    ):
        if self.points:  #unnec
            return self.points

        is_blank = self.get_poss_val(direc, word, idx)  # i guess i could use has_poss here
        if is_blank is not None:
            if is_blank:
                return 0

            try:
                pp = Settings.points[word[idx]][1]
            except (KeyError, IndexError):
                lo.e('Could not get point value of "{}". Using 0.'.format(word[idx]))
                pp = 0

            self.points = 0
            if self.multiplier:
                mval = self.multiplier[0]
                mtyp = self.multiplier[1]
                if mtyp == 'l':
                    pp *= mval
            return pp

        lo.e('Node has no points or poss points for ({}, {}, {})'.format(direc, word, idx))
        lo.w('node -> {}'.format(self))
        #should probably raise bigger error here

        return 0

    def str_pos(self):
        return '[{:2d},{:2d}]'.format(*self.pos)

    def __str__(self):
        return '<Node>: {} v: {:1s}'.format(self.str_pos(), self.value or '_')

    def __repr__(self):
        return self.__str__()


cdef class Board:
    cdef:
        board_t default_board
        bint new_game
        #nodelist_t nodes
        nb_t nodes
        Py_ssize_t nodes_rl, nodes_cl

    #def __cinit__(self, board=Settings.board, default_board=Settings.default_board):
    def __cinit__(self, board_t board, board_t default_board):
        cdef:
            int r, c
            #int db_mid_r = default_board.shape[0] // 2
            #int db_mid_c = default_board.shape[1] // 2
            #str db_mid, b_mid
            str dbval, bval
            Node node

        self.default_board = default_board

        self.nodes = np.empty_like(default_board, dtype=Node)
        self.nodes_rl = board.shape[0]
        self.nodes_cl = board.shape[1]

        self.new_game = False

        # db_mid = default_board[db_mid_r, db_mid_c]
        # b_mid = board[db_mid_r, db_mid_c]
        #
        # if db_mid == 'x' and not b_mid:
        #     self.new_game = True
        # else:
        #     self.new_game = False

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
                if not dbval:
                    dbval = None
                else:
                    dbval = dbval.strip()
                    if dbval == 'x' and not bval:
                        self.new_game = True

                node = Node(r, c, bval, dbval)
                self.nodes[r, c] = node

        # todo i should be using pointers here
        for r in range(self.nodes_rl):
            for c in range(self.nodes_cl):
                self._set_node(r, c)


    #@lru_cache(1023)  # todo reenable?
    cdef Node get(self, int x, int y):
        #f = filter(lambda obj: obj.x == x and obj.y == y, self.nodes)
        #return next(f, None)
        return self.nodes[x, y]

    cdef Node get_by_attr(self, str attr, v):
        return filter(lambda obj: getattr(obj, attr) == v, self.nodes)

    #@lru_cache()
    cdef nodelist_t get_row(self, int x):
        return self.nodes[x]

    #@lru_cache()
    cdef nodelist_t get_col(self, int y):
        return self.nodes[:,y]

    #@cython.boundscheck(False)
    cdef void _set_node(self, int r, int c):
        cdef:
            Node node, p
            nodelist_t nodes_up, nodes_down, nodes_left #, nodes_right
            #list up_tups = [], down_tups = [], left_tups = [], right_tups = []  # List[Tuple[int, int]]
            str up_word = '', down_word = '', left_word = '', right_word = ''
            bint has_edge = False

        node = self.nodes[r, c]

        if r > 0:
            node.up = self.nodes[r-1, c]
            if node.up.value is not None:
                node.has_edge = True
        elif r < self.nodes_rl - 1:
            node.down = self.nodes[r+1, c]
            if node.down.value is not None:
                node.has_edge = True
        elif c > 0:
            node.left = self.nodes[r, c-1]
            if node.left.value is not None:
                node.has_edge = True
        elif c < self.nodes_cl - 1:
            node.right = self.nodes[r, c+1]
            if node.right.value is not None:
                node.has_edge = True

        nodes_up = self.nodes[:r, c]
        nodes_down = self.nodes[r+1:, c]
        nodes_left = self.nodes[r, :c]
        nodes_right = self.nodes[r, c+1:]

        for p in reversed(nodes_up):
            if not p.value: break
            else:
                #up_tups.append(p.pos)
                up_word += p.value

        #up_tups.reverse()
        up_word = up_word[::-1]

        for p in reversed(nodes_left):
            if not p.value: break
            else:
                #left_tups.append(p.pos)
                left_word += p.value

        #left_tups.reverse()
        left_word = left_word[::-1]

        for p in nodes_down:
            if not p.value: break
            else:
                #down_tups.append(p.pos)
                down_word += p.value

        for p in nodes_right:
            if not p.value: break
            else:
                #right_tups.append(p.pos)
                right_word += p.value

        # node.up_vals = up_tups
        # node.down_vals = down_tups
        # node.left_vals = left_tups
        # node.right_vals = right_tups

        node.up_word = up_word
        node.down_word = down_word
        node.left_word = left_word
        node.right_word = right_word


# # cdef class NW:
# #     cdef str value
#
# def get_word(
#     list nodes  # type: List[Node]
# ) -> str:
#     cdef:
#         str nwv, res = ''
#         # NW nw
#         object nw
#         Py_ssize_t new_idx = len(nodes)
#         int i
#
#     for i in range(new_idx):
#         nw = nodes[i]
#         nwv = nw.value
#         res += nwv
#
#     return res
#     #return ''.join(nw.value or '+' for nw in nodes)


#@cython.profile(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef list check_and_get(
    node_list,  # List[Node]
    str direc,
    str chk_dir,
    str word,
    #string word,
    us word_len,
    us start
):
    cdef:
        us blanks, pos, i = 0
        int bef_idx, aft_idx
        Py_ssize_t new_idx = 0
        str bef_word, aft_word, le, new_word
        object node_bef, node_aft, no
        dict other_nodes, other_words
        tuple bef_nodes, aft_nodes

    bef_idx = start - 1
    if bef_idx >= 0:
        node_bef = node_list[bef_idx]
        if node_bef.value:
            return

    aft_idx = start + word_len
    if aft_idx < Settings.shape[direc]:
        node_aft = node_list[aft_idx]
        if node_aft.value:
            return

    ls = list(Settings.letters)
    blanks = Settings.blanks

    new_word_list = []  # List[Dict]
    set_poss_list = []  # type: List[Tuple[Node, Tuple[str, str, int, bool]]]

    #cdef (str, str, us) pos_tup

    for i in range(word_len):
        pos = start + i

        no = node_list[pos]

        #le = word.at(i)
        le = word[i]

        nval = no.value
        if nval:
            if nval != le:
                return

        else:
            is_blank = False

            if le not in ls:
                if blanks > 0:
                    ls.remove('?')
                    blanks -= 1
                    is_blank = True
                else:
                    # todo skip forward if the next letter doesnt exist?
                    return
            else:
                ls.remove(le)

            # set_success = no.set_poss_val(direc, word, i, is_blank)
            # if not set_success:
            #     #todo: i can probably skip stuff now
            set_poss_list.append((no, (direc, word, i, is_blank)))  # todo do i need this now?

            other_nodes = no.get_adj_vals(chk_dir)

            bef_nodes = other_nodes['bef']
            aft_nodes = other_nodes['aft']

            if bef_nodes or aft_nodes:
                other_words = no.get_adj_words(chk_dir)
                bef_word = other_words['bef']
                aft_word = other_words['aft']

                new_word = bef_word + le + aft_word

                if new_word not in Settings.words:
                    return

                new_idx = len(bef_word)

                # set_success = no.set_poss_val(chk_dir, new_word, new_idx, is_blank)
                # if not set_success:
                #     #todo: i can probably skip stuff now
                set_poss_list.append((no, (chk_dir, new_word, new_idx, is_blank)))

                new_word_list.append({
                    'word': new_word,
                    'nodes': list(bef_nodes) + [no] + list(aft_nodes),
                    'direc': chk_dir
                })

    for p in set_poss_list:
        pno, pda = p
        set_success = pno.set_poss_val(*pda)

    return new_word_list


#@cython.profile(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef list can_spell(Node no, str word, str direc, us word_len):
    cdef:
        us start, idx, dir_len
        #array node_list, d_nodes, spell_words, tup_nodes
        str chkdir

    if direc == 'row':
        idx = no.y
        node_list = no.get_row()
        chk_dir = 'col'
    else:
        idx = no.x
        node_list = no.get_col()
        chk_dir = 'row'

    dir_len = Settings.shape[direc]

    spell_words = []  # type: List[dict]

    start = idx - word_len + 1
    if start < 0:
        start = 0

    while (start + word_len) <= dir_len and start <= idx:
        new_words = check_and_get(node_list, direc, chk_dir, word, word_len, start)

        if new_words is not None:
            d_nodes = node_list[start:start + word_len]
            d_pts = no.get_points(word=word, nodes=d_nodes, direc=direc, new_words=new_words)

            tup_nodes = [n.pos for n in d_nodes]
            tup_new_words = [{'word': n['word'], 'nodes': [nn.pos for nn in n['nodes']]} for n in new_words]

            new_d = {
                'pts': d_pts,
                'word': word,
                'nodes': tup_nodes,
                'new_words': tup_new_words
            }

            #todo: insert reverse too?
            spell_words.append(new_d)

        start += 1

    return spell_words

#@cython.profile(True)
cdef void check_words(Node no, str word):
    cdef:
        str direction # enum?
        Py_ssize_t word_len = len(word)  # todo add word length to dict

    for direction in ('row', 'col'):
        can_spell(no, word, direction, word_len)


@cython.boundscheck(False)
#@cython.wraparound(False)
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

@cython.boundscheck(False)
@cython.wraparound(False)
cdef str _print_result(
    dict o,
    bint no_words = False
):
    cdef:
        str ow
        str pts = str(o['pts'])
        #Py_ssize_t owl

    if not no_words:
        ow = o['word']
    else:
        ow = '<hidden> (' + str(len(o['word'])) + ')'

    cdef str s = '\n'.join([
        'pts  : ' + pts,
        'word : ' + ow,  # todo: hide nodes? show length?
        'nodes: ' + _print_node_range(o["nodes"])
    ])

    # todo why are these tabs like 6 or 8?
    s += '\nnew_words:'

    cdef:
        list nw = o['new_words']  # list[dict]
        Py_ssize_t n, nw_l = len(nw)
        dict nw_d
        list nw_nodes
        str nw_nodes_str, nw_word

    if not no_words:
        for n in range(nw_l):
            nw_d = nw[n]
            nw_nodes = nw_d["nodes"]
            nw_word = nw_d["word"]
            nw_nodes_str = _print_node_range(nw_nodes)
            s += f'\n\t{nw_nodes_str}  {nw_word}'
    else:
        s += ' <hidden>'

    return s


cdef _add_results(
    list res_list  # type: Optional[List[dict]]
):
    if not res_list: return
    for res in res_list:
        Settings.word_info.append(res)


# def run_worker(
#     tuple data  # type: Tuple[Node, str]
# ):
#     no = data[0]
#     w = data[1]
#
#     # w = data
#     #
#     # if len(w) > 8:
#     #     word_name = w[:7] + '.'
#     # else:
#     #     word_name = w
#     #
#     # current = mp.current_process()
#     # #print(current.name)
#     # #newname = 'w %s' % word_name
#     # current.name = 'w %s' % word_name
#
#     return check_words(no, w)

#
# #pool_node = None  # type: Optional[Node]
# #lock = mp.Lock()
# def _pool_handler():
#     #global pool_node
#     #pool_node = n
#     signal.signal(signal.SIGINT, signal.SIG_IGN)
#
#     current = mp.current_process()
#     current.name = 'proc_%s' % current.name.split('-')[-1]
#
#
# #pool = mp.Pool(7, initializer=_pool_handler)

#@cython.profile(True)
cdef void check_node(Node no):
    cdef str w

    # if Settings.use_pool:  #todo: is this fixable for profiling?
    #     n = Settings.cpus
    #     #n = 2
    #     #pool = mp.Pool(n, initializer=_pool_handler, initargs=[pool_node])
    #     #pool = mp.Pool(n, initializer=_pool_handler)
    #     pool = Pool(n, initializer=_pool_handler)
    #     try:
    #         #pool_res = pool.map(run_worker, (w for w in Settings.search_words))  # type: List[List[dict]]
    #         pool_res = pool.map(run_worker, ((no, w) for w in Settings.search_words))  # type: List[List[dict]]
    #         pool.close()
    #         pool.join()
    #         #pool.terminate()
    #         #pool.restart()
    #     except KeyboardInterrupt:
    #         lo.e('User interrupt, terminating.')
    #         pool.terminate()
    #         #pool.join()
    #         sys.exit(1)
    #
    #     for x in pool_res:
    #         _add_results(x)
    #
    # else:

    for w in Settings.search_words:
        reses = check_words(no, w)
        _add_results(reses)


cdef void show_solution(
    bint no_words=False
):
    # todo mark blanks

    if not Settings.word_info:
        print('\nNo solution.')
    else:
        newlist = sorted(Settings.word_info, key=lambda k: k['pts'], reverse=True)
        best_data = newlist[:1][0]
        if lo.is_enabled('s'):
            #todo fix this
            top10 = []  # type: List[dict]
            seen = [(best_data['pts'], best_data['word'])]
            for n in newlist:
                if len(top10) == 10:
                    break
                seen_tup = (n['pts'], n['word'])
                if seen_tup in seen:
                    continue
                seen.append(seen_tup)
                top10.append(n)

            top10.reverse()

            print()
            lo.s('-- Top 10 --\n')
            for sidx, s in enumerate(top10):
                lo.s('Choice #{}\n{}\n'.format(sidx + 1, _print_result(s, no_words)))

            lo.s('-- Best --\n{}'.format(_print_result(best_data, no_words)))

        if not no_words:  # todo: print xs instead?
            solved_board = Settings.board.copy()

            for ni, node_tup in enumerate(best_data.get('nodes', [])):
                node = Settings.node_board.get(*node_tup)
                if not node.value:
                    solved_board[node.y][node.x] = '\x1b[33m' + best_data['word'][ni] + '\x1b[0m'

            print('\n' + '-' * ((Settings.shape['row'] * 2) - 1 + 4))

            for row in solved_board.iterrows():
                row_data = row[1].to_list()
                row_str = []
                for rl in row_data:
                    rl_len = len(rl)
                    rl_val = rl

                    if rl_len == 0:
                        rl_val = ' '
                    elif rl_len == 1:
                        rl_val = rl.upper()

                    row_str.append(rl_val)

                print('| ' + ' '.join(row_str) + ' |')

            print('-' * ((Settings.shape['row'] * 2) - 1 + 4))

        else:
            print('\n<solution hidden> ({})'.format(len(best_data["word"])))

        print('\nPoints: {}'.format(best_data["pts"]))


# todo: set default for dict? put in settings? move all settings out to options?
cdef void solve(
    list letters,  # type: List[str]
    str dictionary
):
    #todo make fnc exception
    try:
        wordlist = open(str(Path(_.WORDS_DIR, dictionary + '.txt'))).read().splitlines()
    except FileNotFoundError as exc:
        lo.c('Could not find file: {}'.format(exc.filename))
        sys.exit(1)

    words = set(wordlist)
    blanks = letters.count('?')

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

    # --

    # if Settings.use_pool:  #todo: is this fixable for profiling?
    #     lo.s('Using multiprocessing...')
    #     Settings.cpus = max(min(mp.cpu_count() - 1, len(Settings.search_words)), 1)

    cdef Board full = Settings.node_board
    cdef Node no
    cdef int r, c

    if full.new_game:
        lo.s(' = Fresh game = ')
        no = next(full.get_by_attr('is_start', True), None)
        if no:
            check_node(no)

    else:
        search_nodes = np.zeros_like(full.nodes, dtype=bool)

        if _.SEARCH_NODES is None:
            search_nodes[:] = True

        elif isinstance(_.SEARCH_NODES, list):
            if all(isinstance(t, int) for t in _.SEARCH_NODES):
                search_nodes[_.SEARCH_NODES, :] = True

            elif all(isinstance(t, list) and len(t) == 2 for t in _.SEARCH_NODES):
                xs = [v[0] for v in _.SEARCH_NODES]
                ys = [v[1] for v in _.SEARCH_NODES]

                try:
                    search_nodes[xs, ys] = True
                except IndexError:
                    lo.c('Could not locate nodes for board {}: {}'.format(full.nodes.shape, _.SEARCH_NODES))
                    sys.exit(1)

            else:
                lo.c('Incompatible search node type: {}'.format(type(_.SEARCH_NODES)))
                sys.exit(1)

        else:
            lo.c('Incompatible search node type: {}'.format(type(_.SEARCH_NODES)))
            sys.exit(1)

        for r in range(full.nodes_rl):
            for c in range(full.nodes_cl):
                if search_nodes[r, c]:
                    no = full.nodes[r, c]
                    if no.value or not no.has_edge:
                        search_nodes[r, c] = False

        cdef int num_valid = np.count_nonzero(search_nodes)
        cdef int si = 0

        for r in range(full.nodes_rl):
            for c in range(full.nodes_cl):
                if search_nodes[r, c] is True:
                    no = full.nodes[r, c]
                    if lo.is_enabled('s'):
                        si += 1
                        lo.s('Checking Node (%2s, %2s) - #%3i / %3i', r, c, si, num_valid)
                        check_node(no)


cpdef main(
    str filename=None,
    str dictionary=DICTIONARY,
    bint no_words=False,
    list exclude_letters=None,  # type: List[str]
    bint overwrite=False,
    bint no_multiprocess=False,
    str log_level=DEFAULT_LOGLEVEL,
    **_kwargs
):
    log_level = log_level.upper()
    if log_level != DEFAULT_LOGLEVEL:
        lo.set_level(log_level)

    if filename is not None:
        this_board_dir = Path(_.BOARD_DIR, filename)

        try:
            board = pd.read_pickle(Path(this_board_dir, _.BOARD_FILENAME))  # type: pd.DataFrame
            letters = pd.read_pickle(Path(this_board_dir, _.LETTERS_FILENAME))  # type: List[str]
        except FileNotFoundError as exc:
            lo.c('Could not find file: {}'.format(exc.filename))
            sys.exit(1)

    else:
        board = pd.DataFrame(_.BOARD)
        letters = _.LETTERS

    board = board.to_numpy(dtype=str)

    if exclude_letters:
        for el in exclude_letters:
            letters.remove(el.upper())

    # need?
    shape = {
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
        default_board = pd.read_pickle(board_name).to_numpy(dtype=str)
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
    Settings.shape = shape
    Settings.points = points

    if lo.is_enabled('s'):
        lo.s('Game Board:\n{}'.format(board))
        lo.s('Letters:\n{}'.format(letters))
        print()
    else:
        print('Running...')

    full = Board(board=board, default_board=default_board)
    Settings.node_board = full

    # if no_multiprocess:
    #     Settings.use_pool = False

    md5_board = md5(board.to_json().encode()).hexdigest()[:9]
    md5_letters = md5(''.join(sorted(letters)).encode()).hexdigest()[:9]

    solution_filename = Path(_.SOLUTIONS_DIR, '{}_{}.pkl.gz'.format(md5_board, md5_letters))

    if overwrite:
        has_cache = False
    else:
        has_cache = solution_filename.exists()

    if not has_cache:
        solve(letters, dictionary)
        #dill.dump(Settings.word_info, gzip.open(str(solution_filename), 'wb'))  # todo remove nodes?
        pickle.dump(Settings.word_info, gzip.open(str(solution_filename), 'wb'))  # todo remove nodes?
    else:
        lo.s('Found existing solution')

        #solution = dill.load(gzip.open(str(solution_filename)))  # type: List[Dict[str, Any]]
        solution = pickle.load(gzip.open(str(solution_filename)))  # type: List[Dict[str, Any]]
        Settings.word_info = solution

    show_solution(no_words)

    if not has_cache and lo.is_enabled('e'):
        print()
        #lo.e('board.get: %s', get_word.cache_info())
