#!/usr/bin/env python

"""Parse and solve a scrabble board."""

# -- TODOS

#todo: profiling
# combine stuffz
# change return [] to yields?
# if word was found in a different check, skip
# change to f'' formatting
# if someone elses word is a blank, dont count it
# why is multiprocessing _C getting called?
# if blank and already have another letter, will assume not blank
# fix empty boards in both files
# recommend choices based on letters left, opened tiles, end game, blanks, etc
# allow letters to use option, and num
# cache responses
# why does this take long to close?
# add must include

__version__ = 1.0

DEFAULT_LOGLEVEL = 'SUCCESS'  # need?

DICTIONARY = 'wwf'


def parse_args():
    from parsing import parser_init

    parser = parser_init(
        description=__doc__,
        usage='%(prog)s [options] filename',
        log_level=DEFAULT_LOGLEVEL,
        version=__version__
    )

    parser.add_argument('filename', type=str, default='',
        help='File path for the image')

    parser.add_argument('-n', '--no-words', action='store_true',
        help='Hide actual words')

    parser.add_argument('-o', '--overwrite', action='store_true',
        help='Overwrite existing cache')

    parser.add_argument('-d', '--dictionary', type=str, default=DICTIONARY,
        help='Dictionary/wordlist name to use for solving (default: %(default)s)')

    parser.add_argument('-e', '--exclude-letters', type=lambda x: x.split(','), metavar='L [,L...]',
        help='Letters to exclude from rack for solution')

    parser.add_argument('-m', '--no-multiprocess', action='store_true',
        help='Do not use multiprocessing')

    return parser.parse_args()


ARGS = None
if __name__ == '__main__':
    ARGS = parse_args()

import gzip
import json
import signal
import sys
from hashlib import md5

import pickle
#import dill
#dill.detect.trace(True)
import multiprocessing as mp
#from pathos.multiprocessing import ProcessPool as Pool

from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from functools import lru_cache

import pandas as pd
from logs import log_init

import settings as _

#import pyximport; pyximport.install(pyimport=True)

#import builtins
#profile = getattr(builtins, 'profile', lambda x: x)


lo = log_init(DEFAULT_LOGLEVEL)


#sys.setrecursionlimit(20000)

class Settings:
    word_info = []  # type: List[Dict[str, Any]]

    use_pool = True  # type: bool
    cpus = 1  # type: int

    board = pd.DataFrame()
    default_board = pd.DataFrame()
    shape = {'row': 0, 'col': 0}  # type: Dict[str, int]
    letters = []  # type: List[str]
    blanks = 0  # type: int
    points = {}  # type: Dict[str, List[int]]
    words = set()  # type: Set[str]
    search_words = set()  # type: Set[str]

    node_board = None  # type: Board


class Node:
    def __init__(self,
            x,  # type: int
            y,  # type: int
            multiplier=None,  # type: Optional[str]
            value=None  # type: Optional[str]
    ):
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
        self.points = None
        if self.value:
            try:
                self.points = Settings.points[self.value][1]
            except (KeyError, IndexError):
                lo.e('Could not get point value of "{}". Using 0.'.format(self.value))
                self.points = 0

        # word -> idx = blank
        #self.poss_words = {}

        # direc -> word -> idx = is_blank
        # letter is just word[idx]
        self.poss_values = {}  # type: Dict[str, Dict[str, Dict[int, bool]]]
        self.poss_values = {
            'row': {},
            'col': {}
        }

        self.up = (self.x - 1, self.y) if self.x >= 0 else None  # type: Optional[Tuple[int, int]]
        self.down = (self.x + 1, self.y) if self.x < Settings.shape['row'] else None  # type: Optional[Tuple[int, int]]
        self.left = (self.x, self.y - 1) if self.y >= 0 else None  # type: Optional[Tuple[int, int]]
        self.right = (self.x, self.y + 1) if self.y < Settings.shape['col'] else None  # type: Optional[Tuple[int, int]]

        self.row_vals = {'bef': [], 'aft': []}  # type: Dict[str, List[Tuple[int, int]]]
        self.col_vals = {'bef': [], 'aft': []}  # type: Dict[str, List[Tuple[int, int]]]


    def get_up(self): return self.get_cell(self.up)
    def get_down(self): return self.get_cell(self.down)
    def get_left(self): return self.get_cell(self.left)
    def get_right(self): return self.get_cell(self.right)


    @lru_cache(255)
    def get_row(self):
        return Settings.node_board.get_row(self.x)

    @lru_cache(255)
    def get_col(self):
        return Settings.node_board.get_col(self.y)

    def get_row_vals(self):
        return {
            'bef': [Settings.node_board.get(*n) for n in self.row_vals['bef']],
            'aft': [Settings.node_board.get(*n) for n in self.row_vals['aft']]
        }

    def get_col_vals(self):
        return {
            'bef': [Settings.node_board.get(*n) for n in self.col_vals['bef']],
            'aft': [Settings.node_board.get(*n) for n in self.col_vals['aft']]
        }

    @lru_cache(1023)
    def get_adj_vals(self,
            direc: str,
    ):
        if direc == 'row':
            return self.get_row_vals()
        elif direc == 'col':
            return self.get_col_vals()
        else:
            lo.c('Direction needs to be "row" or "col"')


    def get_cell(self,
            typ  # type: Optional[Union[str, Tuple[int, int]]]
    ):
        if isinstance(typ, str):
            cell_tup = getattr(self, typ)
        else:
            cell_tup = typ
        if cell_tup:
            return Settings.node_board.get(x=cell_tup[0], y=cell_tup[1])


    def has_edge(self):
        up = self.get_up()
        down = self.get_down()
        left = self.get_left()
        right = self.get_right()

        return (isinstance(up, Node) and up.value is not None) or \
               (isinstance(down, Node) and down.value is not None) or \
               (isinstance(left, Node) and left.value is not None) or \
               (isinstance(right, Node) and right.value is not None)

    @lru_cache(2047)
    def _get_poss_word_dict(self,
            direc,  # type: str
            word  # type: str
    ):
        return self.poss_values.get(direc, {}).get(word, {})

    def get_poss_val(self,
            direc,  # type: str
            word,  # type: str
            idx  # type: int
    ):
        return self._get_poss_word_dict(direc, word).get(idx)  # ''?

    def set_poss_val(self,
            direc,  # type: str
            word,  # type: str
            idx,  # type: int
            is_blank=False  # type: bool
    ):
        pv_info = self.poss_values[direc]
        if word not in pv_info:
            pv_info[word] = {idx: is_blank}
        else:
            pv_info_word = pv_info[word]
            if idx not in pv_info_word:
                pv_info_word[idx] = is_blank
            else:
                lo.v('Already parsed: {}, {} at {} ({} v {})'.format(direc, word, idx, is_blank, pv_info_word[idx]))
                return False
        return True

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


# class Word:  # todo
#     def __init__(self,
#             nodes  # type: List[Node]
#     ):
#         self.nodes = nodes


class Board:
    def __init__(self, board=Settings.board, default_board=Settings.default_board):
        self.default_board = default_board
        self.board = board

        self.new_game = False

        self.nodes = []  # type: List[Node]

        bvals = self.board.values  # type: List[List[Optional[str]]]

        for board_r in self.default_board.iterrows():
            r_num = board_r[0]  # type: int
            r_data = board_r[1]  # type: List[Optional[str]]
            b_row = bvals[r_num]
            for c_num, d_cell in enumerate(r_data):

                board_val = b_row[c_num]

                if not board_val:
                    board_val = None
                else:
                    board_val = board_val.strip()
                    if not board_val:
                        board_val = None
                    else:
                        board_val = board_val.upper()

                if not d_cell:
                    d_cell = None
                else:
                    d_cell = d_cell.strip()
                    if d_cell == 'x' and not board_val:
                        self.new_game = True

                node = Node(r_num, c_num, d_cell, board_val)
                self.nodes.append(node)

        self.set_nodes()

    @lru_cache(1023)
    def get(self,
            x,  # type: int
            y   # type: int
    ):
        f = filter(lambda obj: obj.x == x and obj.y == y, self.nodes)
        return next(f, None)

    def get_by_attr(self,
            attr,  # type: str
            v  # type: Any
    ):
        return filter(lambda obj: getattr(obj, attr) == v, self.nodes)

    @lru_cache()
    def get_row(self,
            x  # type: int
    ):
        return list(self.get_by_attr('x', x))

    @lru_cache()
    def get_col(self,
            y  # type: int
    ):
        return list(self.get_by_attr('y', y))

    def set_nodes(self):
        for n in self.nodes:
            for direc in ('row', 'col'):
                if direc == 'row':
                    nodes = self.get_row(n.x)
                    slice_val = n.y
                    node_vals = n.row_vals
                else:  # col
                    nodes = self.get_col(n.y)
                    slice_val = n.x
                    node_vals = n.col_vals

                bef = nodes[:slice_val]
                aft = nodes[slice_val + 1:]

                bef_tups = []  # type: List[Tuple[int, int]]
                aft_tups = []  # type: List[Tuple[int, int]]

                for p in reversed(bef):
                    if not p.value: break
                    else:
                        bef_tups.append(p.pos)

                bef_tups.reverse()

                for p in aft:
                    if not p.value: break
                    else:
                        aft_tups.append(p.pos)

                node_vals['bef'] = bef_tups
                node_vals['aft'] = aft_tups

                # for c in nodes:
                #     n.row.append(c.pos)
                #     if c.value is not None:
                #         n.row_vals.append(tup)
                #     else:
                #         n.row_vals.append(None)


def get_word(
        nodes  # type: List[Node]
        # direc='',  # type: str
        # word='',  # type: str
        # idx=0  # type: int
):
    # todo should not need other args
    # return ''.join(
    #     nw.value or
    #     (word[idx] if nw.has_poss_val(direc, word, idx) else None) or
    #     '+' for nw in nodes
    # )
    return ''.join(nw.value or '+' for nw in nodes)


#@profile
def check_and_get(
        node_list,  # type: List[Node]
        direc,  # type: str
        chk_dir,  # type: str
        word,  # type: str
        word_len,  # type: int
        start  # type: int
):
    #todo add before method

    bef_idx = start - 1
    if bef_idx >= 0:
        node_bef = node_list[bef_idx]
        if node_bef and node_bef.value:
            lo.d('Prior node exists for %s at %s (%s), skipping.' % (direc, bef_idx, node_bef.value))
            return None

    aft_idx = start + word_len
    if aft_idx < Settings.shape[direc]:
        node_aft = node_list[aft_idx]
        if node_aft and node_aft.value:
            lo.d('Following node exists for %s at %s (%s), skipping.', direc, aft_idx, node_aft.value)
            return None

    ls = list(Settings.letters)
    blanks = Settings.blanks

    new_word_list = []  # type: List[Dict]
    set_poss_list = []  # type: List[Tuple[Node, Tuple[str, str, int, bool]]]

    for i in range(word_len):
        pos = start + i

        try:
            no = node_list[pos]
        except IndexError:
            # todo should never happen
            lo.e('no space {} at {}+{} for {}, {}'.format(direc, start, i, word, len(node_list)))
            return None

        le = word[i]
        nval = no.value
        if nval:
            if nval != le:
                lo.d('mismatch {} vs {}'.format(nval, le))
                return None

        else:
            is_blank = False

            if le not in ls:
                if blanks > 0:
                    ls.remove('?')
                    blanks -= 1
                    is_blank = True
                else:
                    lo.d('missing letter {}'.format(le))
                    # todo skip forward if the next letter doesnt exist?
                    return None
            else:
                ls.remove(le)

            # set_success = no.set_poss_val(direc, word, i, is_blank)
            # if not set_success:
            #     #todo: i can probably skip stuff now
            #     lo.d('oops')
            set_poss_list.append((no, (direc, word, i, is_blank)))  # todo do i need this now?

            other_nodes = no.get_adj_vals(chk_dir)
            #lo.d(no.get_adj_vals.cache_info())

            bef_nodes = other_nodes['bef']
            aft_nodes = other_nodes['aft']

            if bef_nodes or aft_nodes:
                lo.v(other_nodes)

                bef_word = get_word(bef_nodes)
                aft_word = get_word(aft_nodes)

                new_word = bef_word + le + aft_word

                if new_word not in Settings.words:
                    #lo.d('invalid new word: {}'.format(new_word))
                    return None
                #else:
                #   lo.d('good new word: {}'.format(new_word))

                new_idx = len(bef_word)

                # set_success = no.set_poss_val(chk_dir, new_word, new_idx, is_blank)
                # if not set_success:
                #     #todo: i can probably skip stuff now
                #     lo.d('oops')
                set_poss_list.append((no, (chk_dir, new_word, new_idx, is_blank)))

                new_word_list.append({
                    'word': new_word,
                    'nodes': bef_nodes + [no] + aft_nodes,
                    'direc': chk_dir,
                    #'idx': new_idx
                })

    for p in set_poss_list:
        pno, pda = p
        set_success = pno.set_poss_val(*pda)
        if not set_success:
            #todo: i can probably skip stuff now
            lo.d('oops')

    return new_word_list


#@profile
def can_spell(
        no,  # type: Node
        word,  # type: str
        direc,  # type: str
        word_len=None  # type: Optional[int]
):
    lo.v('{} - {}'.format(direc, word))

    if direc == 'row':
        idx = no.y
        node_list = no.get_row()
        chk_dir = 'col'
    elif direc == 'col':
        idx = no.x
        node_list = no.get_col()
        chk_dir = 'row'
    else:
        lo.c('Direction needs to be "row" or "col"')
        return

    dir_len = Settings.shape[direc]

    if word_len is None:
        word_len = len(word)

    spell_words = []  # type: List[dict]

    start = max(0, idx - word_len + 1)
    while (start + word_len) <= dir_len and start <= idx:
        #lo.d('%s, %s, %s, %s', start, idx, word_len, dir_len)
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


def check_words(
        no,  # type: Node
        word  # type: str
):
    data = []

    word_len = len(word)  # todo add word length to dict

    for direction in ('row', 'col'):
        spell_words = can_spell(no, word, direction, word_len)

        if spell_words:
            lo.i('--> YES: %s: %s', direction, word)
            # combine words with idxs?

            for word_dict in spell_words:
                lo.i(word_dict)
                data.append(word_dict)

    return data


def _print_node_range(
        n  # type: List[Tuple[int, int]]
):
    return '[{:2d},{:2d}] : [{:2d},{:2d}]'.format(n[0][0], n[0][1], n[-1][0], n[-1][1])


def _print_result(
        o,  # type: dict
        no_words=False  # type: bool
):
    s = '\n'.join([
        'pts  : {}'.format(o["pts"]),
        'word : {}'.format(o["word"] if not no_words else "<hidden> (" + str(len(o["word"])) + ")"),  # todo: hide nodes? show length?
        'nodes: {}'.format(_print_node_range(o["nodes"])),
    ])

    # todo why are these tabs like 6 or 8?
    s += '\nnew_words:'
    if not no_words:
        for n in o['new_words']:
            s += '\n\t{}  {}'.format(_print_node_range(n["nodes"]), n["word"])
    else:
        s += ' <hidden>'

    return s


def _add_results(
        res_list  # type: Optional[List[dict]]
):
    if not res_list: return
    for res in res_list:
        Settings.word_info.append(res)


def run_worker(
        data  # type: Tuple[Node, str]
):
    no = data[0]
    w = data[1]

    # w = data
    #
    # if len(w) > 8:
    #     word_name = w[:7] + '.'
    # else:
    #     word_name = w
    #
    # current = mp.current_process()
    # #print(current.name)
    # #newname = 'w %s' % word_name
    # current.name = 'w %s' % word_name

    return check_words(no, w)


#pool_node = None  # type: Optional[Node]
#lock = mp.Lock()
def _pool_handler():
    #global pool_node
    #pool_node = n
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    current = mp.current_process()
    current.name = 'proc_%s' % current.name.split('-')[-1]


#pool = mp.Pool(7, initializer=_pool_handler)

def check_node(
        no  # type: Node
):
    if Settings.use_pool:  #todo: is this fixable for profiling?
        n = Settings.cpus
        #n = 2
        #pool = mp.Pool(n, initializer=_pool_handler, initargs=[pool_node])
        pool = mp.Pool(n, initializer=_pool_handler)
        #pool = Pool(n, initializer=_pool_handler)
        try:
            #pool_res = pool.map(run_worker, (w for w in Settings.search_words))  # type: List[List[dict]]
            pool_res = pool.map(run_worker, ((no, w) for w in Settings.search_words))  # type: List[List[dict]]
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            lo.e('User interrupt, terminating.')
            pool.terminate()
            #pool.join()
            sys.exit(1)

        for x in pool_res:
            _add_results(x)

    else:
        for w in Settings.search_words:
            reses = check_words(no, w)
            _add_results(reses)


def show_solution(no_words=False  # type: bool
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
def solve(
        letters,  # type: List[str]
        dictionary  # type: str
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

    if Settings.use_pool:  #todo: is this fixable for profiling?
        lo.i('Using multiprocessing...')
        Settings.cpus = max(min(mp.cpu_count() - 1, len(Settings.search_words)), 1)

    full = Settings.node_board

    if full.new_game:
        lo.s(' = Fresh game = ')
        no = next(full.get_by_attr('is_start', True), None)
        if no:
            check_node(no)

    else:
        if _.SEARCH_NODES is None:
            full_nodes = full.nodes
        elif isinstance(_.SEARCH_NODES, list):
            if all(isinstance(t, int) for t in _.SEARCH_NODES):
                full_nodes = [nl for r in _.SEARCH_NODES for nl in full.get_row(r)]
            elif all(isinstance(t, tuple) and len(t) == 2 for t in _.SEARCH_NODES):
                full_nodes = []
                for nl in _.SEARCH_NODES:
                    fr = full.get(nl[0], nl[1])
                    if not fr:
                        lo.c('Could not locate node: {}'.format(nl))
                        sys.exit(1)
                    full_nodes.append(fr)
            else:
                lo.c('Incompatible search node type: {}'.format(type(_.SEARCH_NODES)))
                sys.exit(1)
        else:
            lo.c('Incompatible search node type: {}'.format(type(_.SEARCH_NODES)))
            sys.exit(1)

        full_nodes_len = len(full_nodes)
        for no in full_nodes:
            if no and not no.value and no.has_edge():
                if lo.is_enabled('s'):
                    lo.s('Checking Node (%2s, %2s) - #%s / %s',
                          no.x, no.y, full_nodes.index(no) + 1, full_nodes_len
                    )
                check_node(no)


def main(
        filename=None,  # type: str
        dictionary=DICTIONARY,  # type: str
        no_words=False,  # type: bool
        exclude_letters=None,  # type: List[str]
        overwrite=False,  # type: bool
        no_multiprocess=False,  # type: bool
        log_level=DEFAULT_LOGLEVEL,  # type: str
        **_kwargs
):
    start = timer()

    #global Settings
    #global Node

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

    if exclude_letters:
        for el in exclude_letters:
            letters.remove(el.upper())

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
        default_board = pd.read_pickle(board_name)  # type: pd.DataFrame
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
        lo.v('Default Board:\n{}'.format(Settings.default_board))
        lo.s('Game Board:\n{}'.format(board))
        lo.s('Letters:\n{}'.format(letters))
        print()
    else:
        print('Running...')

    full = Board(board=board, default_board=default_board)
    Settings.node_board = full

    if no_multiprocess:
        Settings.use_pool = False

    md5_board = md5(board.to_json().encode()).hexdigest()[:9]
    md5_letters = md5(''.join(sorted(letters)).encode()).hexdigest()[:9]

    solution_filename = Path(_.SOLUTIONS_DIR, '{}_{}.pkl.gz'.format(md5_board, md5_letters))

    if overwrite:
        has_cache = False
    else:
        has_cache = solution_filename.exists()
        if not has_cache:
            lo.v('No existing solution found')

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

    if not has_cache and lo.is_enabled('v'):
        print()
        lo.v('board.get: %s', Board.get.cache_info())
        lo.v('node.row: %s', Node.get_row.cache_info())
        lo.v('node.col: %s', Node.get_col.cache_info())
        lo.v('node.adj_vals: %s', Node.get_adj_vals.cache_info())
        lo.v('node.lpts: %s', Node._letter_points.cache_info())

    if lo.is_enabled('i'):
        end = timer()
        print()
        lo.i('Time: {}'.format(round(end - start, 1)))


if __name__ == '__main__':
    dargs = vars(ARGS)
    main(**dargs)
