#!/usr/bin/env python

import json
from multiprocessing import (cpu_count, current_process, Pool)
from pprint import pformat
from timeit import default_timer as timer
from typing import Any, Dict, Iterator, List, Optional, Tuple

from setup import (
    BOARD,
    DEFAULT_BOARD,
    LETTERS,
    LOG_LEVEL,
    SEARCH_WORDS as _SW,
    USE_POOL,
)
from utils import log_init

#import builtins
#profile = getattr(builtins, 'profile', lambda x: x)

# -- TODOS

#todo: profiling
#combine stuffz
#change return [] to yields?
#if word was found in a different check, skip
#change to f'' formatting
# if someone elses word is a blank, dont count it
# why is multiprocessing _C getting called?
# if blank and already have another letter, will assume not blank

# -- globals

BLANKS = LETTERS.count('?')

WORD_LIST = open('data/wordlist.txt').read().splitlines()
WORDS = set(WORD_LIST)

SEARCH_WORDS = _SW

SHAPE = {
    'row': BOARD.shape[0],
    'col': BOARD.shape[1]
}

POINTS = json.load(open('data/points.json'))

lo = log_init(LOG_LEVEL, skip_main=False)

# --

if lo.is_enabled('s'):
    lo.i('\nDefault Board:\n{}'.format(DEFAULT_BOARD))
    lo.s('\nGame Board:\n{}'.format(BOARD))
    lo.s('\nLetters: {}'.format(LETTERS))
    print()
else:
    print('Running...')


class Node:
    def __init__(self, x: int, y: int, multiplier: Optional[str] = None, value: Optional[str] = None):
        self.x = x
        self.y = y

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
            self.points = POINTS.get(self.value)[1]

        # word -> idx = blank
        #self.poss_words = {}

        # direc -> word -> idx = is_blank
        # letter is just word[idx]
        self.poss_values: Dict[str, Dict[str, Dict[int, bool]]] = {
            'row': {},
            'col': {}
        }

        self.up: Optional[Node] = None
        self.left: Optional[Node] = None
        self.down: Optional[Node] = None
        self.right: Optional[Node] = None
        self.row: List['Node'] = []
        self.col: List['Node'] = []

    def has_edge(self) -> bool:
        return (isinstance(self.left, Node) and self.left.value is not None) or \
               (isinstance(self.right, Node) and self.right.value is not None) or \
               (isinstance(self.up, Node) and self.up.value is not None) or \
               (isinstance(self.down, Node) and self.down.value is not None)


    def _get_poss_word_dict(self, direc: str, word: str) -> Dict[int, bool]:
        return self.poss_values.get(direc, {}).get(word, {})

    def has_poss_val(self, direc: str, word: str, idx: int) -> bool:
        return idx in self._get_poss_word_dict(direc, word)

    def get_poss_val(self, direc: str, word: str, idx: int) -> Optional[bool]:
        return self._get_poss_word_dict(direc, word).get(idx) # ''?

    def set_poss_val(self, direc: str, word: str, idx: int, is_blank: bool = False) -> bool:
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


    def get_points(
            self, nodes: List['Node'], direc: str, word: str, new_words: Optional[List[Dict]] = None, **_kw
    ) -> int:  # todo combine this optional thing into one type

        if _kw:
            lo.e(f'Extra kw args: {_kw}')

        pts = self._points_from_nodes(nodes, direc, word)
        if new_words:
            for nw_dict in new_words:
                new_pts = self._points_from_nodes(**{k: v for k, v in nw_dict.items() if k != 'idx'})
                pts += new_pts


        if len([x for x in nodes if x.value is None]) == 7:  # todo: or is it all letters?
            pts += 35

        return pts

    @staticmethod
    def _points_from_nodes(nodes: List['Node'], direc: str, word: str, **_kw) -> int:
        if _kw:
            lo.e(f'Extra kw args: {_kw}')
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

    def _letter_points(self, direc: str, word: str, idx: int) -> int:
        if self.points: #unnec
            return self.points

        is_blank = self.get_poss_val(direc, word, idx) # i guess i could use has_poss here
        if is_blank is not None:
            if is_blank:
                return 0

            pp = POINTS.get(word[idx])[1]
            if self.multiplier:
                mval = self.multiplier[0]
                mtyp = self.multiplier[1]
                if mtyp == 'l':
                    pp *= mval
            return pp

        lo.e(f'Node has no points or poss points for ({direc}, {word}, {idx})')
        lo.w(f'node -> {self}')
        #should probably raise bigger error here

        return 0


    def get_other_nodes(self, check_typ: str) -> Tuple[List['Node'], List['Node']]:
        if check_typ == 'row':
            nodes = self.row
            slice_val = self.y
        elif check_typ == 'col':
            nodes = self.col
            slice_val = self.x
        else:
            raise Exception('Incorrect check_typ: {}'.format(check_typ))

        bef = nodes[:slice_val]
        aft = nodes[slice_val+1:]

        return bef, aft

    def get_other_word_nodes(
            self, check_typ: str = None, nodes: Tuple[List['Node'], List['Node']] = None
    ) -> Tuple[List['Node'], List['Node']]:

        if nodes is None:
            if check_typ is None:
                raise Exception('check_typ required with no nodes given')
            bef, aft = self.get_other_nodes(check_typ)
        else:
            bef, aft = nodes

        nodes_bef: List[Node] = []
        nodes_aft: List[Node] = []

        for n in reversed(bef):
            if not n.value:
                break
            else:
                nodes_bef.append(n)

        nodes_bef.reverse()

        for n in aft:
            if not n.value:
                break
            else:
                nodes_aft.append(n)

        return nodes_bef, nodes_aft


    def __str__(self):
        return '<Node>: ({},{}) v: "{}"'.format(self.x, self.y, self.value)

    def __repr__(self):
        return self.__str__()


class Word: #todo
    def __init__(self, nodes: List[Node]):
        self.nodes = nodes



class Board:
    def __init__(self, board_mults = DEFAULT_BOARD, game_board = BOARD) -> None:
        self.default_board = board_mults
        self.board = game_board

        self.new_game = False

        self.nodes: List[Node] = []

        bvals: List[List[Optional[str]]] = self.board.values

        for board_r in self.default_board.iterrows():
            r_num: int = board_r[0]
            r_data: List[Optional[str]] = board_r[1]
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

        for n in self.nodes:
            n.up = self.get(n.x - 1, n.y)
            n.left = self.get(n.x, n.y - 1)
            n.right = self.get(n.x, n.y + 1)
            n.down = self.get(n.x + 1, n.y)
            n.row = self.get_row(n.x)
            n.col = self.get_col(n.y)

    def get(self, x: int, y: int) -> Optional[Node]:
        f = filter(lambda obj: obj.x == x and obj.y == y, self.nodes)
        return next(f, None)

    def get_by_attr(self, attr: str, v: Any) -> Iterator:
        return filter(lambda obj: getattr(obj, attr) == v, self.nodes)

    def get_row(self, x: int) -> List[Node]:
        return list(self.get_by_attr('x', x))

    def get_col(self, y: int) -> List[Node]:
        return list(self.get_by_attr('y', y))


def get_word(nodes: List[Node], direc: str = '', word: str = '', idx: int = 0) -> str:
    return ''.join([
        nw.value or
        (word[idx] if nw.has_poss_val(direc, word, idx) else None) or
        '+' for nw in nodes
    ])

#@profile
def check_and_get(
        node_list: List[Node], direc: str, chk_dir: str, word: str, word_len: int, start: int
) -> Optional[List[Dict]]:
    #todo add before method

    lo.d(start)

    bef_idx = start - 1
    if bef_idx >= 0:
        node_bef = node_list[bef_idx]
        if node_bef and node_bef.value:
            lo.d(f'Prior node exists for {direc} at {start}, skipping.')
            return None

    aft_idx = start + word_len
    if aft_idx < SHAPE[direc]:
        node_aft = node_list[aft_idx]
        if node_aft and node_aft.value:
            lo.d(f'Following node exists for {direc}, skipping.')
            return None


    ls = list(LETTERS)
    blanks = BLANKS

    new_word_list: List[Dict] = []

    set_poss_list = []

    for i in range(word_len):
        pos = start + i

        try:
            no = node_list[pos]
        except IndexError:
            lo.e(f'no space {direc} at {start}+{i} for {word}, {len(node_list)}')
            # todo should never happen
            return None

        le = word[i]
        nval = no.value
        if nval:
            if nval != le:
                lo.d(f'mismatch {nval} vs {le}')
                return None

        else:
            is_blank = False

            if le not in ls:
                if blanks > 0:
                    ls.remove('?')
                    blanks -= 1
                    is_blank = True
                else:
                    lo.d(f'missing letter {le}')
                    return None
            else:
                ls.remove(le)

            # set_success = no.set_poss_val(direc, word, i, is_blank)
            # if not set_success:
            #     #todo: i can probably skip stuff now
            #     lo.d('oops')
            set_poss_list.append((no, (direc, word, i, is_blank)))

            bef_nodes, aft_nodes = no.get_other_word_nodes(chk_dir)

            if bef_nodes or aft_nodes:
                bef_word = get_word(bef_nodes) # should not need other args
                aft_word = get_word(aft_nodes)

                new_word = bef_word + le + aft_word

                if new_word not in WORDS:
                    lo.d(f'invalid new word {new_word}')
                    return None

                new_idx = len(bef_word)

                # set_success = no.set_poss_val(chk_dir, new_word, new_idx, is_blank)
                # if not set_success:
                #     #todo: i can probably skip stuff now
                #     lo.d('oops')
                set_poss_list.append((no, (chk_dir, new_word, new_idx, is_blank)))

                new_word_list.append({
                    'nodes': bef_nodes + [no] + aft_nodes,
                    'word': new_word,
                    'direc': chk_dir,
                    'idx': new_idx
                })

    for p in set_poss_list:
        pno, pda = p
        set_success = pno.set_poss_val(*pda)
        if not set_success:
            #todo: i can probably skip stuff now
            lo.d('oops')

    return new_word_list

#@profile
def can_spell(no: Node, word: str, direc: str) -> List[Dict]:
    lo.v(f'{direc} - {word}')

    if direc == 'row':
        idx = no.y
        node_list = no.row
        chk_dir = 'col'

    elif direc == 'col':
        idx = no.x
        node_list = no.col
        chk_dir = 'row'
    else:
        raise TypeError('Direction needs to be "row" or "col"')

    spell_words: List[Dict] = []

    dir_len = SHAPE[direc]
    word_len = len(word)

    start = max(0, idx - word_len + 1)
    while (start + word_len) <= dir_len and start <= idx:
        new_words = check_and_get(node_list, direc, chk_dir, word, word_len, start)
        if new_words is not None:
            new_d = {
                'nodes': node_list[start:start+word_len],
                'word': word,
                'direc': direc,
                'idx': idx - start,
                'new_words': new_words
            }
            #todo: insert reverse too?
            spell_words.append(new_d)
        start += 1

    return spell_words


def check_words(w: str, no: Node) -> List[dict]:
    data = []

    for direction in ('row', 'col'):
        spell_words = can_spell(no, w, direction)

        if spell_words:
            lo.w('--> YES: %s: %s', direction, w)
            # combine words with idxs?

            for word_dict in spell_words:
                tot_pts = no.get_points(**{k: v for k, v in word_dict.items() if k != 'idx'})
                # todo, could do this beforehand

                new_info ={
                    'pts': tot_pts,
                    'cell_letter': w[word_dict['idx']],
                    'cell_pos': (no.x, no.y),
                    'cell': no
                }

                this_info = {**word_dict, **new_info}

                lo.w(this_info)
                data.append(this_info)

    return data

def run_worker(data: Tuple[str, Node]):
    w = data[0]
    no = data[1]

    if len(w) > 8:
        word_name = w[:7] + '.'
    else:
        word_name = w

    current = current_process()
    newname = 'w %s' % word_name
    current.name = newname

    return check_words(w, no)


def check_node(no: Optional[Node]):
    if not no: return

    if lo.is_enabled('s'):
        print()
        lo.s('checking...\n')

    if USE_POOL:  #todo: is this fixable for profiling?
        n = max(cpu_count() - 1, 1)
        #n = 2
        pool = Pool(n)
        pool_res: List[List[dict]] = pool.map(run_worker, ((w, no) for w in SEARCH_WORDS))

        for x in pool_res:
            for y in x:
                word_info.append(y)
    
    else:
        for w in SEARCH_WORDS:
            reses = check_words(w, no)
            for res in reses:
                word_info.append(res)


def set_search_words():
    global SEARCH_WORDS

    if SEARCH_WORDS: return

    if BLANKS:
        SEARCH_WORDS = WORDS
    else:
        # todo: faster if str? why gen slower?
        SEARCH_WORDS = {
            w for w in WORDS if any([l in w for l in LETTERS])
        }


# todo: nodelist needs main
# [{'word': 'xyz', 'nodes': [nodelist], 'pts': 0}]
word_info: List[Dict[str, Any]] = []

def main() -> None:
    start = timer()

    full = Board()

    set_search_words()

    if full.new_game:
        lo.s(' = Fresh game = ')
        no = next(full.get_by_attr('is_start', True), None)
        if no:
            lo.s('**** Node (%2s, %2s: %1s) - #1 / 1',
                 no.x, no.y, no.value or '_'
            )
            check_node(no)

    else:
        # -- set search nodes

        full_nodes = full.nodes

        #full_nodes = full.get_row(0)
        #full_nodes = full.get_row(0) + full.get_row(1) + full.get_row(2)

        # full_nodes = [
        #     full.get(6, 1),
        #     full.get(7, 1),
        # ]

        # --

        full_nodes_len = len(full_nodes)
        for no in full_nodes:
            if no:
                lo.s('**** Node (%2s, %2s: %1s) - #%s / %s',
                     no.x, no.y, no.value or '_', full_nodes.index(no) + 1, full_nodes_len
                )
                if not no.value and no.has_edge():
                    check_node(no)

    if not word_info:
        print('No solution.')
    else:
        newlist = sorted(word_info, key=lambda k: k['pts'], reverse=True)
        best = newlist[:1]

        if lo.is_enabled('s'):
            print('=========')

            for s in newlist[:10][::-1]:
                lo.s('\n{}'.format(pformat(s)))

            print('---------')

            lo.s('\n{}'.format(pformat(best)))

        solved_board = BOARD.copy()
        best_data = best[0]

        for ni, node in enumerate(best_data.get('nodes', [])):
            if not node.value:
                solved_board[node.y][node.x] = '\x1b[33m' + best_data['word'][ni] + '\x1b[0m'

        print('\n' + '-' * ((SHAPE['row']*2) -1 + 4))

        for row in solved_board.iterrows():
            row_data = row[1].to_list()
            print('| ' + ' '.join(rl.upper() if len(rl) == 1 else rl for rl in row_data) + ' |')

        print('-' * ((SHAPE['row']*2) -1 + 4))

        print(f'\nPoints: {best_data["pts"]}')

    end = timer()
    print('\nTime: {}'.format(round(end - start, 1)))

if __name__ == '__main__':
    main()
