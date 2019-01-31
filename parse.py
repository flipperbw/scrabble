import json
import multiprocessing as mp
from pprint import pformat
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional, Tuple, Iterator

import pandas as pd
import regex

from utils import log_init

# -- TODOS

#todo: profiling
#combine stuffz
#change return [] to yields?
#if word was found in a different check, skip

# --
# -- GLOBALS

# - logging

#LOG_LEVEL = 'DEBUG'
#LOG_LEVEL = 'INFO'
#LOG_LEVEL = 'WARNING'
LOG_LEVEL = 'SUCCESS'

# - pool

USE_POOL = True

# - default board

#todo small vs big board
DEFAULT_BOARD = pd.read_pickle('data/default_board.pkl')

#small board
'''
DEFAULT_BOARD =pd.DataFrame([
    ['3l','','3w'] + ['']*5 + ['3w','','3l'],
    ['','2w','','','','2w','','','','2w',''],
    ['3w','','2l','','2l','','2l','','2l','','3w'],
    ['','','','3l','','','','3l','','',''],
    ['','','2l','','','','','','2l','',''],

    ['','2w','','','','x','','','','2w',''],

    ['', '', '2l', '', '', '', '', '', '2l', '', ''],
    ['', '', '', '3l', '', '', '', '3l', '', '', ''],
    ['3w', '', '2l', '', '2l', '', '2l', '', '2l', '', '3w'],
    ['', '2w', '', '', '', '2w', '', '', '', '2w', ''],
    ['3l', '', '3w'] + [''] * 5 + ['3w', '', '3l']
])
'''

# - game board

#BOARD = pd.read_pickle('data/board.pkl')

''' # old1
BOARD = pd.DataFrame([
    ['']*11,
    ['']*11,
    ['']*11,
    ['']*3 + ['J'] + ['']*7,
    #[''] + ['X'] + [''] + ['J'] + [''] + ['Y'] + ['']*5,
    ['']*3 + ['O'] + ['']*7,
    ['']*2 + ['P', 'E', 'R', 'V', 'S'] + ['']*4,
    [''] + ['M', 'A', 'S'] + ['', ''] + ['E'] + ['']*4,
    ['']*6 + ['E', 'W'] + ['']*3,
    ['']*6 + ['K', 'A', 'F'] + [''] + ['Q'],
    ['']*7 + ['Y', 'A', 'G', 'I'],
    ['']*8 + ['X', 'I', 'S']
])
'''

#todo small vs big board
#BOARD = [['' for _ in range(15)] for _ in range(15)]
#BOARD = [['' for _ in range(11)] for _ in range(11)]

'''
BOARD = [ #n1
    [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
    [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
    [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ','C'],
    [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ','H'],
    [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ','g','I'],
    [' ',' ',' ',' ',' ',' ',' ',' ',' ','j',' ',' ',' ','r',' '],
    [' ',' ',' ',' ',' ',' ',' ',' ',' ','o',' ',' ',' ','i','d'],
    [' ',' ',' ',' ',' ','t','h','i','s','t','l','e',' ','n','a'],
    [' ',' ',' ',' ','g','o','a','d',' ','s','i','t','e','d',' '],
    [' ',' ',' ',' ',' ','g','o',' ',' ',' ','t','a','m','e','r'],
    [' ',' ',' ',' ',' ',' ',' ','e','c','r','u',' ',' ','r','e'],
    [' ',' ',' ','f',' ','w','a','x',' ',' ',' ','w','a','s','p'],
    [' ','N','E','O','N','E','D',' ',' ',' ','q','i',' ',' ',' '],
    [' ',' ',' ','i',' ',' ',' ',' ','d','r','u','m','s',' ',' '],
    [' ',' ',' ','l',' ',' ',' ',' ',' ',' ','o','p',' ',' ',' ']
]
'''

#'''
BOARD = [ #n2
    [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
    [' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '],
    [' ',' ',' ',' ',' ',' ',' ',' ','R',' ',' ',' ',' ',' ',' '],
    [' ',' ',' ',' ',' ',' ',' ','Z','A',' ',' ',' ',' ',' ',' '],
    [' ',' ',' ',' ',' ',' ',' ','E','T',' ',' ',' ',' ',' ',' '],
    [' ',' ',' ',' ',' ','c',' ','S','I','R',' ',' ',' ',' ',' '],
    [' ',' ',' ',' ',' ','o',' ','T','O','E',' ',' ',' ',' ',' '],
    [' ',' ',' ','b','e','n','D','S',' ','L',' ',' ',' ',' ',' '],
    ['j','a','n','e',' ','e',' ',' ',' ','A',' ',' ',' ',' ',' '],
    ['i',' ',' ',' ',' ','d','e','t','o','x',' ',' ',' ',' ',' '],
    ['v',' ',' ',' ',' ',' ',' ','o',' ',' ',' ',' ',' ',' ',' '],
    ['e',' ',' ',' ',' ',' ',' ','u',' ',' ',' ',' ',' ',' ',' '],
    ['s','o','l',' ',' ',' ',' ','c',' ','u',' ',' ',' ',' ',' '],
    [' ',' ',' ','w','r','i','g','h','t','s',' ',' ',' ',' ',' '],
    ['a','f','r','o',' ','T','I','E',' ',' ',' ',' ',' ',' ',' ']
]
#'''

'''
BOARD = [ #s1
    list('  C        '),
    list(' fE  c     '),
    list('laMp o     '),
    list(' de  j     '),
    list(' en  o     '),
    list('  TAXIS    '),
    list('     n     '),
    list('glazes     '),
    list('r b        '),
    list('e u        '),
    list('y townish  '),
]
'''
#BOARD = [[' ' for _ in range(11)] for _ in range(11)]

BOARD = pd.DataFrame(BOARD)

# - letters

#LETTERS = open('data/letters.txt', 'r').read().splitlines()
#LETTERS = list('TOTHBYU')
LETTERS = list('UEROYYK')

# - words

WORDS = open('data/wordlist.txt').read().splitlines()
#WORDS = (
#    'ABUT',
#)

# --

POINTS = json.load(open('data/points.json'))

lo = log_init(LOG_LEVEL, skip_main=False)

lo.i('\n{}'.format(DEFAULT_BOARD))
lo.s('\n{}'.format(BOARD))
lo.s('\n{}'.format(LETTERS))
print()


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

        # direc -> word -> idx = letter
        self.poss_values: Dict[str, Dict[str, Dict[int, str]]] = {
            'row': {},
            'col': {}
        }
        self.poss_blank = False  # todo: fix

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

    def get_poss_val(self, direc: str, word: str, idx: int) -> Optional[str]:
        return self.poss_values.get(direc, {}).get(word, {}).get(idx) # ''?

    def set_poss_val(self, direc: str, word: str, idx: int, v: str) -> bool:
        pv_info = self.poss_values[direc]
        if word not in pv_info:
            pv_info[word] = {idx: v}
        else:
            pv_info_word = pv_info[word]
            if idx not in pv_info_word:
                pv_info_word[idx] = v
            else:
                lo.v('Already parsed {} at {}'.format(word, idx))
                return False
        return True

    def poss_points(self, direc: str, word: str, idx: int) -> int:
        val = self.get_poss_val(direc, word, idx)
        if val:
            if self.poss_blank: # todo: fix
                return 0
            pp = POINTS.get(val)[1]
            if self.multiplier:
                mval = self.multiplier[0]
                mtyp = self.multiplier[1]
                if mtyp == 'l':
                    pp *= mval
        elif not self.points:
            lo.e('Node has no points or poss points')
            lo.v('%s %s %s', direc, word, idx)
            lo.e(self)
            pp = 0
        else:
            pp = self.points
        return pp

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

    def get_word_nodes(
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

        for l in bef[::-1]:
            if not l.value:
                break
            else:
                nodes_bef.insert(0, l)

        for l in aft:
            if not l.value:
                break
            else:
                nodes_aft.append(l)

        return nodes_bef, nodes_aft


    def __str__(self):
        return '<Node>: ({},{}) v: "{}" p: "{}"'.format(self.x, self.y, self.value, self.poss_values)

    def __repr__(self):
        return self.__str__()


class Board:
    def __init__(self, board_mults: pd.DataFrame = DEFAULT_BOARD, game_board: pd.DataFrame = BOARD):
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


def get_word(nodes: List[Node], direc: str, word: str, idx: int) -> str:
    return ''.join([nw.value or nw.get_poss_val(direc, word, idx) or '' for nw in nodes])


def check_extra(word_lets: List[str], board_lets: List[Node]) -> Optional[List[str]]:
    word_lets_len = len(word_lets)
    lo.d(word_lets)
    lo.d(board_lets)

    if not word_lets_len:
        return word_lets

    for i, b in enumerate(board_lets):
        bval = b.value
        if i == word_lets_len:
            if bval:
                lo.i('failed on new word')
                return None
            else:
                return word_lets
        elif bval:
            nl_val = word_lets[i]
            if nl_val != '_' and bval != nl_val: # todo: ?
                lo.i('failed on mismatch')
                return None
            else:
                word_lets[i] = '_'
                lo.i('subbing letter {} for {}'.format(bval, nl_val))

    return word_lets

def can_spell(info: Dict[str, Any], direc: str, word: str) -> List[int]:
    lo.i('---')
    lo.i(word)
    lo.i(direc)

    word_bef: str = info['word_bef']
    word_aft: str = info['word_aft']

    lo.d(word_bef)
    lo.d(word_aft)

    extra_bef: List[Node] = info['other_nodes_bef']
    extra_bef_bwrds: List[Node] = info['other_nodes_bef_bwrds']
    extra_aft: List[Node] = info['other_nodes_aft']

    extra_bef_len = info['other_nodes_bef_len']
    extra_aft_len = info['other_nodes_aft_len']

    results = []

    reg = info['reg']

    rr = reg.finditer(word, overlapped=True)
    for r in rr:
        new_word_list = list(word)

        match_st, match_en = r.span()
        match_bef, match_aft = r.groups()

        lo.d('%s,%s', match_st, match_en)
        lo.d('%s,%s', match_bef, match_aft)

        match_bef_len = len(match_bef)
        match_aft_len = len(match_aft)

        idx = match_st
        if match_bef:
            idx += match_bef_len
        
        new_word_list[match_st:idx] = '_' * match_bef_len
        new_word_list[idx+1:match_en] = '_' * match_aft_len

        lo.d(new_word_list)

        new_letters_bef = new_word_list[:match_st]
        #new_letters_mid = new_word_list[match_st:match_en]
        new_letters_mid = [new_word_list[idx]]
        #new_letters_aft = new_word_list[match_en:]
        new_letters_aft = new_word_list[idx + 1:]

        if lo.enabled_levels[10]:
            lo.d([v.value or '_' for v in extra_bef]) # lambda?
            lo.d([v.value or '_' for v in extra_aft])
            lo.d(new_letters_bef)
            lo.d(new_letters_mid)
            lo.d(new_letters_aft)

        if extra_bef_len < len(new_letters_bef):
            lo.i('not enough space before')
            continue
        if extra_aft_len < len(new_letters_aft):
            lo.i('not enough space after')
            continue

        lo.d('extra bef')
        extra_letters_bef = check_extra(new_letters_bef[::-1], extra_bef_bwrds)
        if extra_letters_bef is None:
            continue
        else:
            extra_letters_bef = extra_letters_bef[::-1]

        lo.d('extra aft')
        extra_letters_aft = check_extra(new_letters_aft, extra_aft)
        if extra_letters_aft is None:
            continue

        new_word_list = extra_letters_bef + new_letters_mid + extra_letters_aft
        lo.d(new_word_list)

        cut_word_list = [v for v in new_word_list if v != '_']

        if len(cut_word_list) == 0:
            lo.i('cant use any letters')
            continue

        has_blank = 0

        for letter in LETTERS:
            if len(cut_word_list) == 0:
                break
            elif letter == '?':
                has_blank += 1
            elif letter in cut_word_list:
                cut_word_list.remove(letter)

        for _ in range(has_blank):
            if len(cut_word_list):
                cut_word_list.pop()

        if len(cut_word_list) == 0:
            results.append(idx)
        else:
            lo.d('reject')

    return results

chk_dir_dict = {
    'row': {
        'bef_dir': 'left',
        'aft_dir': 'right',
        'bef_chk': 'up',
        'aft_chk': 'down',
        'chk_typ': 'col',
    },
    'col': {
        'bef_dir': 'up',
        'aft_dir': 'down',
        'bef_chk': 'left',
        'aft_chk': 'right',
        'chk_typ': 'row'
    }
}

def _get_side_words(
        this_cell: Node,
        bef_letters: List[str],
        aft_letters: List[str],
        typ: str,
        word: str,
        n_idx: int,
        chk_typs: dict
) -> List[List[Node]]:

    word_list = []
    w_bef_list: List[Node] = []
    w_aft_list: List[Node] = []

    bef_dir = chk_typs['bef_dir']
    aft_dir = chk_typs['aft_dir']
    bef_chk = chk_typs['bef_chk']
    aft_chk = chk_typs['aft_chk']
    chk_typ = chk_typs['chk_typ']

    #todo: only if dirs are both blank i guess
    #break this into new fnc

    cell_bef: Node = getattr(this_cell, bef_chk)
    cell_aft: Node = getattr(this_cell, aft_chk)

    bef_nodes: List[Node] = []
    aft_nodes: List[Node] = []

    cell_word_nodes = this_cell.get_word_nodes(chk_typ)

    if cell_bef and cell_bef.value:
        bef_nodes = cell_word_nodes[0]
        oldbefword = get_word(bef_nodes, typ, word, n_idx)
        lo.i(bef_nodes)
        lo.i(oldbefword)

    if cell_aft and cell_aft.value:
        aft_nodes = cell_word_nodes[1]
        oldaftword = get_word(aft_nodes, typ, word, n_idx)
        lo.i(aft_nodes)
        lo.i(oldaftword)

    if bef_nodes or aft_nodes:
        new_nodes = bef_nodes + [this_cell] + aft_nodes
        newword = get_word(new_nodes, typ, word, n_idx)

        if newword:
            if newword not in WORDS:
                lo.i('New word not valid: %s', newword)
                return []

            lo.i(newword)
            lo.i(new_nodes)
            word_list.append(new_nodes)

    for side in (bef_dir, aft_dir):
        cell: Node = this_cell

        if side == bef_dir:
            chk_letters = bef_letters
        else:
            chk_letters = aft_letters

        for let_v in chk_letters:
            cell = getattr(cell, side)
            if not cell:  # these should never happen
                lo.w('No cell to check')
                return []

            if cell.value:
                if cell.value != let_v:
                    lo.w('Cell val does not match')
                    return []
            else:
                pos_res = cell.set_poss_val(typ, word, n_idx, let_v) #todo
                if not pos_res:
                    lo.w('we already found this, skipping')
                    return []

                cell_bef = getattr(cell, bef_chk)
                cell_aft = getattr(cell, aft_chk)

                bef_nodes = []
                aft_nodes = []

                cell_word_nodes = cell.get_word_nodes(chk_typ)

                if cell_bef and cell_bef.value:
                    bef_nodes = cell_word_nodes[0]
                    oldbefword = get_word(bef_nodes, typ, word, n_idx)
                    lo.i(bef_nodes)
                    lo.i(oldbefword)

                if cell_aft and cell_aft.value:
                    aft_nodes = cell_word_nodes[1]
                    oldaftword = get_word(aft_nodes, typ, word, n_idx)
                    lo.i(aft_nodes)
                    lo.i(oldaftword)

                if bef_nodes or aft_nodes:
                    new_nodes = bef_nodes + [cell] + aft_nodes
                    newword = get_word(new_nodes, typ, word, n_idx)

                    if newword:
                        if newword not in WORDS:
                            lo.w('New word not valid: %s', newword)
                            return []

                        lo.i(newword)
                        word_list.append(new_nodes)

            if side == bef_dir:
                w_bef_list.insert(0, cell)
            else:
                w_aft_list.append(cell)

    combined_word = w_bef_list + [this_cell] + w_aft_list
    word_list.append(combined_word)

    return word_list


def is_valid(typ: str, this_cell: Node, n_idxs: List[int], word: str) -> List[Tuple[int, List[List[Node]]]]:
    lo.i(word)
    word_l = list(word)

    full_list = []

    chk_typs = chk_dir_dict.get(typ)
    if not chk_typs:
        raise Exception('Incorrect type for valid check: {}'.format(typ))

    for n_idx in n_idxs:
        lo.i(n_idx)

        bef_letters = word_l[:n_idx][::-1]
        cur_letter  = word_l[n_idx]
        aft_letters = word_l[n_idx + 1:]

        lo.d(bef_letters[::-1])
        lo.d(cur_letter)
        lo.d(aft_letters)

        pos_res = this_cell.set_poss_val(typ, word, n_idx, cur_letter) # todo
        if not pos_res:
            continue

        side_words = _get_side_words(this_cell, bef_letters, aft_letters, typ, word, n_idx, chk_typs)

        if side_words:
            full_list.append((n_idx, side_words))

    return full_list


def word_points(nodelist: List[Node], direc: str, word: str, idx: int) -> int:
    let_pts = sum([n.poss_points(direc, word, idx) for n in nodelist])
    mults = [n.multiplier[0] for n in nodelist if not n.value and n.multiplier and n.multiplier[1] == 'w']
    word_mult = 1
    for m in mults:
        word_mult *= m
    let_pts *= word_mult

    if len([x for x in nodelist if x.value is None]) == 7:  # todo: or is it all letters?
        let_pts += 35

    return let_pts


def check_words(w: str, no: Node, node_info: Dict[str, Dict[str, Any]]) -> Optional[dict]:
    this_info = None

    if '?' in LETTERS or any([l in w for l in LETTERS]):  # todo: faster if str? why gen slower?
        for direction in ('row', 'col'):
            spell_words = can_spell(node_info[direction], direction, w)

            if spell_words:
                lo.w('-> canspell: {}: {} - {}'.format(direction, spell_words, w))

                made_words = is_valid(direction, no, spell_words, w)

                if made_words:
                    lo.w('--> YES: %s: %s', direction, w)
                    # combine words with idxs?
                    for ro in made_words:
                        tot_pts = 0
                        d_idx = ro[0]
                        data = ro[1]
                        for d in data:
                            pts = word_points(d, direction, w, d_idx)
                            tot_pts += pts

                        this_info = {
                            'word': w,
                            'pts': tot_pts,
                            'direction': direction,
                            'cell_letter': w[d_idx],
                            'word_idx': d_idx,
                            'cell_pos': (no.x, no.y),
                            'cell': no,
                            'words': [get_word(da, direction, w, d_idx) for da in data],
                            'nodes': data
                        }

                        lo.w(this_info)
                        word_info.append(this_info)
    return this_info

def run_worker(data: Tuple[str, Node, Dict[str, Dict[str, Any]]]):
    w = data[0]
    no = data[1]
    node_info = data[2]

    if len(w) > 8:
        word_name = w[:7] + '.'
    else:
        word_name = w

    current = mp.current_process()
    newname = 'w %s' % word_name
    current.name = newname

    return check_words(w, no, node_info)


def check_node(no: Optional[Node]):
    if not no: return

    print()
    lo.s('checking...\n')

    node_info = {}
    for t in ('row', 'col'):
        node_nodes = no.get_other_nodes(t)
        node_words = no.get_word_nodes(nodes=node_nodes)

        node_info[t] = {
            'other_nodes_bef': node_nodes[0],
            'other_nodes_bef_bwrds': node_nodes[0][::-1],
            'other_nodes_aft': node_nodes[1],
            'other_nodes_bef_len': len(node_nodes[0]),
            'other_nodes_aft_len': len(node_nodes[1]),
            #'word_nodes_bef': node_words[0],
            #'word_nodes_aft': node_words[1],
            'word_bef': get_word(node_words[0], '', '', -1),
            'word_aft': get_word(node_words[1], '', '', -1),
        }

        lo.d(node_info[t]['word_bef'])
        lo.d(node_info[t]['word_aft'])

        node_info[t]['reg'] = regex.compile(r'({}).({})'.format(
            node_info[t]['word_bef'], node_info[t]['word_aft'])
        )

    if USE_POOL:  #todo: is this fixable for profiling?
        n = max(mp.cpu_count() - 1, 1)
        #n = 2
        pool = mp.Pool(n)
        pool_res = pool.map(run_worker, ((w, no, node_info) for w in WORDS))

        for x in pool_res:
            if x:
                word_info.append(x)
    else:
        for w in WORDS:
            res = check_words(w, no, node_info)
            if res:
                word_info.append(res)

# todo: nodelist needs main
# [{'word': 'xyz', 'nodes': [nodelist], 'pts': 0}]
word_info: List[Dict[str, Any]] = []

def main() -> None:
    start = timer()

    full = Board()

    if full.new_game:
        lo.s(' = Fresh game = ')
        no = next(full.get_by_attr('is_start', True), None)
        if no:
            lo.s('**** Node (%2s, %2s: %1s) - #1 / 1',
                 no.x, no.y, no.value or '_'
            )
            check_node(no)

    else:
        # -- specify nodes

        full_nodes = full.nodes
        #full_nodes = full.get_row(6)

        #full_nodes = []
        #full_nodes.append(full.get(8,2))

        # --

        for no in full_nodes:
            if no:
                lo.s('**** Node (%2s, %2s: %1s) - #%s / %s',
                     no.x, no.y, no.value or '_', full_nodes.index(no) + 1, len(full_nodes)
                )
                if not no.value and no.has_edge():
                    check_node(no)

    #todo print completed board with colors
    print('=========')
    if word_info:
        newlist = sorted(word_info, key=lambda k: k['pts'], reverse=True)

        for s in newlist[:10][::-1]:
            lo.s('\n{}'.format(pformat(s)))

        print('---')
        best = newlist[:1]
        lo.s('\n{}'.format(pformat(best)))

    end = timer()
    print('\nTime: {}'.format(round(end - start, 1)))

if __name__ == '__main__':
    main()
