import json
from utils import log_init
from pprint import pformat
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import regex

#todo: profiling
#check for unused
#combine stuffz
#change return [] to yields?

#log_level = 'DEBUG'
#log_level = 'INFO'
#log_level = 'WARNING'
log_level = 'SUCCESS'

lo = log_init(log_level)

words = open('data/wordlist.txt', 'r').read().splitlines()
points = json.load(open('data/points.json', 'r'))

#board = pd.read_pickle('data/board.pkl')
#letters = open('data/letters.txt', 'r').read().splitlines()

#todo small vs big board
default_board = pd.read_pickle('data/default_board.pkl')
# small board
# default_board =pd.DataFrame([
#     ['3l','','3w'] + ['']*5 + ['3w','','3l'],
#     ['','2w','','','','2w','','','','2w',''],
#     ['3w','','2l','','2l','','2l','','2l','','3w'],
#     ['','','','3l','','','','3l','','',''],
#     ['','','2l','','','','','','2l','',''],
#
#     ['','2w','','','','x','','','','2w',''],
#
#     ['', '', '2l', '', '', '', '', '', '2l', '', ''],
#     ['', '', '', '3l', '', '', '', '3l', '', '', ''],
#     ['3w', '', '2l', '', '2l', '', '2l', '', '2l', '', '3w'],
#     ['', '2w', '', '', '', '2w', '', '', '', '2w', ''],
#     ['3l', '', '3w'] + [''] * 5 + ['3w', '', '3l']
# ])

lo.i('\n{}'.format(default_board))

''' # old1
board = pd.DataFrame([
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
board = [['' for _ in range(15)] for _ in range(15)]
#board = [['' for _ in range(11)] for _ in range(11)]

#old2
board[2][14] = 'C'

board[3][14] = 'H'
board[4][13:15] = 'GI'

board[5][9] = 'J'
board[5][13] = 'R'

board[6][9] = 'O'
board[6][13:15] = 'ID'

board[7][5:12] = 'THISTLE'
board[7][13:15] = 'NA'

board[8][4:8] = 'GOAD'
board[8][9:14] = 'SITED'

board[9][5:7] = 'GO'
board[9][10:15] = 'TAMER'

board[10][7:11] = 'ECRU'
board[10][13:15] = 'RE'

board[11][5:8] = 'WAX'
board[11][11:15] = 'WASP'

board[12][11] = 'I'
board[13][11] = 'M'
board[14][11] = 'P'

'''
#old3
board[3][7] = 'Z'
board[4][7] = 'E'
board[5][7] = 'S'
board[6][7] = 'T'
board[7][7] = 'S'

board[9][7]  = 'T'
board[10][7] = 'O'
board[11][7] = 'U'
board[12][7] = 'C'
board[13][7] = 'H'

board[5][5] = 'C'
board[6][5] = 'O'
board[7][5] = 'N'
board[8][5] = 'E'
board[9][5] = 'D'

board[7][3:8] = 'BENDS'

board[8][0:4] = 'JANE'

board[9][0] = 'I'
board[9][5:10] = 'DETOX'

board[10][0] = 'V'
board[11][0] = 'E'
board[12][0:3] = 'SOL'

board[12][7] = 'C'
board[12][9] = 'U'

board[13][3:10] = 'WRIGHTS'
'''

board = pd.DataFrame(board)

lo.s('\n{}'.format(board))

#letters = ['A', 'E', 'Z', 'L', 'D', '?', 'E'] # old1
letters = list('LT QUO WSA') # old2
#letters = list('UOAFEYR') # old3
lo.s('\n{}'.format(letters))
print()

class Node:
    def __init__(self, x, y, multiplier=None, value=None):
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
            self.points = points.get(self.value)[1]

        # direc -> word -> idx = letter
        self.poss_values: Dict[Optional[str], Dict[Optional[str], Dict[Optional[int], str]]] = None
        self.clear_poss_val()
        self.poss_blank = False  # todo: fix

        self.up = None
        self.left = None
        self.down = None
        self.right = None
        self.row: List['Node'] = None
        self.col: List['Node'] = None

    def has_edge(self) -> bool:
        return (self.right and self.right.value) or\
               (self.left and self.left.value) or\
               (self.up and self.up.value) or\
               (self.down and self.down.value)

    def clear_poss_val(self):
        self.poss_values = {
            'row': {},
            'col': {},
        }

    def get_poss_val(self, direc: str = None, word: str = None, idx: int = None) -> Optional[str]:
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
                #todo: Node (12, 7) old2,
                #lo.e('COLLISION! {} to {} for {}:{}'.format(pv_info_word[idx], v, word, idx))
                #we already parsed this word
                return False
        return True

    def poss_points(self, direc: str, word: str, idx: int) -> int:
        val = self.get_poss_val(direc, word, idx)
        if val:
            if self.poss_blank: # todo: fix
                return 0
            pp = points.get(val)[1]
            if self.multiplier:
                mval = self.multiplier[0]
                mtyp = self.multiplier[1]
                if mtyp == 'l':
                    pp *= mval
        elif not self.points:
            lo.e('ERROR: node has no points or poss points')
            lo.v('%s %s %s', direc, word, idx)
            lo.e(self)
            pp = 0
        else:
            pp = self.points
        return pp

    #def row_vals(self):
    #    return [rr.value for rr in self.row]

    #def col_vals(self):
    #    return [cc.value for cc in self.col]

    def get_other_nodes(self, check_typ) -> Tuple[List['Node'], List['Node']]:
        if check_typ == 'row':
            nodes = self.row
            bef = nodes[:self.y]
            aft = nodes[self.y+1:]
        elif check_typ == 'col':
            nodes = self.col
            bef = nodes[:self.x]
            aft = nodes[self.x+1:]
        else:
            raise Exception('Incorrect check_typ: {}'.format(check_typ))

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
    def __init__(self):
        self.default_board = default_board
        self.board = board

        self.nodes: List[Node] = []

        self.setup()

    def setup(self):
        bvals = board.values

        for board_r in default_board.iterrows():
            r_num = board_r[0]
            r_data = board_r[1]
            b_row = bvals[r_num]
            for c_num, cell in enumerate(r_data):
                board_val = b_row[c_num]

                if not board_val:
                    board_val = None
                if not cell:
                    cell = None

                node = Node(r_num, c_num, cell, board_val)
                self.nodes.append(node)

        for n in self.nodes:
            n.up = self.get(n.x - 1, n.y)
            n.left = self.get(n.x, n.y - 1)
            n.right = self.get(n.x, n.y + 1)
            n.down = self.get(n.x + 1, n.y)
            n.row = self.get_row(n.x)
            n.col = self.get_col(n.y)

    def get(self, x, y) -> Optional[Node]:
        f = filter(lambda obj: obj.x == x and obj.y == y, self.nodes)
        return next(f, None)

    def get_row(self, x) -> List[Node]:
        return list(filter(lambda obj: obj.x == x, self.nodes))

    def get_col(self, y) -> List[Node]:
        return list(filter(lambda obj: obj.y == y, self.nodes))


def get_word(nodes: List[Node], direc: str = None, word: str = None, idx: int = None) -> str:
    return ''.join([nw.value or nw.get_poss_val(direc, word, idx) or '' for nw in nodes])


def check_extra(word_lets: List[str], board_lets: List[Node]) -> Optional[List[str]]:
    word_lets_len = len(word_lets)
    lo.d(word_lets)
    lo.d(board_lets)

    for i, b in enumerate(board_lets):
        bval = b.value
        if i >= word_lets_len:
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

def can_spell(
        info: Dict[str, Any],
        direc: str,
        word: str
) -> List[int]:
    lo.i('---')
    lo.i(word)
    lo.i(direc)

    #bef_nodes = info['word_nodes_bef']
    #aft_nodes = info['word_nodes_aft']
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

        if word == 'VOODOOS':
            lo.i(cut_word_list)

        for letter in letters:
            if len(cut_word_list) == 0:
                break
            elif letter == '?':
                has_blank += 1
            elif letter in cut_word_list:
                cut_word_list.remove(letter)

        for blank in range(has_blank):
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
            if newword not in words:
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
                        if newword not in words:
                            lo.w('New word not valid: %s', newword)
                            return []

                        lo.w(newword)
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

# todo: nodelist needs main
# [{'word': 'xyz', 'nodes': [nodelist], 'pts': 0}]
word_info: List[Dict[str, Any]] = []

def main():
    full = Board()

    full_nodes = full.nodes
    #full_nodes = full.get_row(7)

    # full_nodes = [
    #    full.get(4,8),
    #    full.get(6,0),
    #    full.get(8,9),
    #    full.get(2,3)
    # ]

    for no in full_nodes:
        print()
        lo.s('**** Node (%s, %s) - #%s / %s\n', no.x, no.y, full_nodes.index(no) + 1, len(full_nodes))
        if no and not no.value and no.has_edge():
            lo.s('checking...')

            node_row_nodes = no.get_other_nodes('row')
            node_col_nodes = no.get_other_nodes('col')
            node_row_words = no.get_word_nodes(nodes=node_row_nodes)
            node_col_words = no.get_word_nodes(nodes=node_col_nodes)

            node_info = {
                'row': {
                    'other_nodes_bef': node_row_nodes[0],
                    'other_nodes_bef_bwrds': node_row_nodes[0][::-1],
                    'other_nodes_aft': node_row_nodes[1],
                    'other_nodes_bef_len': len(node_row_nodes[0]),
                    'other_nodes_aft_len': len(node_row_nodes[1]),
                    #'word_nodes_bef': node_row_words[0],
                    #'word_nodes_aft': node_row_words[1],
                    'word_bef': get_word(node_row_words[0]),
                    'word_aft': get_word(node_row_words[1]),
                },
                'col': {
                    'other_nodes_bef': node_col_nodes[0],
                    'other_nodes_bef_bwrds': node_col_nodes[0][::-1],
                    'other_nodes_aft': node_col_nodes[1],
                    'other_nodes_bef_len': len(node_col_nodes[0]),
                    'other_nodes_aft_len': len(node_col_nodes[1]),
                    #'word_nodes_bef': node_col_words[0],
                    #'word_nodes_aft': node_col_words[1],
                    'word_bef': get_word(node_col_words[0]),
                    'word_aft': get_word(node_col_words[1]),
                }
            }

            lo.d(node_info['row']['word_bef'])
            lo.d(node_info['row']['word_aft'])
            lo.d(node_info['col']['word_bef'])
            lo.d(node_info['col']['word_aft'])

            node_info['row']['reg'] = regex.compile(r'({}).({})'.format(
                node_info['row']['word_bef'], node_info['row']['word_aft'])
            )
            node_info['col']['reg'] = regex.compile(r'({}).({})'.format(
                node_info['col']['word_bef'], node_info['col']['word_aft'])
            )

            for w in words:
            #for w in (
            #        'AM', 'AS', 'ADE', 'ZZZ',
            #        #'XIJEY', 'EXIJEY', 'EXIJEYZ', 'XIJEYZ', 'XIJE', 'ETIJEYZ', 'TIJEY', 'TIJEYS'
            #        #'AMAS', 'AMASMAS', 'AMADXZCEFE', 'AMASET', 'AMASETE', 'AMASETED', 'AMASETO',
            #        #'KAFE', 'KAFEQ', 'KAFKAFEQKAF', 'KAFIQ', 'AKAFEQ', 'KAFEL', 'AKAF',
            #        #'DEAL', 'DEALT', 'DEALTS'
            #):
                #if w not in ('OUTLAWS', 'AWOL'): continue
                #if w not in ('QUO',): continue #todo not in dictionary?
                #todo: glouts, has uwa and lgo

                if '?' in letters or any([l in w for l in letters]):  #todo: faster if str? why gen slower?
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


    newlist = sorted(word_info, key=lambda k: k['pts'], reverse=True)
    print('=========')
    for s in newlist[:10]:
        lo.s('\n{}'.format(pformat(s)))
    print('---')
    lo.s('\n{}'.format(pformat(newlist[:1])))

if __name__ == '__main__':
    main()
