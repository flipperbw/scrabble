from typing import List, Dict, Any
import pandas as pd
import json

words = open('wordlist.txt', 'r').read().splitlines()
points = json.load(open('points.json', 'r'))
#default_board = pd.read_pickle('default_board.pkl')

#board = pd.read_pickle('board.pkl')
#letters = open('letters.txt', 'r').read().splitlines()

default_board =pd.DataFrame([
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

print(default_board)

board = pd.DataFrame([
    ['']*11,
    ['']*11,
    ['']*11,
    ['']*3 + ['J'] + ['']*7,
    ['']*3 + ['O'] + ['']*7,
    ['']*2 + ['P', 'E', 'R', 'V', 'S'] + ['']*4,
    [''] + ['M', 'A', 'S'] + ['', ''] + ['E'] + ['']*4,
    ['']*6 + ['E', 'W'] + ['']*3,
    ['']*6 + ['K', 'A', 'F'] + [''] + ['Q'],
    ['']*7 + ['Y', 'A', 'G', 'I'],
    ['']*8 + ['X', 'I', 'S']
])
print(board)
letters = ['A', 'E', 'Z', 'L', 'D', '?', 'E']


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

        self.poss_value = None

        self.up = None
        self.left = None
        self.down = None
        self.right = None
        self.row = None
        self.col = None

    def poss_points(self):
        if self.poss_value:
            pp = points.get(self.poss_value)[1]
            if self.multiplier:
                mval = self.multiplier[0]
                mtyp = self.multiplier[1]
                if mtyp == 'l':
                    pp *= mval
        else:
            pp = self.points
        return pp

    def row_vals(self):
        return [rr.value for rr in self.row]

    def col_vals(self):
        return [cc.value for cc in self.col]

    def get_word_nodes(self, typ):
        if typ == 'row':
            nodes = self.row
            bef = nodes[:self.y][::-1]
            aft = nodes[self.y:]
        elif typ == 'col':
            nodes = self.col
            bef = nodes[:self.x][::-1]
            aft = nodes[self.x:]
        else:
            return

        word_nodes = []

        for l in bef:
            if not l.value:
                break
            else:
                word_nodes.insert(0, l)

        for l in aft:
            if not l.value:
                break
            else:
                word_nodes.append(l)

        return word_nodes

    def get_word(self, typ=None, nodes=None):
        if nodes is None:
            nodes = self.get_word_nodes(typ)

        return ''.join([nw.value or nw.poss_value for nw in nodes])


class Board:
    def __init__(self):
        self.default_board = default_board
        self.board = board

        self.nodes = []

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

    def get(self, x, y) -> Node:
        return next(filter(lambda obj: obj.x == x and obj.y == y, self.nodes), None)

    def get_row(self, x):
        return list(filter(lambda obj: obj.x == x, self.nodes))

    def get_col(self, y):
        return list(filter(lambda obj: obj.y == y, self.nodes))


def can_spell(avail, word, lets=None):
    if lets is None:
        lets = letters

    newword = word.replace(avail, '', 1)

    if word == newword or len(newword) == 0:
        return False

    word_list = list(newword)

    has_blank = 0

    for letter in lets:
        if len(word_list) == 0:
            return True
        elif letter == '?':
            has_blank += 1
        elif letter in word_list:
            word_list.remove(letter)

    for blank in range(has_blank):
        if len(word_list):
            word_list.pop()

    return len(word_list) == 0


def is_valid(typ, this_cell, n_idx, word):
    word = list(word)

    word_list = []
    w_bef_list = []
    w_aft_list = []

    if typ == 'row':
        sides = ('left', 'right')
        bef_typ = 'up'
        aft_typ = 'down'
        node_typ = 'col'
    elif typ == 'col':
        sides = ('up', 'down')
        bef_typ = 'left'
        aft_typ = 'right'
        node_typ = 'row'
    else:
        return False

    for side in sides:
        cell = this_cell

        if side in ('left', 'up'):
            let_list = word[:n_idx][::-1]
            #print('bef', let_list)
        elif side in ('right', 'down'):
            let_list = word[n_idx + 1:]
            #print('aft', let_list)
        else:
            return False

        for let_v in let_list:
            cell: Node = getattr(cell, side)
            if not cell:
                return False

            if cell.value:
                if cell.value != let_v:
                    return False
            else:
                cell.poss_value = let_v

                cell_bef: Node = getattr(cell, bef_typ)
                cell_aft: Node = getattr(cell, aft_typ)

                bef_nodes = []
                aft_nodes = []

                if cell_bef and cell_bef.value:
                    bef_nodes = cell_bef.get_word_nodes(node_typ)
                    #oldword = cell_bef.get_word(nodes=oldnodes)

                if cell_aft and cell_aft.value:
                    aft_nodes = cell_aft.get_word_nodes(node_typ)
                    #oldword = cell_aft.get_word(nodes=oldnodes)

                if bef_nodes or aft_nodes:
                    new_nodes = bef_nodes + [cell] + aft_nodes
                    newword = cell.get_word(nodes=new_nodes)

                    if newword:
                        if newword not in words:
                            return False

                        word_list.append(new_nodes)

            if side in ('left', 'up'):
                w_bef_list.insert(0, cell)
            elif side in ('right', 'down'):
                w_aft_list.append(cell)

    word_list.append(w_bef_list + [this_cell] + w_aft_list)

    # check letters before and new word
    #?
    # check letters after and new word

    return word_list


def word_points(nodelist):
    let_pts = sum([n.poss_points() for n in nodelist])
    mults = [n.multiplier[0] for n in nodelist if n.multiplier and n.multiplier[1] == 'w']
    word_mult = 1
    for m in mults:
        word_mult *= m
    let_pts *= word_mult
    return let_pts


full = Board()

#full_nodes = [full.get(3,7)]
full_nodes = full.nodes

#nodelist needs main
# [{'word': 'xyz', 'nodes': [nodelist], 'pts': 0}]
word_info: List[Dict[str, Any]] = []

for no in full_nodes:
    nval = no.value
    if nval:
        n_r_word = no.get_word(typ='row')
        n_c_word = no.get_word(typ='col')

        for w in words:
            #if w != 'DAZZLE': continue
            spell_row = can_spell(n_r_word, w)
            spell_col = can_spell(n_c_word, w)

            if spell_row or spell_col:
                idxs = [i for i, v in enumerate(w) if v == nval]

                if spell_row:
                    for idx in idxs:
                        row_words = is_valid('row', no, idx, w)
                        if row_words:
                            print('row: ' + w)
                            tot_pts = 0
                            for r in row_words:
                                pts = word_points(r)
                                tot_pts += pts
                            print(tot_pts)
                            word_info.append({
                                'word': w,
                                'pts': tot_pts,
                                'nodes': row_words
                            })

                        #if is_valid_row(n, before[::-1], after):
                        #    print(w)

                if spell_col:
                    for idx in idxs:
                        col_words = is_valid('col', no, idx, w)
                        if col_words:
                            print('col: ' + w)
                            tot_pts = 0
                            for c in col_words:
                                pts = word_points(c)
                                tot_pts += pts
                            print(tot_pts)
                            word_info.append({
                                'word': w,
                                'pts': tot_pts,
                                'nodes': col_words
                            })

                        #if is_valid_row(n.up, before[::-1], after):
                        #    print(w)

max_key = max(word_info, key=lambda k: k['pts'])
print(max_key)
