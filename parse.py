#!/usr/bin/env python3

import argparse
import json
from multiprocessing import cpu_count, current_process, Pool
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import pandas as pd
from utils.logs import log_init

import settings as _

#import pyximport; pyximport.install(pyimport=True)

#import builtins
#profile = getattr(builtins, 'profile', lambda x: x)


# -- GLOBALS

DEFAULT_LOGLEVEL = 'SUCCESS'  # need?
lo = log_init(DEFAULT_LOGLEVEL)


# -- TODOS

#todo: profiling
# combine stuffz
# change return [] to yields?
# if word was found in a different check, skip
# change to f'' formatting
# if someone elses word is a blank, dont count it
# why is multiprocessing _C getting called?
# if blank and already have another letter, will assume not blank
# why is main not showing proper process?
# fix empty boards in both files
# recommend choices based on letters left, opened tiles, end game, blanks, etc
# allow letters to use option, and num


class Settings:
    use_pool: bool = _.USE_POOL

    board = pd.DataFrame()
    default_board = pd.DataFrame()
    shape: Dict[str, int] = {}
    letters: List[str] = []
    blanks: int = 0
    points: Dict[str, List[int]] = {}
    words: Set[str] = set()
    search_words: Set[str] = set()


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
            try:
                self.points = Settings.points[self.value][1]
            except (KeyError, IndexError):
                lo.e(f'Could not get point value of "{self.value}". Using 0.')
                self.points = 0

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
        return self._get_poss_word_dict(direc, word).get(idx)  # ''?

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
        if self.points:  #unnec
            return self.points

        is_blank = self.get_poss_val(direc, word, idx)  # i guess i could use has_poss here
        if is_blank is not None:
            if is_blank:
                return 0

            try:
                pp = Settings.points[word[idx]][1]
            except (KeyError, IndexError):
                lo.e(f'Could not get point value of "{word[idx]}". Using 0.')
                pp = 0

            self.points = 0
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
        aft = nodes[slice_val + 1:]

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

    def str_pos(self):
        return f'[{self.x:2d},{self.y:2d}]'

    def __str__(self):
        return '<Node>: {} v: {:1s}'.format(self.str_pos(), self.value or '_')

    #def __repr__(self):
    #    return self.__str__()


class Word:  #todo
    def __init__(self, nodes: List[Node]):
        self.nodes = nodes


class Board:
    def __init__(self, default_board=Settings.default_board, board=Settings.board) -> None:
        self.default_board = default_board
        self.board = board

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

    bef_idx = start - 1
    if bef_idx >= 0:
        node_bef = node_list[bef_idx]
        if node_bef and node_bef.value:
            lo.d(f'Prior node exists for {direc} at {start}, skipping.')
            return None

    aft_idx = start + word_len
    if aft_idx < Settings.shape[direc]:
        node_aft = node_list[aft_idx]
        if node_aft and node_aft.value:
            lo.d(f'Following node exists for {direc}, skipping.')
            return None

    ls = list(Settings.letters)
    blanks = Settings.blanks

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
                bef_word = get_word(bef_nodes)  # should not need other args
                aft_word = get_word(aft_nodes)

                new_word = bef_word + le + aft_word

                if new_word not in Settings.words:
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

    dir_len = Settings.shape[direc]
    word_len = len(word)

    start = max(0, idx - word_len + 1)
    while (start + word_len) <= dir_len and start <= idx:
        new_words = check_and_get(node_list, direc, chk_dir, word, word_len, start)
        if new_words is not None:
            new_d = {
                'nodes': node_list[start:start + word_len],
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
            lo.i('--> YES: %s: %s', direction, w)
            # combine words with idxs?

            for word_dict in spell_words:
                tot_pts = no.get_points(**{k: v for k, v in word_dict.items() if k != 'idx'})
                # todo, could do this beforehand

                new_info = {
                    'pts': tot_pts,
                    'cell_letter': w[word_dict['idx']],
                    'cell_pos': (no.x, no.y),
                    'cell': no
                }

                this_info = {**word_dict, **new_info}

                lo.i(this_info)
                data.append(this_info)

    return data

# todo check logger processname
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


def _print_node_range(n: List[Node]):
    return f'{n[0].str_pos()} : {n[-1].str_pos()}'


def _print_result(o: dict):
    s = '\n'.join([
        f'pts  : {o["pts"]}',
        f'word : {o["word"]}',
        f'nodes: {_print_node_range(o["nodes"])}',
    ])

    # todo why are these tabs like 6 or 8?
    s += '\nnew_words:'
    for n in o['new_words']:
        s += f'\n\t{_print_node_range(n["nodes"])}  {n["word"]}'

    return s


def _add_results(res_list: Optional[List[dict]]):
    if not res_list: return
    for res in res_list:
        word_info.append(res)


def check_node(no: Optional[Node]):
    if not no: return

    if Settings.use_pool:  #todo: is this fixable for profiling?
        n = max(cpu_count() - 1, 1)
        #n = 2
        pool = Pool(n)
        #pool_res: List[List[dict]] = pool.map(run_worker, ((w, no) for w in Settings.search_words))
        pool_res: List[List[dict]] = pool.map(run_worker, [(w, no) for w in Settings.search_words])
        #pool_res = pool.map_async(run_worker, ((w, no) for w in Settings.search_words), callback=_add_results)

        for x in pool_res:
            _add_results(x)

    else:
        for w in Settings.search_words:
            reses = check_words(w, no)
            _add_results(reses)


# todo: nodelist needs main
# [{'word': 'xyz', 'nodes': [nodelist], 'pts': 0}]
word_info: List[Dict[str, Any]] = []


def show_solution():
    if not word_info:
        print('No solution.')
    else:
        newlist = sorted(word_info, key=lambda k: k['pts'], reverse=True)
        best_data = newlist[:1][0]

        if lo.is_enabled('s'):
            #todo fix this
            top10: List[dict] = []
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
                lo.s(f'Choice #{sidx + 1}\n{_print_result(s)}\n')

            lo.s(f'-- Best --\n{_print_result(best_data)}')

        solved_board = Settings.board.copy()

        for ni, node in enumerate(best_data.get('nodes', [])):
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

        print(f'\nPoints: {best_data["pts"]}')


def parse_args() -> argparse.Namespace:
    #TODO: set customnamespace for completion here
    #https://stackoverflow.com/questions/42279063/python-typehints-for-argparse-namespace-objects

    parser = argparse.ArgumentParser(description='Extract text from a scrabble board')

    parser.add_argument('filename', type=str, default=None,
                        help='File path for the image')
    parser.add_argument('-l', '--log-level', type=str, default=DEFAULT_LOGLEVEL.lower(), choices=[l.lower() for l in lo.levels], metavar='<lvl>',
                        help='Log level for output (default: %(default)s)\nChoices: {%(choices)s}')

    return parser.parse_args()


def main(filename: str = None, log_level: str = DEFAULT_LOGLEVEL, word_typ: str = 'wwf', **_kwargs):
    start = timer()

    log_level = log_level.upper()
    if log_level != DEFAULT_LOGLEVEL:
        lo.set_level(log_level)

    if filename is not None:
        this_board_dir = Path(_.BOARD_DIR, filename)

        try:
            board: pd.DataFrame = pd.read_pickle(Path(this_board_dir, _.BOARD_FILENAME))
            letters: List[str] = pd.read_pickle(Path(this_board_dir, _.LETTERS_FILENAME))
        except FileNotFoundError as exc:
            raise Exception(f'Could not find file: {exc.filename}')

    else:
        board = pd.DataFrame(_.BOARD)
        letters = _.LETTERS

    board_size = board.size
    if board_size == 15 * 15:
        board_name = _.DEF_BOARD_BIG
    elif board_size == 11 * 11:
        board_name = _.DEF_BOARD_SMALL
    else:
        raise Exception(f'Board size ({board_size}) has no match')

    shape = {
        'row': board.shape[0],
        'col': board.shape[1]
    }

    try:
        default_board: pd.DataFrame = pd.read_pickle(board_name)
    except FileNotFoundError as exc:
        raise Exception(f'Could not find file: {exc.filename}')

    #todo make fnc exception
    try:
        wordlist = open(Path(_.WORDS_DIR, word_typ + '.txt')).read().splitlines()
    except FileNotFoundError as exc:
        raise Exception(f'Could not find file: {exc.filename}')

    try:
        points: Dict[str, List[int]] = json.load(open(Path(_.POINTS_DIR, word_typ + '.json')))
    except FileNotFoundError as exc:
        raise Exception(f'Could not find file: {exc.filename}')

    words = set(wordlist)
    blanks = letters.count('?')

    if _.SEARCH_WORDS is None:
        if blanks:
            search_words = words
        else:
            # todo: faster if str? why gen slower?
            search_words = {w for w in words if any(l in w for l in letters)}
            #search_words = {w for w in words if any([l in w for l in letters])}
    elif isinstance(_.SEARCH_WORDS, tuple):
        search_words = set(wordlist[_.SEARCH_WORDS[0]: _.SEARCH_WORDS[1]])
    elif isinstance(_.SEARCH_WORDS, set):
        search_words = _.SEARCH_WORDS
    else:
        raise Exception(f'Incompatible search words type: {type(_.SEARCH_WORDS)}')

    # --

    # todo allow extra mods
    # print(board)
    # print(board[10])
    # board[6][10] = 'R'
    # board[8][10] = 'Y'
    # letters = ['F', 'G']

    Settings.board = board
    Settings.default_board = default_board
    Settings.shape = shape
    Settings.letters = letters
    Settings.blanks = blanks
    Settings.points = points
    Settings.words = words
    Settings.search_words = search_words

    # --

    if lo.is_enabled('s'):
        lo.i('Default Board:\n{}'.format(default_board))
        lo.s('Game Board:\n{}'.format(board))
        lo.s('Letters:\n{}'.format(letters))
        print()
    else:
        print('Running...')

    full = Board(default_board=default_board, board=board)

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
                        raise Exception(f'Could not locate node: {nl}')
                    full_nodes.append(fr)
            else:
                raise Exception(f'Incompatible search node type: {type(_.SEARCH_NODES)}')
        else:
            raise Exception(f'Incompatible search node type: {type(_.SEARCH_NODES)}')

        full_nodes_len = len(full_nodes)
        for no in full_nodes:
            if no and not no.value and no.has_edge():
                if lo.is_enabled('s'):
                    lo.s('Checking Node (%2s, %2s: %1s) - #%s / %s',
                         no.x, no.y, no.value or '_', full_nodes.index(no) + 1, full_nodes_len
                         )
                check_node(no)

    show_solution()

    end = timer()
    lo.i('\nTime: {}'.format(round(end - start, 1)))


if __name__ == '__main__':
    args = parse_args()
    dargs = vars(args)
    main(**dargs)
    #main()
