from pathlib import Path
from typing import List, Set, Tuple, Union

# - DIRS

curr_dir = Path(__file__).parent
img_dir = Path(curr_dir, 'images')
data_dir = Path(curr_dir, 'data')

TEMPL_DIR = Path(img_dir, 'templ')

BOARD_DIR = Path(data_dir, 'boards')
WORDS_DIR = Path(data_dir, 'wordlists')
POINTS_DIR = Path(data_dir, 'points')
SOLUTIONS_DIR = Path(data_dir, 'solutions')

DEF_BOARD_BIG = Path(BOARD_DIR, 'default_board_big.pkl')
DEF_BOARD_SMALL = Path(BOARD_DIR, 'default_board_small.pkl')

BOARD_FILENAME = 'board'
LETTERS_FILENAME = 'letters.pkl'


# todo: setup logger?


# - POOL

USE_POOL = True
#USE_POOL = False


# - GAME BOARD

BOARD = [[' '] * 11] * 11

# big
'''
BOARD = [        #
    list('               '),
    list('               '),
    list('               '),
    list('               '),
    list('               '),
    list('               '),
    list('               '),
    list('               '), #
    list('               '),
    list('               '),
    list('               '),
    list('               '),
    list('               '),
    list('               '),
    list('               '),
]
'''

# small
'''
BOARD = [      #
    list('           '),
    list('           '),
    list('           '),
    list('           '),
    list('           '),
    list('           '), #
    list('           '),
    list('           '),
    list('           '),
    list('           '),
    list('           '),
]
'''

# - LETTERS

#LETTERS = list('TOTHBYU')
LETTERS = ['']


# todo wordlist?


# - WORDS

SEARCH_WORDS: Union[None, Tuple[int, int], Set[str]]
#SEARCH_WORDS = (50000, 52000)
#SEARCH_WORDS = {'TAXON'}
SEARCH_WORDS = None


# - NODES

SEARCH_NODES: Union[None, List[int], List[Tuple[int, int]]]
#SEARCH_NODES = [0,1,2] # rows
#SEARCH_NODES = [(6,1), (7,1)] # nodes
SEARCH_NODES = None


# - EXPORT

__all__ = [a for a in globals() if a == a.upper()]
#__all__ = ['DEF_BOARD_BIG', 'DEF_BOARD_SMALL']
