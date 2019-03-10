from typing import Set
from pathlib import Path


# - DIRS

curr_dir = Path(__file__).parent

img_dir = Path(curr_dir, 'images')
TEMPL_DIR = Path(img_dir, 'templ')

data_dir = Path(curr_dir, 'data')
BOARD_DIR = Path(data_dir, 'boards')
DEF_BOARD_BIG = Path(BOARD_DIR, 'default_board_big.pkl')
DEF_BOARD_SMALL = Path(BOARD_DIR, 'default_board_small.pkl')
WORDS_DIR = Path(data_dir, 'wordlists')
POINTS_DIR = Path(data_dir, 'points')

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
LETTERS = ''


# - SEARCH


def _get_search_words(wordlist: list = None):
    if wordlist is None:
        wordlist = []

    #SEARCH_WORDS = wordlist[50000:52000]
    #SEARCH_WORDS = ['TAXON']
    SEARCH_WORDS: Set[str] = set()

    return SEARCH_WORDS



#__all__ = [a for a in globals() if a == a.upper()]
__all__ = ['DEF_BOARD_BIG', 'DEF_BOARD_SMALL']
