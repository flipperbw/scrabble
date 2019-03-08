import json
from typing import Set
from pathlib import Path

import pandas as pd


# - DIRS

CURR_DIR = Path(__file__).parent

img_dir = Path(CURR_DIR, 'images')
templ_dir = Path(img_dir, 'templ')

data_dir = Path(CURR_DIR, 'data')
board_dir = Path(data_dir, 'boards')
def_board_big = Path(board_dir, 'default_board_big.pkl')
def_board_small = Path(board_dir, 'default_board_small.pkl')
words_dir = Path(data_dir, 'wordlists')
points_dir = Path(data_dir, 'points')

board_filename = 'board'
letters_filename = 'letters.pkl'


# - LOGGING

#LOG_LEVEL = 'DEBUG'
#LOG_LEVEL = 'VERBOSE'
#LOG_LEVEL = 'INFO'
#LOG_LEVEL = 'WARNING'
LOG_LEVEL = 'SUCCESS'
#LOG_LEVEL = 'ERROR'


# - POOL

USE_POOL = True
#USE_POOL = False


# - DEFAULT BOARD

#DEFAULT_BOARD_NAME = def_board_big
DEFAULT_BOARD_NAME = def_board_small


# - GAME BOARD

'''
# big
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


# - LETTERS

#LETTERS = list('TOTHBYU')
#LETTERS = 'EIIEB?A' #problem with blanks overwriting
LETTERS = ''


# - SEARCH

# noinspection PyPep8Naming,PyUnusedLocal
def _get_search_words(wordlist: list = None):
    if wordlist is None:
        wordlist = []

    #SEARCH_WORDS = wordlist[50000:52000]
    #SEARCH_WORDS = ['TAXON']
    SEARCH_WORDS: Set[str] = set()

    return SEARCH_WORDS


def setup(folder_name: str = None, word_typ: str = 'wwf', pts_typ: str = 'wwf'):
    if folder_name is not None:
        this_board_dir = Path(board_dir, folder_name)

        try:
            # todo: move this up
            board = pd.read_pickle(Path(this_board_dir, board_filename))
            letters = pd.read_pickle(Path(this_board_dir, letters_filename))
        except FileNotFoundError as exc:
            raise Exception(f'Could not find file: {exc.filename}')

        board_size = board.size
        if board_size == 15 * 15:
            board_name = def_board_big
        elif board_size == 11 * 11:
            board_name = def_board_small
        else:
            raise Exception(f'Board size ({board_size}) has no match')

    else:
        board = pd.DataFrame(BOARD)
        board_name = DEFAULT_BOARD_NAME
        letters = LETTERS

    try:
        default_board = pd.read_pickle(board_name)
    except FileNotFoundError as exc:
        raise Exception(f'Could not find file: {exc.filename}')

    wordlist = open(Path(words_dir, word_typ + '.txt')).read().splitlines()
    search_words = _get_search_words(wordlist)
    words = set(wordlist)

    points = json.load(open(Path(points_dir, pts_typ + '.json')))

    return {
        'log_level': LOG_LEVEL,
        'use_pool': USE_POOL,
        'words': words,
        'search_words': search_words,
        'board': board,
        'default_board': default_board,
        'letters': letters,
        'points': points
    }
