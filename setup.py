import json
import pandas as pd
from typing import Set
import sys

from utils import log_init


# - DIRS

data_dir = 'data/'
board_dir = data_dir + 'boards/'
words_dir = data_dir + 'wordlists/'
points_dir = data_dir + 'points/'

big_board_name = board_dir + 'default_board_big.pkl'
small_board_name = board_dir + 'default_board_small.pkl'


# - LOGGING

#LOG_LEVEL = 'DEBUG'
#LOG_LEVEL = 'VERBOSE'
#LOG_LEVEL = 'INFO'
#LOG_LEVEL = 'WARNING'
LOG_LEVEL = 'SUCCESS'
#LOG_LEVEL = 'ERROR'

logger = log_init(LOG_LEVEL, skip_main=False)


# - POOL

USE_POOL = True
#USE_POOL = False


# - DEFAULT BOARD

#DEFAULT_BOARD_NAME = big_board_name
DEFAULT_BOARD_NAME = small_board_name


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
    #SEARCH_WORDS = ['EUROKY', 'EURO']
    SEARCH_WORDS: Set[str] = set()

    return SEARCH_WORDS


def setup(filename: str = None, word_typ: str = 'wwf', pts_typ: str = 'wwf'):
    if filename is not None:
        this_board_dir = f'{board_dir}{filename}/'

        try:
            board = pd.read_pickle(this_board_dir + 'board.pkl')

            board_size = board.size
            if board_size == 15*15:
                default_board = pd.read_pickle(board_dir + big_board_name)
            elif board_size == 11*11:
                default_board = pd.read_pickle(board_dir + small_board_name)
            else:
                logger.c(f'Board size ({board_size}) has no match')
                sys.exit(1)

            letters = pd.read_pickle(this_board_dir + 'letters.pkl')

        except FileNotFoundError:
            logger.c('Could not find file', exc_info=1)
            sys.exit(1)
    else:
        board = pd.DataFrame(BOARD)
        default_board = pd.DataFrame(DEFAULT_BOARD_NAME)
        letters = LETTERS

    wordlist = open(f'{words_dir}{word_typ}.txt').read().splitlines()
    search_words = _get_search_words(wordlist)
    words = set(wordlist)

    points = json.load(open(f'{points_dir}{pts_typ}.json'))

    return {
        'log_level': LOG_LEVEL,
        'logger': logger,
        'use_pool': USE_POOL,
        'words': words,
        'search_words': search_words,
        'board': board,
        'default_board': default_board,
        'letters': letters,
        'points': points
    }
