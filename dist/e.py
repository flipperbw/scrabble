#!/usr/bin/env python

"""Parse and solve a scrabble board."""

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

    # parser.add_argument('-m', '--no-multiprocess', action='store_true',
    #     help='Do not use multiprocessing')

    return parser.parse_args()


ARGS = None
if __name__ == '__main__':
    ARGS = parse_args()


#import pstats
#import cProfile

#import pyximport
#pyximport.install()

import p


#from p import main

if __name__ == '__main__':
    dargs = vars(ARGS)

    p.main(**dargs)

    #cProfile.runctx("p.main(**dargs)", globals(), locals(), "Profile.prof")

    #s = pstats.Stats("Profile.prof")
    #s.strip_dirs().sort_stats("time").print_stats()
