#!/usr/bin/env python

"""Parse and solve a scrabble board."""

__version__ = 1.0

DEFAULT_LOGLEVEL = 'SUCCESS'  # need?
DICTIONARY = 'wwf'
NUM_RESULTS = 15


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

    parser.add_argument('-r', '--num-results', type=int, default=NUM_RESULTS,
        help='Number of solutions to show [0 for all] (default: %(default)d)')

    parser.add_argument('-p', '--profile', action='store_true',
        help='Profile the app')
    parser.add_argument('-x', '--trace', action='store_true',
        help='Trace the app')

    return parser.parse_args()


ARGS = None
if __name__ == '__main__':
    ARGS = parse_args()

if __name__ == '__main__':
    dargs = vars(ARGS)

    import p

    if dargs['profile'] is True:
        import cProfile
        import pstats

        #cProfile.runctx('p.main(**dargs)', globals(), locals(), 'p.prof')
        cProfile.run('p.main(**dargs)', 'p.prof')

        s = pstats.Stats('p.prof')
        s.sort_stats('time').print_stats(20)

    elif dargs['trace'] is True:
        import line_profiler

        profile = line_profiler.LineProfiler(
            p.parse_nodes,
            # p.set_word_dict,
            # p.rack_check,
            # p.lets_match,

            # p.rack_match,
            # p.check_nodes,
        )
        profile.runctx('p.main(**dargs)', globals(), locals())
        profile.print_stats()

    else:
        p.main(**dargs)
