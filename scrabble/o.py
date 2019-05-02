#!/usr/bin/env python3

"""Extract text from a scrabble board."""

__version__ = 1.0


DEFAULT_LOGLEVEL = 'INFO'


def parse_args():
    #TODO: set customnamespace for completion here
    #https://stackoverflow.com/questions/42279063/python-typehints-for-argparse-namespace-objects

    from .parser_defaults import parser_init

    parser = parser_init(
        description=__doc__,
        usage='%(prog)s [options] filename',
        log_level=DEFAULT_LOGLEVEL,
        version=__version__
    )

    parser.add_argument('filename', type=str, default=None,  # todo type=argparse.FileType('r')
        help='File path for the image')

    parser.add_argument('-o', '--overwrite', action='store_true',
        help='Overwrite existing files')

    parser.add_argument('-p', '--profile', action='store_true',
        help='Profile the app')
    parser.add_argument('-x', '--trace', action='store_true',
        help='Trace the app')

    return parser.parse_known_args() # todo put everywhere


if __name__ == '__main__':
    pargs, ext_args = parse_args()

    from scrabble import ocr

    dargs = vars(pargs)

    if dargs['profile'] is True:
        import cProfile
        import pstats

        fn = 'o.prof'

        cProfile.run('ocr.main(**dargs)', fn)
        s = pstats.Stats(fn)
        s.sort_stats('time').print_stats(20)

    elif dargs['trace'] is True:
        import line_profiler

        profile = line_profiler.LineProfiler(
            ocr.find_letter_match
        )
        profile.runctx('ocr.main(**dargs)', globals(), locals())
        profile.print_stats()

    else:
        ocr.main(**dargs)
