#!/usr/bin/env python

"""Extract text from a scrabble board."""

__version__ = 1.0

#DEFAULT_LOGLEVEL = 'VERBOSE'
DEFAULT_LOGLEVEL = 'INFO'


from parsing import parser_init

def parse_args():
    #TODO: set customnamespace for completion here
    #https://stackoverflow.com/questions/42279063/python-typehints-for-argparse-namespace-objects

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

    return parser.parse_args()

ARGS = None
if __name__ == '__main__':
    ARGS = parse_args()


import ocr


if __name__ == '__main__':
    dargs = vars(ARGS)
    ocr.main(**dargs)
