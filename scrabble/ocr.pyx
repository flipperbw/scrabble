"""Extract text from a scrabble board."""

cimport cython

#cdef object Path, _s
#cv2, np, pd, Image, log_init

import string
import sys
from pathlib import Path
from typing import Dict, Tuple

import cv2

import numpy as np
cimport numpy as cnp

import pandas as pd
from PIL import Image

import settings as _s
from logs import log_init


lo = log_init('INFO')


cnp.import_array()


# ctypedef cnp.int32_t STR_t
# STR = np.int32
ctypedef cnp.uint8_t BOOL_t
BOOL = np.uint8
ctypedef cnp.float32_t FLO_t
FLO = np.float32


# - img parsing

RACK_SPACE = 106

MIN_THRESH = 0.7

LOWER_BLACK = np.array([64, 22, 0], dtype="uint16")
UPPER_BLACK = np.array([78, 36, 24], dtype="uint16")
LOWER_WHITE = np.array([230, 230, 230], dtype="uint16")
UPPER_WHITE = np.array([255, 255, 255], dtype="uint16")

# todo all colors
LOWER_BLACK_GR = np.array([16, 54, 0], dtype="uint16")
UPPER_BLACK_GR = np.array([30, 68, 24], dtype="uint16")
LOWER_BLACK_PU = np.array([52, 0, 66], dtype="uint16")
UPPER_BLACK_PU = np.array([66, 12, 80], dtype="uint16")
LOWER_BLACK_TE = np.array([0, 15, 66], dtype="uint16")
UPPER_BLACK_TE = np.array([12, 29, 80], dtype="uint16")

IMG_CUT_RANGE = {
    'big': {
        'board': {
            'height': (305, 1049),
            'width': (5, -5)
        },
        'letters': {
            'height': (1120, 1204),
            'width': (10, -10)
        }
    },
    'small': {
        'board': {
            'height': (430, 1039),
            'width': (71, -71)
        },
        'letters': {
            'height': (1150, 1234),
            'width': (1, -1)
        }
    },
}

# -


DEFAULT_BOARD_FILES = {
    'big': _s.DEF_BOARD_BIG,
    'small': _s.DEF_BOARD_SMALL
}


class Dirs:
    def __init__(self, img_file: str, img_typ: str):
        self.default_board = pd.read_pickle(DEFAULT_BOARD_FILES[img_typ])

        img_file_root = Path(img_file).stem
        self.this_board_dir = Path(_s.BOARD_DIR, img_file_root)

        if not self.this_board_dir.exists():
            self.this_board_dir.mkdir()

        self.this_board = Path(self.this_board_dir, _s.BOARD_FILENAME)
        self.this_letters = Path(self.this_board_dir, _s.LETTERS_FILENAME)


def cut_img(img: np.ndarray, typ: str, kind: str) -> np.ndarray:
    ranges = IMG_CUT_RANGE[typ][kind]
    height = ranges['height']
    width = ranges['width']

    return img.copy()[height[0]:height[1], width[0]:width[1]]


cdef object get_img(str img):
    image = cv2.imread(Path(img).expanduser().as_posix())

    if not np.any(image):
        lo.c(f'Could not find image, or it\'s empty: {img}')
        return None

    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    colored = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return colored


# todo: pickle this
letter_templates = {}  # type: Dict[str, Dict[str, np.ndarray]]


cdef create_letter_templates():
    for l in string.ascii_lowercase:
        templ_big = cv2.imread(Path(_s.TEMPL_DIR, l + '.png').as_posix(), 0)

        templ_small = cv2.resize(templ_big, (0, 0), fx=1.12, fy=1.13)
        templ_rack = cv2.resize(templ_big, (0, 0), fx=2.1, fy=2.1)

        letter_templates[l] = {
            'big': templ_big,
            'small': templ_small,
            'rack': templ_rack
        }


cdef int TYP = cv2.TM_CCOEFF_NORMED
cdef object mat = cv2.matchTemplate


# @cython.binding(True)
# @cython.linetrace(True)
# cpdef char[:, :] find_letter_match(BOOL_t[:, :] gimg, str typ, float spacing, char[:, :] dest):
cdef char[:, :] find_letter_match(BOOL_t[:, :] gimg, str typ, float spacing, char[:, :] dest):
    cdef object seen = {}  # type: Dict[str, Tuple[str, float]]
    cdef BOOL_t[:, :] template
    cdef Py_ssize_t h, w, row_num, col_num, x, y
    #cdef cnp.ndarray res
    cdef FLO_t[:, :] res
    #cdef cnp.ndarray[FLO_t, ndim=2] res
    cdef cnp.ndarray[BOOL_t, ndim=2] gb
    cdef cnp.ndarray[BOOL_t, ndim=2] tb

    #cdef object res
    #(W-w+1) \times (H-h+1)


    for l, ld in letter_templates.items():
        template = ld[typ]

        h, w = template.shape[:2]

        gb = gimg.base
        tb = template.base

        #TYP
        res = cv2.matchTemplate(image=gb, templ=tb, method=5)
        #res = mat(image=gb, templ=tb, method=5)
        #res = mat(image=gb, templ=tb, res=res, method=5)

        #res = cv2.matchTemplate(np.array(gimg), template.base, cv2.TM_CCOEFF_NORMED)

        #match_locations = res[(res >= MIN_THRESH)]
        match_locations = np.where(res.base >= MIN_THRESH)

        l = l.upper()

        for x, y in zip(match_locations[1], match_locations[0]):
            start = (x, y)
            end = (x + w, y + h)

            colx = (start[0] + end[0]) / 2
            rowx = (start[1] + end[1]) / 2

            col_num = int(colx // spacing)
            row_num = int(rowx // spacing)

            pos = f'{row_num}x{col_num}'
            confidence = res[y][x]

            if pos in seen:
                (exist_l, exist_conf) = seen[pos]

                if confidence <= exist_conf:
                    continue
                else:
                    seen[pos] = (l, confidence)

                    if l == exist_l:
                        continue

                    lo.d(f'overriding {exist_l, exist_conf} with new {l, confidence}')

            seen[pos] = (l, confidence)
            dest[row_num][col_num] = ord(l)

    return dest


def _u(): pass

cdef object create_board(board: np.ndarray, def_board: np.ndarray):
    if def_board.shape[0] == 11:
        typ = 'small'
        spacing = 55.3
    else:
        typ = 'big'
        spacing = 49.6

    black_mask = cv2.inRange(board, LOWER_BLACK, UPPER_BLACK) + \
                 cv2.inRange(board, LOWER_BLACK_GR, UPPER_BLACK_GR) + \
                 cv2.inRange(board, LOWER_BLACK_PU, UPPER_BLACK_PU) + \
                 cv2.inRange(board, LOWER_BLACK_TE, UPPER_BLACK_TE)
    white_mask = cv2.inRange(board, LOWER_WHITE, UPPER_WHITE)
    comb = black_mask + white_mask

    cdef BOOL_t[:, :] gimg = cv2.bitwise_not(comb)

    if lo.is_enabled('d'):
        show_img(gimg.base)

    cdef char[:, :] table = np.full_like(def_board, '', dtype='|S1')

    find_letter_match(gimg, typ, spacing, table)

    df = pd.DataFrame(table.base.astype('<U1'))
    lo.n(f'Board:\n{df}')

    return df


cdef list get_rack(object img):
    black_mask = cv2.inRange(img, LOWER_BLACK, UPPER_BLACK) + cv2.inRange(img, LOWER_BLACK_GR, UPPER_BLACK_GR) + cv2.inRange(img, LOWER_BLACK_PU, UPPER_BLACK_PU)
    gimg = cv2.bitwise_not(black_mask)

    cdef char[:, :] rack = np.array([[''] * 7], dtype='|S1')

    find_letter_match(gimg, 'rack', RACK_SPACE, rack)

    cdef object nrack = rack.base[0].astype('<U1')

    lo.d(f'Letters:\n{nrack}')

    buffer = 20

    mid_y = img.shape[0] // 2
    mid_x = RACK_SPACE // 2

    start_y = mid_y - buffer
    end_y = start_y + (buffer * 2)

    letters = []

    for i, l in enumerate(nrack):
        start_x = (i * RACK_SPACE) + mid_x - buffer
        end_x = start_x + (buffer * 2)

        imgflat = img[start_y:end_y, start_x:end_x].flatten()

        if all(pixel == 255 for pixel in imgflat):
            continue
        elif not l:
            letters.append('?')
        else:
            letters.append(l)

    lo.n(f'Letters:\n{letters}')

    return letters


def show_img(img_array: np.ndarray):
    Image.fromarray(img_array).show()


cdef void cmain(str filename, bint overwrite, str log_level):
    log_level = log_level.upper()
    if log_level != lo.logger.getEffectiveLevel():
        lo.set_level(log_level)

    cv2_image = get_img(filename)
    if cv2_image is None:
        sys.exit(1)

    img_type = 'small'

    check_typ_pixels = cv2_image[650:850, 10:30]
    check_first = check_typ_pixels[0].tolist()
    for row_val in check_typ_pixels:
        if not row_val.tolist() == check_first:
            img_type = 'big'
            break

    direcs = Dirs(filename, img_type)

    this_board = direcs.this_board
    this_letters = direcs.this_letters

    has_board = not overwrite and this_board.exists()
    has_letters = not overwrite and this_letters.exists()

    if has_board and has_letters:
        lo.n('Info exists, skipping (override with overwrite = True)')

        if lo.is_enabled('i'):
            df_board = pd.read_pickle(this_board)
            #df_board = df_board.to_string(justify='left', col_space=2)
            #df_board.columns.rename('x', inplace=True)
            #df_board = df_board.to_string(formatters={'x': '{:<2s}'})
            #df_board.style.set_properties(**{'text-align': 'left'})
            #df_board.style.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])
            #todo why?
            lo.i(f'Board:\n{df_board}')
            lo.i(f'Letters: {pd.read_pickle(this_letters)}')

        lo.s('Done')

        return

    create_letter_templates()

    if not has_board:
        lo.i('Parsing image...')

        cv2_board = cut_img(cv2_image, img_type, 'board')
        if lo.is_enabled('v'):
            show_img(cv2_board)

        lo.i('Creating board...')
        board = create_board(cv2_board, direcs.default_board)
        board.to_pickle(this_board)

    if not has_letters:
        lo.i('Parsing letters...')

        cv2_letters = cut_img(cv2_image, img_type, 'letters')
        if lo.is_enabled('v'):
            show_img(cv2_letters)

        rack = get_rack(cv2_letters)
        pd.to_pickle(rack, this_letters)

    lo.s('Done parsing image.')

def main(filename: str, overwrite: bool = False, log_level: str = _s.DEFAULT_LOGLEVEL, **_kwargs) -> None:
    cmain(filename, overwrite, log_level)
