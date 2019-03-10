#!/usr/bin/env python3

import argparse
import string
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from utils.logs import log_init

#from settings import *
from settings import BOARD_DIR, DEF_BOARD_BIG, DEF_BOARD_SMALL, TEMPL_DIR, BOARD_FILENAME, LETTERS_FILENAME

# -- GLOBALS

#DEFAULT_LOGLEVEL = 'VERBOSE'
DEFAULT_LOGLEVEL = 'INFO'

# img parsing

min_thresh = 0.7

lower_black = np.array([64, 22, 0], dtype="uint16")
upper_black = np.array([78, 36, 24], dtype="uint16")
lower_white = np.array([230, 230, 230], dtype="uint16")
upper_white = np.array([255, 255, 255], dtype="uint16")

rack_space = 106

img_cut_range = {
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

# --

lo = log_init(DEFAULT_LOGLEVEL)

default_board_files = {
    'big': DEF_BOARD_BIG,
    'small': DEF_BOARD_SMALL
}


class Dirs:
    def __init__(self, img_file: str, img_typ: str):
        self.default_board = pd.read_pickle(default_board_files[img_typ])

        img_file_root = Path(img_file).stem
        self.this_board_dir = Path(BOARD_DIR, img_file_root)

        if not self.this_board_dir.exists():
            self.this_board_dir.mkdir()

        self.this_board = Path(self.this_board_dir, BOARD_FILENAME)
        self.this_letters = Path(self.this_board_dir, LETTERS_FILENAME)


def cut_img(img: np.ndarray, typ: str, kind: str) -> np.ndarray:
    ranges = img_cut_range[typ][kind]
    height = ranges['height']
    width = ranges['width']

    return img.copy()[height[0]:height[1], width[0]:width[1]]


def get_img(img: str) -> Optional[np.ndarray]:
    image = cv2.imread(Path(img).expanduser().as_posix())

    if not np.any(image):
        lo.c(f'Could not find image, or it\'s empty: {img}')
        return None

    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# todo: pickle this
letter_templates: Dict[str, Dict[str, np.ndarray]] = {}


def create_letter_templates():
    for l in string.ascii_lowercase:
        templ_big = cv2.imread(Path(TEMPL_DIR, l + '.png').as_posix(), 0)

        templ_small = cv2.resize(templ_big, (0, 0), fx=1.12, fy=1.13)
        templ_rack = cv2.resize(templ_big, (0, 0), fx=2.1, fy=2.1)

        letter_templates[l] = {
            'big': templ_big,
            'small': templ_small,
            'rack': templ_rack
        }


def find_letter_match(gimg: np.ndarray, typ: str, spacing: float, dest: np.ndarray):
    seen: Dict[str, Tuple[str, float]] = {}

    for l, ld in letter_templates.items():
        template = ld[typ]

        h, w = template.shape[:2]

        res = cv2.matchTemplate(gimg, template, cv2.TM_CCOEFF_NORMED)
        match_locations = np.where(res >= min_thresh)

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
            dest[row_num][col_num] = l

    return dest


def create_board(board: np.ndarray, def_board: np.ndarray):
    if def_board.shape[0] == 11:
        typ = 'small'
        spacing = 55.3
    else:
        typ = 'big'
        spacing = 49.6

    black_mask = cv2.inRange(board, lower_black, upper_black)
    white_mask = cv2.inRange(board, lower_white, upper_white)
    comb = black_mask + white_mask

    gimg = cv2.bitwise_not(comb)

    if lo.is_enabled('d'):
        show_img(gimg)

    # noinspection PyTypeChecker
    table = np.full_like(def_board, '', dtype='U2')
    table = find_letter_match(gimg, typ, spacing, table)

    df = pd.DataFrame(table)
    lo.n(f'Board:\n{df}')

    return df


def get_rack(img: np.ndarray):
    black_mask = cv2.inRange(img, lower_black, upper_black)
    gimg = cv2.bitwise_not(black_mask)

    rack = np.array([[''] * 7], dtype='U1')
    rack = find_letter_match(gimg, 'rack', rack_space, rack)[0]

    lo.d(f'Letters:\n{rack}')

    buffer = 20

    mid_y = img.shape[0] // 2
    mid_x = rack_space // 2

    start_y = mid_y - buffer
    end_y = start_y + (buffer * 2)

    letters = []

    for i, l in enumerate(rack):
        start_x = (i * rack_space) + mid_x - buffer
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


def parse_args() -> argparse.Namespace:
    #TODO: set customnamespace for completion here
    #https://stackoverflow.com/questions/42279063/python-typehints-for-argparse-namespace-objects

    parser = argparse.ArgumentParser(description='Extract text from a scrabble board')

    parser.add_argument('-f', '--file', type=str, required=True, dest='filename',
                        help='File path for the image')
    parser.add_argument('-o', '--overwrite', action='store_true',
                        help='Overwrite existing files')
    parser.add_argument('-l', '--log-level', type=str, default=DEFAULT_LOGLEVEL.lower(), choices=[l.lower() for l in lo.levels], metavar='<lvl>',
                        help='Log level for output (default: %(default)s)\nChoices: {%(choices)s}')

    return parser.parse_args()


def main(filename: str, overwrite: bool = False, log_level: str = DEFAULT_LOGLEVEL, **_kwargs):
    log_level = log_level.upper()
    if log_level != DEFAULT_LOGLEVEL:
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


if __name__ == '__main__':
    args = parse_args()
    dargs = vars(args)
    main(**dargs)
