#!/usr/bin/env python

import os
from typing import Tuple, Dict

import argparse
import cv2
import numpy as np
import pandas as pd
from pytesseract import image_to_string  # todo: image_to_boxes?
from PIL import Image
import string

from utils import log_init


# -- GLOBALS

#log_level = 'VERBOSE'
log_level = 'INFO'

img_dir = 'images/'
data_dir = 'data/'

templ_dir = img_dir + 'templ/'
board_dir = data_dir + 'boards/'

default_board_files = {
    'big': board_dir + 'default_board_big.pkl',
    'small': board_dir + 'default_board_small.pkl'
}

tess_conf = '--psm 8 --oem 0 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ"'

color = True
#color = False

#white = 255
#empty = 236

min_thresh = 0.6

lower_black = np.array([64, 22, 0], dtype="uint16")
upper_black = np.array([78, 36, 24], dtype="uint16")
lower_white = np.array([230, 230, 230], dtype="uint16")
upper_white = np.array([255, 255, 255], dtype="uint16")

# autodetect big vs small by looking at left col and seeing if all same color
img_cut_range = {
    'big': {
        'board': {
            'height': (305, 1049),
            'width': (5, -5)
        },
        'letters': {
            'height': (1126, 1198),
            'width': (18, -18)
        }
    },
    'small': {
        'board': {
            'height': (430, 1039),
            'width': (71, -71)
        },
        'letters': {
            'height': (1157, 1229),
            'width': (16, -16)
        }
    },
}

# --

lo = log_init(log_level, skip_main=False)


class Dirs:
    def __init__(self, img_file: str, img_typ: str):
        self.default_board = pd.read_pickle(default_board_files[img_typ])

        img_file_root = os.path.splitext(img_file.split('/')[-1])[0] + '/'

        self.this_board_dir = board_dir + img_file_root

        if not os.path.exists(self.this_board_dir):
            os.makedirs(self.this_board_dir)


def cut_img(img: np.ndarray, typ: str, kind: str) -> np.ndarray:
    ranges = img_cut_range[typ][kind]
    height = ranges['height']
    width = ranges['width']

    return img.copy()[height[0]:height[1], width[0]:width[1]]


def get_img(img: str) -> np.ndarray:
    image = cv2.imread(img)

    if not np.any(image):
        raise Exception(f'Could not find image, or is empty: {img}')

    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if color:
        #return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        #return gray
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def create_board(board: np.ndarray, def_board: np.ndarray):
    # noinspection PyTypeChecker
    table = np.full_like(def_board, '', dtype='U2')

    if def_board.shape[0] == 11:
        small = True
        spacing = 55.3
    else:
        small = False
        spacing = 49.6

    black_mask = cv2.inRange(board, lower_black, upper_black)
    white_mask = cv2.inRange(board, lower_white, upper_white)
    comb = black_mask + white_mask

    gimg = cv2.bitwise_not(comb)

    if lo.is_enabled('v'):
        show_img(gimg)

    seen: Dict[str, Tuple[str, float]] = {}

    for l in string.ascii_lowercase:
        template = cv2.imread(templ_dir + l + '.png', 0)
        if small:
            template = cv2.resize(template, (0,0), fx=1.12, fy=1.13)
        h, w = template.shape[:2]
        
        res = cv2.matchTemplate(gimg, template, cv2.TM_CCOEFF_NORMED)
        match_locations = np.where(res >= min_thresh)

        l = l.upper()

        for x, y in zip(match_locations[1], match_locations[0]):
            start = (x, y)
            end = (x + w, y + w)

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
            table[row_num][col_num] = l

    df = pd.DataFrame(table)
    lo.s(f'Board:\n{df}')

    return df


def get_letters(img: np.ndarray):
    #all_letters = image_to_string(img, config=tess_conf)
    #lo.i(all_letters)

    bg_cop = img.copy()

    letters = []

    for i in range(0, 7 * 107, 107):
        bg_tl_x = i
        bg_tl_y = 0
        bg_br_x = i + 69
        bg_br_y = 66

        blet = bg_cop[bg_tl_y:bg_br_y, bg_tl_x:bg_br_x]

        blet[np.where(blet >= 50)] = 255

        text = image_to_string(blet, config=tess_conf)
        if not text:
            #todo fix for no letters
            text = '?'

        letters.append(text)

    lo.s(f'Letters:\n{letters}')

    return letters


def show_img(img_array: np.ndarray):
    Image.fromarray(img_array).show()


def parse_args() -> argparse.Namespace:
    #TODO: set customnamespace for completion here
    #https://stackoverflow.com/questions/42279063/python-typehints-for-argparse-namespace-objects

    parser = argparse.ArgumentParser(description='Extract text from a scrabble board')

    parser.add_argument('-f', '--file', type=str, required=True,
                        help='File path for the image')
    parser.add_argument('-t', '--type', type=str, choices=('big', 'small'), default='big', dest='img_type',
                        help='Type of board to parse (default: big)')
    parser.add_argument('-o', '--overwrite', action='store_true',
                        help='Overwrite existing files')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    #todo allow any log level to be passed

    return parser.parse_args()


def main(file: str, img_type: str, overwrite: bool = False, verbose: bool = False, **_kwargs):
    if verbose and lo.logger.getEffectiveLevel() > lo.get_level('VERBOSE'):
        lo.set_level('VERBOSE')

    direcs = Dirs(file, img_type)

    this_board = direcs.this_board_dir + 'board.pkl'
    this_letters = direcs.this_board_dir + 'letters.pkl'

    has_board = not overwrite and os.path.exists(this_board)
    has_letters = not overwrite and os.path.exists(this_board)

    if has_board and has_letters:
        lo.s('Info exists, skipping (override with overwrite = True)')

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

    cv2_image = get_img(file)

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
        if lo.is_enabled('d'):
            show_img(cv2_letters)

        rack = get_letters(cv2_letters)
        pd.to_pickle(rack, this_letters)

    lo.s('Done')


if __name__ == '__main__':
    args = parse_args()
    dargs = vars(args)
    main(**dargs)
