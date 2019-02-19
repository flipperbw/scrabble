#!/usr/bin/env python

import os
from typing import Union, List, Tuple

import argparse
import cv2
import numpy as np
import pandas as pd
from pytesseract import image_to_string  # todo: image_to_boxes?
from PIL import Image

from utils import log_init


# -- GLOBALS

log_level = 'VERBOSE'
#log_level = 'INFO'

img_dir = 'images/'
data_dir = 'data/'

box_dir = img_dir + 'boxes/'
board_dir = data_dir + 'boards/'

default_board_files = {
    'big': board_dir + 'default_board_big.pkl',
    'small': board_dir + 'default_board_small.pkl'
}

tess_conf = '--psm 8 --oem 0 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ"'

# color = True
color = False

white = 255
empty = 236

# autodetect big vs small by looking at left col and seeing if all same color
img_cut_range = {
    'big': {
        'board': {
            'height': (305, 1050),
            'width': (5, -5)
        },
        'letters': {
            'height': (1126, 1198),
            'width': (18, -18)
        }
    },
    'small': {
        'board': {
            'height': (393, 1044),
            'width': (50, -50)
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

        self.this_img_dir = box_dir + img_file_root
        self.this_board_dir = board_dir + img_file_root

        self.make_dirs()

    def make_dirs(self):
        for this_dir in (self.this_img_dir, self.this_board_dir):
            if not os.path.exists(this_dir):
                os.makedirs(this_dir)


def cut_img(img: np.ndarray, typ: str, kind: str) -> np.ndarray:
    ranges = img_cut_range[typ][kind]
    height = ranges['height']
    width = ranges['width']

    return img.copy()[height[0]:height[1], width[0]:width[1]]


def get_img(img: str) -> np.ndarray:
    image = cv2.imread(img)

    if not np.any(image):
        raise Exception(f'Could not find image, or is empty: {img}')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if color:
        return cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
    else:
        return gray


def is_blank(val: Union[int, List[int]], c: int = white):
    if color:
        if not isinstance(val, list):
            lo.e('Expected a list, got an int')
            return None

        return all(va == c for va in val)
    else:
        return val == c


def get_rows(img: np.ndarray):
    img_copy: np.ndarray = img.copy()

    rowspace = 49
    # wanted_row = 7
    wanted_row = 0

    img_copy = img_copy[((wanted_row * rowspace) + 1):742]
    # img_copy = img_copy[193:742]

    return img_copy

def get_boxes(box_rows: np.ndarray):
    box_rows_copy: np.ndarray = box_rows.copy()

    #start_pxl_w = 3
    #end_pxl_w = -3

    check_y = 37
    check_x = 22

    top_box_x = 0
    top_box_y = 0
    bottom_box_x = 0
    bottom_box_y = 0

    new_check = True

    found_top = False

    skip_rows = 0

    col_cnt = 0
    skipped_cols = 0

    #max_top = 0
    max_bottom = 0

    _boxes = []

    for rn, trow in enumerate(box_rows_copy):
        if rn < skip_rows:
            continue

        #row_cut = trow[start_pxl_w:end_pxl_w]
        row_cut = trow

        if new_check:
            if not found_top:
                top_box_y = rn

            for cn, rcol in enumerate(row_cut):
                if not is_blank(rcol):
                    top_box_y = max(rn, top_box_y)
                    found_top = True

            if found_top:
                lo.d('== New {} =='.format((rn / 50) + 1))
                lo.d('start box Y at {}'.format(top_box_y))

                new_check = False
                skip_rows += check_y

        else:
            skip_cols = 0

            found_top = False
            found_bottom = False
            found_left = False
            found_right = False

            for cn, rcol in enumerate(row_cut):
                if cn < skip_cols:
                    continue

                if not found_left:
                    if skipped_cols > 0:
                        top_box_x = skipped_cols + 1
                        found_left = True
                    elif not is_blank(rcol):
                        #top_box_x = cn + start_pxl_w
                        top_box_x = cn
                        found_left = True

                    if found_left:
                        lo.d('start box X at {}'.format(top_box_x))

                        col_cnt = 0
                        skip_cols = top_box_x + 40

                else:
                    if col_cnt > 14:
                        bottom_box_x = top_box_x + 48
                        found_right = True
                        skipped_cols = bottom_box_x
                    elif is_blank(rcol):
                        #bottom_box_x = cn + start_pxl_w - 1
                        bottom_box_x = cn - 1
                        found_right = True
                        skipped_cols = 0
                    else:
                        col_cnt += 1

                    if found_right:
                        lo.d('end box X at {}'.format(bottom_box_x))

                        check_vals = box_rows_copy[rn:(rn + (49 - check_y)), (cn - check_x)]
                        for r, v in enumerate(check_vals):
                            if is_blank(v):
                                bottom_box_y = max(r, (45 - check_y)) + rn
                                found_bottom = True
                                break

                        if not found_bottom:
                            bottom_box_y = len(check_vals) + rn

                        lo.d('end box Y at {}'.format(bottom_box_y))

                        max_bottom = max(max_bottom, bottom_box_y)
                        skip_rows = max_bottom + 1

                        # b_type = 'mult'
                        # bottom_box_y - 7

                        this_box = ((top_box_x, top_box_y), (bottom_box_x, bottom_box_y),)
                        _boxes.append(this_box)

                        lo.d(box_rows_copy[this_box[1][1] -10][this_box[1][0] - 22])

                        lo.d('------')

                        found_left = False
                        found_right = False
                        found_bottom = False

                new_check = True

    for b in _boxes:
        if color:
            cv2.rectangle(box_rows_copy, b[0], b[1], (255, 0, 0), 1)
        else:
            cv2.rectangle(box_rows_copy, b[0], b[1], 0, 1)

    return _boxes, box_rows_copy


def write_boxes(
        _boxes: List[Tuple[Tuple[int, int], Tuple[int, int]]], _rows: np.ndarray, i_dir: str, overwrite=False
):
    text_y_t = 13
    text_y_b = 2
    text_x_l = 5
    text_x_r = 5

    for ba, b in enumerate(_boxes):
        b_s = b[0]
        b_e = b[1]

        b_s_x = b_s[0]
        b_s_y = b_s[1]
        b_e_x = b_e[0]
        b_e_y = b_e[1]

        start = _rows[b_s_y + 22][b_s_x + 7]

        if not is_blank(start, c=empty):
            fn = i_dir + '{}_nat.png'.format(ba)
            fa = i_dir + '{}.png'.format(ba)

            if not overwrite and os.path.exists(fn) and os.path.exists(fa):
                continue

            crop_img = _rows[
               b_s_y + text_y_t:b_e_y - text_y_b,
               b_s_x + text_x_l:b_e_x - text_x_r
            ]

            if color:
                crop_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY).copy()
            else:
                crop_gray = crop_img.copy()

            cv2.imwrite(fn, crop_gray)

            has_white = crop_gray[np.where(crop_gray == 255)]
            if len(has_white):
                crop_gray[np.where(crop_gray >= 203)] = 0
                crop_gray[np.where(crop_gray != 0)] = 255
            else:
                crop_gray[np.where(crop_gray >= 50)] = 255

            cv2.imwrite(fa, crop_gray)

            #blur_gray = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            #blur_gray = cv2.medianBlur(crop_gray, 3)
            #fb = _dir + '{}_blur.png'.format(ba)
            #cv2.imwrite(fb, blur_gray)


def create_board(def_board: np.ndarray, i_dir: str):
    # noinspection PyTypeChecker
    table = np.full_like(def_board, '', dtype='U2')

    num_rows, num_cols = table.shape
    num_tiles = num_rows * num_cols

    # todo: ignore x on start

    for ix in range(num_tiles):
        fina = '{}.png'.format(ix)
        fint = '{}_nat.png'.format(ix)
        na = i_dir + fina
        nat = i_dir + fint

        iu = cv2.imread(na, cv2.IMREAD_GRAYSCALE)
        if not np.any(iu):
            continue

        row = (ix // num_rows)
        col = ix % num_cols

        text = image_to_string(iu, config=tess_conf)

        if text in ('TW', 'DW', 'TL', 'DL', '+'):
            defb_text = def_board[row][col]

            if text == 'TW' and defb_text != '3w':
                lo.e(f'problem: {text} vs {defb_text}')
            elif text == 'DW' and defb_text != '2w':
                lo.e(f'problem: {text} vs {defb_text}')
            elif text == 'TL' and defb_text != '3l':
                lo.e(f'problem: {text} vs {defb_text}')
            elif text == 'DL' and defb_text != '2l':
                lo.e(f'problem: {text} vs {defb_text}')
            elif text == '+' and defb_text != 'x':
                lo.e(f'problem: {text} vs {defb_text}')

            os.remove(na)
            os.remove(nat)

        elif len(text) == 1:
            table[row][col] = text

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


def show_img(img_array):
    Image.fromarray(img_array).show()


def parse_args():
    parser = argparse.ArgumentParser(description='Extract text from a scrabble board')

    parser.add_argument('-f', '--file', type=str, required=True,
                        help='File path for the image')
    parser.add_argument('-t', '--type', type=str, choices=('big', 'small'), default='big',
                        help='Type of board to parse (default: big)')
    parser.add_argument('-o', '--overwrite', action='store_true',
                        help='Overwrite existing files')

    return parser.parse_args()


def main(img_file, img_typ, overwrite):
    direcs = Dirs(img_file, img_typ)

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

    cv2_image = get_img(img_file)

    if not has_board:
        lo.i('Parsing image...')

        cv2_board = cut_img(cv2_image, img_typ, 'board')

        rows = get_rows(cv2_board)
        boxes, rows_copy = get_boxes(rows)
        if lo.is_enabled('v'):
            show_img(rows_copy)

        lo.i('Writing box files...')
        write_boxes(boxes, rows, direcs.this_img_dir, overwrite=overwrite)

        lo.i('Creating board...')
        board = create_board(direcs.default_board, direcs.this_img_dir)
        board.to_pickle(this_board)

    if not has_letters:
        lo.i('Parsing letters...')

        cv2_letters = cut_img(cv2_image, img_typ, 'letters')
        if lo.is_enabled('v'):
            show_img(cv2_letters)

        rack = get_letters(cv2_letters)
        pd.to_pickle(rack, this_letters)

    lo.s('Done')


if __name__ == '__main__':
    args = parse_args()
    main(args.file, args.type, args.overwrite)
