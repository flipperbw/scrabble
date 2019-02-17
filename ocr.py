#!/usr/bin/env python

import os
import sys
from typing import Union, List, Tuple

import cv2
import numpy as np
import pandas as pd
from pytesseract import image_to_string  # todo: image_to_boxes?
#from PIL import Image

from utils import log_init


# -- GLOBALS

log_level = 'INFO'

img_file = sys.argv[1]

img_dir = 'images/'
data_dir = 'data/'

box_dir = img_dir + 'boxes/'
board_dir = data_dir + 'boards/'

big_board_file = board_dir + 'default_board_big.pkl'
small_board_file = board_dir + 'default_board_small.pkl'

tess_conf = '--psm 8 --oem 0 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ"'

#overwrite = True
overwrite = False

# color = True
color = False

white = 255
empty = 236

# --

lo = log_init(log_level, skip_main=False)

class Dirs:
    def __init__(self):
        self.big_board = pd.read_pickle(big_board_file)
        self.small_board = pd.read_pickle(small_board_file)

        img_file_root = os.path.splitext(img_file)[0] + '/'
        self.this_img_dir = box_dir + img_file_root
        self.this_board_dir = board_dir + img_file_root

        self.make_dirs()

    def make_dirs(self):
        for this_dir in (self.this_img_dir, self.this_board_dir):
            if not os.path.exists(this_dir):
                os.makedirs(this_dir)

direcs = Dirs()


def get_img() -> np.ndarray:
    img = img_dir + img_file

    image = cv2.imread(img)

    if not np.any(image):
        lo.e(f'Could not find image, or is empty: {img}')
        sys.exit(1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray[305:-130, 2:-2]

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

    start_pxl_w = 3
    end_pxl_w = -3

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

        row_cut = trow[start_pxl_w:end_pxl_w]

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
                        top_box_x = cn + start_pxl_w
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
                        bottom_box_x = cn + start_pxl_w - 1
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


def write_boxes(_boxes: List[Tuple[Tuple[int, int], Tuple[int, int]]], _rows: np.ndarray):
    text_y_t = 13
    text_y_b = 2
    text_x_l = 5
    text_x_r = 5

    bi = 1

    for ba, b in enumerate(_boxes):
        b_s = b[0]
        b_e = b[1]

        b_s_x = b_s[0]
        b_s_y = b_s[1]
        b_e_x = b_e[0]
        b_e_y = b_e[1]

        start = _rows[b_s_y + 22][b_s_x + 7]

        if not is_blank(start, c=empty):
            fn = direcs.this_img_dir + '{}_nat.png'.format(ba)
            fa = direcs.this_img_dir + '{}.png'.format(ba)

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

            #crop_gray = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            #crop_gray = cv2.medianBlur(crop_gray, 3)

            #display(Image.fromarray(crop_gray))

            #fn = 'i_{}.png'.format(bi)
            #cv2.imwrite(fn, crop_gray)

            bi += 1


def create_board():
    # todo get board.shape, set correct board
    default_board = direcs.big_board

    # noinspection PyTypeChecker
    table = np.full_like(default_board, '', dtype='U2')

    num_rows, num_cols = table.shape
    num_tiles = num_rows * num_cols

    # todo: ignore x on start

    for ix in range(num_tiles):
        fina = '{}.png'.format(ix)
        fint = '{}_nat.png'.format(ix)
        na = direcs.this_img_dir + fina
        nat = direcs.this_img_dir + fint

        iu = cv2.imread(na, cv2.IMREAD_GRAYSCALE)
        if not np.any(iu):
            continue

        row = (ix // num_rows)
        col = ix % num_cols

        text = image_to_string(iu, config=tess_conf)

        if text in ('TW', 'DW', 'TL', 'DL', '+'):
            defb_text = default_board[row][col]

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


def get_letters(img):
    #bottomgray = img[794:, 2:-2]
    bottomgray = img[823:-8, 16:-16]

    all_letters = image_to_string(bottomgray, config=tess_conf)
    lo.i(all_letters)

    bg_cop = bottomgray.copy()

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


def main():
    this_board = direcs.this_board_dir + 'board.pkl'
    this_letters = direcs.this_board_dir + 'letters.pkl'

    has_board = not overwrite and os.path.exists(this_board)
    has_letters = not overwrite and os.path.exists(this_board)

    if has_board and has_letters:
        lo.s('Info exists, skipping (override with overwrite = True)')

        if lo.is_enabled('i'):
            lo.i(f'Board:\n{pd.read_pickle(this_board)}')
            lo.i(f'Letters: {pd.read_pickle(this_letters)}')

        lo.s('Done')

        return

    cv2_image = get_img()
    #Image.fromarray(cv2_image).show()

    if not has_board:
        rows = get_rows(cv2_image)
        boxes, rows_copy = get_boxes(rows)
        #Image.fromarray(rows_copy).show()

        write_boxes(boxes, rows)

        board = create_board()
        board.to_pickle(this_board)

    if not has_letters:
        rack = get_letters(cv2_image)
        pd.to_pickle(rack, this_letters)

    lo.s('Done')


if __name__ == '__main__':
    main()
