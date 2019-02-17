#!/usr/bin/env python

import os
import sys

import cv2
import numpy as np
import pandas as pd
import pytesseract
#from PIL import Image

img_file = sys.argv[1]

img_dir = 'images/'
data_dir = 'data/'

box_dir = img_dir + 'boxes/'
board_dir = data_dir + 'boards/'

big_board_file = data_dir + 'default_board_big.pkl'
small_board_file = data_dir + 'default_board_small.pkl'

tess_conf = '--psm 8 --oem 0 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ"'

# color = True
color = False

big_board = pd.read_pickle(big_board_file)
small_board = pd.read_pickle(small_board_file)

img_file_root = os.path.splitext(img_file)[0] + '/'
this_img_dir = box_dir + img_file_root
this_board_dir = board_dir + img_file_root

for this_dir in (this_img_dir, this_board_dir):
    if not os.path.exists(this_dir):
        os.makedirs(this_dir)


def get_img():
    img = img_dir + img_file

    image = cv2.imread(img)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray[305:-130, 2:-2]

    if color:
        return cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
    else:
        return gray


cv2_image = get_img()
#Image.fromarray(cv2_image).show()


white = 255
empty = 236


def is_blank(val, c=white) -> bool:
    if color:
        return all(va == c for va in val)
    else:
        return val == c


def get_rows():
    _rows = cv2_image.copy()

    rowspace = 49
    # wanted_row = 7
    wanted_row = 0

    _rows = _rows[((wanted_row * rowspace) + 1):742]
    # _rows = _rows[193:742]

    return _rows

def get_boxes(box_rows):
    _rows_copy = box_rows.copy()

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
    found_bottom = False
    found_left = False
    found_right = False

    skip_rows = 0

    skip_cols = 0
    col_cnt = 0
    skipped_cols = 0

    max_top = 0
    max_bottom = 0

    _boxes = []

    for rn, trow in enumerate(_rows_copy):
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
                # print('== New {} =='.format((rn / 50) + 1))

                # print('start box Y at {}'.format(top_box_y))

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
                        # print('start box X at {}'.format(top_box_x))

                        col_cnt = 0
                        skip_cols = top_box_x + 40

                else:
                    if col_cnt > 14:
                        bottom_box_x = (top_box_x + 48)
                        found_right = True
                        skipped_cols = bottom_box_x
                    elif is_blank(rcol):
                        bottom_box_x = (cn + start_pxl_w - 1)
                        found_right = True
                        skipped_cols = 0
                    else:
                        col_cnt += 1

                    if found_right:
                        # print('end box X at {}'.format(bottom_box_x))

                        check_vals = _rows_copy[rn:(rn + (49 - check_y)), (cn - check_x)]
                        for r, v in enumerate(check_vals):
                            if is_blank(v):
                                bottom_box_y = max(r, (45 - check_y)) + rn
                                found_bottom = True
                                break

                        if not found_bottom:
                            bottom_box_y = len(check_vals) + rn
                            found_bottom = True

                        # print('end box Y at {}'.format(bottom_box_y))

                        max_bottom = max(max_bottom, bottom_box_y)
                        skip_rows = max_bottom + 1

                        # b_type = 'mult'
                        # bottom_box_y - 7

                        this_box = ((top_box_x, top_box_y), (bottom_box_x, bottom_box_y),)
                        _boxes.append(this_box)

                        # print(rows[this_box[1][1] -10][this_box[1][0] - 22])
                        # display(Image.fromarray(rows[this_box[0][1]:this_box[1][1], this_box[0][0]:this_box[1][0]]))

                        # print('------')

                        found_left = False
                        found_right = False
                        found_bottom = False

                new_check = True

    for b in _boxes:
        if color:
            cv2.rectangle(_rows_copy, b[0], b[1], (255, 0, 0), 1)
        else:
            cv2.rectangle(_rows_copy, b[0], b[1], 0, 1)

    return _boxes, _rows_copy

#%% show boxes

rows = get_rows()
boxes, rows_copy = get_boxes(rows)
#Image.fromarray(rows_copy).show()

#%%

def write_boxes():
    text_y_t = 13
    text_y_b = 2
    text_x_l = 5
    text_x_r = 5

    bi = 1

    for ba, b in enumerate(boxes):
        b_s_x = b[0][0]
        b_s_y = b[0][1]
        b_e_x = b[1][0]
        b_e_y = b[1][1]

        start = rows[b_s_y + 22][b_s_x + 7]

        if not is_blank(start, c=empty):
            fn = this_img_dir + '{}_nat.png'.format(ba)
            fa = this_img_dir + '{}.png'.format(ba)

            if os.path.exists(fn):
                continue

            crop_img = rows[
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


write_boxes()

# todo get board.shape, set correct board
default_board = big_board

table = np.full_like(default_board, '', dtype='U2')

num_rows, num_cols = table.shape
num_tiles = num_rows * num_cols

# todo: ignore x on start

for ix in range(num_tiles):
    fina = '{}.png'.format(ix)
    fint = '{}_nat.png'.format(ix)
    na = this_img_dir + fina
    nat = this_img_dir + fint

    iu = cv2.imread(na, cv2.IMREAD_GRAYSCALE)
    if not np.any(iu):
        continue

    row = (ix // num_rows)
    col = ix % num_cols

    text = pytesseract.image_to_string(iu, config=tess_conf)

    if text in ('TW', 'DW', 'TL', 'DL', '+'):
        if text == 'TW' and default_board[row][col] != '3w':
            print('problem')
        elif text == 'DW' and default_board[row][col] != '2w':
            print('problem')
        elif text == 'TL' and default_board[row][col] != '3l':
            print('problem')
        elif text == 'DL' and default_board[row][col] != '2l':
            print('problem')
        elif text == '+' and default_board[row][col] != 'x':
            print('problem')

        os.remove(na)
        os.remove(nat)

    elif len(text) == 1:
        table[row][col] = text

df = pd.DataFrame(table)
print(df)
df.to_pickle(this_board_dir + 'board.pkl')


#bottomgray = cv2_image[794:, 2:-2]
bottomgray = cv2_image[823:-8, 16:-16]

#cv2.imwrite(o, gray)

print(pytesseract.image_to_string(bottomgray, config=tess_conf))

bg_cop = bottomgray.copy()

letters = []

for i in range(0, 7 * 107, 107):
    bg_tl_x = i
    bg_tl_y = 0
    bg_br_x = i + 69
    bg_br_y = 66

    blet = bg_cop[bg_tl_y:bg_br_y, bg_tl_x:bg_br_x]

    blet[np.where(blet >= 50)] = 255

    text = pytesseract.image_to_string(blet, config=tess_conf)
    if not text:
        #todo fix for no letters
        text = '?'

    letters.append(text)

print(letters)

pd.to_pickle(letters, this_board_dir + 'letters.pkl')
