from PIL import Image
import pytesseract
import cv2
import os
import numpy as np
import pandas

i = 'scr.png'
o = 'gray.png'
t = 'template.png'

image = cv2.imread(i)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = gray[305:-130, 2:-2]

#cv2.imwrite(o, gray)

backtorgb = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)

#templ = cv2.imread(t, 0)
#trgb = cv2.cvtColor(templ, cv2.COLOR_GRAY2RGB)

#display(Image.fromarray(backtorgb))



def get_boxes():
    # color = True
    color = False

    start_pxl_w = 3
    end_pxl_w = -3

    if color:
        rows = backtorgb.copy()
    else:
        rows = gray.copy()

    rowspace = 49
    # wanted_row = 7
    wanted_row = 0

    rows = rows[((wanted_row * rowspace) + 1):742]
    # rows = rows[193:742]

    check_y = 37
    check_x = 22

    white = 255
    empty = 236

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

    def is_blank(val, c=white):
        if color:
            return all(va == c for va in val)
        else:
            return val == c

    boxes = []

    for rn, row in enumerate(rows):
        if rn < skip_rows:
            continue

        row_cut = row[start_pxl_w:end_pxl_w]

        if new_check:
            if not found_top:
                top_box_y = rn

            for cn, col in enumerate(row_cut):
                if not is_blank(col):
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

            for cn, col in enumerate(row_cut):
                if cn < skip_cols:
                    continue

                if not found_left:
                    if skipped_cols > 0:
                        top_box_x = skipped_cols + 1
                        found_left = True
                    elif not is_blank(col):
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
                    elif is_blank(col):
                        bottom_box_x = (cn + start_pxl_w - 1)
                        found_right = True
                        skipped_cols = 0
                    else:
                        col_cnt += 1

                    if found_right:
                        # print('end box X at {}'.format(bottom_box_x))

                        check_vals = rows[rn:(rn + (49 - check_y)), (cn - check_x)]
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
                        boxes.append(this_box)

                        # print(rows[this_box[1][1] -10][this_box[1][0] - 22])
                        # display(Image.fromarray(rows[this_box[0][1]:this_box[1][1], this_box[0][0]:this_box[1][0]]))

                        # print('------')

                        found_left = False
                        found_right = False
                        found_bottom = False

                new_check = True

                # if color:

    # else:
    # rows_copy = cv2.cvtColor(rows, cv2.COLOR_GRAY2RGB)

    rows_copy = rows.copy()

    for b in boxes:
        if color:
            cv2.rectangle(rows_copy, b[0], b[1], (255, 0, 0), 1)
        else:
            cv2.rectangle(rows_copy, b[0], b[1], 0, 1)



conf = '--psm 8 --oem 0 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ"'


#ignore plus on start

middle_x = [''] * 14

middle_x[3] = '2w'
middle_x[10] = '2w'

middle_y = [['']] * 15

middle_y[3] = ['2w']
middle_y[7] = ['x']
middle_y[11] = ['2w']

quad_tl = np.array([[''] * 7] * 7, dtype='U2')

quad_tl[0][3] = '3w'
quad_tl[0][6] = '3l'
quad_tl[1][2] = '2l'
quad_tl[1][5] = '2w'
quad_tl[2][1] = '2l'
quad_tl[2][4] = '2l'
quad_tl[3][0] = '3w'
quad_tl[3][3] = '3l'
quad_tl[4][2] = '2l'
quad_tl[4][6] = '2l'
quad_tl[5][1] = '2w'
quad_tl[5][5] = '3l'
quad_tl[6][0] = '3l'
quad_tl[6][4] = '2l'

quad_bl = np.flip(quad_tl, 0)
quad_tr = np.flip(quad_tl, 1)
quad_br = np.flip(quad_tl, (0,1))

quad_left = np.concatenate((quad_tl, quad_bl))
quad_right = np.concatenate((quad_tr, quad_br))

full = np.concatenate((quad_left, quad_right), axis=1)
full = np.insert(full, 7, middle_x, axis=0)
full = np.insert(full, [7], middle_y, axis=1)



df_default = pandas.DataFrame(full)



table = np.array([[''] * 15] * 15)


conf = '--psm 8 --oem 0 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ"'

# for ix in range(43):
for ix in range(225):
    fold = 'boxes'
    fina = '{}.png'.format(ix)
    fint = '{}_nat.png'.format(ix)
    na = '{}/{}'.format(fold, fina)
    nat = '{}/{}'.format(fold, fint)

    iu = cv2.imread(na, cv2.IMREAD_GRAYSCALE)
    if not np.any(iu):
        continue

    text = pytesseract.image_to_string(iu, config=conf)

    if text in ('TW', 'DW', 'TL', 'DL'):
        row = (ix // 15)
        col = ix % 15

        if text == 'TW' and full[row][col] != '3w':
            print('problem')
        elif text == 'DW' and full[row][col] != '2w':
            print('problem')
        elif text == 'TL' and full[row][col] != '3l':
            print('problem')
        elif text == 'DL' and full[row][col] != '2l':
            print('problem')

        os.rename(na, '{}/old/{}'.format(fold, fina))
        os.rename(nat, '{}/old/{}'.format(fold, fint))

    elif len(text) == 1:
        row = (ix // 15)
        col = ix % 15

        table[row][col] = text

df = pandas.DataFrame(table)
print(df)



#bottomgray = gray[794:, 2:-2]
bottomgray = gray[823:-8, 16:-16]

#cv2.imwrite(o, gray)

#backtorgb = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)

#templ = cv2.imread(t, 0)
#trgb = cv2.cvtColor(templ, cv2.COLOR_GRAY2RGB)


print(pytesseract.image_to_string(bottomgray, config=conf))

bg_cop = bottomgray.copy()

letters = []

for i in range(0, 7 * 107, 107):
    bg_tl_x = i
    bg_tl_y = 0
    bg_br_x = i + 69
    bg_br_y = 66

    blet = bg_cop[bg_tl_y:bg_br_y, bg_tl_x:bg_br_x]

    blet[np.where(blet >= 50)] = 255

    text = pytesseract.image_to_string(blet, config=conf)
    print(text)

    letters.append(text)

print(letters)



#letter_vals




wordlist = open('wordlist.txt', 'r')
words = wordlist.read().splitlines()


print(df)
print(df_default)


df.to_pickle('board.pkl')
df_default.to_pickle('default_board.pkl')
