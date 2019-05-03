# cython: warn.maybe_uninitialized=True, warn.undeclared=True, warn.unused=True, warn.unused_arg=True, warn.unused_result=True, infer_types.verbose=True

"""Extract text from a scrabble board."""

cimport cython

from scrabble cimport logger as log
from scrabble.logger cimport can_log, lox, loi, lon, los, loc
#lod, lov, loe, clos

from scrabble.utils cimport print_board_clr

cdef object pickle, sys, Path, tpg, cv2, np, Image, _s , log_init

import pickle
import sys
from pathlib import Path
import typing as tpg

import cv2
import numpy as np
from PIL import Image

import scrabble.settings as _s

#from cpython.object cimport PyObject_Call
#from cpython.ref cimport PyObject, Py_TYPE, PyTypeObject
#from cpython.type cimport PyType_Check
#from cpython.function cimport PyFunction_Type

cnp.import_array()


DEF MAX_NODES = 15


cdef type BOOL = <type>np.uint8
cdef type UINT16 = <type>np.uint16  # move to init
#what happens if i use a DEF here?

cdef object npa = np.array
cdef object npz = np.zeros

cdef object cv2_inrange = cv2.inRange
cdef object cv2_imread = cv2.imread
cdef object cv2_resize = cv2.resize


# - img parsing

cdef int RACK_SPACE = 106
cdef float MIN_THRESH = 0.7

#[UINT16_t, ndim=1]
cdef cnparr LOWER_BLACK = npa([64, 22, 0], UINT16)
cdef cnparr UPPER_BLACK = npa([78, 36, 24], UINT16)
cdef cnparr LOWER_WHITE = npa([230, 230, 230], UINT16)
cdef cnparr UPPER_WHITE = npa([255, 255, 255], UINT16)

# todo all colors
cdef cnparr LOWER_BLACK_GR = npa([16, 54, 0], UINT16)
cdef cnparr UPPER_BLACK_GR = npa([30, 68, 24], UINT16)
cdef cnparr LOWER_BLACK_PU = npa([52, 0, 66], UINT16)
cdef cnparr UPPER_BLACK_PU = npa([66, 12, 80], UINT16)
cdef cnparr LOWER_BLACK_PA = npa([38, 0, 66], UINT16)
cdef cnparr UPPER_BLACK_PA = npa([52, 10, 80], UINT16)
cdef cnparr LOWER_BLACK_TE = npa([0, 15, 66], UINT16)
cdef cnparr UPPER_BLACK_TE = npa([12, 29, 80], UINT16)

#sm/big, board/lets, height/wid: start/end
cdef int _img_cut_range[2][2][2][2]
cdef int[:, :, :, :] IMG_CUT_RANGE = _img_cut_range
#cdef int[:, :, :, :] IMG_CUT_RANGE = npz((2, 2, 2, 2), np.intc)

IMG_CUT_RANGE[0][0][0][0] = 430
IMG_CUT_RANGE[0][0][0][1] = 1039
IMG_CUT_RANGE[0][0][1][0] = 71
IMG_CUT_RANGE[0][0][1][1] = -71

IMG_CUT_RANGE[0][1][0][0] = 1150
IMG_CUT_RANGE[0][1][0][1] = 1234
IMG_CUT_RANGE[0][1][1][0] = 1
IMG_CUT_RANGE[0][1][1][1] = -1

IMG_CUT_RANGE[1][0][0][0] = 305
IMG_CUT_RANGE[1][0][0][1] = 1049
IMG_CUT_RANGE[1][0][1][0] = 5
IMG_CUT_RANGE[1][0][1][1] = -5

IMG_CUT_RANGE[1][1][0][0] = 1120
IMG_CUT_RANGE[1][1][0][1] = 1204
IMG_CUT_RANGE[1][1][1][0] = 10
IMG_CUT_RANGE[1][1][1][1] = -10

#cdef int* mom2calc = [1, 2, 3] ?
#cdef int _img_cut_range[2][2][2][2]
#cdef int[] IMG_CUT_RANGE = cnp.PyArray_ZEROS(4, dims_vlets, cnp.NPY_INT32, 0)
#cdef int[] IMG_CUT_RANGE = npz((2, 2, 2, 2), np.


# todo put loadfile into here, maybe make utils file
# this can be converted todo

cdef class Dirs:
    def __cinit__(self, str img_file):
        cdef object img_file_root = Path(img_file).stem
        self.this_board_dir = Path(_s.BOARD_DIR, img_file_root)

        if not <bint>self.this_board_dir.exists():
            self.this_board_dir.mkdir()

        self.this_board = Path(self.this_board_dir, _s.BOARD_FILENAME)
        self.this_letters = Path(self.this_board_dir, _s.LETTERS_FILENAME)


cdef cnparr get_img(str img_name):
    # cdef cnparr[BOOL_t, ndim=3] image
    # cdef BOOL_t[:, :, :] colored
    cdef object img_path = Path(img_name).expanduser().as_posix()
    cdef cnparr image

    image = <cnparr>cv2_imread(img_path)

    if not np.any(image):
        loc(f'Could not find image, or it\'s empty: {img_name}')
        sys.exit(1)

    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return <cnparr>cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


cdef cnparr cut_img(cnparr img, bint is_big, bint is_lets):
    cdef int[:, :] r = IMG_CUT_RANGE[is_big, is_lets]
    return img[r[0,0]:r[0,1], r[1,0]:r[1,1]]


# todo: pickle this
cdef dict letter_templates = {}  # type: tpg.Dict[str, tpg.Dict[str, np.ndarray]]

@cython.wraparound(False)
cdef void create_letter_templates(bint is_big) except *:
    cdef str templ_dir_b = _s.TEMPL_DIR.as_posix() + '/'
    cdef str fbase = '.png'
    cdef str templ_f
    cdef object l

    #cdef unsigned char l
    #cdef unsigned char* templ_dir_c
    #templ_dir_c += b'/'
    #cdef str ls = 'abcdefghijklmnopqrstuvwxyz'
    #cdef str l
    # cdef Py_UCS4 l
    # cdef Py_UCS4* fname = '_.png'
    # cdef Py_UCS4* tem = templ_dir_b
    # cdef const unsigned char* fbase = '.png'
    # cdef unsigned char l

    #cdef cnparr[BOOL_t, ndim=2] templ_big, templ_small, templ_rack
    cdef BOOL_t[:, ::1] templ_big, templ_small, templ_rack
    cdef list tmp_list

    #cdef char *p1 = NULL

    cdef dict kw = {'flags': 0}
    cdef tuple dsize = (0, 0)
    cdef dict res_kw_s = {'dsize': dsize, 'fx': 1.12, 'fy': 1.13}
    cdef dict res_kw_r = {'dsize': dsize, 'fx': 2.1, 'fy': 2.1}

    for l in 'abcdefghijklmnopqrstuvwxyz':
        #fname[0] = l
        #templ_f = tem + fname

        templ_f = <str>((<str>templ_dir_b) + (<str>l) + (<str>fbase))

        (<dict>kw)['filename'] = templ_f

        #templ_big = PyObject_Call(ccc, (), kw)  # todo do this elsewhere
        #templ_big2 = ccc(templ_f, 0)
        #templ_big = <cnparr>cv_read(filename=templ_f, flags=0)
        #templ_big = cv_imread(**kw)   # todo is typetest faster than the extra refs?
        templ_big = <cnparr>cv2_imread(**kw)

        (<dict>res_kw_s)['src'] = templ_big.base
        (<dict>res_kw_r)['src'] = templ_big.base

        templ_small = cv2_resize(**res_kw_s)
        templ_rack = cv2_resize(**res_kw_r)

        if is_big:
            tmp_list = [templ_big]
        else:
            tmp_list = [templ_small]
        tmp_list.append(templ_rack)

        #board, rack
        letter_templates[l] = tmp_list


#cdef int TYP = cv2.TM_CCOEFF_NORMED
#cdef object mat = cv2.matchTemplate


# @cython.binding(True)
# @cython.linetrace(True)
# cpdef char[:, :] find_letter_match(BOOL_t[:, :] gimg, str typ, float spacing, char[:, :] dest):

@cython.wraparound(False)
cdef void find_letter_match(
    object gimg, bint is_rack, float spacing, Py_UCS4[:, ::1] dest
) except *:
    cdef:
        seen_tup seen[MAX_NODES][MAX_NODES]
        seen_tup *old_tup

        Py_UCS4 l
        object ld

        Py_ssize_t h, w, row_num, col_num, x, y
        float colx, rowx
        FLO_t confidence

        FLO_t[:, :] res
        #cnparr res
        #cnparr[FLO_t, ndim=2] res
        #cdef object res
        #(W-w+1) \times (H-h+1)

        #cnparr[BOOL_t, ndim=2] gb
        #cnparr[BOOL_t, ndim=2] tb
        BOOL_t[:, ::1] tb

        tuple match_locations
        INTC_t[:] match_x
        INTC_t[:] match_y

        BOOL_t match_type = cv2.TM_CCOEFF_NORMED
        object mat = cv2.matchTemplate
        object npw = np.where
        #PyArray_Where

    cdef Py_ssize_t sx, sy
    for sx in range(MAX_NODES):
        for sy in range(MAX_NODES):
            seen[sx][sy] = [0, 0]

    for l, ld in letter_templates.items():
        l = l.upper()

        tb = ld[is_rack]

        h = tb.shape[0]
        w = tb.shape[1]

        #gb = gimg.base
        #tb = template.base

        res = mat(image=gimg, templ=tb.base, method=match_type)
        #res = mat(image=gb, templ=tb, res=res, method=match_type)
        #res = mat(np.array(gimg), template.base, cv2.TM_CCOEFF_NORMED)

        #match_locations = res[(res >= MIN_THRESH)]
        match_locations = npw(res.base >= MIN_THRESH)

        match_x = (<tuple>match_locations)[0]
        match_y = (<tuple>match_locations)[1]

        for x, y in zip(match_y, match_x):
            colx = (x + x + w) / 2
            rowx = (y + y + h) / 2

            col_num = int(colx // spacing)
            row_num = int(rowx // spacing)

            confidence = res[y, x]

            old_tup = &seen[row_num][col_num]  # todo diff without pointer?
            if not old_tup.l:
                old_tup.l = l
                old_tup.conf = confidence
            else:
                if confidence <= old_tup.conf:
                    continue

                if l != old_tup.l and can_log('x'):
                    lox(f'overriding {old_tup.l, old_tup.conf} with new {l, confidence}')

                old_tup.l = l
                old_tup.conf = confidence

            dest[row_num, col_num] = l


def _u(): pass


cpdef void show_img(cnparr img_array):
    Image.fromarray(img_array).show()


# todo: SWITCH TO ONE SINGLE SHARED BOARD AND RACK OBJECT


@cython.wraparound(False)
cdef Py_UCS4[:, ::1] create_board(cnparr board, bint is_big):
    cdef float spacing
    #cdef int shape[2]
    cdef NINTP shape[2]

    if is_big:
        shape[:] = [15, 15]
        spacing = 49.6
    else:
        shape[:] = [11, 11]
        spacing = 55.3

    cdef cnparr[BOOL_t, ndim=2] black_mask, white_mask, comb
    #cdef object black_mask, white_mask, comb

    black_mask = cv2_inrange(board, LOWER_BLACK, UPPER_BLACK) + \
                 cv2_inrange(board, LOWER_BLACK_GR, UPPER_BLACK_GR) + \
                 cv2_inrange(board, LOWER_BLACK_PU, UPPER_BLACK_PU) + \
                 cv2_inrange(board, LOWER_BLACK_PA, UPPER_BLACK_PA) + \
                 cv2_inrange(board, LOWER_BLACK_TE, UPPER_BLACK_TE)
    white_mask = cv2_inrange(board, LOWER_WHITE, UPPER_WHITE)
    comb = black_mask + white_mask

    #cdef BOOL_t[:, :] gimg = cv2.bitwise_not(comb)
    #cdef cnparr[BOOL_t, ndim=2] gimg = cv2.bitwise_not(comb)
    cdef object gimg = cv2.bitwise_not(comb)  # type: np.ndarray

    if can_log('d'):
        show_img(gimg)

    #cdef char[:, ::1] table = npz(shape, '|S1')
    #cdef BOOL_t[:, ::1] table = cnp.PyArray_ZEROS(2, shape, cnp.NPY_UBYTE, 0)
    cdef Py_UCS4[:, ::1] table = cnp.PyArray_ZEROS(2, shape, cnp.NPY_UINT32, 0)

    find_letter_match(gimg, False, spacing, table)

    if can_log('n'):
        #pd.DataFrame(table.base.astype('<U1'))
        #{table.base.view("U1")}')
        lon('Board:')
        print_board_clr(table.base.astype(BOOL), log.KS_MAG)

    return table


@cython.wraparound(False)
cdef list get_rack(BOOL_t[:, :, :] img):
    cdef object img_base  # type: np.ndarray
    cdef object black_mask  # type: np.ndarray

    img_base = img.base
    black_mask = (
        <cnparr>(cv2_inrange(img_base, LOWER_BLACK, UPPER_BLACK) +
        cv2_inrange(img_base, LOWER_BLACK_GR, UPPER_BLACK_GR)) +
        cv2_inrange(img_base, LOWER_BLACK_PU, UPPER_BLACK_PU) +
        cv2_inrange(img_base, LOWER_BLACK_PA, UPPER_BLACK_PA) +
        cv2_inrange(img_base, LOWER_BLACK_TE, UPPER_BLACK_TE)
    )

    # cdef cnparr[BOOL_t, ndim=2] gimg = cv2.bitwise_not(black_mask)
    cdef cnparr gimg = cv2.bitwise_not(black_mask)

    if can_log('d'):
        show_img(gimg)

    cdef NINTP dims_rack[2]
    dims_rack[:] = [1, 7]

    #todo interchangeable
    # rack[0, 0] = 68
    # rack[0, 1] = b'D'

    #cdef char[:, ::1] rack3 = npz(dims_rack, '|S1')
    #cdef char[:, ::1] rack2 = npz((1,7), '|S1')
    #cdef char[:, ::1] rack = cnp.PyArray_ZEROS(2, dims_rack, cnp.NPY_UINT8, 0)
    #cdef BOOL_t[:, ::1] rack = cnp.PyArray_ZEROS(2, dims_rack, cnp.NPY_UBYTE, 0)
    #cdef cnp.uint32_t[:, ::1] rack2 = cnp.PyArray_ZEROS(2, dims_rack, cnp.NPY_UINT32, 0)
    #cdef char[:, ::1] rack3 = cnp.PyArray_ZEROS(2, dims_rack, cnp.NPY_UINT8, 0)
    cdef Py_UCS4[:, ::1] rack = cnp.PyArray_ZEROS(2, dims_rack, cnp.NPY_UINT32, 0)

    find_letter_match(gimg, True, RACK_SPACE, rack)

    cdef int buffer = 20

    #type long?
    cdef int mid_y = img.shape[0] // 2
    cdef int mid_x = RACK_SPACE // 2

    cdef Py_ssize_t start_y = mid_y - buffer
    cdef Py_ssize_t end_y = start_y + (buffer * 2)

    cdef list letters = []  # type: tpg.List[str]
    #cdef Py_UCS4* letters

    cdef Py_ssize_t i, start_x, end_x, p
    cdef Py_UCS4 l

    #cdef cnparr imgflat
    cdef BOOL_t[:] imgflat
    cdef bint allwhite
    cdef BOOL_t pixel
    cdef object npravel = np.ravel

    for i in range(rack.shape[1]):
        start_x = (i * RACK_SPACE) + mid_x - buffer
        end_x = start_x + (buffer * 2)

        imgflat = <cnparr>npravel(img[start_y:end_y, start_x:end_x]) # flatten()

        allwhite = True
        for p in range(imgflat.shape[0]):
            pixel = imgflat[p]
            if pixel != 255:
                allwhite = False
                break
        if allwhite is True:
            continue

        l = rack[0, i]
        if not l:
            letters.append('?')
        else:
            letters.append(l)  # vs encode? chr

    lon(f'Letters:\n{letters}')

    return letters


cdef void cmain(str filename, bint overwrite, str log_level) except *:
    cdef Py_UCS4 log_lvl_ch
    log_level = log_level.lower()
    if log_level == 'spam':
        log_lvl_ch = 'x'
    else:
        log_lvl_ch = (<str>log_level)[0]
    log.lo_lvl = log.lvl_alias[log_lvl_ch]

    cdef bint is_big = False
    cdef cnparr cv2_image = <cnparr>get_img(filename)  # bool_t ndim3

    # cdef BOOL_t[:, :, :] check_typ_pixels = cv2_image[650:850, 10:30]
    # cdef BOOL_t[:, :] row_val
    # cdef list check_first = check_typ_pixels.base[0].tolist()
    # cdef list row_val_l
    # cdef Py_ssize_t cti

    cdef cnparr check_typ_pixels = cv2_image[650:850, 10:30]  # type: np.ndarray
    cdef BOOL_t[:, :, :] check_typ_pixels_v = check_typ_pixels
    cdef list check_first = check_typ_pixels[0].tolist()
    cdef cnparr row_val
    cdef Py_ssize_t ci
    for ci in range(check_typ_pixels_v.shape[0]):
        row_val = check_typ_pixels[ci]
        if not row_val.tolist() == check_first:
            is_big = True
            break

    # for cti in range(check_typ_pixels.shape[0]):
    #     row_val = check_typ_pixels[cti]
    #     row_val_l = row_val.base.tolist()
    #     if not row_val_l == check_first:
    #         img_type = 'big'
    #         break

    cdef Dirs direcs = Dirs(filename)

    cdef object this_board
    cdef object this_letters
    this_letters = direcs.this_letters  # type: Path
    this_board = direcs.this_board  # type: Path

    cdef bint has_board = not overwrite and this_board.exists()
    cdef bint has_letters = not overwrite and this_letters.exists()

    cdef Py_UCS4[:, ::1] board
    cdef list rack

    if has_board and has_letters:
        lon('Info exists, skipping (override with overwrite = True)')

        if can_log('i'):
            board = pickle.load(this_board.open('rb')) # open better? todo
            rack = pickle.load(this_letters.open("rb"))

            #df_board = df_board.to_string(justify='left', col_space=2)
            #df_board.columns.rename('x', inplace=True)
            #df_board = df_board.to_string(formatters={'x': '{:<2s}'})
            #df_board.style.set_properties(**{'text-align': 'left'})
            #df_board.style.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])
            #todo why?

            loi(f'Board:\n{board.base}')
            loi(f'Letters: {rack}')

        los('Done')

        return

    create_letter_templates(is_big)

    #cdef cnparr[BOOL_t, ndim=3] cv2_board, cv2_letters
    cdef cnparr cv2_board, cv2_letters_b
    cdef BOOL_t[:, :, :] cv2_letters
    cdef object f

    if not has_board:
        loi('Parsing image...')

        cv2_board = cut_img(cv2_image, is_big, False)
        if can_log('d'):
            show_img(cv2_board)

        loi('Creating board...')
        board = create_board(cv2_board, is_big)
        #this_board.write_bytes(pickle.dumps(board.base))
        with this_board.open('wb') as f:
            #np.save
            pickle.dump(board.base, f)

    #cdef Py_UCS4[:] rack

    if not has_letters:
        loi('Parsing letters...')

        cv2_letters_b = cut_img(cv2_image, is_big, True)
        cv2_letters = cv2_letters_b
        if can_log('d'):
            show_img(cv2_letters_b)

        rack = get_rack(cv2_letters)

        #this_letters.write_bytes(pickle.dumps(rack))
        with this_letters.open('wb') as f:
            pickle.dump(rack, f)

    los('Done parsing image.')

def main(filename: str, overwrite: bool = False, log_level: str = _s.DEFAULT_LOGLEVEL, **_kw) -> None:
    cmain(filename, overwrite, log_level)
