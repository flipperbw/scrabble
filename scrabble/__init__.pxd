# from scrabble cimport p, logger
#
# __all__ = ['p', 'logger']

cimport numpy as cnp

ctypedef cnp.uint8_t BOOL_t
ctypedef cnp.uint16_t UINT16_t
ctypedef cnp.int_t INTC_t
ctypedef cnp.npy_intp INTP_t
ctypedef cnp.float32_t FLO_t
ctypedef cnp.ndarray cnparr
# ctypedef cnp.int32_t STR_t


#__all__ = ['BOOL_t', 'FLO_t', 'cnparr']
