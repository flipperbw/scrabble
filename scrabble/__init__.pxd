# from scrabble cimport p, logger
#
# __all__ = ['p', 'logger']

cimport numpy as cnp


ctypedef cnp.ndarray cnparr

ctypedef cnp.int_t INTC_t
ctypedef cnp.intp_t SIZE_t
ctypedef cnp.npy_intp NINTP
ctypedef cnp.uint8_t BOOL_t
ctypedef cnp.uint16_t UINT16_t
ctypedef unsigned short STRU_t  # diff from above?
ctypedef cnp.int32_t STR_t
ctypedef cnp.uint32_t UINT32_t
ctypedef cnp.float32_t FLO_t


ctypedef unsigned char uchr
ctypedef unsigned char* uchrp
ctypedef const unsigned char cuchr
ctypedef const unsigned char* cuchrp
ctypedef const char cchr
ctypedef const char* cchrp

ctypedef const void * c_void


#__all__ = ['BOOL_t', 'FLO_t', 'cnparr']
