from typing import Dict
import os
import sys

import coloredlogs
import logging
import verboselogs

#logger.d("Houston, we have a %s", "thorny problem", exc_info=1)

#todo logging blank does not reset color?

levels = {
    #'NOTSET':   '_',  # 0
    'SPAM':     'x',  # 5
    'DEBUG':    'd',  # 10
    'VERBOSE':  'v',  # 15
    'INFO':     'i',  # 20
    'NOTICE':   'n',  # 25
    'WARNING':  'w',  # 30
    'SUCCESS':  's',  # 35
    'ERROR':    'e',  # 40
    'CRITICAL': 'c',  # 50
    #'ALWAYS':   'a',  # 60 # todo
}

if hasattr(sys, 'frozen'): #support for py2exe
    _srcfile = "logging%s__init__%s" % (os.sep, __file__[-4:])
elif __file__[-4:].lower() in ['.pyc', '.pyo']:
    _srcfile = __file__[:-4] + '.py'
else:
    _srcfile = __file__
_srcfile = os.path.normcase(_srcfile)

def log_init(level, **_kwargs):
    if isinstance(level, str):
        level = level.upper()
        #if level = 'DISABLED': # todo

        if not hasattr(logging, level):
            raise TypeError('Invalid level requested: {}'.format(level))

    level_styles = {}

    for ln, la in levels.items():
        logging.addLevelName(LogWrapper.get_level(ln), la)
        level_styles[la] = coloredlogs.DEFAULT_LEVEL_STYLES[ln.lower()]

    #logger = logging.getLogger(__name__)
    logger = verboselogs.VerboseLogger(__name__)

    # 'black', 'blue', 'cyan', 'green', 'magenta', 'red', 'white' and 'yellow'
    level_styles['i'] = {'color': 'cyan'}
    level_styles['info'] = {'color': 'cyan'}

    field_styles = coloredlogs.DEFAULT_FIELD_STYLES

    field_styles['lineno'] = {}
    field_styles['funcName'] = {'color': 'white', 'faint': True}
    field_styles['levelname'] = {}

    cl_config = {
        'level': level,
        'stream': sys.stdout,
        'level_styles': level_styles,
        'field_styles': field_styles,
        'fmt': '[%(lineno)3d] %(funcName)15s (%(levelname)s):  %(message)s',
    }
    # logger=logger,

    if 'PYCHARM_HOSTED' in os.environ:
        cl_config['isatty'] = True

    coloredlogs.install(**cl_config)

    return LogWrapper(logger)

def empty(*_args, **_kwargs):
    return

class LogWrapper(object):
    def __init__(self, logger):
        self.logger = logger

        self.enabled_levels: Dict[int, bool] = {}
        for ln in levels:
            got_level = self.get_level(ln)
            self.enabled_levels[got_level] = self.logger.isEnabledFor(got_level)


    def x(self, msg, *args, **kwargs):
        self._log(verboselogs.SPAM, msg, args, **kwargs)

    def d(self, msg, *args, **kwargs):
        if not self._is_enabled(10):
            self.d = empty
            return
        self._log(logging.DEBUG, msg, args, **kwargs)

    def v(self, msg, *args, **kwargs):
        self._log(verboselogs.VERBOSE, msg, args, **kwargs)

    def i(self, msg, *args, **kwargs):
        self._log(logging.INFO, msg, args, **kwargs)

    def n(self, msg, *args, **kwargs):
        self._log(verboselogs.NOTICE, msg, args, **kwargs)

    def w(self, msg, *args, **kwargs):
        self._log(logging.WARNING, msg, args, **kwargs)

    def s(self, msg, *args, **kwargs):
        self._log(verboselogs.SUCCESS, msg, args, **kwargs)

    def e(self, msg, *args, **kwargs):
        self._log(logging.ERROR, msg, args, **kwargs)

    def c(self, msg, *args, **kwargs):
        self._log(logging.CRITICAL, msg, args, **kwargs)


    def l(self, level, msg, *args, **kwargs):
        self._log(level, msg, args, **kwargs)

    @staticmethod
    def get_level(level) -> int:
        if isinstance(level, int):
            return level

        level_int = getattr(logging, level.upper(), None)
        if not isinstance(level_int, int):
            #if logging.raiseExceptions:
            raise TypeError('Invalid level: {}'.format(level))

        return level_int

    def is_enabled(self, level) -> bool:
        level = self.get_level(level)
        return self._is_enabled(level)

    def _is_enabled(self, level: int) -> bool:
        #todo: cache lookup?
        if level in self.enabled_levels:
            return self.enabled_levels[level]
        else:
            new_level = self.logger.isEnabledFor(level)
            self.enabled_levels[level] = new_level
            return new_level

    def _log(self, level, msg, args, exc_info=None, extra=None):
        if not self._is_enabled(level):
            return
        if _srcfile:
            try:
                fn, lno, func = self.find_caller()
            except ValueError:
                fn, lno, func = "(unknown file)", 0, "(unknown function)"
        else:
            fn, lno, func = "(unknown file)", 0, "(unknown function)"
        if exc_info:
            if not isinstance(exc_info, tuple):
                exc_info = sys.exc_info()
        record = self.logger.makeRecord(
            self.logger.name, level, fn, lno, msg, args, exc_info, func, extra
        )
        self.logger.handle(record)

    @staticmethod
    def find_caller():
        f = logging.currentframe()
        if f is not None:
            f = f.f_back
        rv = "(unknown file)", 0, "(unknown function)"
        while hasattr(f, "f_code"):
            co = f.f_code
            filename = os.path.normcase(co.co_filename)
            if filename == _srcfile:
                f = f.f_back
                continue
            rv = (co.co_filename, f.f_lineno, co.co_name)
            break
        return rv