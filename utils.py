import logging
import os
import sys
from functools import partialmethod, partial
from typing import Dict

import coloredlogs
import verboselogs
from humanfriendly.terminal import ANSI_RESET, ansi_style

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


class LogWrapper:
    def __init__(self, logger: logging.Logger, skip_main: bool):
        self.logger = logger

        if hasattr(sys, 'frozen'):  # support for py2exe
            srcfile = "logging%s__init__%s" % (os.sep, __file__[-4:])
        elif __file__[-4:].lower() in ['.pyc', '.pyo']:
            srcfile = __file__[:-4] + '.py'
        else:
            srcfile = __file__
        self._srcfile = os.path.normcase(srcfile)

        self.skip_main = skip_main

        self.enabled_levels: Dict[int, bool] = {}

        self._set_enabled()

    def _set_enabled(self, wrappers=False):
        for ln, la in levels.items():
            got_level = self.get_level(ln)
            enabled_lv = self.logger.isEnabledFor(got_level)
            self.enabled_levels[got_level] = enabled_lv

            if wrappers:
                if enabled_lv:
                    setattr(self, la, partial(self.l, got_level))
                else:
                    setattr(self, la, lambda *_a, **_k: None)

                #setattr(self, la, lambda gl=got_level, msg=None, *args, **kwargs: fnc(gl, msg, *args, **kwargs))

    def set_level(self, lv):
        self.logger.setLevel(lv)
        self._set_enabled(True)

    def x(self, *_ar, **_kw): pass
    def d(self, *_ar, **_kw): pass
    def v(self, *_ar, **_kw): pass
    def i(self, *_ar, **_kw): pass
    def n(self, *_ar, **_kw): pass
    def w(self, *_ar, **_kw): pass
    def s(self, *_ar, **_kw): pass
    def e(self, *_ar, **_kw): pass
    def c(self, *_ar, **_kw): pass

    def l(self, level: int, msg, *args, **kwargs):
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

    def _log(self, level: int, msg, args, exc_info=None, extra=None):
        if not self._is_enabled(level):
            return
        if self._srcfile:
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

        if record.processName == 'MainProcess':
            record.processName = 'Main'

        self.logger.handle(record)

        #if not record.msg: # TODO
        #    sys.stdout.write(ANSI_RESET)

    def find_caller(self) -> tuple:
        f = logging.currentframe()
        if f is not None:
            f = f.f_back
        rv = "(unknown file)", 0, "(unknown function)"
        while hasattr(f, "f_code"):
            co = f.f_code
            filename = os.path.normcase(co.co_filename)
            if filename == self._srcfile or \
                    (self.skip_main and co.co_name == 'main'):
                f = f.f_back
                continue
            rv = (co.co_filename, f.f_lineno, co.co_name)
            break
        return rv


def log_init(level, skip_main=True, **_kwargs) -> LogWrapper:
    if isinstance(level, str):
        level = level.upper()
        #if level = 'DISABLED': # todo

        if not hasattr(logging, level):
            raise TypeError('Invalid level requested: {}'.format(level))

    level_styles = {}

    for ln, la in levels.items():
        conv_level = LogWrapper.get_level(ln)
        logging.addLevelName(conv_level, la)
        setattr(logging, la.upper(), conv_level)
        level_styles[la] = coloredlogs.DEFAULT_LEVEL_STYLES[ln.lower()]

    #logger = logging.getLogger(__name__)
    logger = verboselogs.VerboseLogger(__name__)

    logger.setLevel(level)

    # 'black', 'blue', 'cyan', 'green', 'magenta', 'red', 'white' and 'yellow'  # todo
    level_styles['i'] = {'color': 'cyan', 'bright': True}
    level_styles['info'] = {'color': 'cyan', 'bright': True}
    level_styles['d'] = {'color': 'cyan'}
    level_styles['debug'] = {'color': 'cyan'}

    field_styles = coloredlogs.DEFAULT_FIELD_STYLES

    #pass maxlen as var forthese fields to set fixed width
    field_styles['levelname'] = {}
    field_styles['process'] = {'color': 'black', 'bright': True}
    field_styles['processName'] = {'color': 'black', 'bright': True}
    field_styles['funcName'] = {'color': 'white', 'faint': True}
    field_styles['lineno'] = {}

    form = (
        '{r}'
        '(%(levelname).1s) %(processName)-10.10s {gray}|{r} %(funcName)16s [%(lineno)3d]:  %(message)s'
        '{r}'
    ).format(r=ANSI_RESET, gray=ansi_style(color='black', bright=True))

    #use a formatter to set levelname color = msg (setFormatter?)
    #https://github.com/xolox/python-coloredlogs/blob/master/coloredlogs/__init__.py#L1113

    cl_config = {
        'logger': logger,
        'level': level,
        'stream': sys.stdout,
        'level_styles': level_styles,
        'field_styles': field_styles,
        'fmt': form
    }

    if 'PYCHARM_HOSTED' in os.environ:
        cl_config['isatty'] = True

    coloredlogs.install(**cl_config)

    return _create_wrapper(logger, skip_main)


def _create_wrapper(logger, skip_main) -> LogWrapper:
    for ln, la in levels.items():
        l_i = getattr(logging, ln)
        if logger.isEnabledFor(l_i):
            setattr(LogWrapper, la, partialmethod(LogWrapper.l, l_i))

    base = LogWrapper(logger, skip_main)

    return base
