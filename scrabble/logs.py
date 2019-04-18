import logging
import os
import sys
from functools import partial
from typing import Dict

import coloredlogs
import verboselogs
from humanfriendly.terminal import ANSI_RESET, ansi_style


class partialmethod(partial):
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return partial(self.func, instance,
                       *(self.args or ()), **(self.keywords or {}))



#logger.d("Houston, we have a %s", "thorny problem", exc_info=1)

#todo logging blank does not reset color?

LEVELS = {
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

DEFAULT_LOGLEVEL = 20


class LogWrapper:
    def __init__(self, logger: verboselogs.VerboseLogger, skip_main: bool):
        self.logger = logger

        if hasattr(sys, 'frozen'):  # support for py2exe
            srcfile = "logging%s__init__%s" % (os.sep, __file__[-4:])
        elif __file__[-4:].lower() in ['.pyc', '.pyo']:
            srcfile = __file__[:-4] + '.py'
        else:
            srcfile = __file__

        self._srcfile = os.path.normcase(srcfile)

        self.skip_main = skip_main

        self.levels = LEVELS
        self.enabled_levels = {}  # type: Dict[int, bool]

        self._set_enabled()

    def _set_enabled(self, wrappers: bool = False):
        for ln, la in LEVELS.items():
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

    def l(self, level, msg, *args, **kwargs):
        self._log(level, msg, args, **kwargs)

    @staticmethod
    def get_level(level):
        if isinstance(level, int):
            return level

        level_int = getattr(logging, level.upper(), None)
        if not isinstance(level_int, int):
            #if logging.raiseExceptions:
            raise TypeError('Invalid level: {}'.format(level))

        return level_int

    def is_enabled(self, level):
        level = self.get_level(level)
        return self._is_enabled(level)

    def _is_enabled(self, level):
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
        if self._srcfile:
            try:
                fn, lno, func = self.find_caller()
            except ValueError:
                fn, lno, func = "(unk file)", 0, "(unk function)"
        else:
            fn, lno, func = "(unk file)", 0, "(unk function)"
        if exc_info:
            if not isinstance(exc_info, tuple):
                exc_info = sys.exc_info()

        record = self.logger.makeRecord(
            self.logger.name, level, fn, lno, msg, args, exc_info, func, extra
        )

        if record.processName == 'MainProcess':
            record.processName = 'Main'

        sys.stdout.write('\r')
        self.logger.handle(record)

        #if not record.msg: # TODO
        #    sys.stdout.write(ANSI_RESET)

    def find_caller(self):
        f = logging.currentframe()
        if f is not None:
            f = f.f_back
        rv = "(unk file)", 0, "(unk function)"
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

def _set_styles():
    #print(coloredlogs.DEFAULT_LEVEL_STYLES)

    level_styles = {
        la: coloredlogs.DEFAULT_LEVEL_STYLES[ln.lower()] for ln, la in LEVELS.items()
    }

    # 'black', 'blue', 'cyan', 'green', 'magenta', 'red', 'white' and 'yellow'  # todo
    level_styles['d'] = {'color': 'cyan'}  # make gray
    level_styles['debug'] = level_styles['d']
    level_styles['i'] = {'color': 'cyan', 'bright': True}
    level_styles['info'] = level_styles['i']
    level_styles['e'] = {'color': 'red', 'bold': True}
    level_styles['error'] = level_styles['c']
    level_styles['c'] = {'color': 'red'}
    level_styles['critical'] = level_styles['c']

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

    return level_styles, field_styles, form

def log_init(level = DEFAULT_LOGLEVEL, skip_main= False, **_kwargs):
    """Initialize logger.
    Level can be an integer, or a string in:
        SPAM, DEBUG, VERBOSE, INFO, NOTICE, WARNING, SUCCESS, ERROR, CRITICAL
    """
    #NOTSET, ALWAYS

    if isinstance(level, str):
        level = level.upper()
        #if level = 'DISABLED': # todo

        if not hasattr(logging, level):
            raise TypeError('Invalid level requested: {}'.format(level))

    for ln, la in LEVELS.items():
        conv_level = LogWrapper.get_level(ln)
        logging.addLevelName(conv_level, la)
        setattr(logging, la.upper(), conv_level)

    logger = verboselogs.VerboseLogger(__name__)

    level_styles, field_styles, form = _set_styles()

    cl_config = {
        'logger': logger,
        'level': 0,
        'stream': sys.stdout,
        'level_styles': level_styles,
        'field_styles': field_styles,
        'fmt': form
    }

    if 'PYCHARM_HOSTED' in os.environ:
        cl_config['isatty'] = True

    coloredlogs.install(**cl_config)

    logger.setLevel(level)

    return _create_wrapper(logger, skip_main)


def _create_wrapper(logger, skip_main):
    for ln, la in LEVELS.items():
        l_i = getattr(logging, ln)
        if logger.isEnabledFor(l_i):
            setattr(LogWrapper, la, partialmethod(LogWrapper.l, l_i))

    base = LogWrapper(logger, skip_main)

    return base
