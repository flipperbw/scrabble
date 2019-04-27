"""Parse command line options"""

import argparse
import os
import re
import sys
import textwrap
from typing import Union

import verboselogs
from colored import attr, fg, stylize

from .logs import DEFAULT_LOGLEVEL, LEVELS


#https://i.stack.imgur.com/KTSQa.png

BLUE = fg(153)
RED = fg(160)
GREEN = fg(120)
GRAY = fg(245)
RESET = attr('reset')

ANSI_ESCAPE = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -/]*[@-~]')


class CustomArgumentParser(argparse.ArgumentParser):
    def print_help(self, file=None):
        super().print_help(file)
        if self.epilog and set(self.epilog) == {'\n'}:
            self._print_message(self.epilog, file)

    def exit(self, status=0, message=None):
        if message:
            self._print_message('\n{}\n'.format(stylize(message, RED)), sys.stderr)
        sys.exit(status)

    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(2, 'Error: {}'.format(message))


class CustomHelpFormatter(argparse.HelpFormatter):
    def __init__(self, prog):
        if sys.stdout.isatty():
            width = min(self._terminal_size()[0] - 2, 120)
        else:
            width = 120
        max_help = 40
        super().__init__(prog, max_help_position=max_help, width=width)

    def add_usage(self, usage, actions, groups, prefix=None):
        if prefix is None:
            prefix = '{}\n  '.format(stylize("Usage:", BLUE))
        return super().add_usage(usage, actions, groups, prefix)

    @staticmethod
    def _terminal_size():
        sizes = os.popen('stty size').read().split()
        if len(sizes) != 2:
            rows = 120
            cols = 120
        else:
            rows, cols = sizes
        return int(cols), int(rows)

    def _split_lines(self, text, width):
        split_text = text.split('\n')
        #return [line for para in split_text for line in ansiwrap.wrap(para, width)]
        return [line for para in split_text for line in textwrap.wrap(para, width)]

    def _fill_text(self, text, width, indent):
        split_text = text.split('\n')
        return '\n'.join(
            line for para in split_text for line in textwrap.wrap(
            #line for para in split_text for line in ansiwrap.wrap(
                para, width, initial_indent=indent, subsequent_indent=indent
            )
        )

    #def add_argument(self, action):
    #       invocation_length = max(len(ANSI_ESCAPE.sub('', s)) for s in invocations)

    def _format_action_invocation(self, action):
        if not action.option_strings or action.nargs == 0:
            #act_fmt = stylize(super()._format_action_invocation(action), GREEN)
            act_fmt = super()._format_action_invocation(action)
            if action.nargs == argparse.ONE_OR_MORE:
                return '{} [...]'.format(act_fmt)
            return act_fmt
        else:
            default = self._get_default_metavar_for_optional(action)
            args_string = self._format_args(action, default)
            #return ', '.join(stylize(a, GREEN) for a in action.option_strings) + ' %s' % args_string
            return ', '.join(action.option_strings) + ' %s' % args_string


def parser_init(
    description: str, usage: str = None, add_debug: bool = True, log_level: Union[int, str] = DEFAULT_LOGLEVEL, version: float = 1.0, **kwargs
):
    pargs = {
        'description': stylize(description, GRAY),
        'formatter_class': CustomHelpFormatter,
        'epilog': '\n'
    }

    if usage is not None:
        pargs['usage'] = stylize(usage, BLUE)
        #usage=f'{stylize("%(prog)s [options] username [...]", BLUE)}',  # todo

    if add_debug:
        pargs['add_help'] = False

    for k, v in kwargs.items():
        pargs[k] = v

    p = CustomArgumentParser(**pargs)

    p._positionals.title = 'Required'
    p._optionals.title = 'Flags'

    if add_debug:
        grp_debug = p.add_argument_group(title='Debug')

        if isinstance(log_level, int):
            log_level = verboselogs.logging.getLevelName(log_level)

        grp_debug.add_argument(
            '-l', '--log-level', type=str, default=str(log_level).lower(), choices=[l.lower() for l in LEVELS], metavar='<lvl>',
            help='Log level for output (default: %(default)s)\nChoices: {%(choices)s}'
        )
        grp_debug.add_argument(
            '-v', '--version', action='version', version='{}'.format(version),
            help="Show the version number and exit"
        )
        grp_debug.add_argument(
            '-h', '--help', action='help', default=argparse.SUPPRESS,
            help='Show this help message and exit'
        )

    return p

    # try:
    # except SystemExit as err:
    #     if err.code == 2:
    #         parser.print_help()
    #         sys.exit(2)
