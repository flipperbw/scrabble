import sys

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Compiler.Options import get_directive_defaults

get_directive_defaults = get_directive_defaults()


get_directive_defaults['language_level'] = 3
get_directive_defaults['infer_types'] = True
get_directive_defaults['boundscheck'] = False

comp_directives = {
    'language_level': '3',
    'infer_types': True,
    'boundscheck': False,
#move to only p
    'warn.undeclared': True,
    'warn.maybe_uninitialized': True,
    'warn.unused': True,
    'warn.unused_arg': True,
    'warn.unused_result': True,
}

extra_compile_args = ["-Wall", "-Wextra", "-Wno-cpp"]
define_macros: list = []

if '--prof' in sys.argv:
    get_directive_defaults['linetrace'] = True
    get_directive_defaults['binding'] = True

    comp_directives['linetrace'] = True
    comp_directives['binding'] = True

    define_macros.append(('CYTHON_TRACE', '1'))

    sys.argv.remove('--prof')


ext_modules = [
    # Extension(
    #     "p",
    #     ["p.pyx"],
    #     extra_compile_args=['-fopenmp'],
    #     extra_link_args=['-fopenmp'],
    # ),
    Extension(
        "*",
        ["*.pyx"],
        extra_compile_args=extra_compile_args,
        define_macros=define_macros
    )
]


ext_options = {
    "compiler_directives": comp_directives,
    "annotate": True,
}


setup(
    name='Scrabble parser',
    ext_modules=cythonize(ext_modules, **ext_options)
)
