from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

from Cython.Compiler.Options import get_directive_defaults

get_directive_defaults = get_directive_defaults()

# todo auto disable this

get_directive_defaults['language_level'] = 3
get_directive_defaults['infer_types'] = True
get_directive_defaults['boundscheck'] = False

# get_directive_defaults['linetrace'] = True
# get_directive_defaults['binding'] = True


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
        extra_compile_args=["-Wall", "-Wextra", "-Wno-cpp"],
        #define_macros=[('CYTHON_TRACE', '1')]
    )
]


ext_options = {
    "compiler_directives": {
        'language_level': '3',
        'infer_types': True,
        'boundscheck': False,

        # 'linetrace': True,
        # 'binding': True,
    },
    "annotate": True,
}


setup(
    name='Scrabble parser',
    ext_modules=cythonize(ext_modules, **ext_options)
)
