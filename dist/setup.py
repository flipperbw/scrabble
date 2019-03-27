from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

#import Cython.Compiler.Options
#Cython.Compiler.Options.get_directive_defaults()['profile'] = False
#Cython.Compiler.Options.get_directive_defaults()['linetrace'] = True
#Cython.Compiler.Options.get_directive_defaults()['binding'] = True


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
        extra_compile_args=["-Wall", "-Wextra", "-Wno-cpp"]
    )
]


ext_options = {
    "compiler_directives": {
        'language_level': '3',
        #"profile": True,
        #'linetrace': True
    },
    "annotate": True,
}

setup(
    name='Scrabble parser',
    ext_modules=cythonize(ext_modules, **ext_options)
)
