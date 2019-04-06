import sys

from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize
from Cython.Compiler import Options

#import json
import numpy

#x = Options.get_directive_defaults()
#print(json.dumps(x, indent=4, sort_keys=True))
#print(Options)

Options.buffer_max_dims = 4
Options.closure_freelist_size = 16

# noinspection PyDictCreation
comp_directives = {
    "auto_pickle": False,
    "autotestdict": False,
    "boundscheck": False,

    # "c_string_encoding": "",
    # "c_string_type": "bytes",
    # "control_flow.dot_annotate_defs": false,
    # "control_flow.dot_output": "",

    #"fast_gil": True,
    "infer_types": True,
    "infer_types.verbose": True,
    "initializedcheck": False,
    #"language_level": '3str',
    "language_level": '3',
    #"np_pythran": True,

    # "optimize.inline_defnode_calls": true,
    # "optimize.unpack_method_calls": true,
    # "optimize.unpack_method_calls_in_pyinit": false,

    "warn.maybe_uninitialized": True,
    "warn.undeclared": True,
    "warn.unused": True,
    "warn.unused_arg": True,
    "warn.unused_result": True,

    #"wraparound": true
}

comp_directives['autotestdict'] = True
comp_directives['boundscheck'] = True
comp_directives['initializedcheck'] = True

extra_compile_args = [
    #"-Wall",
    "-Wextra",
    #'-fopenmp'
    "-ffast-math",  # speed?
    "-O3"
    #"-O1"
]

#define_macros = [('CYTHON_NO_PYINIT_EXPORT', '1')]
define_macros: list = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]

if '--prof' in sys.argv:
    comp_directives['profile'] = True
    #comp_directives['binding'] = True

    sys.argv.remove('--prof')

elif '--trace' in sys.argv:
    comp_directives['linetrace'] = True
    comp_directives['binding'] = True

    define_macros.append(('CYTHON_TRACE', '1'))

    sys.argv.remove('--trace')

ext_modules = [
    # Extension("p", ["p.pyx"],
    #     extra_compile_args=['-fopenmp'],
    #     extra_link_args=['-fopenmp'],
    # ),
    Extension(
        "*",
        ["*.pyx"],
        extra_compile_args=extra_compile_args,
        define_macros=define_macros,
        include_dirs=[numpy.get_include()],
        #extra_link_args=['-fopenmp']
    )
]


ext_options = {
    "compiler_directives": comp_directives,
    "annotate": True,
    #"cache": True # ?
}


setup(
    name='Scrabble parser',
    ext_modules=cythonize(ext_modules, **ext_options)
)


"""
"allow_none_for_extension_args": true,
"always_allow_keywords": false,
"annotation_typing": true,
"auto_cpdef": false,
"auto_pickle": null,
"autotestdict": true,
"autotestdict.all": false,
"autotestdict.cdef": false,
"binding": null,
"boundscheck": true,
"c_string_encoding": "",
"c_string_type": "bytes",
"callspec": "",
"ccomplex": false,
"cdivision": false,
"cdivision_warnings": false,
"control_flow.dot_annotate_defs": false,
"control_flow.dot_output": "",
"embedsignature": false,
"emit_code_comments": true,
"fast_getattr": false,
"fast_gil": false,
"formal_grammar": false,
"infer_types": null,
"infer_types.verbose": false,
"initializedcheck": true,
"iterable_coroutine": false,
"language_level": null,
"linetrace": false,
"nogil": false,
"nonecheck": false,
"np_pythran": false,
"old_style_globals": false,
"optimize.inline_defnode_calls": true,
"optimize.unpack_method_calls": true,
"optimize.unpack_method_calls_in_pyinit": false,
"optimize.use_switch": true,
"overflowcheck": false,
"overflowcheck.fold": true,
"preliminary_late_includes_cy28": false,
"profile": false,
"py2_import": false,
"remove_unreachable": true,
"set_initial_path": null,
"test_assert_path_exists": [],
"test_fail_if_path_exists": [],
"type_version_tag": true,
"unraisable_tracebacks": true,
"warn": null,
"warn.maybe_uninitialized": false,
"warn.multiple_declarators": true,
"warn.undeclared": false,
"warn.unreachable": true,
"warn.unused": false,
"warn.unused_arg": false,
"warn.unused_result": false,
"wraparound": true
"""
