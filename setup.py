import sys

from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize
from Cython.Compiler import Options
from Cython.Distutils import build_ext
#from Cython.Distutils import build_ext, Extension

import compileall

import numpy  # todo need?

#import json
#x = Options.get_directive_defaults()
#print(json.dumps(x, indent=4, sort_keys=True))
#print(Options)

MOD_DIR = 'scrabble'

Options.buffer_max_dims = 5
Options.closure_freelist_size = 4096
#Options.closure_freelist_size = 2 ** 19
Options.annotate = True
#Options.clear_to_none = False

# noinspection PyDictCreation
comp_directives = {
    'allow_none_for_extension_args': False,
    #'annotation_typing': True,

    "auto_pickle": False,
    "autotestdict": False,
    "boundscheck": False,

    # "c_string_encoding": "",
    # "c_string_type": "bytes",
    # "control_flow.dot_annotate_defs": false,
    # "control_flow.dot_output": "",

    "embedsignature": True,

    #'fast_gil': True,
    #'final': True,  # todo check

    "initializedcheck": False,

    #'internal': True,

    "language_level": '3',  # '3str',
    #"np_pythran": True,

    #"set_initial_path": null,

    # "optimize.inline_defnode_calls": true,
    # "optimize.unpack_method_calls": true,
    # "optimize.unpack_method_calls_in_pyinit": false,

    'overflowcheck.fold': False,

    #"wraparound": true
}


extra_compile_args = [
    #"-Wall",
    "-Wextra",
    "-ffast-math",  # speed?
    "-O3"
    #"-O1"
    #'-fopenmp'
]

define_macros: list = [
    ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
    #('CYTHON_FAST_PYCALL', '1'),
    #('CYTHON_NO_PYINIT_EXPORT', '1')
]

debug_macros: list = []


if '--profile' in sys.argv:
    comp_directives['profile'] = True
    #comp_directives['binding'] = True

    #debug_macros.append(('CYTHON_TRACE_NOGIL', '1'))

    sys.argv.remove('--profile')

elif '--trace' in sys.argv:
    #comp_directives['profile'] = True
    # comp_directives['linetrace'] = True
    # comp_directives['binding'] = True

    #debug_macros.append(('CYTHON_TRACE', '1'))
    debug_macros.append(('CYTHON_TRACE_NOGIL', '1'))

    sys.argv.remove('--trace')

all_macros = define_macros + debug_macros

#include_dirs=['..', '.', './scrabble', '/home/brett/scrabble', '/home/brett/scrabble/scrabble']
include_dirs=['.']

extensions = [
    # Extension(
    #     #f"{MOD_DIR}.*",
    #     "*",
    #     [f"{MOD_DIR}/*.pyx"],
    #     #include_dirs=[numpy.get_include()],  # '/home/brett/scrabble/scrabble'
    #     #include_dirs=['.', MOD_DIR, './scrabble', '/home/brett/scrabble/scrabble', '/home/brett/scrabble'],  # '/home/brett/scrabble/scrabble'
    #     include_dirs=['.', MOD_DIR, numpy.get_include()],
    #     extra_compile_args=extra_compile_args,
    #     define_macros=define_macros,
    # )

    Extension(
        "scrabble.p",
        #"p",
        [f"{MOD_DIR}/p.pyx"],
        include_dirs=include_dirs + [numpy.get_include()],
        extra_compile_args=extra_compile_args,
        define_macros=all_macros,
        #extra_compile_args=['-fopenmp'],
        #extra_link_args=['-fopenmp'],
    ),

    Extension(
        "scrabble.ocr",
        [f"{MOD_DIR}/ocr.pyx"],
        include_dirs=include_dirs + [numpy.get_include()],
        extra_compile_args=extra_compile_args,
        define_macros=all_macros,
    ),

    Extension(
        "scrabble.logger",
        [f"{MOD_DIR}/logger.pyx"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        define_macros=debug_macros,
    ),

    # Extension(
    #     f"{MOD_DIR}/ocr", [f"{MOD_DIR}/ocr.pyx"],
    #     extra_compile_args=extra_compile_args,
    #     define_macros=define_macros,
    #     include_dirs=[
    #         numpy.get_include(),
    #         '/usr/include/opencv',
    #         '/usr/include/opencv2',
    #     ],
    #
    #     library_dirs=['/usr/lib', '/usr/lib/x86_64-linux-gnu'],
    #     libraries=['opencv_imgproc', 'opencv_saliency']
    #
    #     #extra_link_args=['-fopenmp']
    #     # extra_link_args=[
    #     #     '-lopencv_shape',
    #     #     '-lopencv_stitching',
    #     #     '-lopencv_superres',
    #     #     '-lopencv_videostab',
    #     #     '-lopencv_aruco',
    #     #     '-lopencv_bgsegm',
    #     #     '-lopencv_bioinspired',
    #     #     '-lopencv_ccalib',
    #     #     '-lopencv_datasets',
    #     #     '-lopencv_dpm',
    #     #     '-lopencv_face',
    #     #     '-lopencv_freetype',
    #     #     '-lopencv_fuzzy',
    #     #     '-lopencv_hdf',
    #     #     '-lopencv_line_descriptor',
    #     #     '-lopencv_optflow',
    #     #     '-lopencv_video',
    #     #     '-lopencv_plot',
    #     #     '-lopencv_reg',
    #     #     '-lopencv_saliency',
    #     #     '-lopencv_stereo',
    #     #     '-lopencv_structured_light',
    #     #     '-lopencv_phase_unwrapping',
    #     #     '-lopencv_rgbd',
    #     #     '-lopencv_viz',
    #     #     '-lopencv_surface_matching',
    #     #     '-lopencv_text',
    #     #     '-lopencv_ximgproc',
    #     #     '-lopencv_calib3d',
    #     #     '-lopencv_features2d',
    #     #     '-lopencv_flann',
    #     #     '-lopencv_xobjdetect',
    #     #     '-lopencv_objdetect',
    #     #     '-lopencv_ml',
    #     #     '-lopencv_xphoto',
    #     #     '-lopencv_highgui',
    #     #     '-lopencv_videoio',
    #     #     '-lopencv_imgcodecs',
    #     #     '-lopencv_photo',
    #     #     '-lopencv_imgproc',
    #     #     '-lopencv_core'
    #     # ]
    # ),
]

#compileall.compile_dir(MOD_DIR, maxlevels=0, optimize=2, workers=4)

ext_options = {
    "compiler_directives": comp_directives,
    'include_path': include_dirs,
    #"annotate": True,
    #"cache": True  #todo ?
}

setup_ext = cythonize(extensions, **ext_options)

setup(
    name='scrabble',
    #version, description, etc
    cmdclass = {'build_ext': build_ext},
    ext_modules=setup_ext,
    packages=[MOD_DIR],
    package_data = {
        'scrabble': ['*.pxd', '*.pyx']
    },
)

#"unraisable_tracebacks": true, # todo

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
"unraisable_tracebacks": true, # todo
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
