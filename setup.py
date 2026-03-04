from setuptools import setup, Extension

try:
    from Cython.Build import cythonize

    ext_modules = cythonize(
        [Extension("battleship._fast_board", ["battleship/_fast_board.pyx"])],
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    )
except ImportError:
    ext_modules = []

setup(
    name="battleship",
    packages=["battleship"],
    ext_modules=ext_modules,
)
