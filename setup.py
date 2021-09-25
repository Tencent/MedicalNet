#!/usr/bin/env python

import os

# Always prefer setuptools over distutils
from setuptools import find_packages, setup

# https://packaging.python.org/guides/single-sourcing-package-version/
# http://blog.ionelmc.ro/2014/05/25/python-packaging/
import med3d

_PATH_ROOT = os.path.dirname(__file__)


def _load_requirements(path_dir=_PATH_ROOT, comment_char='#'):
    with open(os.path.join(path_dir, 'requirements.txt')) as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = [ln[:ln.index(comment_char)] if comment_char in ln else ln for ln in lines]
    return reqs


# https://packaging.python.org/discussions/install-requires-vs-requirements /
# keep the meta-data here for simplicity in reading this file... it's not obvious
# what happens and to non-engineers they won't know to look in init ...
# the goal of the project is simplicity for researchers, don't want to add too much
# engineer specific practices
setup(
    name='MedicalNet',
    version=med3d.__version__,
    description=med3d.__docs__,
    author=med3d.__author__,
    author_email=med3d.__author_email__,
    url=med3d.__homepage__,
    license=med3d.__license__,
    packages=find_packages(exclude=['tests', 'docs']),
    long_description_content_type='text/markdown',
    include_package_data=True,
    zip_safe=False,
    keywords=['deep learning', 'pytorch', 'AI'],
    python_requires='>=3.6',
    setup_requires=[],
    install_requires=_load_requirements(_PATH_ROOT),
    classifiers=[
        'Environment :: Console',
        'Natural Language :: English',
        # How mature is this project? Common values are
        #   3 - Alpha, 4 - Beta, 5 - Production/Stable
        'Development Status :: 3 - Alpha',
        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Information Analysis',
        # Pick your license as you wish
        # 'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
    ],
)
