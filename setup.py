from __future__ import print_function
from setuptools import setup
import codecs
import os
import re

here = os.path.abspath(os.path.dirname(__file__))


# Get the version number from _version.py, and exe_path (learn from tombo)
verstrline = open(os.path.join(here, 'ccsmeth', '_version.py'), 'r').readlines()[-1]
vsre = r"^VERSION = ['\"]([^'\"]*)['\"]"
mo = re.search(vsre, verstrline)
if mo:
    __version__ = mo.group(1)
else:
    raise RuntimeError('Unable to find version string in "ccsmeth/_version.py".')


def read(*parts):
    # intentionally *not* adding an encoding option to open
    return codecs.open(os.path.join(here, *parts), 'r').read()


long_description = read('README.rst')


with open('requirements.txt', 'r') as rf:
    required = rf.read().splitlines()


setup(
    name='ccsmeth',
    packages=['ccsmeth', 'ccsmeth.utils'],
    keywords=['methylation', 'pacbio', 'neural network'],
    version=__version__,
    url='https://github.com/PengNi/ccsmeth',
    download_url='https://github.com/PengNi/ccsmeth/archive/refs/tags/{}.tar.gz'.format(__version__),
    license='BSD-3-Clause-Clear license',
    author='Peng Ni',
    # install_requires=['numpy>=1.15.3',
    #                   'statsmodels>=0.9.0',
    #                   'scikit-learn>=0.20.1',
    #                   'torch>=1.2.0,<=1.7.0',
    #                   ],
    install_requires=required,
    author_email='543943952@qq.com',
    description='Detecting DNA methylation from PacBio CCS reads',
    long_description=long_description,
    entry_points={
        'console_scripts': [
            'ccsmeth=ccsmeth.ccsmeth:main',
            ],
        },
    platforms=['Linux', 'MacOS'],
    zip_safe=False,
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'License :: OSI Approved :: BSD License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        ],
)
