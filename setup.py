
from os import path
import sys

from setuptools import find_packages, setup
from setuptools.command.install import install as InstallCommandBase

pkg_slug = 'mountaintop'

try:
    version_file = path.join(pkg_slug, 'VERSION')
    __version__ = '0.0.0'
    for content in open(version_file, 'r', encoding='utf8').readlines():
        content = content.strip()
        if len(content) > 0:
            __version__ = content
            break
except FileNotFoundError:
    __version__ = '0.0.0'

with open('requirements.txt') as f:
    base_dep = []
    for line in f.readlines():
        content = line.strip()
        # remove blank lines and comments
        if len(content) > 0 and content[0] != '#' and '-e git://' not in content:
            base_dep.append(content)



setup(
    name=pkg_slug,
    packages=find_packages(exclude=[
        '*.tests', '*.tests.*', 'tests.*', 'tests', 'test', 'docs', 'examples'
    ]),
    version=__version__,
    include_package_data=True,
    author='yinanxu',
    author_email='yinanxu0@gmail.com',
    description='make research work more friendly',
    # long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yinanxu0/mountaintop',
    install_requires=base_dep,
    # extras_require=extras_dep,
    # setup_requires=[
    #     'setuptools>=18.0',
    #     'pytest-runner',
    #     'black==19.3b0',
    #     'isort==4.3.21',
    # ],
    cmdclass={
        'install': InstallCommandBase,
    },
    tests_require=['pytest'],
    python_requires='>=3.7',
    entry_points={'console_scripts': [
            'mountaintop=mountaintop.bin.main:main', 
            'mt=mountaintop.bin.main:main'
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)