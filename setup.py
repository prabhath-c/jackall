from setuptools import setup, find_packages

setup(
    name='jackall',
    version='0.1.0',
    description='A versatile Python toolkit for multitasking in computational materials science',
    author='Prabhath Chilakalapudi',
    author_email='p.chilakalapudi@mpie.de',
    url='https://github.com/prabhath-c/jackall',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20',
        'pandas>=1.2',
        'matplotlib>=3.3',
        'pyiron'
    ]
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)