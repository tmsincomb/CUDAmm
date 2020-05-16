# CUDAmm
Matrix multiplication analysis with CUDA

# Usage
""" Matrix Multiplication Comparison!

This program will auto generate the columns for you so you just need to
input the matrix dimensions M N O for matrixes (MxN) & (NxO). Safe to assume the
N will be the same since matrix multipliation isn't possible without it.

Usage:
    ./project.py  (-h | --help)
    ./project.py  M N O
    ./project.py  [--CUDA | --THREADS=<numeric_value>] M N O

Arguments:
    M                                Matrix A row count
    N                                Matrix A column count & matrix B row count
    O                                Matrix B column count

Options:
    -h, --help                       Prints out usage examples.
    -c, --CUDA                       Use GPU with 16 x 16 block size
    -t, --THREADS=<numeric_value>    Number of threads

Terminal Examples:
    ./project.py -c     3 3 3        Square matrix multipliation with GPU. Hardcoded block 16x16
    ./project.py -t 6   3 3 3        Square matrix multipliation with Multithreading of 6 threads
    ./project.py        4 3 7        Rectangular matrix multipliation (4 x 3) * (3 x 7)
"""
