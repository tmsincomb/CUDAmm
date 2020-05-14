#!/home/605/sincomb/anaconda3/bin/python3
""" Matrix Multiplication Comparison!
This program will auto generate the columns for you so you just need to
input the matrix dimensions M N O for matrixes (MxN) & (NxO). Safe to assume the
N will be the same since matrix multipliation isn't possible without it.

Usage:
    ./project.py  (-h | --help)
    ./project.py  [--CUDA] M N O

Arguments:
    M                            Matrix A row count
    N                            Matrix A column count & matrix B row count
    O                            Matrix B column count

Options:
    -h, --help                   Prints out usage examples.
    -c, --CUDA                   Tells algo to use CUDA [default: False]

Terminal Examples:
    ./project.py -c 3 3 3        Square matrix multipliation with GPU
    ./project.py 3 3 3           Square matrix multipliation with only CPU
    ./project.py 4 3 7           Rectangular matrix multipliation
"""
import math # Built-in Math library
from time import time # Built-in time keeping library

from docopt import docopt # Easy Command Line I/O Library
import numpy as np # Linear alg library
import pandas as pd # Matrix processing
try:
    import pycuda.driver as cuda # Access GPU specifics
    import pycuda.autoinit # Automatically inits backend GPU stuffs for you
    from pycuda.compiler import SourceModule # Complie cuda kernels
except:
    print('WARNING :: You need an NVIDIA GPU for this to work!')


class MatrixMultiplication:

    # Inspired by :)
    # https://stackoverflow.com/questions/13896560/multiply-rectangular-matrices-in-cuda
    src_module = """
        __global__ void dot(int width, int height, const float *A, const float *B, float *C){

            unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
            unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
            float tmp_value = 0.0;

            // Makes sure we don't spill over grid parameters
            if ((row > height) || (col > width)) return;

            for(int i=0; i<height; ++i)
                tmp_value += A[row * height + i] * B[width * i + col];

            C[row*width + col] = tmp_value;
        }
    """

    def __init__(self, matrixA, matrixB, use_cuda=False, dim_block=16):
        self.module = SourceModule(self.src_module)
        self.matrixA = matrixA
        self.matrixB = matrixB
        self.use_cuda = use_cuda
        self.dim_block = dim_block

    @property
    def dot(self):
        if self.use_cuda:
            return self.gpu_dot()
        else:
            return self.cpu_dot()

    def cpu_dot(self):
        # split row iterations by threads; use hw1
        # ijk

    def gpu_dot(self):

        # Dimensions of the resulting matrix!
        width = self.matrixB.shape[1] # O
        height = self.matrixA.shape[0] # M

        # Create resulting dot matrix of (M x O) from (M x N)*(N * O)
        self.matrixC = np.empty([height, width])

        # Best default format type for PYCUDA with GPU
        self.matrixA = self.matrixA.astype(np.float32)
        self.matrixB = self.matrixB.astype(np.float32)
        self.matrixC = self.matrixC.astype(np.float32)

        # Allocate GPU memory for matrixes
        matrixA_mem_alloc = cuda.mem_alloc(self.matrixA.nbytes)
        matrixB_mem_alloc = cuda.mem_alloc(self.matrixB.nbytes)
        matrixC_mem_alloc = cuda.mem_alloc(self.matrixC.nbytes)

        # Copy matrixes to allocated GPU memory
        cuda.memcpy_htod(matrixA_mem_alloc, self.matrixA)
        cuda.memcpy_htod(matrixB_mem_alloc, self.matrixB)

        print(self.matrixA)
        print(self.matrixB)
        print(self.matrixC)

        # Dynamic grid for none squared matrix multiplication
        dim_grid_x = math.ceil(width / self.dim_block)
        dim_grid_y = math.ceil(height / self.dim_block)

        print(dim_grid_x)
        print(dim_grid_y)

        # Make sure grid is usable
        if (width % self.dim_block != 0) and (height % self.dim_block != 0):
            grid=(dim_grid_x+1, dim_grid_y+1, 1) # if matrix is smaller than block size
        else:
            grid=(dim_grid_x, dim_grid_y, 1)

        # Call specific function from CUDA kernel
        dot_product = self.module.get_function("dot");

        # Dot product of matrixA with matrixB using GPU
        dot_product(
            np.int32(width),
            np.int32(height),
            matrixA_mem_alloc,
            matrixB_mem_alloc,
            matrixB_mem_alloc,
            block=(self.dim_block,self.dim_block,1),
            grid=grid
        );

        # Copies completed dot product from GPU memory to normal memory
        cuda.memcpy_dtoh(self.matrixC, matrixC_mem_alloc)

        return self.matrixC


def main():
    args = docopt(__doc__) # grab command inputs into a dictionary
    print(args)

    use_cuda = args['--CUDA']
    M = int(args['M']) # matrixA rows
    N = int(args['N']) # matrixA cols & matrixB rows
    O = int(args['O']) # matrixB cols

    np.random.seed(seed=42) # make sure you get the same result each time.
    matrixA = np.random.rand(M,N)
    matrixB = np.random.rand(N,O)

    mm = MatrixMultiplication(matrixA, matrixB, use_cuda)

    start = time()
    mm_dot = mm.dot
    elapsed_time = time() - start

    # Numpy built-in matrix multiplication :: Sanity check
    numpy_dot = matrixA.dot(matrixB)

    print(mm_dot)
    print(numpy_dot)
    print(np.around(mm_dot, decimals=5) == np.around(numpy_dot, decimals=5))

if __name__ == '__main__':
    main()
