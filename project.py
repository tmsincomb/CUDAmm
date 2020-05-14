#!/home/605/sincomb/anaconda3/bin/python3
""" Matrix Multiplication Comparison

Usage:
    ./project.py  (-h | --help)
    ./project.py  [--CUDA] M N O

Arguments:
    M
    N
    O

Options:
    -h, --help                      Prints out usage examples.
    -c, --CUDA                      Tells algo to use CUDA [default: False]

Terminal Examples:
    ./project.py -c 3 3 3
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
    src_module = """
        __global__ void dot(int width, int height, const float *A, const float *B, float *C){

            unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
            unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
            float tmp_value = 0.0;

            // Makes sure we don't spill over grid parameters
            if ((row > height) || (col > width)) return;

            for(int i=0; i<width; ++i)
                tmp_value += A[row * width + i] * B[width * i + col];

            C[row*n + col] = tmp_value;
        }
    """

    def __init__(self, matrixA, matrixB, use_cuda=False, dim_block=16):
        self.module = SourceModule(src_module)
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
        pass

    def gpu_dot(self):

        width = self.matrixB.shape[1] # O
        height = self.matrixA.shape[0] # M

        # Create resulting dot matrix of (M x O) from (M x N)*(N * O)
        self.matrixC = np.empty([height, width])
        self.matrixC = self.matrixC.astype(np.float32)

        # Allocate GPU memory for matrixes
        a_gpu = cuda.mem_alloc(self.matrixA.nbytes)
        b_gpu = cuda.mem_alloc(self.matrixB.nbytes)
        c_gpu = cuda.mem_alloc(self.matrixC.nbytes)

        # Copy matrixes to allocated GPU memory
        cuda.memcpy_htod(a_gpu, self.matrixA)
        cuda.memcpy_htod(b_gpu, self.matrixB)

        # Dynamic grid for none squared matrix multiplication
        dim_grid_x = math.ceil(width / self.dim_block)
        dim_grid_y = math.ceil(height / self.dim_block)

        # set grid size
        if (width % self.dim_block != 0) and (height % self.dim_block != 0):
            grid=(dim_grid_x+1, dim_grid_y+1, 1)
        else:
            grid=(dim_grid_x, dim_grid_y, 1)

        # Call specific function from CUDA kernel
        dot_product = mod.get_function("dot");

        # Dot product of matrixA with matrixB using GPU
        dot_product(
            np.int32(width),
            np.int32(height)
            a_gpu,
            b_gpu,
            c_gpu,
            block=(self.dim_block,self.dim_block,1),
            grid=grid
        );

        # Copies completed dot product from GPU memory to normal memory
        cuda.memcpy_dtoh(self.matrixC, matrixC)

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
    print(mm_dot == numpy_dot)

if __name__ == '__main__':
    main()
