#!/home/605/sincomb/anaconda3/bin/python3
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
import math # Built-in Math library
from time import time # Built-in time keeping library

from docopt import docopt # Easy Command Line I/O Library
import numpy as np # Linear alg library
from multiprocessing.dummy import Pool as ThreadPool # dynamic multithreading
from multiprocessing import Process, sharedctypes
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
        __global__ void dot(int M, int N, int O, const float *A, const float *B, float *C){

            unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
            unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
            float tmp_value = 0;

            // Makes sure we don't spill over grid parameters
            if ((row < M) && (col < O)) {
                for(int i=0; i < N; ++i)
                    tmp_value += A[row * N + i] * B[O * i + col];
                // Makes sure threads are behaving. Random byte glitches WILL happen in giant matrixes.
                __syncthreads();
                C[row * O + col] = tmp_value;
            }
        }
    """

    def __init__(self, matrixA, matrixB):
        self.matrixA = matrixA
        self.matrixB = matrixB

    @property
    def dot(self):
        return self.matrixA.dot(self.matrixB)

    def cpu_dot(self, threads:int=None):
        """ Use allocated amount of threads to divide up the matrix rows for multipliation.

        :param int threads: [default: None] # will auto to max the CPU will allow.
        :returns: numpy matrix
        """

        def _dot(row_start, batch_size, M, N, O, matrixA, matrixB):
            """ Multiply Matrixes via batch & multithreading/processing """
            # Chunch size; number of rows per thread
            row_end = row_start + batch_size
            # Allocated memory for array accessed.
            matrixC = np.ctypeslib.as_array(shared_array)
            # Populate resulting matrix by specific rows
            for i in range(row_start, row_end):
                # Don't want to continue if row max dimension hit
                if i == M: break # Row end may exceed if batch not perfectly divisable.
                for k in range(0, N):
                    for j in range(0, O):
                        matrixC[i][j] += matrixA[i][k] * matrixB[k][j]

        # Dimensions of the resulting matrix!
        M, N = self.matrixA.shape # (M x N)
        O = self.matrixB.shape[1] # (N x O)

        # Create resulting dot matrix of (M x O) from (M x N)*(N * O)
        matrixC = np.zeros([M, O])
        result = np.ctypeslib.as_ctypes(matrixC)
        shared_array = sharedctypes.RawArray(result._type_, result)

        # Python has a 1 GIL limit where you can only use 1 thread per core.
        # I.E 6 cores == 6 threads
        batch_size = 1;
        if (M > threads):
            batch_size = math.ceil(M / threads)
        jobs = []
        for row_start in range(0, M, batch_size):
            process = Process(
                target=_dot, # Function you want to use.
                args=(row_start, batch_size,
                      M, N, O,
                      self.matrixA.copy(), self.matrixB.copy()))
            # Add next process job
            jobs.append(process)

        # Signals to close for multithreaded backend to be ready for finishing what's left.
        for job in jobs:
            job.start()
        # Wait for the pool to be done.
        for jon in jobs:
            job.join()

        # Pull shared matrix address amongst the cores/threads/processes
        matrixC = np.ctypeslib.as_array(shared_array)

        return matrixC

    @property
    def gpu_dot(self):#, dim_block:int=16):
        # Initialize gpu CUDA kernels
        self.module = SourceModule(self.src_module)

        # Dimensions of the resulting matrix!
        height, N = self.matrixA.shape # M, N
        width = self.matrixB.shape[1] # O

        # Create resulting dot matrix of (M x O) from (M x N)*(N * O)
        self.matrixC = np.zeros([height, width])

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

        # DEBUG
        # print(self.matrixA)
        # print(self.matrixB)
        # print(self.matrixC)

        dim_block = 16 # Stable block size for tuckoo

        dim_grid_x = math.ceil((width) / dim_block)
        dim_grid_y = math.ceil((height) / dim_block)

        # Call specific function from CUDA kernel
        dot_product = self.module.get_function("dot");

        # Dot product of matrixA with matrixB using GPU
        dot_product(
            np.int32(height),
            np.int32(N),
            np.int32(width),
            matrixA_mem_alloc,
            matrixB_mem_alloc,
            matrixC_mem_alloc,
            block=(dim_block, dim_block, 1),
            grid=(dim_grid_x, dim_grid_y),
        );

        # Copies completed dot product from GPU memory to normal memory
        cuda.memcpy_dtoh(self.matrixC, matrixC_mem_alloc)

        return self.matrixC


def main():
    args = docopt(__doc__) # grab command inputs into a dictionary
    print(args)

    dim_block = int(args['--CUDA']) if args.get('--CUDA') else None
    threads = int(args['--THREADS']) if args.get('--THREADS') else None
    M = int(args['M']) # matrixA rows
    N = int(args['N']) # matrixA cols & matrixB rows
    O = int(args['O']) # matrixB cols

    np.random.seed(seed=42) # make sure you get the same result each time.
    matrixA = np.random.rand(M,N)
    matrixB = np.random.rand(N,O)

    # Initialize class attributes
    mm = MatrixMultiplication(matrixA, matrixB)

    start = time()
    ### GPU ###
    if dim_block:
        # print('GPU!')
        dot = mm.gpu_dot # GPU multithreaded matrix multipliation
    ### CPU ###
    elif threads:
        # print('THREAD!')
        dot = mm.cpu_dot(threads=threads) # CPU multithreaded matrix multipliation
    else:
        # Numpy built-in matrix multiplication
        dot = mm.dot
    dot_elapsed_time = time() - start

    # Numpy built-in matrix multiplication :: Sanity check
    numpy_dot = matrixA.dot(matrixB)
    # if dim_block:
    #     numpy_dot = numpy_dot.astype(np.float32)

    # print(dot) # My matrix multipliation
    # print(numpy_dot) # Builti-in matrix multipliation
    # print(np.around(dot, decimals=2) == np.around(numpy_dot, decimals=2)) # Check individual cells for correctness

    ### This one will be a false, false in large matrixes due to byte changes that add up. ###
    print(np.array_equal(np.around(dot, decimals=2), np.around(numpy_dot, decimals=2))) #

    print(round(dot_elapsed_time, 2)) # Time for my dot product in seconds


if __name__ == '__main__':
    main()
