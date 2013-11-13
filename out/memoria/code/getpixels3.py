#!/opt/python2.7/bin/python
import sys
import numpy as np
import Image

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

def GPU():
    im = Image.open(sys.argv[1])
    print sys.argv[1], ": ", im.format, im.size, im.mode, '\n'

    pixels = np.array(im.getdata()).reshape(im.size[0], im.size[1], 3)
    print pixels
    print pixels.size
    #pixels = np.array(im)
    
    #gpu = cuda.mem_alloc(pixels.nbytes)
    #cuda.memcpy_htod(gpu, pixels)

    kernel = SourceModule("""
        #define MAX_PIXEL_VALUE 255
        #define THRESHOLD 50

        __global__ void process_pixel(int *pixels, int N)
        {
            // int id = blockDim.x*blockIdx.x + threadIdx.x;
            int idx = threadIdx.x;
            if (id < N) {
               if ( pixels[id] > THRESHOLD ) {
                   pixels[id] = MAX_PIXEL_VALUE;
               }
            }
            /*
            if ((r[id] > THRESHOLD) && (g[id] > THRESHOLD) && (b[id] > THRESHOLD)) {
                r[id] = MAX_PIXEL_VALUE;
                g[id] = MAX_PIXEL_VALUE;
                b[id] = MAX_PIXEL_VALUE;
            }
            */
        }
        """)

    func = kernel.get_function("process_pixel")
    func(cuda.InOut(pixels), np.int32(pixels.size), 
         block=(128,1,1), grid=(1,1) )

    #newpixels = np.zeros_like(pixels)
    #cuda.memcpy_dtoh(newpixels, gpu)
    
    print pixels
    im2 = Image.fromarray(np.uint8(pixels))
    im2.save("post.png")

if __name__ == "__main__":
    GPU()
