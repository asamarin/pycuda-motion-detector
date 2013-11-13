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

    #pixels = np.array(im.getdata()).reshape(im.size[0], im.size[1], 3)
    pixels = np.array(im.getdata())
    #print pixels

    red = np.array(pixels[:,0])
    green = np.array(pixels[:,1])
    blue = np.array(pixels[:,2])
    #red = []
    #green = []
    #blue = [] 
    #for i in pixels:
    #    red += [j for j in i[:,0]]
    #    green += [j for j in i[:,1]]
    #    blue += [j for j in i[:,2]]
    print "R = ", red
    print "G = ", green
    print "B = ", blue

    #r = np.array(red)
    #g = np.array(green)
    #b = np.array(blue)

    r_gpu = cuda.mem_alloc(red.nbytes)
    cuda.memcpy_htod(r_gpu, red)
    
    g_gpu = cuda.mem_alloc(green.nbytes)
    cuda.memcpy_htod(g_gpu, green)

    b_gpu = cuda.mem_alloc(blue.nbytes)
    cuda.memcpy_htod(b_gpu, blue)
 
    mod = SourceModule("""
        #define MAX_PIXEL_VALUE 255
        #define THRESHOLD 50

        __global__ void process_pixel(int* r, int* g, int* b) {

            int idx = threadIdx.x;

            if ((r[idx] > THRESHOLD) && (g[idx] > THRESHOLD) && (b[idx] > THRESHOLD)) {
                r[idx] = MAX_PIXEL_VALUE;
                g[idx] = MAX_PIXEL_VALUE;
                b[idx] = MAX_PIXEL_VALUE;
            }
        }
        """)

    func = mod.get_function("process_pixel")
    func(r_gpu, g_gpu, b_gpu, block=(4,128,1))

    r = np.empty_like(red)
    cuda.memcpy_dtoh(r, r_gpu)
    g = np.empty_like(green)
    cuda.memcpy_dtoh(g, g_gpu)
    b = np.empty_like(blue)
    cuda.memcpy_dtoh(b, b_gpu)

    print "R = ", r
    print "G = ", g
    print "B = ", b

#    im2 = Image.fromarray(np.uint8(newpixels))
#    im2.save("post.png")

if __name__ == "__main__":
    GPU()
