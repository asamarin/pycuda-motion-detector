#!/opt/python2.7/bin/python

import re
import numpy as np

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

class CUDAHandler:

    def __init__(self):
        self.host = []
        self.device = []
        self.kernel = None
        self.func_name = ""
        self.operands = 0

    def _createGPUArray(self, i):
        try:
            self.host.append(np.array(i).astype(np.float32))
            self.device.append(cuda.mem_alloc(self.host[-1].nbytes))
        except ValueError:
            print "[CUDAHandler Error] Invalid datatype. All of its items mus be floats or integers."
            return
        except:
            print "[CUDAHandler Error] An error occurred. GPU array could not be create."
            return

    def loadData(self, **data):
            try:
                self.operands = len(data["input"])
                for i in data["input"] + data["output"]:
                   self._createGPUArray(i)            
                   cuda.memcpy_htod(self.device[-1], self.host[-1])
            except KeyError:
                print "[CUDAHandler Error] I/O mismatch."
                return
            except:
                print "[CUDAHandler Error] Input data could not be transfer to GPU."
                return

    def getFromGPU(self):
        for i in xrange(1, self.operands):
            cuda.memcpy_dtoh(self.host[-i], self.device[-i])        
        return self.host[self.operands:]

    def setKernel(self, kernel_str):
        self.kernel = SourceModule(kernel_str)
        regex = re.compile(r"\s*__global__\s+\w+\s+(\w+)")
        match = regex.search(kernel_str)
        try:
            self.func_name = match.groups()[0]
        except AttributeError:
            print "[CUDAHandler Error] Could not retrieve main kernel function name"
            raise

    # TODO Pasar por argumento un **kwargs directamente, serian:
    #      - hilos por bloque
    #      - numero de bloques (bloques por grid)
    #      - shape del bloque (tupla (x, y, z))
    #      - shape del grid (tupla (x, y))
    # Mirar lo de prepared_call() en vez de get_function()
    def Launch(self, blockdim, griddim=None):
        func = self.kernel.get_function(self.func_name)
        apply(func, self.device, {block:blockdim, grid:griddim})
