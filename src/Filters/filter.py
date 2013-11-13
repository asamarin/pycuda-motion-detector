import sys

import Image
from abc import ABCMeta, abstractmethod

from Filters.CUDAHandler import CUDAHandler

# --------------------------------------- #

class Filter:
    """ Clase padre de filtros """
    __metaclass__ = ABCMeta

    # Constantes para los valores minimos y maximos de los pixeles
    MAX_PIXEL_VALUE = 255
    MIN_PIXEL_VALUE = 0
    
    # CUDAHandler
    cuda = None

    # Constantes para indicar con que dispositivo se debe procesar el filtro
    CPU = 0
    CUDA = 1
    OPENCL = 2

    # Diccionario para mapear los modos de procesamiento (CPU, GPU, ...) con
    # los respectivos metodos abstractos que definiran posteriormente los filtros
    processing_method = {} 
        
    # Atributos
    images = []
    post_img = None
    
    def __init__(self, *images):
        for im in images:
            self.images.append(im)
        self.cuda = CUDAHandler()

        # El diccionario de metodos se inicializa en el constructor ya que
        # daba problemas crearlo directamente en los atributos
        self.processing_method = { 
            Filter.CPU    : self._processCPU,
            Filter.CUDA   : self._processCUDA,
            Filter.OPENCL : self._processCPU
        }

    # TODO Esquema de colores como parametro
    def newPostImg(self, mode, size):
        self.post_img = Image.new(mode, size)

    def fetchResult(self):
        return self.post_img

    @abstractmethod
    def _processCPU(self):
        pass

    @abstractmethod
    def _processCUDA(self):
        pass

    def Apply(self, mode):
        # Se instancia una nueva imagen postprocesada, que sera la que contenga
        # el resultado de haber aplicado el filtro a la/s imagen/es original/es
        self.newPostImg(self.images[0].mode, (self.images[0].size[0], self.images[0].size[1]))
        # Ahora se llama al metodo de procesado elegido
        self.processing_method[mode]()

# --------------------------------------- #


