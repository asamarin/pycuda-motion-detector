#!/opt/python2.7/bin/python

import sys
import Image
from abc import ABCMeta, abstractmethod

# --------------------------------------- #

class Filter:
    """ Clase padre de filtros """
    __metaclass__ = ABCMeta

    # Constantes para indicar con que dispositivo se debe procesar el filtro
    CPU = 0
    CUDA = 1
    OPENCL = 2

    # Atributos
    images = []
    post_img = None
    
    def __init__(self, *images):
        for im in images:
            self.images.append(im)

    # TODO Esquema de colores como parametro
    def new_post_img(self, mode, size):
        self.post_img = Image.new(mode, size)

    def fetch_result(self):
        return self.post_img

    @abstractmethod
    def Apply(self):
        pass

# --------------------------------------- #

class ErosionFilter(Filter):
    def __init__(self, *images):
        super(ErosionFilter, self).__init__(*images)
        
    def Apply(self):
        pass

# --------------------------------------- #

class DifferenceFilter(Filter):
    def __init__(self, *images):
        super(DifferenceFilter, self).__init__(*images)

    def Apply(self, mode):
        self.new_post_img(self.images[0].mode, 
                          (self.images[0].size[0], self.images[0].size[1]))
        for x in xrange(self.images[0].size[0]):
            for y in xrange(self.images[0].size[1]):
                # "diff" resultara ser una tupla de 3 elementos (en imagenes RGB) con la 
                # diferencia en valor absoluto por cada canal en ese pixel, comparado con el
                # mismo pixel de la imagen anterior
                diff = tuple([abs(a - b) for a,b in zip(self.images[0].getpixel((x, y)), self.images[1].getpixel((x, y)))])
                # img.putpixel((x, y), value)
                self.post_img.putpixel((x, y), diff)

# --------------------------------------- #

im1 = Image.open(sys.argv[1])
im2 = Image.open(sys.argv[2])
print sys.argv[1], ": ", im1.format, im1.size, im1.mode, '\n'
diferencia = DifferenceFilter(im1, im2)
diferencia.Apply(Filter.CPU)
post = diferencia.fetch_result()
post.save("post.png", "PNG")
