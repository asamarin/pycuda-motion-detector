import ImageChops
from filter import Filter

class DifferenceFilter(Filter):
    
    def __init__(self, *images):
        self.images = []
        super(DifferenceFilter, self).__init__(*images)

    def _processCPU(self):
        self.post_img = ImageChops.difference(self.images[0], self.images[1])

    def _processCUDA(self):
        cuda.loadData(input=self.images, output=self.post_img)
        from Kernels.difference_kernel import diffkernel
        cuda.setKernel(diffkernel)
        cuda.Launch((4,4,1))
