from filter import Filter

class ThresholdFilter(Filter):

    # Valor discriminante (menores que este -> MIN_PIXEL_VALUE mayores que este -> MAX_PIXEL_VALUE)
    level = 127

    def __init__(self, *images, **kwargs):
        self.images = []
        super(ThresholdFilter, self).__init__(*images)
        try:
            self.level = kwargs['level']
        except KeyError:
            pass
        
    def _processCPU(self):
        grayscaled = self.images[0].convert("L")
        self.post_img = grayscaled.point(lambda x: Filter.MAX_PIXEL_VALUE if x >= self.level else Filter.MIN_PIXEL_VALUE)

    def _processCUDA(self):
        pass

