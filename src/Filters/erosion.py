import itertools
from filter import Filter

class ErosionFilter(Filter):
    
    # Mascara de aplicacion del filtro; en el constructor de
    # la clase se explica con que se rellena esta lista
    mask = []

    _kernel = """
    #define   ROWS_BLOCKDIM_X 32
    #define   ROWS_BLOCKDIM_Y 8
    #define   ROWS_HALO_STEPS 1

    __global__ void erosion_kernel(
        float *d_Dst,
        float *d_Src,
        int imageW,
        int imageH,
        int pitch
    ){
        __shared__ float s_Data[ROWS_BLOCKDIM_Y][(2 * ROWS_HALO_STEPS) + ROWS_BLOCKDIM_X];

        //Offset to the left halo edge
        const int baseX = (blockIdx.x * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
        const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;

        d_Src += baseY * pitch + baseX;
        d_Dst += baseY * pitch + baseX;

        //Load main data
        #pragma unroll
        for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
            s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = d_Src[i * ROWS_BLOCKDIM_X];

        //Load left halo
        #pragma unroll
        for(int i = 0; i < ROWS_HALO_STEPS; i++)
            s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX >= -i * ROWS_BLOCKDIM_X ) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;

        //Load right halo
        #pragma unroll
        for(int i = 0; i < ROWS_HALO_STEPS; i++)
            s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;

        //Compute and store results
        __syncthreads();

        #pragma unroll
        for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS; i++){
            float result = 0;

            #pragma unroll
            for(int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
                result = (s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j] > 0) ? 255 : 0; 

            d_Dst[i * ROWS_BLOCKDIM_X] = result;
        }
    }
    """

    def __init__(self, *images):
        self.images = []
        super(ErosionFilter, self).__init__(*images)

        # La mascara contendra tuplas con las sumas que deben efectuarse a la 
        # posicion (x, y) del pixel actual que estamos tratando: por ejemplo, 
        # sumar (1, 0) significa (x + 1, y + 0), osea el pixel situado justo 
        # debajo, y por contra sumar (0, -1) implica (x + 0, y - 1), es decir 
        # el pixel situado justo a la izquierda del actual
        self.mask = [(i, j) for i, j in itertools.permutations([-1, 0, 1], 2) if abs(i) != abs(j)]
        
    def _processCPU(self):
        surrounding_pixels = []
        for row in xrange(self.images[0].size[0]):
            for col in xrange(self.images[0].size[1]):
                try:
                    surrounding_pixels = [self.images[0].getpixel((row + i, col + j)) for i, j in self.mask]
                except IndexError:
                    # Si se produjo alguna excepcion de indexado (debido a que estamos situados en algun borde
                    # de la imagen), construimos la lista de pixels adyacentes replicando el pixel actual
                    # TODO Intentar reconocer que pixel es el que tiro la excepcion, y replicar solo ese
                    surrounding_pixels = list(itertools.repeat(self.images[0].getpixel((row, col)), len(self.mask)))
                finally:
                    # La funcion implicita "all()" devuelve True si _todos_ los elementos de un objeto iterable
                    # son asimismo True (o distintos de 0 si los elementos son numericos, como es el caso)
                    if all(surrounding_pixels):
                        self.post_img.putpixel((row, col), Filter.MAX_PIXEL_VALUE)
                    else:
                        self.post_img.putpixel((row, col), Filter.MIN_PIXEL_VALUE)

    def _processCUDA(self):
        cuda.copyToGPU(self.images[0])
        cuda.setKernel(self._kernel)
        # TODO Dividir los grids atendiendo al ancho y alto de la imagen
        # cuda.Launch(...)
        cuda.getFromGPU()

