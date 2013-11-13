#!/usr/bin/env python

import Image
import ImageChops

from videoHandler import VideoHandler
from Filters.filter import Filter
from Filters.erosion import ErosionFilter
from Filters.difference import DifferenceFilter
from Filters.threshold import ThresholdFilter

class MotionDetector:
   
    pre_video = None
    post_video = None

    def __init__(self, video_path):
        self.pre_video = VideoHandler(video_path)
        # TODO cambiar el constructor del videohandler
        self.post_video = VideoHandler()

    def launchCPU(self):
        frames_PIL = [Image.fromstring("RGB", self.pre_video.getSize(), i.tostring()) for i in self.pre_video.getAllFrames()]
        for i in xrange(1, len(frames_PIL)):
            # Diferencia
            diferencia = DifferenceFilter(frames_PIL[i], frames_PIL[i + 1])
            diferencia.Apply(Filter.CPU)
            tmp = diferencia.fetchResult()

            # Threshold
            threshold = ThresholdFilter(tmp, level=20)
            threshold.Apply(Filter.CPU)
            tmp2 = threshold.fetchResult()

            # Erosion
            erosion = ErosionFilter(tmp2)
            erosion.Apply(Filter.CPU)
            post = erosion.fetchResult()

            # TODO Mergeado en una clase aparte
            r, g, b = frames_PIL[i + 1].split()
            tmp = ImageChops.add(r, post)
            merged = Image.merge("RGB", (tmp, g, b))

            self.post_video.appendFrame(merged)
            print "#",

        # Terminado, salvamos resultado
        self.post_video.Save("./out.mpg")
    
    def launchCUDA(self):
        pass

