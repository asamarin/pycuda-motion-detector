#!/usr/bin/env python

import sys
import Image
import ImageChops

from motionDetector import MotionDetector
from Filters.filter import Filter
from Filters.erosion import ErosionFilter
from Filters.difference import DifferenceFilter
from Filters.threshold import ThresholdFilter

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'Usage:', sys.argv[0], '<image1> <image2'
        sys.exit(1)
    
    motion = MotionDetector(sys.argv[1])
    motion.launchCPU()
