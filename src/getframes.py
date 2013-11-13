#!/usr/bin/python2.7

import sys
import cv
import numpy as np
import Image

#files = sys.argv[1:]

#for f in files:
capture = cv.CaptureFromFile(sys.argv[1])
print capture

print cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH)
print cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT)

for i in xrange(100):
    frame = cv.QueryFrame(capture)

if frame:
    im = Image.fromstring("RGB", (frame.width, frame.height), frame.tostring())
    pixels = np.array(im.getdata())
    im.save("frame.png")
    print pixels
