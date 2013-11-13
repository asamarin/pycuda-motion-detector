#!/usr/bin/python2.7

import cv
import sys
import Image

capture = cv.CaptureFromFile(sys.argv[1])
#frame_converted = cv.CreateImage((frame.width, frame.height), frame.depth, frame.channels)
frame = cv.CreateImage((int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH)), 
                        int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT))), 
                        8L, 3)
for i in xrange(1000):
    frame_converted = cv.CvtColor(cv.QueryFrame(capture), frame, cv.CV_BGR2RGB)
    im = Image.fromstring("RGB", (frame.width, frame.height), frame.tostring())
    im.save("frames/frame" + str(i) + ".jpg")
