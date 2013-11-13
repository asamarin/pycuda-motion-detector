import cv

class VideoHandler:
    FIRST_FRAME = 1
    LAST_FRAME = -1

    video = None
    frames = []

    def __init__(self, video_path=None):
        if video_path is not None:
            self.video = cv.CaptureFromFile(video_path)
            while(1):
                frame = cv.QueryFrame(self.video)
                if frame:
                    self.frames.append(frame)
                else:
                    break

    def getWidth(self):
        return cv.GetCaptureProperty(self.video, cv.CV_CAP_PROP_FRAME_WIDTH)

    def getHeight(self):
        return cv.GetCaptureProperty(self.video, cv.CV_CAP_PROP_FRAME_HEIGHT)

    def getFPS(self):
        return cv.GetCaptureProperty(self.video, cv.CV_CAP_PROP_FPS)

    def getSize(self):
        """ Tupla compuesta por el ancho y el alto, convertidos a valores enteros """
        return (int(self.getWidth()), int(self.getHeight()))

    def getFrameCount(self):
        # Buggy:
        # return cv.GetCaptureProperty(self.video, cv.CV_CAP_PROP_FRAME_COUNT)
        return len(self.frames)

    def getAllFrames(self):
        return self.frames

    def getFrameRange(self, frame_range):
        return self.frames[frame_range[0]:frame_range[1]]

    def setAllFrames(self, frames):
        self.frames = frames

    def appendFrame(self, frame):
        self.frames.append(frame)

    def Save(self, filename):
        writer = cv.CreateVideoWriter(filename, cv.CV_FOURCC('M', 'J', 'P', 'G'), self.getFPS(), self.getSize(), 1)
        for i in self.frames:
            cv_im = cv.CreateImageHeader(i.size, cv.IPL_DEPTH_8U, 3)
            cv.SetData(cv_im, i.tostring())
            cv.WriteFrame(writer, cv_im)
