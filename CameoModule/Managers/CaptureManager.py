import time
import numpy as np
import cv2 as cv


class CaptureManager(object):

    def __init__(self, capture, previewWindowManager=None, shouldMirrorPreview=False, shouldConvertBitDepth10To8 = True):
        """
        Capture manager class.
        :param capture:                 instance of the VideoCapture object (opencv)
        :param previewWindowManager:    instance of the WindowManager object
        :param shouldMirrorPreview:     if the frame shown is to be flipped
        """

        self.previewWindowManager = previewWindowManager
        self.shouldMirrorPreview = shouldMirrorPreview
        self.shouldConvertBitDepth10To8 = shouldConvertBitDepth10To8

        self._capture = capture
        self._channel = 0
        self._enteredFrame = False
        self._frame = None
        self._imageFilename = None

        # Video
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None

        # Frame rate
        self._startTime = None
        self._framesElapsed = 0
        self._fpsEstimate = None

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        if self._channel != value:
            self._channel = value
            self._frame = None

    @property
    def frame(self):

        if self._enteredFrame and self._frame is None:
            ret, self._frame = self._capture.retrieve(self._frame, self.channel)  # Get frame

            if self.shouldConvertBitDepth10To8 and self._frame is not None and self._frame.dtype == np.uint16:
                self._frame = (self._frame >> 2).astype(np.uint8)

        return self._frame

    @property
    def isWritingImage(self):
        return self._imageFilename is not None

    @property
    def isWritingVideo(self):
        return self._videoFilename is not None

    def enterFrame(self):
        """
        Capture the next frame, if any.
        """

        # But first, check that any previous frame was exited.
        assert not self._enteredFrame, 'previous enterFrame() had no matching exitFrame()'

        if self._capture is not None:
            self._enteredFrame = self._capture.grab()  # Synchronize the camera

    def exitFrame(self):
        """
        Draw to the window. Write to files. Release the frame.
        """

        # Check whether any grabbed frame is retrievable.
        # The getter may retrieve and cache the frame.
        if self.frame is None:
            self._enteredFrame = False
            return

        # Update the FPS estimate and related variables.
        if self._framesElapsed == 0:
            self._startTime = time.time()
        else:
            timeElapsed = time.time() - self._startTime
            self._fpsEstimate = self._framesElapsed / timeElapsed

        self._framesElapsed += 1

        # Draw to the window, if any.
        if self.previewWindowManager is not None:
            if self.shouldMirrorPreview:
                mirroredFrame = cv.flip(self._frame.copy(), 1)
                self.previewWindowManager.show(mirroredFrame, self._fpsEstimate)
            else:
                self.previewWindowManager.show(self._frame.copy(), self._fpsEstimate)

        # Write to the image file, if any.
        if self.isWritingImage:
            cv.imwrite(self._imageFilename, self._frame)
            self._imageFilename = None

        # Write to the video file, if any.
        self._writeVideoFrame()

        # Release the frame.
        self._frame = None
        self._enteredFrame = False

    def writeImage(self, filename):
        """
        Write the next exited frame to an image file.
        """

        self._imageFilename = filename

    def startWritingVideo(self, filename, encoding=cv.VideoWriter_fourcc('M', 'J', 'P', 'G')):
        """
        Start writing exited frames to a video file.
        """

        self._videoFilename = filename
        self._videoEncoding = encoding

    def stopWritingVideo(self):
        """
        Stop writing exited frames to a video file.
        """

        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None

    def _writeVideoFrame(self):

        if not self.isWritingVideo:
            return

        if self._videoWriter is None:
            fps = self._capture.get(cv.CAP_PROP_FPS)

            if fps <= 0.0:

                # The capture's FPS is unknown so use an estimate.
                if self._framesElapsed < 20:
                    # Wait until more frames elapse so that the
                    # estimate is more stable.
                    return

                else:
                    fps = self._fpsEstimate

            size = (int(self._capture.get(cv.CAP_PROP_FRAME_WIDTH)), int(self._capture.get(cv.CAP_PROP_FRAME_HEIGHT)))
            self._videoWriter = cv.VideoWriter(self._videoFilename, self._videoEncoding, fps, size)

        self._videoWriter.write(self._frame)
