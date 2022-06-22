import cv2 as cv
from Common import Colors


class WindowManager(object):

    def __init__(self, windowName, keypressCallback=None):
        """
        Window manager class.
        :param windowName      : window name
        :param keypressCallback: callback function
        """

        self.keypressCallback = keypressCallback

        self._windowName = windowName
        self._isWindowCreated = False

        # Text
        self._elapsedFrameToShowText = 0

        self._textToShow = None
        self._textOrigin = None
        self._textColor = None
        self._fontScale = None
        self._textFont = None
        self._textThickeness = None

    @property
    def isWindowCreated(self):
        return self._isWindowCreated

    @property
    def isPuttingText(self):
        return self._textToShow is not None

    def createWindow(self):
        cv.namedWindow(self._windowName)
        self._isWindowCreated = True

    def show(self, frame, fps):

        if self._textToShow:

            # Put text to the image, if any.
            if self.isPuttingText and self._elapsedFrameToShowText != 0:
                cv.putText(frame, self._textToShow, org=self._textOrigin, fontFace=self._textFont,
                           fontScale=self._fontScale, color=self._textColor, thickness=self._textThickeness)

                self._elapsedFrameToShowText -= 1

                if self._elapsedFrameToShowText == 0:
                    self._textToShow = None

        if fps:
            cv.putText(frame, f"FPS: {round(fps, 2)}", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, Colors.RED, 2)

        cv.imshow(self._windowName, frame)

    def screenshotText(self, text):
        """
        Setting text to show in the image.
        """

        self._textToShow = text
        self._textOrigin, self._textFont, self._fontScale, self._textColor, self._textThickeness = getScreenshot()
        self._elapsedFrameToShowText = 20

    def screencastText(self, text):
        """
        Setting text to show in the image.
        """

        self._textToShow = text
        self._textOrigin, self._textFont, self._fontScale, self._textColor, self._textThickeness = getScreencast()
        self._elapsedFrameToShowText = 20

    def destroyWindow(self):
        cv.destroyWindow(self._windowName)
        self._isWindowCreated = False

    def processEvents(self):
        keycode = cv.waitKey(1)
        if self.keypressCallback is not None and keycode != -1:
            self.keypressCallback(keycode)


def getScreenshot():
    return (250, 450), cv.FONT_HERSHEY_SIMPLEX, 1, Colors.GREEN, 2


def getScreencast():
    return (400, 450), cv.FONT_HERSHEY_SIMPLEX, 1, Colors.ORANGE, 2
