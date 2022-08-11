import cv2 as cv

from CameoModule.Managers.CaptureManager import CaptureManager
from CameoModule.Managers.WindowManager import WindowManager
from Common import Keycode
from CameoModule.CurveFilters import CurveFilters
from Funtionalities import Filters, Depth


class Cameo(object):
    """
    Cameo object
    """

    def __init__(self):
        self._windowManager = WindowManager('Cameo', self.onKeypress)
        self._captureManager = CaptureManager(cv.VideoCapture(0), self._windowManager, shouldMirrorPreview=True)
        self._curveFilter = CurveFilters.BGRPortraCurveFilter()

    def run(self):
        """
        Run the main loop.
        """

        self._windowManager.createWindow()

        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame

            if frame is not None:
                #CurveFilters.strokeEdges(frame, frame, inverting=True)
                self._curveFilter.apply(frame, frame)

            self._captureManager.exitFrame()
            self._windowManager.processEvents()

    def onKeypress(self, keycode):
        """
        Handle a keypress.
        space  -> Take a screenshot.
        tab    -> Start/stop recording a screencast.
        escape -> Quit.
        """

        if keycode == Keycode.SPACE:
            self._windowManager.screenshotText("Captured a screenshot!")
            self._captureManager.writeImage('screenshot.png')

        elif keycode == Keycode.TAB:

            if not self._captureManager.isWritingVideo:
                self._windowManager.screencastText("Start a video!")
                self._captureManager.startWritingVideo('screencast.avi')

            else:
                self._windowManager.screencastText("Stop a video!")
                self._captureManager.stopWritingVideo()

        elif keycode == Keycode.ESCAPE:
            self._windowManager.destroyWindow()


class CameoDepth(Cameo):
    """
    Cameo Depht object.
    Use to capture frame from depht camera.
    """

    def __init__(self):
        super().__init__()

        self._windowManager = WindowManager('Cameo', self.onKeypress)

        # device = cv2.CAP_OPENNI2 # Uncomment for Microsoft Kinect via OpenNI2
        device = cv.CAP_OPENNI2_ASUS  # Uncomment for Asus Xtion or Occipital Structure via OpenNI2
        self._captureManager = CaptureManager(cv.VideoCapture(device), self._windowManager, True)
        self._curveFilter = CurveFilters.BGRPortraCurveFilter()

    def run(self):
        """Run the main loop."""

        self._windowManager.createWindow()

        while self._windowManager.isWindowCreated:

            self._captureManager.enterFrame()
            self._captureManager.channel = cv.CAP_OPENNI_DISPARITY_MAP
            disparityMap = self._captureManager.frame

            self._captureManager.channel = cv.CAP_OPENNI_VALID_DEPTH_MASK
            validDepthMask = self._captureManager.frame

            self._captureManager.channel = cv.CAP_OPENNI_BGR_IMAGE
            frame = self._captureManager.frame

            if frame is None:
                # Failed to capture a BGR frame.
                # Try to capture an infrared frame instead.
                self._captureManager.channel = cv.CAP_OPENNI_IR_IMAGE
                frame = self._captureManager.frame

            if frame is not None:

                # Make everything except the median layer black.
                mask =  Depth.createMedianMask(disparityMap, validDepthMask)
                frame[mask == 0] = 0

                if self._captureManager.channel == cv.CAP_OPENNI_BGR_IMAGE:
                    # A BGR frame was captured.
                    # Apply filters to it.
                    Filters.strokeEdges(frame, frame)
                    self._curveFilter.apply(frame, frame)

            self._captureManager.exitFrame()
            self._windowManager.processEvents()
