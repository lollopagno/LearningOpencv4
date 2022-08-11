import cv2 as cv

from CameoModule.Managers.CaptureManager import CaptureManager
from CameoModule.Managers.WindowManager import WindowManager
from Common import Keycode
from CameoModule.CurveFilters import CurveFilters
from Funtionalities import Filters

class Cameo(object):

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
