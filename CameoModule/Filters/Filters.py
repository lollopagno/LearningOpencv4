import cv2 as cv
import numpy as np
from CameoModule import Utilities as utils


def strokeEdges(src, dst, inverting=False, blurKsize=7, edgeKsize=5):

    blurredSrc = cv.medianBlur(src, blurKsize)  # Blurred image (LPF)
    graySrc = cv.cvtColor(blurredSrc, cv.COLOR_BGR2GRAY)  # Conversion to grayscale image

    cv.Laplacian(graySrc, cv.CV_8U, graySrc, ksize=edgeKsize)  # Laplacian edge detector

    # normalizedInverseAlpha = (255 - graySrc) / 255
    if inverting:
        graySrc = cv.bitwise_not(graySrc)  # Inversion bit-per-bt grayscale image

    normalized = graySrc / 255  # Normaliced image to [0,1]
    channels = cv.split(src)  # Splitting channels of the image

    for channel in channels:
        channel[:] = channel * normalized

    cv.merge(channels, dst)


class VFuncFilter(object):
    """
    A filter that applies a function to V (or all of BGR).
    """

    def __init__(self, vFunc=None, dtype=np.uint8):
        length = np.iinfo(dtype).max + 1
        self._vLookupArray = utils.createLookupArray(vFunc, length)

    def apply(self, src, dst):
        """
        Apply the filter with a BGR or gray source/destination.
        """

        srcFlatView = np.ravel(src)
        dstFlatView = np.ravel(dst)
        utils.applyLookupArray(self._vLookupArray, srcFlatView, dstFlatView)


class VCurveFilter(VFuncFilter):
    """
    A filter that applies a curve to V (or all of BGR).
    """

    def __init__(self, vPoints, dtype=np.uint8):
        VFuncFilter.__init__(self, utils.createCurveFunc(vPoints), dtype)


class BGRFuncFilter(object):
    """
    A filter that applies different functions to each of BGR.
    """

    def __init__(self, vFunc=None, bFunc=None, gFunc=None, rFunc=None, dtype=np.uint8):
        length = np.iinfo(dtype).max + 1

        self._bLookupArray = utils.createLookupArray(utils.createCompositeFunc(bFunc, vFunc), length)
        self._gLookupArray = utils.createLookupArray(utils.createCompositeFunc(gFunc, vFunc), length)
        self._rLookupArray = utils.createLookupArray(utils.createCompositeFunc(rFunc, vFunc), length)

    def apply(self, src, dst):
        """
        Apply the filter with a BGR source/destination.
        """

        b, g, r = cv.split(src)

        utils.applyLookupArray(self._bLookupArray, b, b)
        utils.applyLookupArray(self._gLookupArray, g, g)
        utils.applyLookupArray(self._rLookupArray, r, r)

        cv.merge([b, g, r], dst)


class BGRCurveFilter(BGRFuncFilter):
    """
    A filter that applies different curves to each of BGR.
    """

    def __init__(self, vPoints=None, bPoints=None, gPoints=None, rPoints=None, dtype=np.uint8):
        BGRFuncFilter.__init__(self,
                               utils.createCurveFunc(vPoints),
                               utils.createCurveFunc(bPoints),
                               utils.createCurveFunc(gPoints),
                               utils.createCurveFunc(rPoints), dtype)


class BGRPortraCurveFilter(BGRCurveFilter):
    """
    A filter that applies Portra-like curves to BGR.
    """

    def __init__(self, dtype=np.uint8):
        BGRCurveFilter.__init__(
            self,
            vPoints=[(0, 0), (23, 20), (157, 173), (255, 255)],
            bPoints=[(0, 0), (41, 46), (231, 228), (255, 255)],
            gPoints=[(0, 0), (52, 47), (189, 196), (255, 255)],
            rPoints=[(0, 0), (69, 69), (213, 218), (255, 255)],
            dtype=dtype)


class BGRProviaCurveFilter(BGRCurveFilter):
    """
    A filter that applies Provia-like curves to BGR.
    """

    def __init__(self, dtype=np.uint8):
        BGRCurveFilter.__init__(
            self,
            bPoints=[(0, 0), (35, 25), (205, 227), (255, 255)],
            gPoints=[(0, 0), (27, 21), (196, 207), (255, 255)],
            rPoints=[(0, 0), (59, 54), (202, 210), (255, 255)],
            dtype=dtype)


class BGRVelviaCurveFilter(BGRCurveFilter):
    """
    A filter that applies Velvia-like curves to BGR.
    """

    def __init__(self, dtype=np.uint8):
        BGRCurveFilter.__init__(
            self,
            vPoints=[(0, 0), (128, 118), (221, 215), (255, 255)],
            bPoints=[(0, 0), (25, 21), (122, 153), (165, 206), (255, 255)],
            gPoints=[(0, 0), (25, 21), (95, 102), (181, 208), (255, 255)],
            rPoints=[(0, 0), (41, 28), (183, 209), (255, 255)],
            dtype=dtype)


class BGRCrossProcessCurveFilter(BGRCurveFilter):
    """
    A filter that applies cross-process-like curves to BGR.
    """

    def __init__(self, dtype=np.uint8):
        BGRCurveFilter.__init__(
            self,
            bPoints=[(0, 20), (255, 235)],
            gPoints=[(0, 0), (56, 39), (208, 226), (255, 255)],
            rPoints=[(0, 0), (56, 22), (211, 255), (255, 255)],
            dtype=dtype)
