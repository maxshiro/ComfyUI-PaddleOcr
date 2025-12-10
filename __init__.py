from .paddle_ocr import *
from .closed_shape_detector import *

NODE_CLASS_MAPPINGS = {
    "OcrBoxMask": OcrBoxMask,
    "OcrImageText": OcrImageText,
    "OcrBlur": OcrBlur,
    "ClosedShapesDetector": ClosedShapesDetector,
    "ConnectedComponentsDetector": ConnectedComponentsDetector,
}

