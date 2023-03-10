from PIL import ImageFilter, Image
import numpy as np
import base64
from io import BytesIO

BASE64_PREAMBLE = "data:image/png;base64,"

def to_np(blob):
    return np.asarray(blob).astype(np.uint8)

def generate_mask(label_id, segmentation=None, segments_info=None):
    BLUR = 5
    BLUR_SIZE = BLUR * BLUR

    on = 255
    off = 0
    map_bits = np.vectorize(lambda p: on if p == label_id else off)
    mapped_np_img = map_bits(np.asarray(segmentation))
    mask = Image.fromarray(mapped_np_img.astype("uint8"), "L")
    mask = mask.filter(ImageFilter.MaxFilter(BLUR_SIZE))
    return mask.convert("RGB")


def np_to_pil(I, fmt="RGB"):
    return Image.fromarray(np.asarray(I).astype(np.uint8), fmt)


def pil_to_np(pil_img):
    return np.asarray(pil_img)


def pil_to_b64(pil_img):
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return BASE64_PREAMBLE + str(img_str)[2:-1]


def b64_to_pil(b64_str):
    return Image.open(BytesIO(base64.b64decode(b64_str.replace(BASE64_PREAMBLE, ""))))

