"""Font matching must measure transformations rather than silently change type."""
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from text_fidelity import match_font, FONT_DIR


def specimen(filename, text, shear=0, tracking=0):
    font = ImageFont.truetype(str(filename), 100)
    alpha = Image.new('L', (2000, 180))
    draw = ImageDraw.Draw(alpha)
    cursor = 20.
    if tracking:
        for char in text:
            draw.text((cursor, 10), char, font=font, fill=255)
            cursor += font.getlength(char) + tracking
    else:
        draw.text((20, 10), text, font=font, fill=255)
    alpha = np.array(alpha)
    if shear:
        alpha = cv2.warpAffine(alpha, np.float32([[1, shear, abs(shear)*180], [0, 1, 0]]), (2060, 180))
    ys, xs = np.nonzero(alpha > 0)
    alpha = alpha[ys.min():ys.max()+1, xs.min():xs.max()+1]
    alpha = cv2.resize(alpha, None, fx=.72, fy=.72, interpolation=cv2.INTER_AREA)
    h, w = alpha.shape
    rgb = np.full((h+40, w+40, 3), 240, np.uint8)
    rgb[20:20+h, 20:20+w] = np.rint(240-alpha[...,None].astype(float)/255*220).astype(np.uint8)
    return rgb, {'text': text, 'box': [20, 20, w, h], 'color': '#141414'}


def test_sheared_chinese_headline_returns_font_and_native_shear():
    rgb, spec = specimen(FONT_DIR/'LayerStudioSansSC-Black.ttf', '安心健康生活', shear=-.24)
    result = match_font(rgb, spec)
    assert result['key'] == 'LayerStudioSansSC-Black', result
    assert result['score'] > .85, result
    assert abs(result.get('shear', 0) + .24) < .05, result


def test_english_numerals_find_actual_bold_italic_font():
    rgb, spec = specimen('C:/Windows/Fonts/arialbi.ttf', 'VITAMIN 85%')
    result = match_font(rgb, spec)
    assert result['key'] == 'arial-bold-italic', result
    assert result['score'] > .85, result


def test_letter_spacing_is_recovered_as_editable_tracking():
    rgb, spec = specimen(FONT_DIR/'LayerStudioSansSC-Bold.ttf', '安心健康生活', tracking=7)
    result = match_font(rgb, spec)
    assert result['key'] == 'LayerStudioSansSC-Bold', result
    assert result['score'] > .85, result
    assert abs(result.get('tracking', 0)-7) < 2, result
