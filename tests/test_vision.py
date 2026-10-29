import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'app'))


def test_dark_screenshot_bars_are_cropped_but_dark_photo_is_preserved():
    import vision
    im = Image.new('RGB', (200, 320), 'black')
    ImageDraw.Draw(im).rectangle((0, 60, 199, 259), fill='#bcddee')
    crop, box = vision.crop_black_bars(im)
    assert crop.size == (200, 200)
    assert box == (0, 60, 200, 260)
    dark = Image.new('RGB', (200, 200), '#101010')
    assert vision.crop_black_bars(dark)[0].size == dark.size


def test_phone_ui_glyphs_do_not_prevent_cropping_the_central_advertisement():
    import vision
    im=Image.new('RGB',(300,650),'black');d=ImageDraw.Draw(im)
    d.rectangle((0,170,299,469),fill='#70aadd')
    d.text((20,10),'12:18   5G',fill='white',font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',24))
    d.rectangle((210,520,285,558),fill='red')
    d.text((10,485),'PRICE 126.28',fill='white',font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20))
    cropped,box=vision.crop_black_bars(im)
    assert box==(0,170,300,470)
    assert cropped.size==(300,300)


def test_glyph_mask_removes_ink_without_erasing_entire_text_rectangle():
    import vision
    im = Image.new('RGB', (300, 100), '#ead3a2')
    font = ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc', 36)
    ImageDraw.Draw(im).text((20, 20), '自动处理', fill='#172027', font=font)
    mask, spec = vision.text_region(np.array(im), [[10, 10], [190, 10], [190, 78], [10, 78]], '自动处理', .99)
    assert 600 < np.count_nonzero(mask) < 8500
    assert mask[10, 10] == 0
    assert max(spec['color_rgb']) < 100
    assert spec['box'][0] >= 18 and spec['box'][1] >= 22
    repaired = vision.remove_text(np.array(im), mask)
    assert np.mean(np.abs(repaired[mask > 0].astype(float) - np.array([234, 211, 162]))) < 12


def test_separated_highlight_recomposes_original_alpha_and_rgb():
    import vision
    rng = np.random.default_rng(12)
    rgba = rng.integers(0, 256, (25, 40, 4), dtype=np.uint8)
    base, light = vision.split_highlights(rgba)
    ab, al = base[..., 3:4] / 255., light[..., 3:4] / 255.
    a = al + ab * (1 - al)
    got = light[..., :3] * al + base[..., :3] * ab * (1 - al)
    expected = rgba[..., :3] * (rgba[..., 3:4] / 255.)
    assert np.max(np.abs(a - rgba[..., 3:4] / 255.)) < .009
    assert np.max(np.abs(got - expected)) < 2.1


def test_disconnected_foreground_parts_keep_position():
    import vision
    alpha = np.zeros((120, 160), np.uint8)
    alpha[10:50, 15:45] = 255
    alpha[70:110, 100:145] = 255
    parts = vision.components(alpha)
    assert len(parts) == 2
    assert sorted([p['box'] for p in parts]) == [(15, 10, 30, 40), (100, 70, 45, 40)]


def test_oversized_input_is_rejected_before_decoding(tmp_path):
    import vision
    p = tmp_path / 'large.png'
    Image.new('RGB', (5100, 100)).save(p)
    with pytest.raises(ValueError, match='尺寸'):
        vision.load_image(p)


def test_binary_black_text_is_recognized_as_ink():
    import vision
    im=Image.new('RGB',(200,70),'white')
    ImageDraw.Draw(im).text((15,8),'TEST',fill='black',font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',40))
    rgb=np.where(np.array(im)>127,255,0).astype(np.uint8)
    mask,spec=vision.text_region(rgb,[[5,0],[140,0],[140,65],[5,65]],'TEST',1)
    assert np.count_nonzero(mask)>500
    assert spec['color']=='#000000'
    reverse=255-rgb
    white_mask,white_spec=vision.text_region(reverse,[[5,0],[140,0],[140,65],[5,65]],'TEST',1)
    assert white_spec['color']=='#ffffff'
    assert np.count_nonzero(white_mask)==np.count_nonzero(mask)


def test_rotated_and_product_text_preserves_original_pixels():
    import vision
    spec={'box':[10,10,40,20],'rotation':10,'text':'编辑文字'}
    fg=np.zeros((1000,100),np.uint8)
    assert '旋转' in vision.text_preservation_reason(spec,fg)
    spec['rotation']=0
    fg[10:30,10:50]=255
    assert '小字' in vision.text_preservation_reason(spec,fg)
    assert vision.text_preservation_reason(spec,np.zeros_like(fg)) is None
