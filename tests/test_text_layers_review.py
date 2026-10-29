"""Independent regression checks for text extraction integration."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import text_layers


def test_native_background_repair_must_not_erase_neighboring_letters():
    neighbor=np.zeros((30,70),np.uint8);neighbor[10:20,42:59]=255
    changed=np.zeros((20,50),bool);changed[3:17,2:42]=True
    repair=(10,5,np.zeros((20,50,3),np.uint8),changed)
    assert text_layers.repair_overlaps(repair,neighbor)
    neighbor[:]=0;neighbor[26:]=255
    assert not text_layers.repair_overlaps(repair,neighbor)


def test_disjoint_masks_with_overlapping_crop_bounds_do_not_restore_previous_text():
    backdrop = np.full((80,80,3), 240, np.uint8)
    first = np.zeros((80,80), np.uint8)
    first[30:40,30:40] = 255
    second = np.zeros_like(first)
    second[20:25,20:25] = 255
    second[50:55,50:55] = 255
    # Region 2's raster bounds enclose region 1, but neither owns any pixels
    # of the other. Export order must not put region 1 back into the base.
    source = backdrop.copy()
    source[(first > 0) | (second > 0)] = 20
    regions = [{'index':index,'mask':mask,'spec':{'text':text,'color':'#141414'},
                'repair_method':'synthetic known backdrop'}
               for index,mask,text in [(0,first,'A'),(1,second,'B')]]
    prepared = {'clean':backdrop,'regions':regions,'union':np.maximum(first,second)}
    clean, rendered = text_layers.render(source,prepared,1,1)
    assert len(rendered) == 2
    assert np.array_equal(clean,backdrop), 'text was restored below its independent group'
    composite = clean.copy()
    for region in rendered:
        x,y = region['left'],region['top']
        for layer in region['layers']:
            rgba = layer['rgba']
            alpha = rgba[...,3:4] / 255
            target = composite[y:y+rgba.shape[0],x:x+rgba.shape[1]]
            target[:] = np.round(rgba[...,:3] * alpha + target * (1-alpha)).astype(np.uint8)
    assert np.max(np.abs(composite.astype(int)-source)) <= 2
