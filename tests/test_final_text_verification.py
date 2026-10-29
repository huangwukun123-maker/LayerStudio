from pathlib import Path
import sys
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
from engine import verify_text_composite

def test_final_stack_loss_is_replaced_by_explicit_source_region(tmp_path):
    source=np.full((90,180,3),[195,176,110],np.uint8)
    source[30:35,40:125]=[245,245,225]
    Image.fromarray(np.full_like(source,[195,176,110])).save(tmp_path/'preview.png')
    region={'name':'Low contrast','alternative':{'box':[35,25,95,20],'region_id':'one'},
            'rasters':[{'file':'old.png','left':35,'top':25,'editableStroke':True}]}
    manifest={'text_regions':[region], 'text_audit':[{'region_id':'one','mode':'appearance'}],'warnings':[]}
    assert verify_text_composite(tmp_path,manifest,source)==1
    assert region['contains_background'] and manifest['text_audit'][0]['mode']=='preserved'
    asset=region['rasters'][0]
    assert asset['editableStroke'] is False and '含背景' in asset['name']
    rgba=np.array(Image.open(tmp_path/asset['file']))
    x,y=asset['left'],asset['top'];visible=rgba[...,3]>0
    assert np.array_equal(rgba[...,:3][visible],source[y:y+rgba.shape[0],x:x+rgba.shape[1]][visible])

def test_bounded_colour_rounding_does_not_fail_final_stack_verification(tmp_path):
    source=np.full((30,60,3),180,np.uint8)
    Image.fromarray(source-3).save(tmp_path/'preview.png')
    manifest={'text_regions':[{'alternative':{'box':[5,5,30,10],'region_id':'ok'},'rasters':[{}]}],
              'text_audit':[],'warnings':[]}
    assert verify_text_composite(tmp_path,manifest,source)==0

def test_disclosed_contour_reconstruction_is_not_replaced_with_source_noise(tmp_path):
    source=np.full((30,60,3),180,np.uint8)
    Image.fromarray(source-80).save(tmp_path/'preview.png')
    manifest={'text_regions':[{'alternative':{'box':[5,5,30,10],'region_id':'brand'},
                'rasters':[{}],'contour_reconstructed':True}], 'text_audit':[],'warnings':[]}
    assert verify_text_composite(tmp_path,manifest,source)==0
