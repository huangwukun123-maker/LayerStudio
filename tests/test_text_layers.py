import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))

def fixture():
    yy,xx=np.mgrid[:150,:360]
    bg=np.stack((180+xx*.08,210+yy*.05,240-xx*.03),axis=2).round().astype('uint8')
    core=np.zeros((150,360),'uint8')
    cv2.putText(core,'AB 75%',(20,88),cv2.FONT_HERSHEY_SIMPLEX,1.8,255,4,cv2.LINE_AA)
    shadow=cv2.warpAffine(core,np.float32([[1,0,3],[0,1,5]]),(360,150))
    rgb=bg.copy()
    for a,c in [(shadow,[15,55,130]),(core,[250,250,255])]:
        t=a[...,None]/255;rgb=np.round(rgb*(1-t)+np.array(c)*t).astype('uint8')
    mask=cv2.dilate(core,np.ones((5,5),'uint8'))
    region={'spec':{'text':'AB 75%','box':[20,44,224,48],'color':'#fafaff'},'mask':mask,'quad':[[18,42],[255,42],[255,95],[18,95]]}
    return rgb,bg,region

def over(bg,rgba):
    a=rgba[...,3:4]/255
    return np.round(rgba[...,:3]*a+bg*(1-a)).astype('uint8')

def test_extracted_styled_text_recomposes_without_an_opaque_rectangle():
    import text_layers
    rgb,bg,region=fixture()
    prepared=text_layers.prepare(rgb,[region])
    clean,layers=text_layers.render(rgb,prepared,1,1)
    assert len(layers)==1
    combined=clean.copy()
    for layer in layers[0]['layers']:
        x,y=layers[0]['left'],layers[0]['top'];a=layer['rgba']
        combined[y:y+a.shape[0],x:x+a.shape[1]]=over(combined[y:y+a.shape[0],x:x+a.shape[1]],a)
        assert (a[...,3]==0).any(), 'text must have transparent gaps'
    assert np.abs(combined.astype(int)-rgb).max()<=2
    assert len(layers[0]['layers'])==2, 'colored shadow and glyph face need separate appearance layers'
    assert np.array_equal(clean[110:],rgb[110:]), 'visible background outside text changed'
    assert np.mean(np.abs(clean[50:90,30:240].astype(int)-bg[50:90,30:240]))<9

def test_overlapping_regions_have_unique_pixel_ownership_and_no_ghost():
    import text_layers
    rgb,bg,region=fixture()
    other={**region,'spec':{**region['spec'],'text':'75%','box':[100,44,150,48]},'quad':[[100,42],[255,42],[255,95],[100,95]]}
    prepared=text_layers.prepare(rgb,[region,other])
    union=sum((r['mask']>0).astype(int) for r in prepared['regions'])
    assert union.max()==1
    clean,layers=text_layers.render(rgb,prepared,1,1)
    combined=clean.copy()
    for item in layers:
        x,y=item['left'],item['top']
        for layer in item['layers']:
            a=layer['rgba'];combined[y:y+a.shape[0],x:x+a.shape[1]]=over(combined[y:y+a.shape[0],x:x+a.shape[1]],a)
    assert np.abs(combined.astype(int)-rgb).max()<=2

def test_text_extraction_scales_and_preserves_unrelated_product_edges():
    import text_layers
    rgb,_,region=fixture();rgb[115:,:]=[190,30,20]
    prepared=text_layers.prepare(rgb,[region])
    high=cv2.resize(rgb,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
    clean,layers=text_layers.render(high,prepared,3,3)
    assert np.array_equal(high[345:],clean[345:])
    assert all(l['rgba'].shape[2]==4 for item in layers for l in item['layers'])
    assert layers[0]['left']<100 and layers[0]['top']<160

def test_super_resolution_must_not_invent_marks_inside_original_letter_shapes():
    import text_layers
    from PIL import Image
    original=np.full((60,100,3),220,np.uint8);original[20:40,30:60]=25
    high=np.array(Image.fromarray(original).resize((300,180),Image.Resampling.LANCZOS))
    hallucinated=high.copy();hallucinated[72:86,100:118]=255
    mask=np.zeros((60,100),np.uint8);mask[18:42,28:62]=255
    result=text_layers.preserve_text_edges(hallucinated,Image.fromarray(original),mask)
    assert np.array_equal(result[72:86,100:118],high[72:86,100:118])
    assert np.array_equal(result[:30],hallucinated[:30])

def test_letter_core_stays_opaque_when_recolored():
    import text_layers
    bg=np.full((50,80,3),(100,180,220),np.uint8)
    source=bg.copy();source[15:35,25:55]=[200,230,245]
    mask=np.zeros((50,80),np.uint8);mask[13:37,23:57]=255
    prepared={'clean':bg,'regions':[{'index':0,'mask':mask,'spec':{'color':'#c8e6f5'},'repair_method':'known'}]}
    clean,regions=text_layers.render(source,prepared,1,1)
    item=regions[0];rgba=item['layers'][-1]['rgba']
    assert rgba[22-item['top'],35-item['left'],3]==255,'solid letter was exported semi-transparent'


def test_separately_rebuilt_brand_has_no_duplicate_under_its_layer():
    import text_layers
    background=np.full((50,100,3),[80,130,220],np.uint8)
    source=background.copy();source[18:28,30:70]=[255,255,255]
    erase=np.zeros((50,100),np.uint8);erase[15:32,25:75]=255
    prepared=text_layers.prepare(source,[],inpaint=lambda image,mask:background.copy(),extra_erase_mask=erase)
    clean,layers=text_layers.render(source,prepared,1,1)
    assert layers==[]
    assert np.array_equal(clean,background),'a separately rebuilt brand remains baked into the backing'


def test_background_tone_must_not_replace_the_known_dark_ink_as_face_colour():
    import text_layers
    source=np.full((24,100,3),[225,216,190],np.uint8)
    source[6:18,20:30]=[40,40,32]
    rgba=np.dstack((source,np.full((24,100),255,np.uint8)))
    layers=text_layers._split_appearance(rgba,source,{'color':'#282820'})
    assert len(layers)==2
    assert layers[-1]['rgba'][10,25,3]==255
    assert layers[-1]['rgba'][10,60,3]==0


def test_single_color_label_keeps_its_panel_and_exports_ink_not_backdrop_error():
    import text_layers
    bg=np.full((95,320,3),[155,220,235],np.uint8)
    bg[25:76,10:305]=[238,204,140]
    ink=np.zeros((95,320),np.uint8)
    cv2.putText(ink,'PURE WATER',(22,62),cv2.FONT_HERSHEY_SIMPLEX,.85,255,2,cv2.LINE_AA)
    source=np.round(bg*(1-ink[...,None]/255)).astype(np.uint8)
    region={'spec':{'text':'PURE WATER','color':'#252525','box':[20,40,205,28]},
            'quad':[[19,39],[224,39],[224,65],[19,65]],'mask':cv2.dilate(ink,np.ones((3,3),np.uint8))}
    wrong_model=lambda image,mask:np.where(mask[...,None]>0,np.array([190,205,210],np.uint8),image)
    prepared=text_layers.prepare(source,[region],inpaint=wrong_model)
    assert 'mono_color' in prepared['regions'][0]
    clean,layers=text_layers.render(source,prepared,1,1)
    assert np.array_equal(clean[28:35,25:220],bg[28:35,25:220]),'yellow panel became a learned-model block'
    for item in layers:
        for layer in item['layers']:
            rgba=layer['rgba'];visible=rgba[...,3]>150
            if np.any(visible):assert np.max(rgba[...,:3][visible])<60,'panel background entered the glyph layer'


def test_unreliable_complex_cutout_preserves_all_strokes_and_discloses_background():
    import text_layers
    source,bg,region=fixture()
    prepared=text_layers.prepare(source,[region],inpaint=lambda rgb,mask:bg.copy())
    clean,layers=text_layers.render(source,prepared,1,1)
    assert layers[0]['contains_background'] is True
    assert len(layers[0]['layers'])==1
    assert '含背景' in layers[0]['layers'][0]['name']
    combined=clean.copy();item=layers[0];rgba=item['layers'][0]['rgba']
    x,y=item['left'],item['top']
    combined[y:y+rgba.shape[0],x:x+rgba.shape[1]]=over(combined[y:y+rgba.shape[0],x:x+rgba.shape[1]],rgba)
    assert np.array_equal(combined,source),'fidelity fallback must not remove faint glyph strokes'
