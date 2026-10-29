import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'app'))
import vision


def test_mixed_brand_keeps_its_original_lettering_but_normal_title_stays_editable():
    foreground = np.zeros((800,800),np.uint8)
    brand = {'text':'Acme品牌','box':[22,14,220,53],'rotation':0}
    title = {'text':'官方旗舰店','box':[262,23,191,37],'rotation':0}
    assert '品牌' in (vision.text_preservation_reason(brand,foreground) or '')
    assert vision.text_preservation_reason(title,foreground) is None


def test_brand_on_a_product_is_kept_embedded_instead_of_duplicated():
    foreground=np.zeros((800,800),np.uint8)
    foreground[14:67,22:242]=255
    brand={'text':'Acme品牌','box':[22,14,220,53],'rotation':0}
    reason=vision.text_preservation_reason(brand,foreground)
    assert reason and '产品' in reason
    assert not reason.startswith('品牌标志'), 'do not extract branding from opaque foreground products'


def scene():
    rgb = np.zeros((100,380,3),np.uint8)
    rgb[:] = np.linspace([209,137,72],[120,76,30],380).astype(np.uint8)
    image=Image.fromarray(rgb)
    draw=ImageDraw.Draw(image)
    font=ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc',32)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
    draw.text((12,20),'Acme',font=font,fill=(252,244,228))
    draw.text((123,12),'®',font=small,fill=(252,244,228))
    draw.text((160,24),'正常文字',font=font,fill=(252,244,228))
    return np.asarray(image),rgb


def test_preserved_brand_mask_covers_registration_mark_without_eating_neighbor():
    rgb,bg=scene()
    quad=[[12,25],[123,25],[123,60],[12,60]]
    spec={'box':[12,25,111,35],'color':'#fcf4e4'}
    excluded=np.zeros(rgb.shape[:2],np.uint8)
    excluded[:,158:]=255
    mask=vision.preserved_text_mask(rgb,quad,spec,excluded)
    mark=np.any(rgb[10:34,124:145]!=bg[10:34,124:145],axis=2)
    assert mark.any()
    assert np.all(mask[10:34,124:145][mark]>0), 'registration mark was cut'
    assert not mask[:,158:].any(), 'neighboring title was included'
    assert not mask[80:].any(), 'background outside lettering was included'


def test_transparent_brand_layer_recomposes_without_background_box():
    rgb,bg=scene()
    mask=np.any(rgb!=bg,axis=2).astype(np.uint8)*255
    rgba=vision.foreground_from_background(rgb,bg,mask)
    alpha=rgba[...,3:4].astype(float)/255
    composited=np.round(rgba[...,:3]*alpha+bg*(1-alpha))
    assert np.max(np.abs(composited-rgb))<=2
    assert not rgba[...,3][mask==0].any()
    assert np.any((rgba[...,3]>0)&(rgba[...,3]<255)), 'anti-aliased edges should stay transparent'


def test_synthetic_gradient_banner_does_not_become_part_of_logo_mask():
    # A reproducible fixture keeps private customer artwork out of the repository.
    yy,xx=np.mgrid[:160,:320]
    rgb=np.stack((60+xx*.12,100+yy*.1,160+xx*.08),axis=2).round().astype(np.uint8)
    cv2.putText(rgb,'DEMO',(22,55),cv2.FONT_HERSHEY_SIMPLEX,1.4,(252,244,228),3,cv2.LINE_AA)
    excluded=np.zeros(rgb.shape[:2],np.uint8)
    excluded[:,258:]=255
    mask=vision.preserved_text_mask(rgb,[[20,12],[243,12],[243,68],[20,68]],
                                    {'box':[22,14,220,53],'color':'#fcf4e4'},excluded)
    assert not mask[:,:8].any(), 'flat gradient to the left of logo was selected'
    assert not mask[80:].any(), 'blue area below banner was selected'


def test_upscaler_receives_original_logo_while_replaceable_title_is_erased():
    rgb,bg=scene()
    all_text=cv2.dilate(np.any(rgb!=bg,axis=2).astype(np.uint8)*255,np.ones((5,5),np.uint8))
    keep=all_text.copy(); keep[:,150:]=0
    prepared=vision.textless_for_enhancement(rgb,all_text,keep)
    assert np.array_equal(prepared[keep>0],rgb[keep>0]), 'logo was removed before enhancement'
    title_ink=np.any(rgb[:,160:]!=bg[:,160:],axis=2)
    before=np.mean(np.abs(rgb[:,160:].astype(float)-bg[:,160:])[title_ink])
    after=np.mean(np.abs(prepared[:,160:].astype(float)-bg[:,160:])[title_ink])
    assert after<before*.15, 'replaceable title should not remain in enhancement input'


def test_colored_card_under_logo_is_retained_in_the_compositing_background():
    rgb=np.full((400,400,3),210,np.uint8)
    rgb[10:79,20:251]=[30,100,200]
    foreground=np.zeros((400,400),np.uint8)
    assert vision.find_decor(rgb,foreground)[0], 'fixture must be a detectable card'
    logo=np.zeros((400,400),np.uint8);logo[25:55,50:200]=255
    shapes,removed=vision.find_decor(rgb,foreground,protected=logo)
    assert not shapes, 'a filled shape above the transparent logo would hide or recolor it'
    assert not removed.any(), 'card must remain in the logo compositing background'
