import sys
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageFont,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
import vision

def test_gradient_behind_removed_title_does_not_have_inpaint_smears():
    yy,xx=np.mgrid[:160,:350]
    bg=np.stack((200-xx*.1+yy*.02,225-xx*.05,240-yy*.1),axis=2).round().astype(np.uint8)
    rgb=bg.copy();rgb[35:90,30:290]=[20,20,20]
    mask=np.zeros((160,350),np.uint8);mask[31:94,26:294]=255
    repaired=vision.remove_text(rgb,mask,regions=[[30,35,260,55]])
    assert np.max(np.abs(repaired[40:85,40:280].astype(int)-bg[40:85,40:280]))<=2
    assert np.array_equal(repaired[110:],rgb[110:]), 'unrelated visible background changed'

def test_texture_and_protected_object_are_not_flattened_by_gradient_repair():
    rng=np.random.default_rng(55)
    rgb=rng.integers(50,220,(130,250,3),dtype=np.uint8)
    mask=np.zeros((130,250),np.uint8);mask[40:75,35:190]=255
    expected=cv2.inpaint(rgb,mask,5,cv2.INPAINT_TELEA)
    repaired=vision.remove_text(rgb,mask,regions=[[35,40,155,35]])
    assert np.array_equal(repaired,expected)

def test_heavy_headline_and_serif_text_match_distinct_font_outlines():
    from text_fidelity import match_font,FONT_DIR
    for filename,expected in [('LayerStudioSansSC-Black.ttf','LayerStudioSansSC-Black'),('C:/Windows/Fonts/simsun.ttc','songti')]:
        font=ImageFont.truetype(str(FONT_DIR/filename) if ':' not in filename else filename,50)
        im=Image.new('RGB',(440,120),(210,235,240))
        ImageDraw.Draw(im).text((20,15),'安心健康用水',font=font,fill=(25,25,25))
        a=np.array(im);ys,xs=np.nonzero(a[...,0]<100)
        box=[int(xs.min()),int(ys.min()),int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)]
        result=match_font(a,{'text':'安心健康用水','box':box,'color':'#191919'})
        assert result['key']==expected, result
        # SimSun has strong size-specific hinting: rendering at 50 and 100 px
        # need not yield identical thin strokes, but it must win over sans.
        assert result['score']>(.85 if expected.endswith('Black') else .60)

def test_collapsed_ocr_spaces_do_not_force_the_wrong_heavy_font():
    from text_fidelity import match_font,FONT_DIR
    font=ImageFont.truetype(str(FONT_DIR/'LayerStudioSansSC-Regular.ttf'),40)
    im=Image.new('RGB',(650,95),(240,225,170))
    ImageDraw.Draw(im).text((10,10),'简易安装    适配水龙头',font=font,fill=(30,30,30))
    a=np.array(im);ys,xs=np.nonzero(a[...,0]<120)
    spec={'text':'简易安装 适配水龙头','box':[int(xs.min()),int(ys.min()),int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)],'color':'#1e1e1e'}
    result=match_font(a,spec)
    assert result['key']=='LayerStudioSansSC-Regular'
    assert result['text']=='简易安装    适配水龙头'

def test_binary_black_text_does_not_match_the_white_background_as_ink():
    from text_fidelity import match_font,FONT_DIR
    im=Image.new('L',(650,150),255)
    ImageDraw.Draw(im).text((10,10),'安心健康用水',font=ImageFont.truetype(str(FONT_DIR/'LayerStudioSansSC-Black.ttf'),100),fill=0)
    gray=np.where(np.array(im)<128,0,255).astype(np.uint8)
    a=np.repeat(gray[...,None],3,axis=2);ys,xs=np.nonzero(gray==0)
    spec={'text':'安心健康用水','box':[int(xs.min()),int(ys.min()),int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)],'color':'#000000'}
    result=match_font(a,spec)
    assert result['key']=='LayerStudioSansSC-Black'
    # The binary source bbox excludes the font's faint antialias fringe.
    assert result['score']>.90

def test_unreliable_font_or_background_must_not_be_declared_safe_to_rebuild():
    from text_fidelity import assess_rebuild,FONT_DIR
    font=ImageFont.truetype(str(FONT_DIR/'LayerStudioSansSC-Black.ttf'),50)
    for textured in [False,True]:
        rng=np.random.default_rng(2)
        bg=rng.integers(110,255,(150,420,3),dtype=np.uint8) if textured else np.full((150,420,3),[195,225,235],np.uint8)
        im=Image.fromarray(bg);ImageDraw.Draw(im).text((20,35),'健康用水',font=font,fill=(25,25,25))
        a=np.array(im);mask,spec=vision.text_region(a,[[15,35],[230,35],[230,110],[15,110]],'健康用水',.99)
        result=assess_rebuild(a,spec,mask,np.zeros_like(mask))
        if textured:
            assert not result['safe'] and result['repair'] is None
        else:
            assert result['safe'] and result['repair'] is not None
        wrong={**spec,'text':'随机字体XYZ'}
        result=assess_rebuild(a,wrong,mask,np.zeros_like(mask))
        assert not result['safe'] and result['repair'] is None

def test_matching_font_does_not_allow_silent_loss_of_a_colored_shadow():
    from text_fidelity import assess_rebuild,FONT_DIR
    im=Image.new('RGB',(430,160),(85,150,220))
    font=ImageFont.truetype(str(FONT_DIR/'LayerStudioSansSC-Black.ttf'),50)
    draw=ImageDraw.Draw(im)
    draw.text((23,49),'健康用水',font=font,fill=(15,45,110))
    draw.text((20,40),'健康用水',font=font,fill=(245,250,255))
    a=np.array(im)
    mask,spec=vision.text_region(a,[[15,40],[230,40],[230,115],[15,115]],'健康用水',.99)
    mask[36:125,15:235]=255
    result=assess_rebuild(a,spec,mask,np.zeros_like(mask))
    assert not result['safe'], 'a known font does not reconstruct its drop shadow'
