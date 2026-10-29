"""Compare visible glyph shapes with available fonts instead of forcing YaHei."""
from pathlib import Path
from functools import lru_cache
import json,os,sys,shutil,re
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont

APP_DIR=Path(sys.executable).resolve().parent if getattr(sys,'frozen',False) else Path(__file__).resolve().parent
FONT_DIR=APP_DIR/'fonts'

# Windows supplies these fonts; they are matched locally, never redistributed.
# Keep the same entries in prepare_fonts.py so the PSD writer can resolve keys.
LATIN_FONTS=[
    ('arial','arial.ttf','ArialMT','Arial'),
    ('arial-bold','arialbd.ttf','Arial-BoldMT','Arial Bold'),
    ('arial-italic','ariali.ttf','Arial-ItalicMT','Arial Italic'),
    ('arial-bold-italic','arialbi.ttf','Arial-BoldItalicMT','Arial Bold Italic'),
    ('arial-black','ariblk.ttf','Arial-Black','Arial Black'),
    ('times','times.ttf','TimesNewRomanPSMT','Times New Roman'),
    ('times-bold','timesbd.ttf','TimesNewRomanPS-BoldMT','Times New Roman Bold'),
    ('times-italic','timesi.ttf','TimesNewRomanPS-ItalicMT','Times New Roman Italic'),
    ('calibri','calibri.ttf','Calibri','Calibri'),
    ('calibri-bold','calibrib.ttf','Calibri-Bold','Calibri Bold'),
    ('calibri-italic','calibrii.ttf','Calibri-Italic','Calibri Italic'),
    ('impact','impact.ttf','Impact','Impact'),
]

@lru_cache(maxsize=1)
def catalog():
    path=FONT_DIR/'catalog.json'
    entries=json.loads(path.read_text(encoding='utf-8')) if path.is_file() else []
    keys={item['key'] for item in entries}
    for key,file,psd,label in LATIN_FONTS:
        if key not in keys:
            entries.append({'key':key,'file':file,'psd_name':psd,'label':label,'bundled':False,'latin_only':True})
    return entries

def font_path(item):
    return (FONT_DIR if item['bundled'] else Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts')/item['file']

@lru_cache(maxsize=64)
def load_font(filename):
    return ImageFont.truetype(filename,100)


@lru_cache(maxsize=192)
def _render_glyphs(filename,text,tracking=0):
    """Render at 100 px. Tracking uses individual advances, matching PSD export."""
    font=load_font(filename)
    box=font.getbbox(text)
    width=max(1,int(np.ceil(font.getlength(text)+abs(tracking)*len(text)+20)))
    canvas=Image.new('L',(width,max(1,box[3]-box[1]+4)))
    draw=ImageDraw.Draw(canvas)
    if tracking:
        cursor=2-box[0]
        for char in text:
            draw.text((cursor,2-box[1]),char,font=font,fill=255)
            cursor+=font.getlength(char)+tracking
    else:
        draw.text((2-box[0],2-box[1]),text,font=font,fill=255)
    bounds=canvas.getbbox()
    return np.array(canvas.crop(bounds)) if bounds else None


def render_match_mask(match,width,height):
    """Render a match's glyph silhouette at the requested box for later analysis.

    shear is the x/y affine slope before fitting the bbox (negative leans right).
    tracking is additional pixels per character at the native 100 px font size.
    """
    item=next((item for item in catalog() if item['key']==match['key']),None)
    if not item:return None
    raw=_render_glyphs(str(font_path(item)),match['text'],float(match.get('tracking',0)))
    if raw is None:return None
    return _fit_glyphs(raw,float(match.get('shear',0)),width,height)


def _fit_glyphs(raw,shear,width,height):
    if shear:
        rh,rw=raw.shape
        offset=max(0,-shear*(rh-1))+2
        raw=cv2.warpAffine(raw,np.float32([[1,shear,offset],[0,1,0]]),
                           (int(np.ceil(rw+abs(shear)*(rh-1)+4)),rh),flags=cv2.INTER_LINEAR)
        # boundingRect scans the byte mask directly, without allocating two
        # int64 arrays for every ink pixel in every shear/spacing candidate.
        left,top,ink_width,ink_height=cv2.boundingRect(raw)
        if not ink_width:return np.zeros((height,width),np.uint8)
        raw=raw[top:top+ink_height,left:left+ink_width]
    return np.array(Image.fromarray(raw).resize((width,height),Image.Resampling.LANCZOS))


def match_font(rgb,spec):
    x,y,w,h=spec['box'];crop=rgb[y:y+h,x:x+w]
    if not crop.size:return None
    gray=cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
    threshold,mask=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    color=np.array([int(spec['color'][i:i+2],16) for i in (1,3,5)])
    if float(np.dot(color,[.299,.587,.114]))<=threshold:mask=255-mask
    target=mask>0;best=None;ranked=[];scores={};filenames={}
    has_cjk=any(ord(char)>=0x2e80 for char in spec['text'])
    variants=[spec['text']]
    if ' ' in spec['text']:
        variants=list(dict.fromkeys(re.sub(r' +',' '*n,spec['text']) for n in range(1,7)))

    def score_candidate(item,text,tracking,shear):
        nonlocal best
        candidate_key=(item['key'],text,float(tracking),round(float(shear),4))
        if candidate_key in scores:return scores[candidate_key]
        filename=filenames.setdefault(item['key'],str(font_path(item)))
        raw=_render_glyphs(filename,text,float(tracking))
        if raw is None:return None
        rendered=_fit_glyphs(raw,shear,w,h)>127
        score=float(np.count_nonzero(rendered&target)/max(1,np.count_nonzero(rendered|target)))
        result={'key':item['key'],'label':item['label'],'psd_name':item['psd_name'],
                'score':score,'text':text,'shear':round(float(shear),4),'tracking':float(tracking)}
        # Prefer simpler transforms when the measured difference is negligible.
        if best is None or score>best['score']+1e-6:best=result
        scores[candidate_key]=result
        return result

    for item in catalog():
        if has_cjk and item.get('latin_only'):continue
        filename=font_path(item)
        if not filename.is_file():continue
        family_best=None
        for text in variants:
            for shear in (0,-.1,-.2,-.3,-.4,.1,.2,.3,.4):
                result=score_candidate(item,text,0,shear)
                if result and (family_best is None or result['score']>family_best['score']):family_best=result
        if family_best:ranked.append((family_best,item))
    # Refine likely families only: full Cartesian search over all fonts, spaces,
    # tracking and shear was needlessly slow on pages containing many labels.
    for candidate,item in sorted(ranked,key=lambda pair:pair[0]['score'],reverse=True)[:4]:
        text=candidate['text'];start_shear=candidate['shear']
        for tracking in (0,-8,-4,4,8,12):
            for shear in (start_shear,start_shear-.05,start_shear+.05):
                score_candidate(item,text,tracking,shear)
    if best:
        item=next(item for item in catalog() if item['key']==best['key'])
        chosen=best.copy()
        for tracking in np.arange(chosen['tracking']-3,chosen['tracking']+3.01,1):
            for shear in np.arange(chosen['shear']-.04,chosen['shear']+.0401,.02):
                score_candidate(item,chosen['text'],tracking,shear)
    return best

def export_fonts(directory,keys):
    """Bundle TypeLayer fonts and notices; installation remains a manual choice."""
    entries=[item for item in catalog() if item['key'] in keys and item['bundled']]
    if not entries:return True
    destination=Path(directory)/'fonts';destination.mkdir(exist_ok=True)
    for item in entries:
        src=font_path(item);shutil.copy2(src,destination/item['file'])
    for notice in FONT_DIR.glob('*OFL.txt'):shutil.copy2(notice,destination/notice.name)
    (destination/'字体说明.txt').write_text('本目录为PSD文字层使用的完整字体（附OFL许可）。若Photoshop提示缺失字体，请选中.ttf文件，右键安装后重新打开PSD。匹配字体不代表已确定原设计使用的字体。',encoding='utf-8-sig')
    return False

def assess_rebuild(rgb,spec,mask,excluded):
    """Require evidence for BOTH glyph identity and the hidden backdrop.

    There are no brand names, slogans or sample coordinates in this decision.
    A low-confidence result is preserved, never silently replaced with YaHei.
    """
    import vision
    match=match_font(rgb,spec)
    result={'safe':False,'match':match,'repair':None,'reason':''}
    if not match or match['score']<.80:
        result['reason']='字体或文字效果不能可靠匹配'
        return result
    fitted=vision.smooth_brand_background(rgb,rgb,mask,cv2.bitwise_or(mask,excluded),spec['box'][3]*1.8)
    if fitted is None:
        result['reason']='文字后方存在纹理、物体边界或缺少可靠底色，不能可靠补背景'
        return result
    # Matching a glyph silhouette does not reproduce its colored shadow,
    # outline or gradient. Test whether the observed ink can be explained by
    # one foreground color composited over the validated background.
    source=rgb[mask>0].astype(float);backdrop=fitted[mask>0].astype(float)
    foreground=np.array([int(spec['color'][i:i+2],16) for i in (1,3,5)])
    direction=foreground-backdrop;delta=source-backdrop
    alpha=np.sum(delta*direction,axis=1)/np.maximum(np.sum(direction*direction,axis=1),1)
    residual=np.linalg.norm(delta-np.clip(alpha,0,1)[:,None]*direction,axis=1)
    visible=np.linalg.norm(delta,axis=1)>20
    styled=visible&((alpha<-.12)|(alpha>1.35)|(residual>25))
    if np.count_nonzero(styled)>max(10,np.count_nonzero(visible)*.04):
        result['reason']='存在不能可靠重建的多色填充、描边或阴影'
        return result
    changed=np.any(fitted!=rgb,axis=2);ys,xs=np.nonzero(changed)
    if not len(xs):
        result['reason']='未获得可靠的去字区域'
        return result
    x0,y0,x1,y1=int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)
    result.update(safe=True,repair=(x0,y0,fitted[y0:y1,x0:x1].copy(),changed[y0:y1,x0:x1].copy()))
    return result

def apply_repairs(rgb,repairs):
    result=rgb.copy()
    for x,y,pixels,changed in repairs:
        section=result[y:y+pixels.shape[0],x:x+pixels.shape[1]]
        section[changed]=pixels[changed]
    return result
