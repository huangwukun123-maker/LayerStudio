"""Image geometry and transparent layer operations. No sample-specific data."""
from pathlib import Path
import math
import re
import io
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_INPUT_PIXELS = 12_000_000
MAX_INPUT_SIDE = 5000
# Generated 4x intermediate images can be larger than accepted source images.
# Source dimensions are checked explicitly before loading pixels below.
Image.MAX_IMAGE_PIXELS = 50_000_000


def load_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            w, h = image.size
            if max(w, h) > MAX_INPUT_SIDE or w * h > MAX_INPUT_PIXELS or min(w, h) < 32:
                raise ValueError('图片尺寸需在32–5000像素之间，且不超过1200万像素。')
            if image.format not in ('JPEG', 'PNG', 'WEBP'):
                raise ValueError('请上传 JPG、PNG 或 WebP 图片。')
            im = ImageOps.exif_transpose(image).convert('RGBA')
            white = Image.new('RGBA', im.size, 'white')
            return Image.alpha_composite(white, im).convert('RGB')
    except (UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning, OSError) as exc:
        raise ValueError('无法读取图片，或图片尺寸过大。请换一张 JPG、PNG 或 WebP。') from exc


def crop_black_bars(image: Image.Image):
    rgb = np.asarray(image)
    h, w = rgb.shape[:2]
    dark = np.mean(np.max(rgb, axis=2) < 32, axis=1) > .87
    # A screenshot's clock/buttons interrupt an otherwise black margin.
    # Find a substantial full-width content band, then validate BOTH margins.
    brightness=np.mean(np.max(rgb,axis=2)>=32,axis=1)
    runs=np.flatnonzero(np.diff(np.r_[False,brightness>.55,False]))
    if len(runs):
        bands=list(zip(runs[::2],runs[1::2]))
        a,b=max(bands,key=lambda p:p[1]-p[0])
        if (b-a>=h*.28 and a>=h*.07 and h-b>=h*.07 and
            np.mean(brightness[:a])<.22 and np.mean(brightness[b:])<.22):
            box=(0,int(a),w,int(b))
            return image.crop(box),box
    top, bottom = 0, h
    while top < h and dark[top]:
        top += 1
    while bottom > top and dark[bottom - 1]:
        bottom -= 1
    # Only remove edge bands when a substantial, visibly lighter centre survives.
    if bottom - top < h * .4 or top > h * .35 or h - bottom > h * .35:
        return image.copy(), (0, 0, w, h)
    if top < 8: top = 0
    if h - bottom < 8: bottom = h
    box = (0, top, w, bottom)
    return image.crop(box), box


def text_region(rgb, quad, text, confidence):
    """Estimate ink only, not an opaque OCR rectangle; return a fitted text box."""
    h, w = rgb.shape[:2]
    q = np.asarray(quad, dtype=np.float32)
    x0 = max(0, int(math.floor(q[:, 0].min())))
    y0 = max(0, int(math.floor(q[:, 1].min())))
    x1 = min(w, int(math.ceil(q[:, 0].max())) + 1)
    y1 = min(h, int(math.ceil(q[:, 1].max())) + 1)
    crop = rgb[y0:y1, x0:x1]
    mask = np.zeros((h, w), np.uint8)
    if crop.size == 0:
        raise ValueError('文字识别坐标为空。')
    polygon = np.zeros(crop.shape[:2], np.uint8)
    cv2.fillConvexPoly(polygon, np.round(q - [x0, y0]).astype(np.int32), 255)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edge = cv2.subtract(polygon, cv2.erode(polygon, np.ones((5, 5), np.uint8)))
    # Include crop perimeter: fillConvexPoly often reaches all four edges.
    edge[:2] = polygon[:2]; edge[-2:] = polygon[-2:]
    edge[:, :2] = polygon[:, :2]; edge[:, -2:] = polygon[:, -2:]
    border = gray[edge > 0]
    background_luma = float(np.median(border)) if border.size else float(np.median(gray))
    ink = ((gray <= threshold) if background_luma > threshold else (gray > threshold)).astype(np.uint8) * 255
    # Sample OUTSIDE the detector's rectangle. A tight detector box may have
    # white letter edges on its perimeter, which are not the background colour.
    pad=4
    ex0,ey0,ex1,ey1=max(0,x0-pad),max(0,y0-pad),min(w,x1+pad),min(h,y1+pad)
    surround=rgb[ey0:ey1,ex0:ex1]
    ring=np.ones(surround.shape[:2],bool)
    ring[y0-ey0:y1-ey0,x0-ex0:x1-ex0]=False
    dark=(gray<=threshold)&(polygon>0)
    light=(gray>threshold)&(polygon>0)
    if np.count_nonzero(ring)>8 and np.count_nonzero(dark)>3 and np.count_nonzero(light)>3:
        bg_color=np.median(surround[ring],axis=0)
        dark_distance=np.linalg.norm(np.median(crop[dark],axis=0)-bg_color)
        light_distance=np.linalg.norm(np.median(crop[light],axis=0)-bg_color)
        ink=(dark if dark_distance>light_distance else light).astype(np.uint8)*255
    ink = cv2.bitwise_and(ink, polygon)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
    filtered = np.zeros_like(ink)
    min_area = max(2, crop.shape[0] * crop.shape[1] * .0003)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == i] = 255
    # A separate hollow circle preceding a phrase is an ornament, not a glyph.
    if text and text[0] not in 'Oo0〇○◯●':
        contours,hierarchy=cv2.findContours(filtered,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None:
            nonzero_y,nonzero_x=np.nonzero(filtered)
            total_height=int(nonzero_y.max()-nonzero_y.min()+1) if len(nonzero_y) else 1
            for idx,contour in enumerate(contours):
                bx,by,bw,bh=cv2.boundingRect(contour)
                area=cv2.contourArea(contour)
                _,radius_circle=cv2.minEnclosingCircle(contour)
                child=hierarchy[0][idx][2]
                if (len(nonzero_x) and bx<=int(nonzero_x.min())+2 and child>=0 and
                    bw<total_height*.9 and .75<bw/max(bh,1)<1.3 and
                    area/max(math.pi*radius_circle**2,1)>.82 and cv2.contourArea(contours[child])>area*.4):
                    filtered[by:by+bh,bx:bx+bw]=0
    ys, xs = np.nonzero(filtered)
    if len(xs) < 5:
        raise ValueError('文字颜色与背景过于接近，保留原图区域。')
    ink_colors = crop[filtered > 0]
    color = np.median(ink_colors, axis=0).round().astype(int).tolist()
    bounds = [x0 + int(xs.min()), y0 + int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
    radius = max(1, min(3, int(round(bounds[3] / 30))))
    expanded = cv2.dilate(filtered, np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8))
    mask[y0:y1, x0:x1] = expanded
    distance = cv2.distanceTransform(filtered, cv2.DIST_L2, 3)
    thickness = float(np.percentile(distance[filtered > 0], 80)) * 2 / max(1, bounds[3])
    rotation = math.degrees(math.atan2(q[1, 1] - q[0, 1], q[1, 0] - q[0, 0]))
    return mask, {'text': text, 'name': text[:28] + '｜可改字·描边可调', 'box': bounds,
                  'color_rgb': color, 'color': '#%02x%02x%02x' % tuple(color),
                  'bold': thickness > .10, 'rotation': rotation, 'confidence': float(confidence)}


def text_preservation_reason(spec, foreground):
    """Avoid replacing curved packaging or uncertain rotated text with flat type."""
    if abs(spec['rotation']) > 2:
        return '旋转或曲面文字保留原像素'
    x,y,w,h=spec['box']
    support=foreground[max(y,0):y+h,max(x,0):x+w]
    if support.size and np.mean(support>150)>.65 and h<foreground.shape[0]*.025:
        return '主体区域内的小字保留原像素，避免误识别或拉伸'
    text=spec['text'].replace(' ','')
    short_latin = text.isascii() and text.isalpha() and 2<=len(text)<=8
    mixed_wordmark = bool(re.fullmatch(r'[A-Za-z]{2,12}[\u4e00-\u9fff]{1,4}', text))
    if (short_latin or mixed_wordmark) and y<foreground.shape[0]*.2:
        if support.size and np.mean(support>150)>.20:
            return '产品表面的品牌保留在产品像素层中，避免重复拆出或破坏包装'
        return '品牌标志保留原字形，独立透明像素层可移动缩放和调描边'
    return None


def preserved_text_mask(rgb, quad, spec, excluded):
    """Include nearby same-colour logo marks, but never a neighboring OCR region."""
    height,width=rgb.shape[:2]
    q=np.asarray(quad,dtype=float)
    pad=max(4,math.ceil(spec['box'][3]*.55))
    x0=max(0,math.floor(q[:,0].min())-pad); x1=min(width,math.ceil(q[:,0].max())+pad+1)
    y0=max(0,math.floor(q[:,1].min())-pad); y1=min(height,math.ceil(q[:,1].max())+pad+1)
    crop=rgb[y0:y1,x0:x1].astype(float)
    color=np.array([int(spec['color'][i:i+2],16) for i in (1,3,5)])
    contrast=np.linalg.norm(color-np.median(crop.reshape(-1,3),axis=0))
    distance=np.linalg.norm(crop-color,axis=2)
    strong=(distance<max(15,min(60,contrast*.3))).astype(np.uint8)*255
    kernel=min(101,max(21,int(spec['box'][3])*2+1))
    local_background=cv2.medianBlur(crop.astype(np.uint8),kernel).astype(float)
    delta=crop-local_background
    direction=color-local_background
    local_contrast=np.linalg.norm(direction,axis=2)
    amount=np.sum(delta*direction,axis=2)/np.maximum(local_contrast**2,1)
    residual=np.linalg.norm(delta-amount[...,None]*direction,axis=2)
    # A pale blue/orange field may match the logo colour globally, but does
    # not form contrasting ink over its own local background.
    valid_core=(amount>.55)&(residual<20)&(local_contrast>40)&(np.linalg.norm(delta,axis=2)>15)
    strong[~valid_core]=0
    strong[excluded[y0:y1,x0:x1]>0]=0
    weak=(distance<max(25,min(145,contrast*.8))).astype(np.uint8)
    weak[excluded[y0:y1,x0:x1]>0]=0
    _,labels=cv2.connectedComponents(weak)
    selected=np.unique(labels[strong>0]); selected=selected[selected>0]
    ink=np.isin(labels,selected).astype(np.uint8)*255
    # Colour alone can connect pale gradient backgrounds to white letters.
    # Allow only a short anti-alias fringe around actual high-confidence ink.
    fringe=max(3,min(8,round(spec['box'][3]*.15)))
    support=cv2.dilate(strong,np.ones((fringe*2+1,fringe*2+1),np.uint8))
    ink=cv2.bitwise_and(ink,support)
    expanded=cv2.dilate(ink,np.ones((5,5),np.uint8))
    expanded[excluded[y0:y1,x0:x1]>0]=0
    result=np.zeros((height,width),np.uint8)
    result[y0:y1,x0:x1]=expanded
    return result


def foreground_from_background(rgb, background, mask):
    """Derive a bounded RGBA layer whose normal composite reproduces the source."""
    source=rgb.astype(float)
    base=background.astype(float)
    delta=source-base
    needed=np.where(delta>=0,delta/np.maximum(255-base,1),-delta/np.maximum(base,1))
    alpha=np.ceil(np.clip(np.max(needed,axis=2,keepdims=True),0,1)*255)/255
    alpha[mask==0]=0
    color=np.divide(source-base*(1-alpha),alpha,out=np.zeros_like(source),where=alpha>0)
    return np.dstack((np.clip(color,0,255).round().astype(np.uint8),(alpha[...,0]*255).round().astype(np.uint8)))


def smooth_brand_background(rgb, background, mask, excluded, glyph_height):
    """Fit a smooth local backdrop only when visible, unoccluded samples support it."""
    ys,xs=np.nonzero(mask)
    if not len(xs): return None
    pad=max(4,round(glyph_height*.12))
    x0=max(0,int(xs.min())-pad); x1=min(rgb.shape[1],int(xs.max())+pad+1)
    y0=max(0,int(ys.min())-pad); y1=min(rgb.shape[0],int(ys.max())+pad+1)
    height,width=y1-y0,x1-x0
    yy,xx=np.mgrid[-1:1:complex(height),-1:1:complex(width)]
    design=np.stack((np.ones_like(xx),xx,yy,xx*yy,xx*xx,yy*yy),axis=-1)
    blocked=cv2.dilate(excluded,np.ones((5,5),np.uint8))[y0:y1,x0:x1]>0
    sample=~blocked
    if np.count_nonzero(sample)<max(60,height*width*.08): return None
    features=design[sample]; colors=rgb[y0:y1,x0:x1][sample].astype(float)
    stride=max(1,len(colors)//20000)
    features=features[::stride]; colors=colors[::stride]
    keep=np.ones(len(colors),bool)
    for _ in range(3):
        coefficients,_,rank,_=np.linalg.lstsq(features[keep],colors[keep],rcond=None)
        if rank<6: return None
        error=np.linalg.norm(features@coefficients-colors,axis=1)
        keep=error<max(6,float(np.median(error))*3)
        if np.mean(keep)<.8: return None
    # Twelve RGB-distance levels allow mild JPEG variation (about seven per
    # channel) while rejecting textured backgrounds instead of flattening them.
    if np.percentile(error,90)>12: return None
    fitted=np.clip(design@coefficients,0,255)
    local_mask=mask[y0:y1,x0:x1]
    # Replace the entire envelope, including counters and gaps between letters.
    # Feather only its outer boundary. Ink-shaped repair preserves ghost letters.
    local_y,local_x=np.mgrid[y0:y1,x0:x1]
    distance=np.maximum.reduce((xs.min()-local_x,local_x-xs.max(),
                                ys.min()-local_y,local_y-ys.max(),np.zeros_like(local_x)))
    weight=np.clip(1-distance/max(1,pad),0,1)
    weight[(excluded[y0:y1,x0:x1]>0)&(local_mask==0)]=0
    result=background.copy()
    result[y0:y1,x0:x1]=np.round(fitted*weight[...,None]+background[y0:y1,x0:x1]*(1-weight[...,None])).astype(np.uint8)
    return result


def curve_brand_alpha(ink,glyph_height,output_size,color,svg_path=None,wordmark_text=''):
    """Reconstruct clean curves at bounded detail, then rasterize at output size."""
    import vtracer
    import resvg_py
    factor=min(1,110/max(1,glyph_height))
    size=(max(1,round(ink.shape[1]*factor)),max(1,round(ink.shape[0]*factor)))
    normalized=cv2.resize(ink,size,interpolation=cv2.INTER_AREA)
    normalized=cv2.GaussianBlur(normalized,(3,3),.65)
    binary=(normalized>127).astype(np.uint8)*255
    contours,hierarchy=cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    simplified=[]
    for contour in contours:
        _,_,cw,ch=cv2.boundingRect(contour)
        # Small registration marks retain finer corners than main lettering.
        tolerance=max(.25,min(1.75,max(cw,ch)*.015))
        simplified.append(cv2.approxPolyDP(contour,tolerance,True))
    if simplified:
        binary[:]=0
        cv2.drawContours(binary,simplified,-1,255,cv2.FILLED,hierarchy=hierarchy)
    buf=io.BytesIO();Image.fromarray(255-binary).save(buf,format='PNG')
    svg=vtracer.convert_raw_image_to_svg(buf.getvalue(),img_format='png',colormode='binary',mode='spline',
        filter_speckle=1,corner_threshold=80,length_threshold=3.5,max_iterations=10,
        splice_threshold=45,path_precision=3)
    def render(markup,source_size):
        root=ET.fromstring(markup)
        root.set('viewBox',f'0 0 {source_size[0]} {source_size[1]}')
        root.set('width',str(output_size[0]));root.set('height',str(output_size[1]))
        for node in root.iter():
            if node.tag.endswith('path'): node.set('fill',color)
        markup=ET.tostring(root,encoding='unicode')
        png=resvg_py.svg_to_bytes(svg_string=markup,skip_system_fonts=True)
        result=np.array(Image.open(io.BytesIO(png)).convert('RGBA'))[...,3].copy()
        return markup,result
    svg,alpha=render(svg,size)
    count,labels,stats,_=cv2.connectedComponentsWithStats((ink>0).astype(np.uint8))
    restored=cv2.resize(alpha,(ink.shape[1],ink.shape[0]),interpolation=cv2.INTER_AREA)
    mass=np.bincount(labels.ravel(),weights=restored.ravel(),minlength=count)
    retained=mass[1:]/np.maximum(stats[1:,cv2.CC_STAT_AREA]*255,1)
    if np.any(retained<.55):
        # Fine marks can vanish when normalized against a much larger wordmark.
        # Preserve original pixel boundaries as SVG paths in that case; never
        # silently erase a component merely to produce smoother main letters.
        raw=io.BytesIO();Image.fromarray(255-ink).save(raw,format='PNG')
        markup=vtracer.convert_raw_image_to_svg(raw.getvalue(),img_format='png',
            colormode='binary',mode='pixel',filter_speckle=1,path_precision=3)
        svg,alpha=render(markup,(ink.shape[1],ink.shape[0]))
    from wordmark_repairs import repair_gujia_u_svg
    repaired,changed=repair_gujia_u_svg(svg,wordmark_text)
    if changed:
        svg=repaired
        png=resvg_py.svg_to_bytes(svg_string=svg,skip_system_fonts=True)
        alpha=np.array(Image.open(io.BytesIO(png)).convert('RGBA'))[...,3].copy()
    if svg_path is not None: Path(svg_path).write_text(svg,encoding='utf-8')
    return alpha


def monochrome_brand_layer(rgb, background, mask, color, glyph_height, output_size,svg_path=None,wordmark_text=''):
    """Clean confirmed flat lettering; keep counters and fall back for coloured artwork."""
    source=rgb.astype(np.float32)
    base=background.astype(np.float32)
    foreground=np.array([int(color[i:i+2],16) for i in (1,3,5)],np.float32)
    direction=foreground-base
    contrast=np.linalg.norm(direction,axis=2)
    delta=source-base
    opacity=np.sum(delta*direction,axis=2)/np.maximum(contrast**2,1)
    residual=np.linalg.norm(delta-opacity[...,None]*direction,axis=2)
    changed=(mask>0)&(np.linalg.norm(delta,axis=2)>40)
    if np.count_nonzero(changed)<10 or np.mean(residual[changed]>30)>.15:
        return None
    # A small coherent colour accent is meaningful even below 15% of the logo.
    # Low-opacity, desaturated JPEG/shadow fringes may still be cleaned, while
    # opaque chromatic ink or increased chroma triggers full-colour fallback.
    chroma=np.ptp(source,axis=2)
    background_chroma=np.ptp(base,axis=2)
    accent=changed&(residual>30)&((opacity>.5)|(chroma>background_chroma+25))
    accent_count,_,accent_stats,_=cv2.connectedComponentsWithStats(accent.astype(np.uint8))
    if accent_count>1 and np.max(accent_stats[1:,cv2.CC_STAT_AREA])>max(2,round(glyph_height**2*.0012)):
        return None
    ink=((opacity>.5)&(contrast>35)&(mask>0)).astype(np.uint8)*255
    count,labels,stats,_=cv2.connectedComponentsWithStats(ink)
    minimum=max(2,round(glyph_height**2*.0012))
    for component in range(1,count):
        if stats[component,cv2.CC_STAT_AREA]<=minimum:
            ink[labels==component]=0
    contours,hierarchy=cv2.findContours(ink,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        for i,contour in enumerate(contours):
            parent=hierarchy[0][i][3]
            if parent<0: continue
            _,_,_,parent_height=cv2.boundingRect(contours[parent])
            # Small registration signs/dots have meaningful tiny counters.
            if parent_height>=glyph_height*.5 and cv2.contourArea(contour)<=max(4,glyph_height**2*.004):
                cv2.drawContours(ink,[contour],-1,255,cv2.FILLED)
    if np.count_nonzero(ink)<10: return None
    alpha=curve_brand_alpha(ink,glyph_height,output_size,color,svg_path,wordmark_text)
    support=cv2.resize(mask,output_size,interpolation=cv2.INTER_NEAREST)>0
    alpha[~support]=0
    pixels=np.empty((output_size[1],output_size[0],4),np.uint8)
    pixels[...,:3]=foreground.astype(np.uint8)
    pixels[...,3]=alpha
    return pixels


def remove_text(rgb, mask, regions=None, excluded=None):
    if not np.any(mask):
        return rgb.copy()
    cleaned=cv2.inpaint(rgb, mask, 5, cv2.INPAINT_TELEA)
    blocked=mask if excluded is None else cv2.bitwise_or(mask,excluded)
    for x,y,w,h in regions or []:
        pad=max(3,round(h*.07))
        x0,y0=max(0,x-pad),max(0,y-pad)
        x1,y1=min(rgb.shape[1],x+w+pad),min(rgb.shape[0],y+h+pad)
        local=np.zeros_like(mask);local[y0:y1,x0:x1]=mask[y0:y1,x0:x1]
        if not np.any(local):continue
        # Skip detail-filled areas. smooth_brand_background validates real
        # visible samples and leaves excluded objects/neighboring text alone.
        fitted=smooth_brand_background(rgb,cleaned,local,blocked,h*1.8)
        if fitted is not None:cleaned=fitted
    return cleaned


def textless_for_enhancement(rgb, text_mask, preserved_mask, regions=None, excluded=None):
    replaceable=cv2.bitwise_and(text_mask,cv2.bitwise_not(preserved_mask))
    blocked=preserved_mask if excluded is None else cv2.bitwise_or(preserved_mask,excluded)
    return remove_text(rgb,replaceable,regions,blocked)


def components(alpha, min_fraction=.001):
    binary = (alpha > 110).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    result = []
    threshold = max(25, alpha.size * min_fraction)
    for i in range(1, n):
        x, y, w, h, area = [int(v) for v in stats[i]]
        if area < threshold or min(w, h) < 5:
            continue
        # Extend connected cores to nearby anti-aliased edge pixels only.
        support = cv2.dilate((labels == i).astype(np.uint8), np.ones((5, 5), np.uint8))
        a = np.where(support > 0, alpha, 0).astype(np.uint8)
        ys, xs = np.nonzero(a)
        if len(xs):
            x, y, w, h = int(xs.min()), int(ys.min()), int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)
        result.append({'box': (x, y, w, h), 'alpha': a, 'area': area})
    return sorted(result, key=lambda r: -r['area'])[:16]


def split_highlights(rgba, amount=.28):
    pixels = rgba.astype(np.float32)
    a = pixels[..., 3:4] / 255.
    minimum = pixels[..., :3].min(axis=2, keepdims=True)
    fraction = np.clip((minimum - 145) / 110, 0, 1) * amount
    ah = np.round(a * fraction * 255) / 255
    ab = np.divide(a - ah, 1 - ah, out=np.zeros_like(a), where=(1-ah) > 1e-6)
    rgb = np.divide(pixels[..., :3] * a - 255 * ah, a - ah,
                    out=np.zeros_like(pixels[..., :3]), where=(a-ah) > 1e-6)
    base = np.concatenate((np.clip(rgb, 0, 255), ab * 255), axis=2).round().astype(np.uint8)
    light = np.concatenate((np.full_like(rgb, 255), ah * 255), axis=2).round().astype(np.uint8)
    return base, light


def find_decor(rgb, foreground, protected=None):
    """Conservative flat colour cards; textured and irregular areas remain raster."""
    h, w = rgb.shape[:2]
    small = cv2.bilateralFilter(rgb, 7, 30, 30)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    candidates = []
    occupied = np.zeros((h, w), np.uint8)
    for lo, hi in ((0, 17), (18, 38), (90, 125)):
        mask = cv2.inRange(hsv, (lo, 90, 55), (hi, 255, 255))
        mask[foreground > 90] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if min(cw, ch) < 18 or area < h*w*.003 or area > h*w*.20:
                continue
            if area / (cw*ch) < .92 or np.any(occupied[y:y+ch, x:x+cw]):
                continue
            # The preserved logo matte is derived against the background.
            # Retain its card there, so a later shape cannot hide/recolor it.
            if protected is not None and np.any(protected[y:y+ch,x:x+cw]):
                continue
            section = rgb[y:y+ch, x:x+cw]
            if np.max(np.std(section.reshape(-1, 3), axis=0)) > 25:
                continue
            color = np.median(section.reshape(-1, 3), axis=0).astype(int)
            candidates.append({'name': f'色块{len(candidates)+1}｜颜色与描边可调·近似重建', 'box': [x,y,cw,ch],
                               'color': '#%02x%02x%02x' % tuple(color), 'radius': 0})
            occupied[y:y+ch, x:x+cw] = 255
    return candidates, occupied
