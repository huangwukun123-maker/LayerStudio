"""Reversible decomposition of unsupported typography, without sample rules.

Visible raster glyphs are not TypeLayers. Their hidden type alternatives are
explicitly approximate. Invisible backing pixels are reconstructed, never
described as recovered original artwork.
"""
import cv2
import numpy as np
from PIL import Image
import vision
import text_backdrop
from text_matte import decontaminate_edges, refine_ink_support, source_fidelity_ok


def repair_overlaps(repair, neighbor_mask):
    x,y,pixels,changed=repair
    return bool(np.any(changed & (neighbor_mask[y:y+pixels.shape[0],x:x+pixels.shape[1]]>0)))


def preserve_text_edges(enhanced, original, mask):
    """Upsample original typography without SR hallucinated holes or stroke tips."""
    height,width=enhanced.shape[:2]
    source=np.array(original.resize((width,height),Image.Resampling.LANCZOS).convert('RGB'))
    support=cv2.resize(mask,(width,height),interpolation=cv2.INTER_NEAREST)
    support=cv2.dilate(support,np.ones((7,7),np.uint8))
    weight=cv2.GaussianBlur(support.astype(np.float32)/255,(7,7),1)[...,None]
    return np.round(enhanced*(1-weight)+source*weight).astype(np.uint8)


def prepare(rgb, regions, inpaint=None, extra_erase_mask=None):
    h,w=rgb.shape[:2]
    owner=np.full((h,w),-1,np.int32)
    prepared=[]
    polygons=[]
    for item in regions:
        polygon=np.zeros((h,w),np.uint8)
        cv2.fillConvexPoly(polygon,np.round(item['quad']).astype(np.int32),255)
        polygons.append(polygon)
    erase_union=np.zeros((h,w),np.uint8) if extra_erase_mask is None else extra_erase_mask.copy()
    complex_union=erase_union.copy();coverage_union=erase_union.copy();mono_zone=np.zeros((h,w),np.uint8);mono_erase=np.zeros((h,w),np.uint8)
    for index,item in enumerate(regions):
        neighbors=np.zeros((h,w),np.uint8)
        for other_index,polygon in enumerate(polygons):
            if other_index!=index:neighbors=cv2.bitwise_or(neighbors,polygon)
        repair=text_backdrop.reconstruct(rgb,item['quad'],item['mask'],excluded=neighbors)
        coverage_union=cv2.bitwise_or(coverage_union,repair['erase_mask'])
        record={**item,'mask':repair['mask'],'index':index,'backdrop':repair}
        # Validate ink against source colours before any learned backdrop can
        # turn a panel tone shift into a solid glyph.
        if inpaint is not None:
            x,y,pixels=repair['repair'];estimate=rgb.copy();estimate[y:y+pixels.shape[0],x:x+pixels.shape[1]]=pixels
            color=[int(item['spec']['color'][i:i+2],16) for i in (1,3,5)]
            refined=refine_ink_support(rgb,estimate,repair['structural_mask'],color)
            if refined['usable']:record['primary_support']=refined['support']
            if refined['is_monochrome']:
                record['mono_color']=refined['ink_color']
                record['mask']=refined['erase_mask']
                record['mask']=cv2.bitwise_and(record['mask'],repair['erase_mask'])
                mono_erase=cv2.bitwise_or(mono_erase,record['mask'])
                mono_zone=cv2.bitwise_or(mono_zone,repair['erase_mask'])
        prepared.append(record)
        if 'mono_color' not in record:complex_union=cv2.bitwise_or(complex_union,repair['erase_mask'])
    erase_union=cv2.bitwise_or(complex_union,mono_erase)
    blended=text_backdrop.blend_reconstruction(rgb,erase_union,inpaint) if inpaint is not None and np.any(erase_union) else None
    learned=blended['image'] if blended is not None else None
    if blended is not None:erase_union=blended['mask']
    if learned is not None and np.any(mono_erase):
        # Thin plain letters need only a local repair. Preserve the real
        # coloured panel around them instead of replacing its entire OCR box.
        local=cv2.inpaint(rgb,mono_erase,3,cv2.INPAINT_NS)
        fringe=round(blended['feather_radius'])
        complex_protected=cv2.dilate(complex_union,np.ones((2*fringe+1,2*fringe+1),np.uint8))
        mono_only=(mono_zone>0)&(complex_protected==0)
        learned[mono_only]=local[mono_only]
        erase_union[mono_only]=mono_erase[mono_only]
    if learned is not None:
        for item in prepared:
            repair=item['backdrop'];quality=repair['quality']
            # The eraser may reconstruct the complete OCR box; only actual
            # edge-bounded text support enters independently movable layers.
            if 'mono_color' not in item:
                item['mask']=repair['structural_mask'].copy()
            repair.update(method='learned_inpaint',repair=(0,0,learned))
    # Resolve shared support once, not separately for each exported raster.
    # Smaller labels own their pixels when a long-line detector covers both.
    for item in sorted(prepared,key=lambda r:np.count_nonzero(r['mask']),reverse=True):
        owner[item['mask']>0]=item['index']
    union=(owner>=0).astype(np.uint8)*255
    # Use the complete reconstructed text backdrop when the learned model is
    # available. Pasting it only through glyph silhouettes creates letter-shaped
    # tone seams. Rebuilt visible gaps are disclosed, not carried on the glyph.
    clean=learned.copy() if learned is not None else (cv2.inpaint(rgb,union,3,cv2.INPAINT_NS) if np.any(union) else rgb.copy())
    for item in prepared:
        mask=(owner==item['index']).astype(np.uint8)*255
        item['mask']=mask
        if not np.any(mask):
            item['repair_method']='重叠区域已归入相邻文字'
            continue
        repair=item['backdrop']
        if repair['method']!='local_inpaint':
            x,y,pixels=repair['repair'];active=mask[y:y+pixels.shape[0],x:x+pixels.shape[1]]>0
            target=clean[y:y+pixels.shape[0],x:x+pixels.shape[1]]
            target[active]=pixels[active]
            item['repair_method']=('按单色字形局部补全底色' if 'mono_color' in item else '本机修复模型估算文字后方背景') if repair['method']=='learned_inpaint' else '按周围可见底色拟合重建'
        else:
            item['repair_method']='按局部字形遮挡范围补全，隐藏纹理为估算'
    return {'regions':prepared,'clean':clean,'union':union,
            'coverage_union':coverage_union,
            'erase_union':erase_union if learned is not None else union,'uses_inpaint':learned is not None}


def _split_appearance(rgba, source, spec):
    color=np.array([int(spec['color'][i:i+2],16) for i in (1,3,5)])
    opaque=(rgba[...,3]>240)&(np.linalg.norm(source.astype(float)-color,axis=2)<65)
    if np.count_nonzero(opaque)>20:
        color=np.median(source[opaque],axis=0)
    distance=np.linalg.norm(source.astype(float)-color,axis=2)
    # This is a visible-colour split, not identification of original PS fx.
    face=np.clip((100-distance)/45,0,1)
    a=rgba[...,3]/255
    main_mass=np.sum(a*face);edge_mass=np.sum(a*(1-face))
    if main_mass<20 or edge_mass<max(15,main_mass*.08):
        return [{'name':'原字形与效果｜透明像素·描边颜色可调','rgba':rgba}]
    face_a=a*face
    edge_a=np.divide(a-face_a,1-face_a,out=np.zeros_like(a),where=face_a<1)
    front=rgba.copy();back=rgba.copy()
    front[...,3]=np.round(face_a*255).astype(np.uint8)
    back[...,3]=np.round(edge_a*255).astype(np.uint8)
    front[front[...,3]==0,:3]=0;back[back[...,3]==0,:3]=0
    return [{'name':'边缘与阴影｜近似分离·可调透明度','rgba':back},
            {'name':'字面｜保留原字形·描边颜色可调','rgba':front}]


def _preserved_region(enhanced,item):
    height,width=enhanced.shape[:2]
    region=item['backdrop']['erase_mask']
    radius=max(4,min(12,round(max(region.shape)*.004)))
    region=cv2.dilate(region,np.ones((2*radius+1,2*radius+1),np.uint8))
    support=cv2.resize(region,(width,height),interpolation=cv2.INTER_NEAREST)
    yy,xx=np.nonzero(support)
    left=max(0,int(xx.min())-2);right=min(width,int(xx.max())+3)
    top=max(0,int(yy.min())-2);bottom=min(height,int(yy.max())+3)
    rgba=np.dstack((enhanced[top:bottom,left:right],support[top:bottom,left:right]))
    rgba[rgba[...,3]==0,:3]=0
    return {'index':item['index'],'spec':item['spec'],'left':left,'top':top,
        'contains_background':True,'repair_method':'透明分离或逐像素保真检查未通过，保留含背景原区域；底图为估算重建',
        'layers':[{'name':'保真原区域｜含背景·需精修，不能直接改字','rgba':rgba}]}


def render(enhanced, prepared, sx, sy):
    """Return a textless base and bounded transparent appearance layers.

    Decomposition is performed against the final enhanced pixels, so the
    composition does not inherit low-resolution interpolation or inpaint blur.
    Only hidden backing pixels use an estimated replacement.
    """
    height,width=enhanced.shape[:2]
    clean=enhanced.copy();rendered=[]
    backdrop=cv2.resize(prepared['clean'],(width,height),interpolation=cv2.INTER_CUBIC)
    if 'erase_union' in prepared:
        erase=cv2.resize(prepared['erase_union'],(width,height),interpolation=cv2.INTER_NEAREST)>0
        clean[erase]=backdrop[erase]
    for item in prepared['regions']:
        mask=cv2.resize(item['mask'],(width,height),interpolation=cv2.INTER_NEAREST)
        ys,xs=np.nonzero(mask)
        if not len(xs):continue
        if prepared.get('uses_inpaint') and 'mono_color' not in item:
            # Complex gradients/extrusions cannot yet be matted reliably.
            # Preserve their actual pixels instead of publishing missing
            # strokes or backdrop-colour fragments as a successful cutout.
            # This is explicitly a source REGION, not transparent glyph ink.
            rendered.append(_preserved_region(enhanced,item))
            continue
        if prepared.get('uses_inpaint'):
            evidence=cv2.resize(item['backdrop']['erase_mask'],(width,height),interpolation=cv2.INTER_NEAREST)
            ys,xs=np.nonzero(evidence)
        left,top,right,bottom=int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)
        # Two transparent pixels around every region make its bounds explicit
        # without adding an opaque OCR rectangle.
        left=max(0,left-2);top=max(0,top-2);right=min(width,right+2);bottom=min(height,bottom+2)
        support=mask[top:bottom,left:right]
        source=enhanced[top:bottom,left:right]
        base=source.copy()
        replacement=backdrop[top:bottom,left:right]
        active=(support>0)&(np.max(np.abs(source.astype(int)-replacement),axis=2)>2)
        base[active]=replacement[active]
        rgba=vision.foreground_from_background(source,base,support)
        if 'mono_color' in item:
            color=np.asarray(item['mono_color'],np.float32)
            direction=color-base.astype(np.float32)
            amount=np.sum((source.astype(np.float32)-base)*direction,axis=2)/np.maximum(np.sum(direction*direction,axis=2),1)
            amount=np.clip(amount,0,1)*(support>0)
            rgba[...,:3]=color.astype(np.uint8)
            rgba[...,3]=np.round(amount*255).astype(np.uint8)
            rgba[rgba[...,3]==0,:3]=0
        # Minimum-alpha compositing alone makes opaque pale lettering partly
        # transparent. Recover opaque contrasting cores so recolouring works;
        # retain the solved antialias fringe and exact source composition.
        core=(np.linalg.norm(source.astype(float)-base,axis=2)>24).astype(np.uint8)
        radius=max(1,round(min(sx,sy)))
        core=cv2.erode(core,np.ones((2*radius+1,2*radius+1),np.uint8))>0
        if 'mono_color' in item:
            core&=(rgba[...,3]>245)&(np.linalg.norm(source.astype(float)-np.array(item['mono_color']),axis=2)<24)
        rgba[core,:3]=source[core];rgba[core,3]=255
        if prepared.get('uses_inpaint'):
            rgba=decontaminate_edges(rgba,source,base,core)
            opacity=rgba[...,3:4]/255
            trial=np.round(rgba[...,:3]*opacity+clean[top:bottom,left:right]*(1-opacity)).astype(np.uint8)
            evidence=cv2.resize(item['backdrop']['structural_mask'],(width,height),interpolation=cv2.INTER_NEAREST)[top:bottom,left:right]
            # Judge expected ink, not reconstructed background/AA fringes.
            # This evidence is derived independently from the source and the
            # estimated backing, so an omitted stroke can still fail the gate.
            direction=np.asarray(item['mono_color'],np.float32)-replacement
            observed=source.astype(np.float32)-replacement
            expected=np.sum(observed*direction,axis=2)/np.maximum(np.sum(direction*direction,axis=2),1)
            residual=np.linalg.norm(observed-np.clip(expected,0,1)[...,None]*direction,axis=2)
            actual=np.sum((trial.astype(np.float32)-replacement)*direction,axis=2)/np.maximum(np.sum(direction*direction,axis=2),1)
            # JPEG chroma noise can legitimately disappear when a plain ink
            # color is recovered. Reject missing ink coverage, not that noise.
            loss=np.clip(expected,0,1)-np.clip(actual,0,1)
            evidence=((evidence>0)&(expected>=.35)&(loss>.20)&(residual<28)&(np.linalg.norm(observed,axis=2)>32)).astype(np.uint8)*255
            if not source_fidelity_ok(source,trial,evidence):
                rendered.append(_preserved_region(enhanced,item))
                continue
        # Raster bounds of two regions can intersect even when their masks do
        # not. Replacing the whole crop here would restore earlier letters.
        target=clean[top:bottom,left:right]
        target[active]=replacement[active]
        rendered.append({'index':item['index'],'spec':item['spec'],'left':left,'top':top,
                         'repair_method':item['repair_method'],
                         'layers':_split_appearance(rgba,source,item['spec'])})
    return clean,rendered
