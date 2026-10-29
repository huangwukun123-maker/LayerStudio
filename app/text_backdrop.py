"""Bounded reconstruction behind text, excluding nearby colored objects.

This estimates hidden pixels; it does not recover the original background.
Multi-color glyph interiors are included even when the OCR seed only captures a
light outline. Smooth local models require a majority consensus of visible
samples. Textured regions retain a labeled local-inpainting fallback.
"""
import math

import cv2
import numpy as np


def refine_complex_support(rgb,estimate,structural_mask,erase_mask):
    """Reject backdrop-color differences from complex glyph candidates.

    Smooth visible gaps estimate the background model's ordinary color error.
    Ink cores must exceed that error; weak antialias pixels are retained only
    immediately beside a core. This is conservative appearance segmentation,
    not proof of the original font or an exact alpha matte.
    """
    source=np.asarray(rgb);backdrop=np.asarray(estimate)
    structural=np.asarray(structural_mask)>0;envelope=np.asarray(erase_mask)>0
    if source.shape!=backdrop.shape or source.shape[:2]!=structural.shape or structural.shape!=envelope.shape:
        raise ValueError('Complex glyph refinement requires matching RGB images and masks')
    delta=np.linalg.norm(source.astype(float)-backdrop.astype(float),axis=2)
    smooth=cv2.GaussianBlur(source.astype(np.float32),(3,3),.8)
    detail=np.max(np.abs(source.astype(float)-smooth),axis=2)
    samples=delta[envelope&~structural&(detail<6)]
    error=10.
    if len(samples)>=30:
        median=float(np.median(samples));mad=float(np.median(np.abs(samples-median)))
        error=min(55.,max(8.,median+3*max(2.,mad)))
    core=(delta>max(24.,error+12))&structural&envelope
    if not np.any(core):
        return np.zeros(structural.shape,np.uint8)
    distance=cv2.distanceTransform((~core).astype(np.uint8),cv2.DIST_L2,3)
    weak=(distance<=1.5)&(delta>max(2.,error*.45))&structural&envelope
    return (core|weak).astype(np.uint8)*255


def blend_reconstruction(rgb,erase_mask,inpaint,feather_radius=None):
    """Reconstruct an expanded region, blending its outer fringe only.

    The original erasure mask always uses 100% reconstructed pixels, so no
    original letter can return through feathering. ``mask`` in the return value
    includes the expanded blend fringe: downstream compositing must preserve
    this whole region instead of clipping it back to ``core_mask``. Feathering
    still reconstructs some previously visible background; it is not recovery
    of original hidden artwork.
    """
    source=np.asarray(rgb)
    original_mask=np.asarray(erase_mask)
    if source.ndim!=3 or source.shape[2]!=3 or original_mask.shape!=source.shape[:2]:
        raise ValueError('Reconstruction blending requires an RGB image and matching two-dimensional mask')
    # Keep expansion narrow around densely packed product labels. A wide ring
    # can hide neighboring logos or physical edges from the background model.
    radius=(max(4,min(12,round(max(source.shape[:2])*.004)))
            if feather_radius is None else float(feather_radius))
    if not math.isfinite(radius) or radius<0:
        raise ValueError('Reconstruction feather radius must be a finite nonnegative number')
    core=(original_mask>0).astype(np.uint8)*255
    if not np.any(core):
        return {'image':source.copy(),'mask':core.copy(),'core_mask':core,'feather_radius':radius}
    if radius==0:
        expanded=core.copy();weight=(core>0).astype(np.float32)
    else:
        distance=cv2.distanceTransform((core==0).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        expanded=(distance<=radius).astype(np.uint8)*255
        fraction=np.clip(1-distance/radius,0,1)
        # Smoothstep has zero derivative at both ends of the transition,
        # avoiding a hard edge where the learned patch meets visible pixels.
        weight=fraction*fraction*(3-2*fraction)
        weight[core>0]=1
    reconstructed=np.asarray(inpaint(source,expanded))
    if reconstructed.shape!=source.shape or not np.isfinite(reconstructed).all():
        raise ValueError('Inpainting returned an invalid RGB reconstruction')
    mixed=np.round(source.astype(np.float32)*(1-weight[...,None])
                   +reconstructed.astype(np.float32)*weight[...,None])
    output=source.copy()
    selected=expanded>0
    output[selected]=np.clip(mixed[selected],0,255).astype(source.dtype)
    return {'image':output,'mask':expanded,'core_mask':core,'feather_radius':radius}


def harmonize_background(rgb,estimate,mask):
    """Match a learned backdrop to visible local tones beneath glyphs only.

    Supply the union of ALL glyph supports, so another letter is not used as a
    color-offset sample. The inpainted quantity is signed source-minus-model
    correction, not scene RGB: the learned geometry remains intact and a nearby
    correctly reconstructed colored object contributes zero correction.
    Unmasked pixels of the estimate are returned byte-for-byte unchanged.
    """
    source=np.asarray(rgb)
    backdrop=np.asarray(estimate)
    hidden=np.asarray(mask)>0
    if source.shape!=backdrop.shape or source.ndim!=3 or source.shape[2]!=3 or hidden.shape!=source.shape[:2]:
        raise ValueError('Background harmonization requires matching RGB images and a two-dimensional mask')
    result=backdrop.copy()
    if not np.any(hidden) or np.all(hidden):
        return result
    # Float planes retain signed offsets without uint8 centering/clipping and
    # avoid losing subtle gradient corrections through repeated quantization.
    correction=source.astype(np.float32)-backdrop.astype(np.float32)
    # A missed stroke outside the mask differs greatly from a plausible
    # background. Such residuals cannot be treated as tone offsets, or ink gets
    # spread back into the newly cleared region. Estimate only modest shifts.
    unreliable=np.linalg.norm(correction,axis=2)>40
    unknown=(hidden|unreliable).astype(np.uint8)*255
    if np.all(unknown):
        return result
    correction[unknown>0]=0
    filled=np.stack([cv2.inpaint(np.ascontiguousarray(correction[...,channel]),unknown,3,cv2.INPAINT_NS)
                     for channel in range(3)],axis=2)
    corrected=np.clip(backdrop.astype(np.float32)+filled,0,255).round().astype(backdrop.dtype)
    result[hidden]=corrected[hidden]
    return result


def _design(height,width):
    yy,xx=np.mgrid[-1:1:complex(height),-1:1:complex(width)]
    return np.stack((np.ones_like(xx),xx,yy,xx*yy,xx*xx,yy*yy),axis=-1)


def _robust_model(features,colors):
    if len(colors)<60:
        return None
    # Deterministic subsampling bounds CPU while retaining spatial coverage.
    step=max(1,len(colors)//5000)
    x=features[::step];y=colors[::step].astype(float)
    rng=np.random.default_rng(7483)
    candidates=[np.arange(len(y))]
    candidates.extend(rng.choice(len(y),min(10,len(y)),replace=False) for _ in range(32))
    best=None
    for chosen in candidates:
        coefficients,_,rank,_=np.linalg.lstsq(x[chosen],y[chosen],rcond=None)
        if rank<6:
            continue
        error=np.linalg.norm(x@coefficients-y,axis=1)
        keep=error<20
        count=int(np.count_nonzero(keep))
        score=(count,-float(np.median(error)))
        if best is None or score>best[0]:
            best=(score,coefficients,keep)
    if best is None or np.mean(best[2])<.30:
        return None
    coefficients,keep=best[1],best[2]
    for _ in range(4):
        coefficients,_,rank,_=np.linalg.lstsq(x[keep],y[keep],rcond=None)
        if rank<6:
            return None
        error=np.linalg.norm(x@coefficients-y,axis=1)
        keep=error<20
    if np.mean(keep)<.52 or np.percentile(error[keep],90)>18:
        return None
    return coefficients,{'inlier_fraction':round(float(np.mean(keep)),4),
                         'sample_error':round(float(np.percentile(error[keep],90)),3),
                         'color_low':np.maximum(0,y[keep].min(axis=0)-8).tolist(),
                         'color_high':np.minimum(255,y[keep].max(axis=0)+8).tolist()}


def _bounded_fit(features,model):
    # One-sided samples around crowded labels can otherwise extrapolate into
    # negative RGB or spurious saturated colors despite a small sample residual.
    coefficients,quality=model
    return np.clip(features@coefficients,quality['color_low'],quality['color_high'])


def _edge_support(crop,envelope,glyph_h):
    # Canny on the color image uses the strongest channel edge: red lettering
    # need not contrast with blue backing in grayscale to be detectable.
    edges=cv2.Canny(crop,25,70)
    count,labels,stats,_=cv2.connectedComponentsWithStats(edges)
    total=stats[:,cv2.CC_STAT_AREA]
    inside=np.bincount(labels[envelope>0],weights=(edges[envelope>0]>0),minlength=count)
    keep=(inside>=total*.80)&(total>=3);keep[0]=False
    edges=(keep[labels]&(envelope>0)).astype(np.uint8)*255
    close_size=5 if glyph_h>=60 else 3
    edges=cv2.morphologyEx(edges,cv2.MORPH_CLOSE,np.ones((close_size,close_size),np.uint8))
    contours,_=cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    structural=np.zeros_like(envelope)
    cv2.drawContours(structural,contours,-1,255,cv2.FILLED)
    # Low-contrast white glyph bodies may merge with a white glow, leaving only
    # their contrasting edge fragments. Bound completion to one glyph-sized
    # component; never fill the convex hull of an entire text line.
    for contour in contours:
        _,_,width,height=cv2.boundingRect(contour)
        if height>=glyph_h*.35 and width<=glyph_h*1.8:
            cv2.drawContours(structural,[cv2.convexHull(contour)],-1,255,cv2.FILLED)
    return cv2.dilate(structural,np.ones((3,3),np.uint8))


def _piecewise_fit(crop,blocked,glyph_h,global_fit):
    """A long headline can cross both a light glow and a dark sky.

    Local overlapping models preserve that varying backdrop instead of fitting
    a single dominant background color across the full line.
    """
    height,width=crop.shape[:2]
    if width<glyph_h*3.5:
        return global_fit
    # Local windows must not re-introduce colored objects rejected by the
    # full-line consensus. Achromatic glow/brightness variation remains usable.
    delta=crop.astype(float)-global_fit
    local_blocked=blocked|(np.ptp(delta,axis=2)>45)
    half=max(20,round(glyph_h*1.25))
    numerator=np.zeros(crop.shape,float);weight=np.zeros(crop.shape[:2],float)
    for center in range(0,width+half,half):
        left=max(0,center-half);right=min(width,center+half+1)
        if right-left<12:continue
        samples=~local_blocked[:,left:right]
        features=_design(height,right-left)
        model=_robust_model(features[samples],crop[:,left:right][samples])
        if model is None:continue
        fitted=_bounded_fit(features,model)
        local_weight=np.maximum(.02,1-np.abs(np.arange(left,right)-center)/(half+1))
        numerator[:,left:right]+=fitted*local_weight[None,:,None]
        weight[:,left:right]+=local_weight
    result=global_fit.copy().astype(float)
    np.divide(numerator,weight[...,None],out=result,where=weight[...,None]>0)
    return np.clip(result,0,255).round().astype(np.uint8)


def reconstruct(rgb,quad,mask,excluded=None,protected=None):
    """Return a bounded repair and complete multi-color glyph support.

    ``excluded`` contains OTHER text regions for background sampling only.
    Optional ``protected`` pixels must not be changed; leave it unset when the
    caller resolves overlapping text supports with a unique ownership map.
    Result ``repair`` is ``(left,top,rgb_crop)``; apply it only where the returned
    full-size ``mask`` is nonzero, after resolving shared pixel ownership.
    ``structural_mask`` supplies source-edge-bounded glyph candidates for a
    stronger background model; ``erase_mask`` supplies its broad OCR envelope.
    Neither the broad erasure envelope nor the background estimate is a glyph
    alpha layer. The input is never mutated. ``method`` labels estimation.
    """
    image_h,image_w=rgb.shape[:2]
    q=np.asarray(quad,np.float32)
    polygon=np.zeros((image_h,image_w),np.uint8)
    cv2.fillConvexPoly(polygon,np.round(q).astype(np.int32),255)
    # Geometry-derived margins, independent of any sample's text or position.
    glyph_h=max(2.,float((np.linalg.norm(q[3]-q[0])+np.linalg.norm(q[2]-q[1]))/2))
    fringe=max(3,min(8,round(glyph_h*.18)))
    envelope=cv2.dilate(polygon,np.ones((fringe*2+1,fringe*2+1),np.uint8))
    support=cv2.bitwise_and(np.asarray(mask,np.uint8),envelope)
    # A white outline encloses the colored letter body. Filling its external
    # contours also covers counters, whose matching backing yields alpha zero
    # when the caller decomposes the source against the reconstructed backdrop.
    closed=cv2.morphologyEx(support,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(closed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    support=np.zeros_like(polygon)
    cv2.drawContours(support,contours,-1,255,cv2.FILLED)
    support=cv2.dilate(support,np.ones((fringe*2+1,fringe*2+1),np.uint8))
    support=cv2.bitwise_and(support,envelope)
    others=np.zeros_like(polygon) if excluded is None else np.asarray(excluded,np.uint8)
    protected=np.zeros_like(polygon) if protected is None else np.asarray(protected,np.uint8)
    support[protected>0]=0
    pad=max(10,math.ceil(glyph_h*.85))
    x0=max(0,math.floor(q[:,0].min())-pad);x1=min(image_w,math.ceil(q[:,0].max())+pad+1)
    y0=max(0,math.floor(q[:,1].min())-pad);y1=min(image_h,math.ceil(q[:,1].max())+pad+1)
    crop=rgb[y0:y1,x0:x1]
    if not crop.size:
        raise ValueError('Text region has no image area')
    local_envelope=envelope[y0:y1,x0:x1]
    structural=_edge_support(crop,local_envelope,glyph_h)
    # Do not let an incorrect luminance seed turn the whole line's sky into
    # foreground. Edge-bounded shapes include closed, differently colored fills.
    support[:]=0
    support[y0:y1,x0:x1]=cv2.bitwise_and(structural,local_envelope)
    support[protected>0]=0
    structural_mask=support.copy()
    erase_mask=envelope.copy();erase_mask[protected>0]=0
    local_support=support[y0:y1,x0:x1]
    blocked=cv2.bitwise_or(envelope,others)[y0:y1,x0:x1]>0
    features=_design(*crop.shape[:2])
    model=_robust_model(features[~blocked],crop[~blocked])
    if model is None:
        repaired=cv2.inpaint(crop,local_support,3,cv2.INPAINT_NS)
        return {'mask':support,'structural_mask':structural_mask,'erase_mask':erase_mask,
                'repair':(x0,y0,repaired),'method':'local_inpaint','quality':None}
    coefficients,quality=model
    fitted=_bounded_fit(features,model).round().astype(np.uint8)
    fitted=_piecewise_fit(crop,blocked,glyph_h,fitted)
    # A seed based on luminance can contain sky/cloud patches and miss the red
    # interior of a white outlined logo. Compare against the validated local
    # backdrop instead, retaining only contrasting ink plus a narrow AA fringe.
    difference=np.linalg.norm(crop.astype(float)-fitted,axis=2)
    threshold=max(10,min(22,quality['sample_error']*1.3))
    ink=((difference>threshold)&(structural>0)&(local_envelope>0)).astype(np.uint8)*255
    ink=cv2.morphologyEx(ink,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    ink=cv2.dilate(ink,np.ones((3,3),np.uint8))
    ink[envelope[y0:y1,x0:x1]==0]=0
    ink[protected[y0:y1,x0:x1]>0]=0
    support[:]=0;support[y0:y1,x0:x1]=ink
    return {'mask':support,'structural_mask':structural_mask,'erase_mask':erase_mask,
            'repair':(x0,y0,fitted),'method':'robust_polynomial','quality':quality}
