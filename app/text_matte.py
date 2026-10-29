"""Conservative local ink-colour recovery for antialiased text edges."""
import cv2
import numpy as np


def source_fidelity_ok(source, composite, region_mask):
    """Reject missing small strokes without averaging them into empty backing.

    ``region_mask`` must cover the original expected text region, including
    pixels missing from the extracted alpha. Eight-level AA differences pass;
    a connected three-pixel defect or eight scattered errors above ten channel
    levels fails. This validates appearance only, not layer editability.
    """
    original=np.asarray(source)
    rendered=np.asarray(composite)
    region=np.asarray(region_mask)>0
    if original.shape!=rendered.shape or original.ndim!=3 or original.shape[2]!=3 or region.shape!=original.shape[:2]:
        raise ValueError('Fidelity check requires equal RGB images and a matching region mask')
    error=np.max(np.abs(original.astype(np.float32)-rendered.astype(np.float32)),axis=2)
    bad=(error>10)&region
    count=int(np.count_nonzero(bad))
    if count<3:return True
    if count>=8:return False
    _,_,stats,_=cv2.connectedComponentsWithStats(bad.astype(np.uint8),connectivity=8)
    return not bool(np.any(stats[1:,cv2.CC_STAT_AREA]>=3))


def refine_ink_support(source, backdrop, candidate, ink_color, radius=1):
    """Find primary single-colour ink without promoting backdrop model errors.

    This estimates the primary ink only. The caller must retain its complex-
    effect path for multi-colour typography; ``usable`` means that real primary
    ink seeds exist, not that shadows or gradients have been recovered.
    Preserve original pixels outside the returned tight support when using it
    for local reconstruction. A broad learned erasure rectangle is not a matte.
    """
    observed=np.asarray(source,dtype=np.float32)
    background=np.asarray(backdrop,dtype=np.float32)
    allowed=np.asarray(candidate)>0
    color=np.asarray(ink_color,dtype=np.float32).reshape(3)
    result={'usable':False,'is_monochrome':False,'core':np.zeros(allowed.shape,bool),
            'support':allowed.astype(np.uint8)*255,'ink_color':color.round().astype(int).tolist()}
    if not np.any(allowed):return result
    ink_distance=np.linalg.norm(observed-color,axis=2)
    backdrop_distance=np.linalg.norm(observed-background,axis=2)
    # A precise structural mask contains mostly ink. Its median must not be
    # mistaken for backing. Prefer the immediately visible source outer ring;
    # use the region median only when a full-crop candidate has no such ring.
    tone_radius=max(3,int(round(radius))*3)
    ring=(cv2.dilate(allowed.astype(np.uint8),np.ones((tone_radius*2+1,tone_radius*2+1),np.uint8))>0)&~allowed
    ring&=ink_distance>35
    local_tone=np.median(observed[ring if np.count_nonzero(ring)>=8 else allowed],axis=0)
    backdrop_distance=np.maximum(backdrop_distance,np.linalg.norm(observed-local_tone,axis=2))
    # OCR's median ink often samples grey AA pixels rather than the true black
    # centre (or cream fringes rather than bright white centres). Follow the
    # ink direction beyond that median instead of selecting only its outline.
    ink_direction=color-local_tone
    ink_strength=float(np.linalg.norm(ink_direction))
    if ink_strength<35:return result
    projection=np.sum((observed-local_tone)*ink_direction,axis=2)/(ink_strength*ink_strength)
    color_residual=np.linalg.norm(observed-local_tone-projection[...,None]*ink_direction,axis=2)
    probable=allowed&(projection>=.8)&(color_residual<=max(22,ink_strength*.15))
    if np.count_nonzero(probable)>=3:
        strongest=probable&(projection>=np.percentile(projection[probable],75))
        color=np.median(observed[strongest],axis=0)
        ink_distance=np.linalg.norm(observed-color,axis=2)
    # Close to ink in the original image, not merely different from a model.
    # This rejects yellow/orange panels even if their reconstruction is wrong.
    seeds=allowed&((ink_distance<=22)|probable)&(backdrop_distance>=35)
    if np.count_nonzero(seeds)<3:return result
    measured=color.copy()
    radius=max(1,int(round(radius)))
    # A failed model may leave the source letters in its purported backdrop.
    # Estimate the immediate AA backing from original visible neighbours after
    # masking real source-colour seeds. This also follows multicolour panels.
    local_erase=cv2.dilate(seeds.astype(np.uint8)*255,np.ones((radius*4+1,radius*4+1),np.uint8))
    local_background=cv2.inpaint(np.asarray(source,dtype=np.uint8),local_erase,3,cv2.INPAINT_NS).astype(np.float32)
    background=local_background
    direction=measured-background
    denominator=np.sum(direction*direction,axis=2)
    alpha=np.sum((observed-background)*direction,axis=2)/np.maximum(denominator,1)
    residual=np.linalg.norm(observed-background-np.clip(alpha,0,1)[...,None]*direction,axis=2)
    distance=cv2.distanceTransform((~seeds).astype(np.uint8),cv2.DIST_L2,5)
    # Anti-alias coverage can extend a few pixels beyond the near-opaque seed.
    # Reconstructed panel colours further away never join the text support.
    fringe=allowed&(distance<=max(2,radius*2))&(alpha>.06)&(alpha<=1.15)&(residual<=22)
    tight=seeds|fringe
    opaque=seeds&(np.linalg.norm(observed-measured,axis=2)<=14)&(alpha>=.92)
    core=cv2.erode(opaque.astype(np.uint8),np.ones((radius*2+1,radius*2+1),np.uint8))>0
    # Keep thin opaque strokes; erosion is only an additional interior check.
    if not np.any(core):core=opaque
    # Plain ink and its AA lie on the visible backing-to-ink colour direction.
    # A substantial different-colour band touching the ink (e.g. dark blue
    # extrusion under a white headline) must retain the complex-effect path.
    # Long labels may cross a gradient or coloured stripe. Classify effects
    # relative to each column's visible local backing rather than one global
    # median, which otherwise calls yellow-to-cyan panels coloured shadows.
    tone_columns=np.broadcast_to(local_tone,(observed.shape[1],3)).copy()
    sample_columns=np.flatnonzero(np.count_nonzero(ring,axis=0)>=2)
    if len(sample_columns)>=2:
        sampled=np.array([np.median(observed[ring[:,x],x],axis=0) for x in sample_columns])
        for channel in range(3):
            tone_columns[:,channel]=np.interp(np.arange(observed.shape[1]),sample_columns,sampled[:,channel])
    tones=tone_columns[None,:,:]
    main_direction=measured-tones
    main_strength=np.maximum(np.linalg.norm(main_direction,axis=2),1.)
    main_projection=np.sum((observed-tones)*main_direction,axis=2)/(main_strength*main_strength)
    main_residual=np.linalg.norm(observed-tones-main_projection[...,None]*main_direction,axis=2)
    neighboring=allowed&(distance<=max(4,radius*4))&~seeds
    substantial=np.linalg.norm(observed-tones,axis=2)>np.maximum(85,main_strength*.30)
    incompatible=(neighboring&substantial&(np.linalg.norm(observed-measured,axis=2)>50)
                  &((main_projection<.05)|(main_residual>45)))
    incompatible_count=int(np.count_nonzero(incompatible))
    monochrome=incompatible_count<max(12,np.count_nonzero(seeds)*.07)
    result.update(usable=True,core=core,support=tight.astype(np.uint8)*255,
                  ink_color=measured.round().astype(int).tolist(),
                  retained_fraction=float(np.count_nonzero(tight)/np.count_nonzero(allowed)),
                  is_monochrome=bool(monochrome),incompatible_pixels=incompatible_count,
                  seed_pixels=int(np.count_nonzero(seeds)),
                  erase_mask=cv2.bitwise_and(cv2.dilate(tight.astype(np.uint8)*255,
                    np.ones((radius*2+1,radius*2+1),np.uint8)),allowed.astype(np.uint8)*255),
                  local_background=local_background.round().astype(np.uint8))
    return result


def decontaminate_edges(rgba, source, backdrop, core):
    """Use nearby opaque ink only where it explains the observed edge.

    Minimum-alpha matting can turn a small backdrop estimation error into a
    saturated foreground fringe. Project onto the nearest trusted core colour,
    but keep the original solution if it cannot explain all source channels
    within eight levels. This is local antialias decontamination, not a global
    assumption that a word, its gradient, and its shadow share one colour.
    Inputs are never mutated; transparent holes and opaque pixels are retained.
    """
    result=rgba.copy()
    original=source.astype(np.float32)
    background=backdrop.astype(np.float32)
    trusted=(np.asarray(core,dtype=bool)&(rgba[...,3]>=240)
             &(np.linalg.norm(original-background,axis=2)>24))
    fringe=(rgba[...,3]>0)&(rgba[...,3]<255)&~np.asarray(core,dtype=bool)
    if not np.any(trusted) or not np.any(fringe):return result
    distance,labels=cv2.distanceTransformWithLabels((~trusted).astype(np.uint8),
        cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
    nearby=fringe&(distance<=6)
    if not np.any(nearby):return result
    palette=np.zeros((int(labels.max())+1,3),np.float32)
    palette[labels[trusted]]=original[trusted]
    foreground=palette[labels[nearby]]
    base=background[nearby];observed=original[nearby]
    direction=foreground-base
    denominator=np.sum(direction*direction,axis=1)
    alpha=np.sum((observed-base)*direction,axis=1)/np.maximum(denominator,1)
    opacity=np.rint(np.clip(alpha,0,1)*255).astype(np.uint8)
    quantized=opacity.astype(np.float32)/255
    recomposed=np.rint(base*(1-quantized[:,None])+foreground*quantized[:,None])
    accepted=((alpha>=0)&(alpha<=1)&(denominator>24*24)
              &(np.max(np.abs(recomposed-observed),axis=1)<=8))
    ys,xs=np.nonzero(nearby);ys,xs=ys[accepted],xs[accepted]
    result[ys,xs,:3]=np.rint(foreground[accepted]).astype(np.uint8)
    result[ys,xs,3]=opacity[accepted]
    return result
