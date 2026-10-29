"""Regression checks for full ink coverage and bounded text-backdrop repair."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
from text_backdrop import reconstruct


def apply(rgb,result):
    clean=rgb.copy()
    x,y,pixels=result['repair']
    section=clean[y:y+pixels.shape[0],x:x+pixels.shape[1]]
    mask=result['mask'][y:y+pixels.shape[0],x:x+pixels.shape[1]]>0
    section[mask]=pixels[mask]
    return clean


def blue_gradient(height=170,width=380):
    yy,xx=np.mgrid[:height,:width]
    return np.stack((70+xx*.05+yy*.12,140+xx*.04+yy*.1,220+xx*.03),axis=2).round().astype(np.uint8)


def test_outlined_colored_letters_include_fill_when_seed_only_finds_outline():
    bg=blue_gradient();rgb=bg.copy()
    cv2.putText(rgb,'ABC',(80,105),cv2.FONT_HERSHEY_SIMPLEX,2,(250,250,250),12,cv2.LINE_AA)
    cv2.putText(rgb,'ABC',(80,105),cv2.FONT_HERSHEY_SIMPLEX,2,(190,30,45),7,cv2.LINE_AA)
    seed=(np.min(rgb,axis=2)>230).astype(np.uint8)*255
    result=reconstruct(rgb,[[72,50],[218,50],[218,113],[72,113]],seed)
    red=(rgb[...,0]>150)&(rgb[...,1]<60)
    assert np.mean(result['mask'][red]>0)>.99,'colored glyph interiors were left behind'
    clean=apply(rgb,result)
    assert np.percentile(np.max(np.abs(clean.astype(int)-bg),axis=2)[red],95)<5
    assert np.array_equal(clean[result['mask']==0],rgb[result['mask']==0])
    assert result['method']=='robust_polynomial'


def test_headline_near_contrasting_object_does_not_pull_object_color_into_sky():
    bg=blue_gradient(240,520);rgb=bg.copy()
    rgb[141:215,235:490]=[195,25,30]
    original=rgb.copy()
    core=np.zeros(rgb.shape[:2],np.uint8)
    cv2.putText(core,'QUALITY 75%',(18,127),cv2.FONT_HERSHEY_SIMPLEX,1.9,255,6,cv2.LINE_AA)
    shadow=cv2.warpAffine(core,np.float32([[1,0,3],[0,1,4]]),(520,240))
    rgb[shadow>0]=[10,45,95];rgb[core>0]=[245,250,255]
    seed=cv2.dilate(core,np.ones((5,5),np.uint8))
    result=reconstruct(rgb,[[12,81],[465,81],[465,134],[12,134]],seed)
    clean=apply(rgb,result)
    assert result['method']=='robust_polynomial'
    assert np.percentile(np.max(np.abs(clean.astype(int)-bg),axis=2)[core>0],95)<8
    assert np.array_equal(clean[145:215,235:490],original[145:215,235:490])


def test_neighboring_text_is_not_used_as_background_or_erased():
    bg=blue_gradient();rgb=bg.copy()
    cv2.putText(rgb,'TEST',(30,82),cv2.FONT_HERSHEY_SIMPLEX,1.5,(250,250,255),5,cv2.LINE_AA)
    cv2.putText(rgb,'NEXT',(30,114),cv2.FONT_HERSHEY_SIMPLEX,.8,(180,20,30),3,cv2.LINE_AA)
    seed=(np.min(rgb,axis=2)>230).astype(np.uint8)*255
    neighbor=np.zeros(rgb.shape[:2],np.uint8);neighbor[88:120,22:115]=255
    result=reconstruct(rgb,[[25,45],[145,45],[145,85],[25,85]],seed,excluded=neighbor,protected=neighbor)
    clean=apply(rgb,result)
    assert not np.any(result['mask'][neighbor>0])
    assert np.array_equal(clean[neighbor>0],rgb[neighbor>0])


def test_textured_backdrop_does_not_claim_a_validated_smooth_fit():
    rng=np.random.default_rng(452)
    rgb=rng.integers(30,230,(130,250,3),dtype=np.uint8)
    seed=np.zeros((130,250),np.uint8)
    cv2.putText(seed,'TEXT',(45,75),cv2.FONT_HERSHEY_SIMPLEX,1.3,255,3)
    rgb[seed>0]=250
    result=reconstruct(rgb,[[39,42],[150,42],[150,81],[39,81]],seed)
    assert result['method']=='local_inpaint'
    clean=apply(rgb,result)
    assert np.array_equal(clean[result['mask']==0],rgb[result['mask']==0])


def test_wrong_luminance_seed_does_not_turn_smooth_glow_into_text():
    bg=blue_gradient(220,520).astype(float)
    yy,xx=np.mgrid[:220,:520]
    glow=65*np.exp(-((xx-80)**2+(yy-65)**2)/(2*90**2))
    bg=np.clip(bg+glow[...,None],0,255).round().astype(np.uint8)
    rgb=bg.copy()
    cv2.putText(rgb,'ABC',(220,132),cv2.FONT_HERSHEY_SIMPLEX,1.8,(15,40,90),10,cv2.LINE_AA)
    cv2.putText(rgb,'ABC',(220,132),cv2.FONT_HERSHEY_SIMPLEX,1.8,(250,250,250),6,cv2.LINE_AA)
    seed=np.zeros(rgb.shape[:2],np.uint8);seed[76:140,25:370]=255
    result=reconstruct(rgb,[[25,76],[370,76],[370,140],[25,140]],seed)
    assert not np.any(result['mask'][82:132,40:170]),'smooth background was classified as glyph ink'
    assert np.all(result['erase_mask'][82:132,40:170]>0), 'background erasure envelope must be separate from glyph alpha'
    assert not np.any(result['structural_mask'][82:132,40:170])


def test_sampling_exclusions_do_not_clip_letters_owned_by_overlapping_regions():
    bg=blue_gradient();rgb=bg.copy()
    cv2.putText(rgb,'ABC',(80,105),cv2.FONT_HERSHEY_SIMPLEX,2,(250,250,250),7,cv2.LINE_AA)
    seed=(np.min(rgb,axis=2)>230).astype(np.uint8)*255
    other_box=np.zeros(seed.shape,np.uint8);other_box[65:100,110:120]=255
    result=reconstruct(rgb,[[72,50],[218,50],[218,113],[72,113]],seed,excluded=other_box)
    assert np.any(result['mask'][other_box>0]), 'sampling exclusions must not override later pixel ownership'


def test_harmonization_removes_model_tone_bias_under_glyphs_only():
    from text_backdrop import harmonize_background
    bg=blue_gradient(150,340)
    mask=np.zeros(bg.shape[:2],np.uint8)
    cv2.putText(mask,'TEXT',(40,97),cv2.FONT_HERSHEY_SIMPLEX,1.8,255,7,cv2.LINE_AA)
    mask=cv2.dilate(mask,np.ones((3,3),np.uint8))
    rgb=bg.copy();rgb[mask>0]=[250,250,250]
    estimate=np.clip(bg.astype(int)+[12,-7,9],0,255).astype(np.uint8)
    corrected=harmonize_background(rgb,estimate,mask)
    error=np.max(np.abs(corrected.astype(int)-bg),axis=2)
    assert np.percentile(error[mask>0],95)<=2
    assert np.array_equal(corrected[mask==0],estimate[mask==0])
    assert np.array_equal(rgb[mask>0],np.tile([250,250,250],(np.count_nonzero(mask),1)))


def test_harmonization_preserves_red_object_edge_without_pulling_red_into_text():
    from text_backdrop import harmonize_background
    bg=blue_gradient(180,360);bg[105:165,150:340]=[195,25,30]
    mask=np.zeros(bg.shape[:2],np.uint8)
    cv2.putText(mask,'TEST',(45,92),cv2.FONT_HERSHEY_SIMPLEX,1.6,255,6,cv2.LINE_AA)
    mask=cv2.dilate(mask,np.ones((3,3),np.uint8))
    rgb=bg.copy();rgb[mask>0]=[250,250,250]
    estimate=np.clip(bg.astype(int)+[8,-5,6],0,255).astype(np.uint8)
    estimate[105:165,150:340]=bg[105:165,150:340]
    corrected=harmonize_background(rgb,estimate,mask)
    assert np.array_equal(corrected[105:165,150:340],bg[105:165,150:340])
    assert np.max(corrected[...,0][mask>0].astype(int)-bg[...,0][mask>0])<=5
    assert np.percentile(np.max(np.abs(corrected.astype(int)-bg),axis=2)[mask>0],95)<=3


def test_harmonization_without_visible_samples_keeps_model_estimate():
    from text_backdrop import harmonize_background
    rgb=np.full((20,30,3),250,np.uint8);estimate=np.full_like(rgb,120)
    assert np.array_equal(harmonize_background(rgb,estimate,np.full((20,30),255,np.uint8)),estimate)


def test_harmonization_does_not_spread_unmasked_ink_fragments_as_tone_corrections():
    from text_backdrop import harmonize_background
    bg=blue_gradient(120,180);rgb=bg.copy()
    mask=np.zeros(bg.shape[:2],np.uint8);mask[35:75,50:110]=255
    rgb[mask>0]=250
    # A missed dark stroke next to the support is a bad sample, not evidence
    # that the entire reconstructed background needs a large dark offset.
    rgb[35:75,47:50]=15
    estimate=np.clip(bg.astype(int)+[12,-7,9],0,255).astype(np.uint8)
    corrected=harmonize_background(rgb,estimate,mask)
    assert np.percentile(np.max(np.abs(corrected.astype(int)-bg),axis=2)[mask>0],95)<=5


def test_learned_patch_feathers_only_outside_full_strength_text_erasure():
    from text_backdrop import blend_reconstruction
    rgb=np.full((70,100,3),100,np.uint8)
    erase=np.zeros((70,100),np.uint8);erase[25:45,35:65]=255
    rgb[erase>0]=0
    seen=[]
    def model(source,mask):
        seen.append(mask.copy())
        result=source.copy();result[mask>0]=200
        return result
    result=blend_reconstruction(rgb,erase,model,feather_radius=8)
    assert len(seen)==1 and np.all(seen[0][erase>0]==255)
    assert np.count_nonzero(seen[0])>np.count_nonzero(erase)
    assert np.all(result['image'][erase>0]==200),'erased original letters must never fade back in'
    assert np.array_equal(result['image'][result['mask']==0],rgb[result['mask']==0])
    samples=result['image'][35,64:74,0].astype(int)
    assert samples[0]==200 and samples[-1]==100
    assert np.all(np.diff(samples)<=0) and np.max(np.abs(np.diff(samples)))<25
    assert np.array_equal(erase,result['core_mask'])


def test_reconstruction_feather_preserves_outside_objects_and_handles_image_edge():
    from text_backdrop import blend_reconstruction
    rgb=blue_gradient(90,150);rgb[55:80,110:145]=[195,25,30]
    erase=np.zeros(rgb.shape[:2],np.uint8);erase[0:15,0:30]=255
    rgb[erase>0]=250
    result=blend_reconstruction(rgb,erase,lambda source,mask:np.full_like(source,[85,150,225]),feather_radius=6)
    assert np.all(result['image'][erase>0]==[85,150,225])
    assert np.array_equal(result['image'][55:80,110:145],rgb[55:80,110:145])
    assert result['mask'].shape==erase.shape and result['mask'].dtype==np.uint8


def test_empty_reconstruction_mask_skips_model_and_returns_independent_image():
    from text_backdrop import blend_reconstruction
    rgb=blue_gradient(40,60)
    def forbidden(source,mask):
        raise AssertionError('nothing needs reconstruction')
    result=blend_reconstruction(rgb,np.zeros(rgb.shape[:2],np.uint8),forbidden)
    assert np.array_equal(result['image'],rgb)
    assert not np.shares_memory(result['image'],rgb)


def test_complex_support_does_not_classify_model_tone_error_as_glyph():
    from text_backdrop import refine_complex_support
    bg=blue_gradient(170,380);source=bg.copy()
    ink=np.zeros(bg.shape[:2],np.uint8)
    cv2.putText(ink,'ABC',(80,110),cv2.FONT_HERSHEY_SIMPLEX,2,255,7,cv2.LINE_AA)
    shadow=cv2.warpAffine(ink,np.float32([[1,0,4],[0,1,5]]),(380,170))
    source[shadow>0]=[15,45,95];source[ink>0]=[250,250,250]
    structural=np.zeros(bg.shape[:2],np.uint8);structural[50:120,70:225]=255
    erase=np.zeros_like(structural);erase[40:132,55:240]=255
    estimate=bg.copy();estimate[erase>0]=np.clip(bg[erase>0].astype(int)+[22,18,-16],0,255)
    refined=refine_complex_support(source,estimate,structural,erase)
    distant=np.zeros_like(erase);distant[53:61,72:216]=255
    assert np.mean(refined[distant>0]>0)<.05,'plain background within a convex hull became text'
    assert np.mean(refined[ink==255]>0)>.95
    assert np.mean(refined[(shadow==255)&(ink==0)]>0)>.90
