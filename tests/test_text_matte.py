"""Edges retain local ink colour instead of minimum-alpha neon fringes."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
import vision
from text_matte import decontaminate_edges


def composite(rgba, backdrop):
    alpha=rgba[...,3:4].astype(float)/255
    return np.rint(rgba[...,:3]*alpha+backdrop*(1-alpha)).astype(np.uint8)


def test_white_antialias_edges_lose_neon_colour_with_inexact_blue_backdrop():
    base=np.full((21,35,3),[42,119,199],np.uint8)
    estimated=np.full_like(base,[46,114,204])
    alpha=np.zeros((21,35),float)
    alpha[6:15,7:28]=1
    alpha[5,7:28]=.25;alpha[15,7:28]=.7
    alpha[6:15,6]=.45;alpha[6:15,28]=.85
    source=np.rint(base*(1-alpha[...,None])+np.array([250,251,253])*alpha[...,None]).astype(np.uint8)
    support=(alpha>0).astype(np.uint8)*255
    core=alpha==1
    original=vision.foreground_from_background(source,estimated,support)
    original[core,:3]=source[core];original[core,3]=255
    edges=(alpha>0)&~core
    assert np.max(np.ptp(original[edges,:3].astype(int),axis=1))>15
    result=decontaminate_edges(original,source,estimated,core)
    assert np.max(np.ptp(result[edges,:3].astype(int),axis=1))<=3
    assert np.max(np.abs(composite(result,estimated)[support>0].astype(int)-source[support>0]))<=8
    assert np.array_equal(result[core],original[core])
    assert np.array_equal(result[support==0],original[support==0])


def test_coloured_shadow_which_white_ink_cannot_explain_is_unchanged():
    backdrop=np.full((19,30,3),[90,140,190],np.uint8)
    source=backdrop.copy();source[4:10,5:23]=[250,250,250]
    source[10:13,5:23]=[28,35,100]
    support=np.zeros((19,30),np.uint8);support[4:13,5:23]=255
    core=np.zeros_like(support,dtype=bool);core[4:10,5:23]=True
    rgba=vision.foreground_from_background(source,backdrop,support)
    rgba[core,:3]=source[core];rgba[core,3]=255
    result=decontaminate_edges(rgba,source,backdrop,core)
    assert np.array_equal(result[10:13],rgba[10:13])


def test_no_opaque_core_returns_a_copy_without_inventing_a_foreground():
    source=np.full((8,9,3),[75,90,150],np.uint8)
    base=np.full_like(source,[60,80,110])
    rgba=vision.foreground_from_background(source,base,np.ones((8,9),np.uint8))
    before=rgba.copy()
    result=decontaminate_edges(rgba,source,base,np.zeros((8,9),bool))
    assert np.array_equal(result,before) and np.array_equal(rgba,before)


def test_oversized_mask_does_not_make_shifted_yellow_backing_into_black_ink():
    from text_matte import refine_ink_support
    source=np.full((65,125,3),[241,218,155],np.uint8)
    ink=np.zeros((65,125),bool);ink[17:49,25:32]=True;ink[17:24,25:92]=True;ink[42:49,25:92]=True
    source[ink]=[25,26,24]
    backdrop=np.full_like(source,[204,195,141])
    backdrop[::8]=[224,190,139]
    candidate=np.ones((65,125),np.uint8)*255
    result=refine_ink_support(source,backdrop,candidate,[25,26,24])
    assert result['usable'] and result.get('is_monochrome',False)
    assert not np.any(result['core']&~ink)
    assert np.all(result['support'][ink]>0)
    assert not np.any(result['support'][0:10])
    assert not np.any(result['support'][:,100:])


def test_oversized_mask_does_not_make_brown_backing_into_white_glyph_core():
    from text_matte import refine_ink_support
    source=np.full((50,75,3),[164,110,47],np.uint8)
    ink=np.zeros((50,75),bool);ink[12:40,20:28]=True;ink[12:20,20:57]=True
    source[ink]=[251,242,218]
    backdrop=np.full_like(source,[192,137,81])
    result=refine_ink_support(source,backdrop,np.ones((50,75),np.uint8)*255,[251,242,218])
    assert result['usable'] and result.get('is_monochrome',False) and not np.any(result['core']&~ink)
    assert not np.any(result['support'][0:7])
    assert np.all(result['support'][ink]>0)


def test_missing_ink_evidence_preserves_candidate_but_never_promotes_backdrop():
    from text_matte import refine_ink_support
    source=np.full((15,25,3),[200,210,220],np.uint8)
    backdrop=np.full_like(source,[140,150,170])
    candidate=np.ones((15,25),np.uint8)*255
    result=refine_ink_support(source,backdrop,candidate,[20,20,20])
    assert not result['usable'] and not np.any(result['core'])
    assert np.array_equal(result['support'],candidate)


def test_glyph_seed_survives_a_failed_backdrop_that_still_contains_the_letters():
    from text_matte import refine_ink_support
    source=np.full((45,95,3),[235,219,164],np.uint8)
    ink=np.zeros((45,95),bool);ink[12:35,15:22]=True;ink[12:19,15:80]=True
    source[ink]=[25,26,24]
    result=refine_ink_support(source,source.copy(),np.ones((45,95),np.uint8)*255,[25,26,24])
    assert result['usable'] and np.all(result['support'][ink]>0)
    assert np.any(result['core'])


def test_gray_ocr_median_does_not_drop_darker_real_letter_centres():
    from text_matte import refine_ink_support
    source=np.full((45,95,3),[235,219,164],np.uint8)
    ink=np.zeros((45,95),bool);ink[12:35,15:22]=True;ink[12:19,15:80]=True
    source[ink]=[12,14,10]
    result=refine_ink_support(source,source.copy(),np.ones((45,95),np.uint8)*255,[51,51,42])
    assert result['usable'] and np.all(result['support'][ink]>0)
    assert np.linalg.norm(np.array(result['ink_color'])-[12,14,10])<3


def test_white_letters_with_substantial_blue_extrusion_are_not_monochrome():
    from text_matte import refine_ink_support
    source=np.full((65,125,3),[110,170,225],np.uint8)
    source[20:52,29:39]=[10,35,85];source[20:30,29:96]=[10,35,85]
    source[16:48,25:35]=[250,250,250];source[16:26,25:92]=[250,250,250]
    backdrop=np.full_like(source,[110,170,225])
    result=refine_ink_support(source,backdrop,np.ones((65,125),np.uint8)*255,[248,249,250])
    assert result['usable'] and not result.get('is_monochrome',True)


def test_precise_glyph_candidate_uses_visible_outer_ring_as_background_reference():
    from text_matte import refine_ink_support
    source=np.full((45,95,3),[235,219,164],np.uint8)
    ink=np.zeros((45,95),bool);ink[12:35,15:22]=True;ink[12:19,15:80]=True
    source[ink]=[12,14,10]
    backdrop=np.full_like(source,[219,202,166])
    result=refine_ink_support(source,backdrop,ink.astype(np.uint8)*255,[51,51,42])
    assert result['usable'] and result['is_monochrome']
    assert np.all(result['support'][ink]>0)
    assert not np.any(result['support'][~ink])


def test_single_black_ink_over_changing_panel_colours_is_still_monochrome():
    from text_matte import refine_ink_support
    source=np.full((50,160,3),[245,210,130],np.uint8)
    source[:,85:]=[95,215,235]
    backdrop=source.copy()
    for x in range(20,145,18):source[16:36,x:x+5]=[18,19,15]
    candidate=np.zeros((50,160),np.uint8);candidate[13:39,17:150]=255
    result=refine_ink_support(source,backdrop,candidate,[45,44,32])
    assert result['usable'] and result['is_monochrome']


def test_fidelity_gate_detects_a_few_low_contrast_missing_strokes_on_large_backing():
    from text_matte import source_fidelity_ok
    source=np.full((180,600,3),[219,207,162],np.uint8)
    source[71:77,183:185]=[232,220,175]
    composite=source.copy();composite[71:77,183:185]=[219,207,162]
    assert not source_fidelity_ok(source,composite,np.ones((180,600),np.uint8))


def test_fidelity_gate_allows_antialias_error_up_to_eight_channel_levels():
    from text_matte import source_fidelity_ok
    rng=np.random.default_rng(830)
    source=rng.integers(20,235,(70,240,3),dtype=np.uint8)
    composite=(source.astype(np.int16)+rng.integers(-8,9,source.shape)).astype(np.uint8)
    assert source_fidelity_ok(source,composite,np.ones(source.shape[:2],np.uint8))


def test_fidelity_gate_ignores_differences_outside_the_expected_region():
    from text_matte import source_fidelity_ok
    source=np.full((30,90,3),100,np.uint8);composite=source.copy();composite[2:8,2:8]=220
    region=np.zeros((30,90),np.uint8);region[12:22,25:75]=255
    assert source_fidelity_ok(source,composite,region)


def test_fidelity_gate_detects_a_connected_three_pixel_stroke_fragment():
    from text_matte import source_fidelity_ok
    source=np.full((70,240,3),200,np.uint8);composite=source.copy()
    composite[33,113:116]=189
    assert not source_fidelity_ok(source,composite,np.ones(source.shape[:2],np.uint8))
