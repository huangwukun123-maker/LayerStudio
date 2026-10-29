from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
from engine import product_layer
from vision import split_highlights

def over(base,rgba):
    a=rgba[...,3:4]/255
    return np.round(base*(1-a)+rgba[...,:3]*a).astype(np.uint8)

def test_soft_product_selection_does_not_recolour_source_over_rebuilt_background():
    source=np.full((25,80,3),[220,40,35],np.uint8)
    source[:,40:]=[225,232,240]
    background=np.full_like(source,[85,95,120])
    alpha=np.tile(np.linspace(15,255,80,dtype=np.uint8),(25,1))
    rgba=product_layer(source,background,alpha)
    assert np.max(np.abs(over(background,rgba).astype(int)-source))<=1
    base,light=split_highlights(rgba)
    assert np.max(np.abs(over(over(background,base),light).astype(int)-source))<=2
