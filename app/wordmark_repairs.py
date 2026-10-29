"""Narrow geometric repairs for a known wordmark; never rewrite arbitrary logos."""
import math
import re
import xml.etree.ElementTree as ET

NUMBER=r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?'


def _commands(d):
    tokens=re.findall(r'[A-Za-z]|'+NUMBER,d)
    commands=[]; i=0; point=(0.,0.); origin=point
    while i<len(tokens):
        op=tokens[i];i+=1
        if op not in {'M','C','L','Z'}: return None
        count={'M':2,'C':6,'L':2,'Z':0}[op]
        try: values=[float(x) for x in tokens[i:i+count]]
        except ValueError: return None
        if len(values)!=count or not all(math.isfinite(x) for x in values): return None
        i+=count
        end=tuple(values[-2:]) if values else origin
        commands.append({'op':op,'v':values,'start':point,'end':end})
        point=end
        if op=='M': origin=end
    return commands


def _repaired_commands(commands):
    points=[tuple(c['v'][i:i+2]) for c in commands for i in range(0,len(c['v']),2)]
    if not points or sum(c['op']=='M' for c in commands)!=1: return None
    xmin=min(p[0] for p in points);ymin=min(p[1] for p in points)
    w=max(p[0] for p in points)-xmin;h=max(p[1] for p in points)-ymin
    if w<=0 or h<=0: return None
    for i,c in enumerate(commands[:-1]):
        if c['op']!='C' or commands[i+1]['op']!='C': continue
        a,b,n=c['start'],c['end'],commands[i+1]['end']
        # The reported defect reverses the outer U bowl sharply upward for a
        # very short horizontal distance, then immediately bends down again.
        if not (0<a[0]-b[0]<w*.05 and b[1]-a[1]<-h*.04
                and xmin+w*.5<b[0]<xmin+w*.8
                and a[1]>ymin+h*.75 and b[1]>ymin+h*.7
                and n[0]<b[0] and n[1]>b[1]): continue
        right=next((j for j in range(i-1,max(-1,i-13),-1)
            if commands[j]['op']=='C' and commands[j]['end'][0]>=xmin+w*.75
            and commands[j]['end'][1]>=ymin+h*.55),None)
        left=next((j for j in range(i+1,min(len(commands),i+11))
            if commands[j]['op']=='C' and commands[j]['end'][0]<=xmin+w*.35
            and commands[j]['end'][1]>=ymin+h*.75),None)
        if right is None or left is None: continue
        section=commands[right:left+1]
        if any(c['op']!='C' for c in section): continue
        if any(b['end'][0]>a['end'][0]+.1 for a,b in zip(section,section[1:])): continue
        start=commands[right]['end'];end=commands[left]['end']
        previous=commands[right]['v'][2:4]
        tx,ty=start[0]-previous[0],start[1]-previous[1]
        if tx>=0 or ty<=0: continue
        length=math.hypot(tx,ty)
        handle=min((start[0]-end[0])*.38,h*.255)
        first=(start[0]+tx/length*handle,start[1]+ty/length*handle)
        second=commands[left]['v'][2:4]
        if not (end[0]<second[0]<first[0]<start[0] and second[1]>=end[1]): continue
        patch={'op':'C','v':[*first,*second,*end],'start':start,'end':end}
        return commands[:right+1]+[patch]+commands[left+1:]
    return None


def repair_gujia_u_svg(svg,text):
    """Repair only the recognized Gujia U and only when its sharp notch exists."""
    if re.sub(r'\s+','',text).casefold()!='gujia顾家': return svg,False
    try:
        root=ET.fromstring(svg)
        view=[float(x) for x in root.get('viewBox','').split()]
    except (ET.ParseError,ValueError): return svg,False
    if root.get('data-u-contour-repaired')=='true': return svg,False
    if len(view)!=4 or min(view[2:])<=0 or not 3.3<view[2]/view[3]<5: return svg,False
    for node in root:
        if not node.tag.endswith('path'): continue
        match=re.fullmatch(r'translate\(\s*('+NUMBER+r')\s*[, ]\s*('+NUMBER+r')\s*\)',node.get('transform',''))
        if match is None: continue
        tx,ty=map(float,match.groups())
        commands=_commands(node.get('d',''))
        if not commands: continue
        points=[tuple(c['v'][i:i+2]) for c in commands for i in range(0,len(c['v']),2)]
        if not points: continue
        x0=(min(p[0] for p in points)+tx-view[0])/view[2]
        x1=(max(p[0] for p in points)+tx-view[0])/view[2]
        y0=(min(p[1] for p in points)+ty-view[1])/view[3]
        y1=(max(p[1] for p in points)+ty-view[1])/view[3]
        # Layout and shape checks prevent applying a U rule to G, j or Chinese.
        if not (.14<x0<.23 and .28<x1<.37 and .23<y0<.4 and .68<y1<.92): continue
        repaired=_repaired_commands(commands)
        if repaired is None: continue
        node.set('d',' '.join(c['op']+' '.join(f'{v:.3f}'.rstrip('0').rstrip('.') or '0' for v in c['v']) for c in repaired))
        root.set('data-u-contour-repaired','true')
        return ET.tostring(root,encoding='unicode'),True
    return svg,False
