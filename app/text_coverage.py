"""Multi-scale OCR coverage without dropping neighboring overlapping text.

The output remains compatible with LocalModels.recognize. Metadata distinguishes
uncertain recognition and duplicate observations from independently editable
regions; confidence never implies that an inferred string is the original text.
"""
import json
import math
from difflib import SequenceMatcher
from pathlib import Path
import unicodedata

import cv2
import numpy as np


def _key(text):
    return ''.join(unicodedata.normalize('NFKC', text).casefold().split())


def _polygon(item):
    return cv2.convexHull(np.asarray(item['quad'], np.float32)).reshape(-1, 2)


def _direction(item):
    points = np.asarray(item['quad'], np.float64)
    vector = points[1] - points[0]
    return vector / max(float(np.linalg.norm(vector)), 1e-8)


def _contains_substring(outer, inner, containment):
    """Only merge aligned pieces at the position described by their text.

    A generic geometric containment test alone merges distinct product labels
    into headlines. Requiring matching content, baseline and proportional
    location avoids that failure and catches cropped duplicate line fragments.
    """
    big, small = _key(outer['text']), _key(inner['text'])
    if len(small) < 2 or len(big) <= len(small) or small not in big or containment < .94:
        return False
    if outer['score'] < .70 and outer['score'] < inner['score'] - .10:
        return False
    direction = _direction(outer)
    a = np.asarray(outer['quad']) @ direction
    b = np.asarray(inner['quad']) @ direction
    aw, bw = float(np.ptp(a)), float(np.ptp(b))
    if aw < 1 or bw < 1:
        return False
    ah = cv2.contourArea(_polygon(outer)) / aw
    bh = cv2.contourArea(_polygon(inner)) / bw
    if not .65 <= bh / max(ah, 1e-6) <= 1.4:
        return False
    if abs(bw / aw - len(small) / len(big)) > .20:
        return False
    measured_center = ((b.min() + b.max()) * .5 - a.min()) / aw
    start = big.find(small)
    while start >= 0:
        if abs(measured_center - (start + len(small) * .5) / len(big)) <= .17:
            return True
        start = big.find(small, start + 1)
    return False


def _duplicate_relation(left, right):
    a, b = _polygon(left), _polygon(right)
    aa, ba = cv2.contourArea(a), cv2.contourArea(b)
    intersection, _ = cv2.intersectConvexConvex(a, b)
    if intersection <= 0:
        return None
    alignment = abs(float(np.dot(_direction(left), _direction(right))))
    if alignment < math.cos(math.radians(12)):
        return None
    containment = intersection / min(aa, ba)
    iou = intersection / max(aa + ba - intersection, 1e-6)
    # Different passes can read the same physical word differently (e.g. GNC /
    # GND). Near-identical polygons describe one region, not two editable words.
    # Preserve competing strings as alternatives for the delivery audit.
    if iou >= .82:
        return 'same'
    # Small-label detectors disagree by a few pixels in height, making IoU
    # misleading even when both describe precisely the same baseline/string.
    if containment >= .90 and min(aa, ba) / max(aa, ba) >= .55:
        left_key, right_key = _key(left['text']), _key(right['text'])
        if min(len(left_key),len(right_key)) >= 4 and SequenceMatcher(None,left_key,right_key).ratio() >= .82:
            return 'same'
    # A stylized prefix can be misread as two characters, defeating the stricter
    # text-similarity rule above. Relax content matching only when essentially
    # the whole smaller polygon describes the same-sized, aligned text band.
    # Keep registered marks and distinct neighboring baselines independent.
    if (containment >= .97 and min(aa,ba)/max(aa,ba) >= .70
            and alignment >= math.cos(math.radians(5))):
        left_key,right_key = _key(left['text']),_key(right['text'])
        direction = _direction(left)
        normal = np.array([-direction[1],direction[0]])
        left_height = aa / max(float(np.ptp(a @ direction)),1e-6)
        right_height = ba / max(float(np.ptp(b @ direction)),1e-6)
        center_offset = abs(float((a.mean(axis=0)-b.mean(axis=0)) @ normal))
        if (center_offset <= .20*min(left_height,right_height)
                and min(len(left_key),len(right_key)) >= 4
                and SequenceMatcher(None,left_key,right_key).ratio() >= .70):
            return 'same'
    if _key(left['text']) == _key(right['text']):
        if iou >= .45 or (containment >= .86 and min(aa, ba) / max(aa, ba) >= .30):
            return 'same'
    if alignment >= math.cos(math.radians(8)):
        if _contains_substring(left, right, intersection / ba):
            return 'left_contains'
        if _contains_substring(right, left, intersection / aa):
            return 'right_contains'
    return None


def _merge_observation(winner, other):
    winner['merged_sources'] = list(dict.fromkeys(winner['merged_sources'] + other['merged_sources']))
    alternatives = winner['alternatives'] + other['alternatives']
    alternatives.append({'text': other['text'], 'score': other['score'], 'source': other['source']})
    seen = set()
    winner['alternatives'] = []
    for alternative in alternatives:
        key = (alternative['text'], alternative['score'], alternative['source'])
        if key not in seen:
            seen.add(key)
            winner['alternatives'].append(alternative)
    return winner


def normalize_regions(detected, width, height, *, min_score=.50):
    """Validate, de-duplicate and order observations in image coordinates.

    Keeps low-confidence (>= min_score), rotated, vertical and distinct
    overlapping lines. No arbitrary 120-region truncation is performed. Invalid
    or empty observations are discarded, without inventing text for them.
    """
    if width <= 0 or height <= 0:
        raise ValueError('OCR image dimensions must be positive')
    candidates = []
    for index, raw in enumerate(detected):
        try:
            text = str(raw.get('text') or '').strip()
            quad = np.array(raw['quad'], dtype=np.float64, copy=True)
            score = float(raw['score'])
        except (ValueError, TypeError, KeyError):
            continue
        if not text or quad.shape != (4,2) or not np.isfinite(quad).all() or not math.isfinite(score):
            continue
        if score < min_score:
            continue
        quad[:,0] = np.clip(quad[:,0], 0, width)
        quad[:,1] = np.clip(quad[:,1], 0, height)
        if cv2.contourArea(cv2.convexHull(quad.astype(np.float32))) < 1:
            continue
        source = str(raw.get('source', 'full'))
        candidate = {**raw, 'text':text, 'quad':quad.tolist(), 'score':min(1.,max(0.,score)),
                     'source':source, 'low_confidence':score < .70,
                     'merged_sources':list(raw.get('merged_sources', [source])),
                     'alternatives':list(raw.get('alternatives', [])), '_order':index}
        candidates.append(candidate)
    # Prefer full lines over partial observations. Exact duplicates still choose
    # the highest confidence observation below, irrespective of source scale.
    candidates.sort(key=lambda item: (-len(_key(item['text'])), -item['score'], item['_order']))
    kept = []
    for candidate in candidates:
        for index, previous in enumerate(kept):
            relation = _duplicate_relation(previous, candidate)
            if not relation:
                continue
            if relation == 'right_contains' or (relation == 'same' and candidate['score'] > previous['score']):
                kept[index] = _merge_observation(candidate, previous)
            else:
                kept[index] = _merge_observation(previous, candidate)
            break
        else:
            kept.append(candidate)
    kept.sort(key=lambda item: (min(p[1] for p in item['quad']), min(p[0] for p in item['quad']), item['_order']))
    for index, item in enumerate(kept):
        item.pop('_order', None)
        item['region_id'] = f'text_{index + 1:03d}'
        item['recognition_conflict'] = any(_key(a['text']) != _key(item['text']) for a in item['alternatives'])
        item['multi_pass_confirmed'] = len(item['merged_sources']) > 1
        item['review_required'] = (item['low_confidence'] or item['recognition_conflict']
                                   or (len(_key(item['text'])) == 1 and not item['multi_pass_confirmed']))
    return kept


def recognize_all(model, rgb, *, min_score=.50, progress=None, debug_path=None):
    """Recognize full image plus four overlapping, enlarged quadrants.

    Each quadrant covers 62% of each image dimension and overlaps its neighbor
    by 24%; enlarging 2x exposes small packaging text to the OCR detector.
    Observations touching an internal crop seam are not added because their
    recognized string may be a truncated fragment. Original image edges remain
    valid. Model objects need only provide recognize(rgb).
    """
    height, width = rgb.shape[:2]
    observations = [{**item, 'source':'full'} for item in model.recognize(rgb)]
    if progress:
        progress(1, 5)
    tile_w, tile_h = min(width, math.ceil(width * .62)), min(height, math.ceil(height * .62))
    rejected_edges = 0
    for index, (left, top) in enumerate(((0,0), (width-tile_w,0), (0,height-tile_h), (width-tile_w,height-tile_h))):
        tile = rgb[top:top+tile_h, left:left+tile_w]
        scaled = cv2.resize(tile, (tile_w*2,tile_h*2), interpolation=cv2.INTER_CUBIC)
        for item in model.recognize(scaled):
            try:
                quad = np.asarray(item['quad'], np.float64)
            except (KeyError, ValueError, TypeError):
                continue
            if quad.shape != (4,2) or not np.isfinite(quad).all():
                continue
            x0,y0 = quad.min(axis=0)
            x1,y1 = quad.max(axis=0)
            if ((left > 0 and x0 <= 2) or (top > 0 and y0 <= 2)
                    or (left+tile_w < width and x1 >= tile_w*2-2)
                    or (top+tile_h < height and y1 >= tile_h*2-2)):
                rejected_edges += 1
                continue
            quad = quad / 2 + np.array([left,top])
            observations.append({**item, 'quad':quad.tolist(), 'source':f'tile{index}'})
        if progress:
            progress(index+2, 5)
    regions = normalize_regions(observations, width, height, min_score=min_score)
    if debug_path is not None:
        Path(debug_path).write_text(json.dumps({'observations':observations, 'regions':regions,
            'low_confidence_count':sum(item['low_confidence'] for item in regions),
            'rejected_tile_edge_count':rejected_edges}, ensure_ascii=False, indent=2), encoding='utf-8')
    return regions
