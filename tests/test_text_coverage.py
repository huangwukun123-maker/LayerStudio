"""Coverage tests exercise meaningful OCR overlap and crop failure cases."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from text_coverage import normalize_regions, recognize_all


def detection(text, box, score=.95, **metadata):
    x, y, w, h = box
    return {'text': text, 'quad': [[x,y],[x+w,y],[x+w,y+h],[x,y+h]],
            'score': score, **metadata}


def test_distinct_overlapping_lines_are_both_preserved():
    values = [detection('主标题', (10,10,120,35)),
              detection('第二行说明', (15,35,125,30), .61)]
    result = normalize_regions(values, 200, 100)
    assert [r['text'] for r in result] == ['主标题', '第二行说明']
    assert result[1]['low_confidence']


def test_repeated_text_at_different_positions_is_not_deduplicated():
    result = normalize_regions([detection('GNC', (10,10,40,20)),
                                detection('GNC', (80,10,40,20))], 200, 100)
    assert len(result) == 2


def test_same_region_from_multiple_scales_keeps_best_and_provenance():
    result = normalize_regions([detection('高钙蛋白粉', (10,10,130,30), .81, source='full'),
                                detection('高钙蛋白粉', (11,11,128,29), .96, source='tile0')], 200, 100)
    assert len(result) == 1
    assert result[0]['score'] == .96
    assert set(result[0]['merged_sources']) == {'full', 'tile0'}


def test_contained_partial_line_is_merged_but_unrelated_overlapping_text_is_kept():
    result = normalize_regions([detection('维A+维E+维C', (10,10,180,30), .92),
                                detection('维E', (78,11,43,28), .98),
                                detection('另一个字', (40,10,100,30), .95)], 220, 100)
    assert {r['text'] for r in result} == {'维A+维E+维C', '另一个字'}
    assert any(a['text'] == '维E' for r in result for a in r['alternatives'])


def test_low_confidence_and_rotated_text_are_retained_without_fabricating_content():
    value = {'text': '包装小字', 'quad': [[12,12],[112,27],[109,46],[9,31]], 'score': .52}
    result = normalize_regions([value, detection('', (1,1,20,20)),
                                detection('outside', (-100,-100,10,10)),
                                detection('noise', (1,1,20,20), .2)], 200, 100)
    assert len(result) == 1
    assert result[0]['text'] == '包装小字' and result[0]['low_confidence']
    assert np.allclose(result[0]['quad'], value['quad'])


def test_crossed_slanted_same_text_does_not_merge_on_axis_aligned_overlap():
    values = [{'text':'TEST','score':.95,'quad':[[10,10],[100,100],[95,105],[5,15]]},
              {'text':'TEST','score':.94,'quad':[[10,100],[100,10],[105,15],[15,105]]}]
    assert len(normalize_regions(values, 120, 120)) == 2


def test_multiscale_coverage_maps_coordinates_and_filters_internal_crop_edges():
    class Model:
        calls = 0
        def recognize(self, rgb):
            self.calls += 1
            if self.calls == 1:
                return [detection('标题', (5,5,30,15))]
            if self.calls == 2:
                # Top-left tile is 62 px at 2x scale. A right-edge fragment
                # is not evidence for a separate original word.
                return [detection('标题', (10,10,60,30), .98),
                        detection('包装', (50,70,40,20), .83),
                        detection('被切断', (110,70,14,20), .99)]
            if self.calls == 5:
                return [detection('下角', (20,20,30,20), .77)]
            return []
    model = Model()
    result = recognize_all(model, np.full((100,100,3), 160, np.uint8))
    assert model.calls == 5
    assert {r['text'] for r in result} == {'标题','包装','下角'}
    by_text = {r['text']:r for r in result}
    assert np.allclose(by_text['包装']['quad'], [[25,35],[45,35],[45,45],[25,45]])
    assert np.allclose(by_text['下角']['quad'], [[48,48],[63,48],[63,58],[48,58]])
    assert by_text['标题']['score'] == .98


def test_source_image_boundaries_are_not_treated_as_internal_tile_seams():
    class Model:
        calls = 0
        def recognize(self, rgb):
            self.calls += 1
            return [detection('边缘字', (0,0,40,16), .9)] if self.calls == 2 else []
    assert [r['text'] for r in recognize_all(Model(),np.zeros((100,100,3),np.uint8))] == ['边缘字']


def test_same_physical_label_with_conflicting_ocr_is_one_region_with_alternatives():
    result = normalize_regions([detection('GNC', (10,10,180,40), 1., source='tile0'),
                                detection('DND', (12,10,178,40), .57, source='tile1'),
                                detection('GND', (9,11,184,40), .53, source='full')], 250, 100)
    assert len(result) == 1
    assert result[0]['text'] == 'GNC'
    assert {a['text'] for a in result[0]['alternatives']} == {'DND','GND'}
    assert result[0]['recognition_conflict'] and result[0]['review_required']


def test_normalizing_does_not_mutate_input_numpy_quad():
    quad = np.array([[-5.,2.],[40.,2.],[40.,20.],[-5.,20.]])
    original = quad.copy()
    normalize_regions([{'text':'TEST','score':.9,'quad':quad}],100,100)
    assert np.array_equal(quad, original)


def test_same_small_line_with_different_height_and_one_ocr_error_is_one_region():
    result = normalize_regions([detection('MPORTED WHEY PROTEN',(10,10,75,15),.94,source='full'),
                                detection('MPORTED WHEY PROTEIN',(11,12,73,11),.96,source='tile3')],150,60)
    assert len(result) == 1
    assert result[0]['text'] == 'MPORTED WHEY PROTEIN'
    assert result[0]['recognition_conflict']


def test_contained_tilted_wordmark_with_two_leading_ocr_errors_is_one_region():
    values = [
        {'text':'Abcde品牌','score':.962,'source':'full',
         'quad':[[10,31],[110,20],[113,45],[13,56]]},
        {'text':'xcde品牌','score':.945,'source':'tile0',
         'quad':[[12,31.5],[100.5,25],[102,47],[13.5,53.5]]},
        detection('R',(99,26.5,6.5,6.5),.984,source='tile1'),
    ]
    result = normalize_regions(values, 160, 90)
    assert {item['text'] for item in result} == {'Abcde品牌','R'}
    wordmark = next(item for item in result if item['text']=='Abcde品牌')
    assert set(wordmark['merged_sources']) == {'full','tile0'}
    assert wordmark['recognition_conflict'] and wordmark['review_required']
    assert wordmark['alternatives'][0]['text'] == 'xcde品牌'


def test_similar_wordmarks_on_adjacent_tilted_baselines_remain_separate():
    values = [
        {'text':'Abcde品牌','score':.962,
         'quad':[[10,31],[110,20],[113,45],[13,56]]},
        {'text':'xcde品牌','score':.945,
         'quad':[[12,53.5],[100.5,47],[102,69],[13.5,75.5]]},
    ]
    result = normalize_regions(values, 160, 100)
    assert len(result) == 2


def test_similar_contained_words_on_distinct_baselines_are_not_weakly_merged():
    # Matching most letters is insufficient when the boxes have different
    # perpendicular centers, even if the shorter box is fully contained.
    values = [detection('Abcde品牌',(10,10,120,40),.96),
              detection('xcde品牌',(10,10,120,28),.94)]
    assert len(normalize_regions(values,160,90)) == 2
