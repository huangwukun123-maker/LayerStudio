"""Full automatic input-to-PSD pipeline, independent of any chat assistant."""
from pathlib import Path
import json
import shutil
import subprocess
import time
import zipfile
import cv2
import numpy as np
from PIL import Image
from models import APP_DIR, LocalModels, NO_WINDOW, model_status
import vision
import text_fidelity
import text_coverage
import text_layers


def save_image(path, array):
    Image.fromarray(array).save(path)


def product_layer(source,background,alpha):
    """Keep soft segmentation from washing out the recomposed photograph."""
    foreground=vision.foreground_from_background(source,background,alpha)
    opacity=np.maximum(foreground[...,3],alpha).astype(float)/255
    color=np.divide(source.astype(float)-background*(1-opacity[...,None]),opacity[...,None],
                    out=np.zeros_like(source,dtype=float),where=opacity[...,None]>0)
    return np.dstack((np.clip(color,0,255).round().astype(np.uint8),np.round(opacity*255).astype(np.uint8)))


def repair_hidden(rgb, mask):
    if not np.any(mask):
        return rgb.copy()
    h, w = rgb.shape[:2]
    factor = min(1, 600/max(h, w))
    size = (max(1, round(w*factor)), max(1, round(h*factor)))
    small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    repaired = cv2.inpaint(small, small_mask, 5, cv2.INPAINT_TELEA)
    full = cv2.resize(repaired, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.where(mask[..., None] > 0, full, rgb).astype(np.uint8)


def verify_text_composite(directory,manifest,source):
    """Check the final stack, including interference from neighboring layers.

    An extraction can pass alone and still be covered by another region.
    Source-region fallbacks are explicit, drawn last, and never counted as
    transparent glyphs or given glyph-only effects.
    """
    preview=np.array(Image.open(directory/'preview.png').convert('RGB'))
    changed=0
    for index,region in enumerate(manifest['text_regions']):
        if not region['rasters']:continue
        # An explicitly disclosed contour reconstruction intentionally differs
        # from noisy source edges; do not undo approved curve/noise repairs.
        if region.get('contour_reconstructed'):continue
        x,y,w,h=region['alternative']['box']
        x=max(0,x);y=max(0,y);right=min(source.shape[1],x+w);bottom=min(source.shape[0],y+h)
        if right<=x or bottom<=y:continue
        error=np.max(np.abs(source[y:bottom,x:right].astype(int)-preview[y:bottom,x:right]),axis=2)
        limit=3 if region.get('contains_background') else 20
        if np.percentile(error,98)<=limit:continue
        changed+=1
        pad=max(6,round(h*.12));left=max(0,x-pad);top=max(0,y-pad)
        right=min(source.shape[1],right+pad);bottom=min(source.shape[0],bottom+pad)
        rgba=np.dstack((source[top:bottom,left:right],np.full((bottom-top,right-left),255,np.uint8)))
        # Pad outside the preserved source area; no pretend transparent holes.
        lpad=min(2,left);tpad=min(2,top);rpad=min(2,source.shape[1]-right);bpad=min(2,source.shape[0]-bottom)
        rgba=cv2.copyMakeBorder(rgba,tpad,bpad,lpad,rpad,cv2.BORDER_CONSTANT,value=[0,0,0,0])
        filename=f'text_verified_{index+1}.png';save_image(directory/filename,rgba)
        region.update(contains_background=True,left=left-lpad,top=top-tpad,
            repair_method='最终合成检查发现笔画或颜色损失，保留含背景原区域，需精修',
            rasters=[{'name':'保真原区域｜合成检查保护·含背景·需精修','file':filename,
                      'left':left-lpad,'top':top-tpad,'editableStroke':False}])
        for audit in manifest['text_audit']:
            if audit['region_id']==region['alternative'].get('region_id'):
                audit.update(mode='preserved',reason=region['repair_method'])
    if changed:
        manifest['text_regions'].sort(key=lambda r:bool(r.get('contains_background')))
        manifest['warnings'].append(f'最终合成检查发现{changed}处文字存在损失，已自动保留含背景原区域并标为需精修；未冒充透明拆字成功。')
    return changed


class Processor:
    def __init__(self):
        self.models = LocalModels()

    def process(self, source, directory, options, update):
        start = time.monotonic()
        directory = Path(directory).resolve()
        if not all(m['ready'] for m in model_status()):
            raise RuntimeError('本机模型不完整，请重新解压完整软件包后运行。')
        update(3, '读取图片与检查尺寸')
        original = vision.load_image(source)
        original_size = original.size
        warnings = []
        if options.get('crop_ui', True):
            original, crop_box = vision.crop_black_bars(original)
            if original.size != original_size:
                warnings.append(f'已裁除上下黑边：保留原图区域 {crop_box}。')
        else:
            crop_box = (0, 0, *original.size)
        original.save(directory/'original.png')
        scale = int(options.get('scale', 3))
        if scale not in (2, 3, 4):
            raise ValueError('清晰化倍率只支持2、3、4倍。')
        final_factor = min(scale, 4096/max(original.size))
        out_w, out_h = [round(n*final_factor) for n in original.size]
        if final_factor < scale:
            warnings.append('为控制内存，输出长边限制为4096像素，已自动调整倍率。')
        work_image = original.copy()
        work_image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        rgb = np.array(work_image)
        h, w = rgb.shape[:2]
        sx, sy = out_w/w, out_h/h

        update(8, '自动识别文字和原位置')
        detected = text_coverage.recognize_all(self.models,rgb,
            progress=lambda n,total:update(8+3*n/total,f'全图与小字复查 {n}/{total}'))
        (directory/'ocr_regions.json').write_text(json.dumps(detected,ensure_ascii=False,indent=2),encoding='utf-8')
        update(11,'检查主体上的包装文字')
        initial_foreground=self.models.foreground(rgb)
        # A foreground detector also selects flat label panels. Excluding the
        # whole panel would hide its valid background samples. Local fit/error
        # checks reject actual product edges; neighboring OCR is always blocked.
        text_excluded=np.zeros((h,w),np.uint8)
        for detected_item in detected:
            cv2.fillConvexPoly(text_excluded,np.round(detected_item['quad']).astype(np.int32),255)
        text_repairs=[];review_required=[];fallback=[];text_audit=[]
        protected_textmask=np.zeros((h,w),np.uint8)
        texts, preserved, textmask = [], [], np.zeros((h, w), np.uint8)
        def scaled_spec(spec,match=None):
            result={k:v for k,v in spec.items() if k!='color_rgb'}
            result['box']=[round(spec['box'][0]*sx),round(spec['box'][1]*sy),
                           max(1,round(spec['box'][2]*sx)),max(1,round(spec['box'][3]*sy))]
            if match:
                result.update(text=match['text'],font_key=match['key'],font_match={**match,'score':round(match['score'],4)},
                              shear=match.get('shear',0),tracking=match.get('tracking',0))
            return result
        for item_index,item in enumerate(detected):
            update(12+5*item_index/max(1,len(detected)),f'检查文字 {item_index+1}/{len(detected)}')
            try:
                localmask, spec = vision.text_region(rgb, item['quad'], item['text'], item['score'])
            except ValueError:
                # Even ambiguous/low-contrast regions get an explicit layer and
                # audit entry, never silently disappear from processing.
                q=np.asarray(item['quad']);lo=np.maximum(0,np.floor(q.min(axis=0)).astype(int))
                hi=np.minimum([w,h],np.ceil(q.max(axis=0)).astype(int))
                bx,by=lo;bw,bh=np.maximum(1,hi-lo)
                localmask=np.zeros((h,w),np.uint8)
                cv2.fillConvexPoly(localmask,np.round(q).astype(np.int32),255)
                color=np.median(rgb[localmask>0],axis=0).astype(int)
                spec={'text':item['text'],'name':item['text'][:28],'box':[int(bx),int(by),int(bw),int(bh)],
                      'color':'#%02x%02x%02x'%tuple(color),'rotation':0,'confidence':item['score'],'bold':False}
            spec['region_id']=item.get('region_id',str(item_index))
            reason=vision.text_preservation_reason(spec,initial_foreground)
            color=np.array([int(spec['color'][i:i+2],16) for i in (1,3,5)])
            if reason and reason.startswith('品牌标志') and np.min(color)>210:
                excluded=np.zeros((h,w),np.uint8)
                for neighbor_index,neighbor in enumerate(detected):
                    if neighbor_index!=item_index:
                        cv2.fillConvexPoly(excluded,np.round(neighbor['quad']).astype(np.int32),255)
                excluded=cv2.dilate(excluded,np.ones((5,5),np.uint8))
                logo_mask=vision.preserved_text_mask(rgb,item['quad'],spec,excluded)
                if np.any(logo_mask):
                    match=text_fidelity.match_font(rgb,spec)
                    preserved.append({'text':item['text'],'mask':logo_mask,'quad':item['quad'],
                                      'color':spec['color'],'glyph_height':spec['box'][3],
                                      'alternative':scaled_spec(spec,match)})
                    textmask=cv2.bitwise_or(textmask,logo_mask)
                    text_audit.append({'region_id':spec['region_id'],'text':spec['text'],'box':spec['box'],'mode':'appearance','reason':'标志原字形轮廓，另附近似可改字替代层'})
                    review_required.append({'text':spec['text'],'box':spec['box'],'mode':'appearance','reason':'标志字形不可由普通字体完全还原；可改字替代层需确认'})
                    continue
            # Tiny detected skew is normalized to horizontal; the fitted box
            # is already in document axes, so applying the angle again is wrong.
            if abs(spec['rotation'])<2:spec['rotation']=0
            # Replace the full fitted text footprint, including anti-aliased
            # outlines/JPEG ringing, so a moved TypeLayer leaves no ghost text.
            bx,by,bw,bh=spec['box']
            pad=max(2,round(bh*.05))
            erase_mask=localmask.copy()
            if bh<h*.09:
                erase_mask[max(0,by-pad):min(h,by+bh+pad),max(0,bx-pad):min(w,bx+bw+pad)]=255
            else:
                erase_mask=cv2.dilate(erase_mask,np.ones((5,5),np.uint8))
            if reason or item['score']<.70 or bh>bw*2 or item.get('recognition_conflict'):
                assessment={'safe':False,'match':text_fidelity.match_font(rgb,spec),'reason':reason or '低置信度或竖排文字需确认'}
            else:
                assessment=text_fidelity.assess_rebuild(rgb,spec,erase_mask,text_excluded)
            if assessment['safe']:
                neighbors=np.zeros((h,w),np.uint8)
                for neighbor_index,neighbor in enumerate(detected):
                    if neighbor_index!=item_index:
                        cv2.fillConvexPoly(neighbors,np.round(neighbor['quad']).astype(np.int32),255)
                neighbors=cv2.dilate(neighbors,np.ones((3,3),np.uint8))
                if text_layers.repair_overlaps(assessment['repair'],neighbors):
                    assessment.update(safe=False,reason='字边紧邻其他文字，采用独立原字形以保护相邻内容')
            if not assessment['safe']:
                reason=assessment['reason']
                reason=reason.replace('保留原像素','').replace('保留在产品像素层中，避免重复拆出或破坏包装','需按包装外观拆分')
                if item.get('review_required'):
                    reason+='；不同识别结果有分歧或置信度偏低，请核对改字替代层内容'
                review_required.append({'text':spec['text'],'reason':reason,'box':spec['box'],'mode':'appearance'})
                fallback.append({'spec':spec,'mask':localmask,'quad':item['quad'],
                                 'alternative':scaled_spec(spec,assessment['match'])})
                text_audit.append({'region_id':spec['region_id'],'text':spec['text'],'box':spec['box'],'mode':'appearance','reason':reason,
                                   'ocr_confidence':item['score'],'recognition_conflict':item.get('recognition_conflict',False)})
                continue
            match=assessment['match'];text_repairs.append(assessment['repair'])
            textmask = cv2.bitwise_or(textmask, erase_mask)
            texts.append(scaled_spec(spec,match))
            text_audit.append({'region_id':spec['region_id'],'text':spec['text'],'box':spec['box'],'mode':'native','reason':'通过字体及底色检查的可改字层'})
        warnings.append('文字内容由OCR自动识别；可靠匹配的文字重建为T层，单色字形拆为透明像素外观。复杂字形无法可靠透明分离时保留含背景原区域并标为需精修。像素层不能直接改字；隐藏的近似字体替代层可改字，但字体、曲面和字效不保证与原设计相同。')
        save_image(directory/'text_mask.png', textmask)
        brand_mask=np.zeros((h,w),np.uint8)
        for item in preserved:
            brand_mask=cv2.bitwise_or(brand_mask,item['mask'])
        # All recognized text, including separately rebuilt brand contours,
        # shares one backdrop erasure. A brand must not remain under its layer.
        brand_erase=np.zeros((h,w),np.uint8)
        for item in preserved:
            polygon=np.zeros((h,w),np.uint8)
            cv2.fillConvexPoly(polygon,np.round(item['quad']).astype(np.int32),255)
            pad=max(3,min(8,round(item['glyph_height']*.18)))
            brand_erase=cv2.bitwise_or(brand_erase,cv2.dilate(polygon,np.ones((2*pad+1,2*pad+1),np.uint8)))
        clean=text_fidelity.apply_repairs(rgb,text_repairs)
        update(17,'重建文字后方底图并检查字形边界')
        prepared_text=text_layers.prepare(clean,fallback,inpaint=self.models.inpaint,extra_erase_mask=brand_erase)
        clean=prepared_text['clean']
        save_image(directory/'textless.png', clean)
        enhancement_input=text_fidelity.apply_repairs(rgb,text_repairs)
        save_image(directory/'enhancement_input.png',enhancement_input)
        save_image(directory/'brand_mask.png',brand_mask)

        update(18, '在本机显卡上增强清晰度')
        enhancement = self.models.enhance(directory/'enhancement_input.png', directory/'enhanced.png',
                                            lambda p: update(18 + p*.30, 'AI清晰化 '+str(round(p))+'%'))
        with Image.open(directory/'enhanced.png') as large:
            enhanced = np.array(large.convert('RGB').resize((out_w,out_h), Image.Resampling.LANCZOS))

        # Contour reconstruction needs the denoised input used by the original
        # brand workflow. Ordinary text still keeps its source glyph edges.
        brand_source=enhanced.copy()
        enhanced=text_layers.preserve_text_edges(enhanced,original,cv2.bitwise_or(prepared_text['coverage_union'],brand_mask))
        save_image(directory/'appearance_source.png',enhanced)
        appearance_source=enhanced.copy()
        warnings.append('文字区域采用原图字形平滑放大，避免超分辨率模型猜出错误笔画；未识别的小字仍可能受增强影响。平滑放大不能恢复原图不存在的细节，可改字替代层则由字体重新绘制。')

        update(50,'拆分全部文字字形、边缘与近似阴影')
        enhanced,rendered_text=text_layers.render(enhanced,prepared_text,sx,sy)
        save_image(directory/'textless_enhanced.png',enhanced)
        text_regions=[]
        for i,item in enumerate(rendered_text):
            assets=[]
            for j,layer in enumerate(item['layers']):
                filename=f'text_{i+1}_{j+1}.png';save_image(directory/filename,layer['rgba'])
                assets.append({'name':layer['name'],'file':filename,'left':item['left'],'top':item['top'],'editableStroke':not item.get('contains_background',False)})
            alternative=fallback[item['index']]['alternative']
            alternative['name']='可改字替代（近似字体，默认隐藏）'
            text_regions.append({'name':f'{i+1:02d} {item["spec"]["text"][:32]}','left':item['left'],'top':item['top'],
                                 'rasters':assets,'alternative':alternative,'repair_method':item['repair_method'],
                                 'contains_background':item.get('contains_background',False)})
            if item.get('contains_background'):
                for audit in text_audit:
                    if audit['region_id']==item['spec']['region_id']:
                        audit.update(mode='preserved',reason='复杂字形无法可靠透明分离；保留含背景原区域以避免缺字、杂色，需精修；附隐藏近似可改字层')
        exported_indices={item['index'] for item in rendered_text}
        for index,item in enumerate(fallback):
            if index in exported_indices:continue
            # Two OCR descriptions can ultimately refer to the same ink. Keep
            # a real editable alternative, but never pretend it has an
            # independently extracted appearance raster.
            text_regions.append({'name':f'重叠识别 {item["spec"]["text"][:25]}｜字形见相邻组',
                'rasters':[],'alternative':{**item['alternative'],'name':'可改字替代（重叠识别，默认隐藏）'},
                'repair_method':'原字形像素已归入相邻文字组，未重复复制'})
            for audit in text_audit:
                if audit['region_id']==item['spec']['region_id']:
                    audit.update(mode='shared',reason='与相邻识别区域使用同一字形像素，已提供独立改字替代层，外观未重复复制')
        # Prevent shape approximation beneath text from changing its backing:
        # these visible surfaces are already retained in the textless image.
        protected_textmask=cv2.bitwise_or(protected_textmask,cv2.bitwise_or(prepared_text['erase_union'],prepared_text['coverage_union']))

        update(51, '自动分割主体与独立区域')
        alpha = self.models.foreground(clean)
        parts = vision.components(alpha)
        if not parts:
            warnings.append('主体边界不明显，未强行拆分；画面保留在背景像素层。')
        if len(parts) == 1:
            warnings.append('识别到一个连通主体；相互连接的产品零件仍在同一层中。')
        if parts:
            warnings.append('主体按自动前景轮廓拆分，透明玻璃、细线和重叠物体可能合并或遗漏。')
        union = np.zeros((h,w), np.uint8)
        for part in parts:
            union = np.maximum(union, part['alpha'])

        update(60, '识别装饰并修补被遮挡背景')
        shapes, deco_mask = vision.find_decor(clean, union, protected=cv2.bitwise_or(brand_mask,protected_textmask))
        # Only transparent foreground areas are removed. Entire OCR rectangles
        # are never copied into product layers or left behind as a duplicate.
        removed = cv2.dilate((union > 8).astype(np.uint8)*255, np.ones((5,5),np.uint8))
        removed = cv2.bitwise_or(removed, deco_mask)
        removed = cv2.bitwise_or(removed, brand_mask)
        bg = repair_hidden(clean, removed)
        for item in preserved:
            repaired=vision.smooth_brand_background(rgb,bg,item['mask'],
                cv2.bitwise_or(removed,textmask),item['glyph_height'])
            if repaired is not None:
                # Keep the feathered repair fringe during the later mix with
                # enhanced pixels, otherwise the original halo reappears.
                repaired_area=np.any(repaired!=bg,axis=2).astype(np.uint8)*255
                removed=cv2.bitwise_or(removed,repaired_area)
                bg=repaired
                warnings.append(f'“{item["text"][:20]}”下方的平滑底色按周围渐变重建，避免补图色块；不属于原始底图恢复。')
        bg = cv2.resize(bg, (out_w,out_h), interpolation=cv2.INTER_CUBIC)
        # Expanded removal masks are needed to estimate hidden pixels, but
        # visible pixels outside the actual foreground must remain unchanged.
        visible_support=cv2.bitwise_or(union,deco_mask)
        high_removed = cv2.resize(visible_support, (out_w,out_h), interpolation=cv2.INTER_CUBIC)
        bg = np.where(high_removed[...,None] > 0, bg, enhanced)
        high_protected=cv2.resize(protected_textmask,(out_w,out_h),interpolation=cv2.INTER_NEAREST)>0
        bg[high_protected]=enhanced[high_protected]
        save_image(directory/'background.png', bg.astype(np.uint8))
        if np.any(removed):
            warnings.append('主体后方不可见背景使用邻近颜色修补，属于重建，不能视为原始背景恢复。')
        if shapes:
            warnings.append('检测到的简单色块已重建为形状层，细微渐变可能与原图不同。')
        for shape in shapes:
            shape['box'] = [round(v*(sx if i%2 == 0 else sy)) for i,v in enumerate(shape['box'])]

        rasters = []
        for i,item in enumerate(preserved):
            high_mask=cv2.resize(item['mask'],(out_w,out_h),interpolation=cv2.INTER_NEAREST)
            ys,xs=np.nonzero(high_mask)
            if not len(xs): continue
            left,top,right,bottom=int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)
            crop=brand_source[top:bottom,left:right]
            backdrop=bg[top:bottom,left:right]
            # Fit the brand matte from the original surrounding colors, not
            # a model reconstruction that can change the glyph/background mix.
            fitted=vision.smooth_brand_background(rgb,rgb,item['mask'],
                cv2.bitwise_or(brand_mask,textmask),item['glyph_height'])
            if fitted is not None:
                backdrop=cv2.resize(fitted,(out_w,out_h),interpolation=cv2.INTER_CUBIC)[top:bottom,left:right]
            support=high_mask[top:bottom,left:right]
            rgba=vision.monochrome_brand_layer(crop,backdrop,support,item['color'],
                item['glyph_height']*sy,(right-left,bottom-top),directory/f'brand_{i+1}.svg',wordmark_text=item['text'])
            denoised=rgba is not None
            if not denoised:
                rgba=vision.foreground_from_background(appearance_source[top:bottom,left:right],
                    bg[top:bottom,left:right],support)
            else:
                warnings.append(f'“{item["text"][:20]}”为纯色标志，已近似重建轮廓并清除字缝残影，细线优先保留原有细节，另附重建SVG；PSD中保留透明像素层和可调描边，不是原始商标矢量或可改字文字层。')
                if 'data-u-contour-repaired="true"' in (directory/f'brand_{i+1}.svg').read_text(encoding='utf-8'):
                    warnings.append('顾家字标的U右下角检测到尖刺，已局部补成连续弧线；这段属于几何重建。')
            asset=Image.fromarray(rgba)
            file=f'brand_{i+1}.png'
            asset.save(directory/file)
            suffix='去噪轮廓·透明像素' if denoised else '原字形·透明像素'
            text_regions.append({'name':f'品牌标志 {item["text"][:20]}','left':left,'top':top,
                'contour_reconstructed':denoised,
                'rasters':[{'name':f'品牌原字形｜{suffix}','file':file,'left':left,'top':top,'editableStroke':True}],
                'alternative':{**item['alternative'],'name':'可改字替代（近似字体，默认隐藏）'},
                'repair_method':'平滑底色与标志轮廓近似重建'})
        update(70, '建立透明主体和可调光影图层')
        for i, part in enumerate(parts):
            x,y,pw,ph = part['box']
            left,top = round(x*sx),round(y*sy)
            right,bottom = min(out_w,round((x+pw)*sx)),min(out_h,round((y+ph)*sy))
            a = cv2.resize(part['alpha'][y:y+ph,x:x+pw], (right-left,bottom-top), interpolation=cv2.INTER_CUBIC)
            if np.any(brand_mask):
                brand_support=cv2.resize(brand_mask,(out_w,out_h),interpolation=cv2.INTER_NEAREST)
                a[brand_support[top:bottom,left:right]>0]=0
            rgba = product_layer(enhanced[top:bottom,left:right],bg[top:bottom,left:right],a)
            layer_name = f'主体{i+1}｜透明像素·原位置'
            if options.get('split_light', True):
                base, light = vision.split_highlights(rgba)
            else:
                base, light = rgba, None
            file = f'object_{i+1}.png'
            save_image(directory/file, base)
            rasters.append({'name':layer_name,'file':file,'group':'products','left':left,'top':top})
            if light is not None and np.any(light[...,3]):
                lf = f'light_{i+1}.png'
                save_image(directory/lf,light)
                rasters.append({'name':f'主体{i+1}亮部｜近似分离·可调透明度','file':lf,
                                'group':'light','left':left,'top':top})
        if options.get('split_light', True) and parts:
            warnings.append('光影层为亮部近似分离，可调透明度；未恢复拍摄时真实灯光或原始阴影参数。')
        warnings.append('AI超分辨率估算细节，无法保证还原原图中已经丢失的真实细节。')
        font_keys={s.get('font_key') for s in texts}|{r['alternative'].get('font_key') for r in text_regions}
        fonts_installed=text_fidelity.export_fonts(directory,font_keys)
        if not fonts_installed:
            warnings.append('字体已附在完整包fonts目录；如Photoshop提示缺失字体，请自行安装后重新打开PSD。')
        warnings = list(dict.fromkeys(warnings))
        manifest = {'width':out_w,'height':out_h,'original':'original.png','background':'background.png',
                    'texts':texts,'text_regions':text_regions,'text_audit':text_audit,'shapes':shapes,
                    'rasters':rasters,'warnings':warnings,'review_required':review_required}
        manifest_path = directory/'manifest.json'
        manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        update(83, '写入原生文字、描边和PSD图层组')
        node = APP_DIR/'runtime'/'node.exe'
        node_path = str(node) if node.is_file() else shutil.which('node')
        if not node_path:
            raise RuntimeError('缺少PSD导出运行组件，请重新解压完整软件包。')
        run = subprocess.run([node_path,str(APP_DIR/'psd_writer.cjs'),str(manifest_path)],
                             capture_output=True,encoding='utf-8',errors='replace',timeout=180,
                             creationflags=NO_WINDOW)
        (directory/'writer.log').write_text(run.stdout+'\n'+run.stderr,encoding='utf-8')
        if run.returncode or not (directory/'result.psd').is_file():
            raise RuntimeError('PSD导出失败，请查看任务目录中的writer.log。')
        if verify_text_composite(directory,manifest,appearance_source):
            update(91,'复核全部文字合成，保留无法可靠分离的原区域')
            manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
            run=subprocess.run([node_path,str(APP_DIR/'psd_writer.cjs'),str(manifest_path)],capture_output=True,
                encoding='utf-8',errors='replace',timeout=180,creationflags=NO_WINDOW)
            (directory/'writer.log').write_text(run.stdout+'\n'+run.stderr,encoding='utf-8')
            if run.returncode:raise RuntimeError('文字合成复核后的PSD导出失败，请查看writer.log。')
        update(95, '核对图层并打包交付文件')
        stats = json.loads((directory/'psd_stats.json').read_text(encoding='utf-8'))
        layers = json.loads((directory/'layers.json').read_text(encoding='utf-8'))
        preserved_count=sum(bool(r['rasters']) and r.get('contains_background',False) for r in text_regions)
        appearance_count=sum(bool(r['rasters']) and not r.get('contains_background',False) for r in text_regions)
        shared_count=len(text_regions)-appearance_count-preserved_count
        report = ['分层工坊 · 全自动处理说明',f'输出尺寸：{out_w} × {out_h}',
                  f'检测文字：{len(detected)} 处；直接显示的原生文字：{len(texts)} 处；透明原字形组：{appearance_count} 处；保真原区域（含背景，需精修）：{preserved_count} 处；隐藏可改字替代层：{len(text_regions)} 处；共享字形区域：{shared_count} 处',
                  f'形状层：{len(shapes)} 个',f'主体区域：{len(parts)} 个',
                  'PS中：双击T图层改字；双击fx调整描边颜色/粗细；主体图层可移动缩放；光影层可调透明度。',
                  '特殊文字：展开05文字中的对应文字组，关闭原字形、字面/阴影或保真原区域，打开默认隐藏的可改字替代，再用文字工具修改。替代字体及效果是近似重建，不是找回原设计源文字。修改前请核对OCR内容，尤其包装小字。',
                  '保真原区域：复杂字形自动透明分离不可靠时保留原图区域，其中含背景，移动会一起移动背景；不能直接改字，不提供假冒只作用于字形的换色/描边。此类区域需精修，不能作为已完整自动拆出透明文字交付。',
                  '原字形透明层：可独立移动、缩放；启用fx中的颜色叠加可换色，启用描边可新增描边并设置颜色和粗细。若需替换已有描边，请先关闭边缘与阴影层。边缘与阴影仅为可见颜色近似分离，不能等同原始立体、投影参数。',
                  ('完整交付包内附本次使用的思源系列字体；若另一台电脑提示缺失字体，请先安装fonts目录中的字体。' if (directory/'fonts').exists() else '本次未使用额外的开源字体文件。'),
                  f'近似替代需确认的文字：{len(review_required)} 处；文字检测仍可能遗漏或误认，不能保证任意图片中的全部文字均被正确识别。',
                  '文字后方的平滑背景按可见颜色拟合补齐，纹理区域保留局部修补；被文字遮挡的原始细节无法保证还原。',
                  '为避免移动文字时带走背景杂色，文字范围内的字缝底色也由本机补图模型重建；该范围可能与原图底色细节略有不同，不属于原始背景恢复。',
                  '', '处理与重建说明：', *['• '+s for s in warnings], '', '全部文字处理清单：',
                  *[f'{i+1}. {s["text"]} ｜ { {"native":"原生文字","appearance":"透明原字形及近似替代","preserved":"保真原区域（含背景，需精修）","shared":"共享原字形及独立改字替代"}[s["mode"]]} ｜ {s["reason"]}' for i,s in enumerate(text_audit)]]
        (directory/'report.txt').write_text('\n'.join(report),encoding='utf-8-sig')
        with zipfile.ZipFile(directory/'delivery.zip','w',zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
            for src,name in [('result.psd','可编辑分层.psd'),('preview.png','合成预览.png'),
                             ('original.png','输入原图.png'),('report.txt','处理说明.txt'),('layers.json','图层结构.json'),
                             ('ocr_regions.json','文字识别清单.json')]:
                archive.write(directory/src,name)
            for svg in directory.glob('brand_*.svg'):
                archive.write(svg,'重建标志矢量/'+svg.name)
            for font_file in (directory/'fonts').glob('*'):
                archive.write(font_file,'fonts/'+font_file.name)
        return {'width':out_w,'height':out_h,'text_count':len(texts),'shape_count':len(shapes),
                'text_region_count':appearance_count,'text_alternative_count':len(text_regions),
                'text_preserved_count':preserved_count,
                'shared_text_count':shared_count,'detected_text_count':len(detected),
                'text_audit':text_audit,
                'pixel_count':stats['pixel_count'],'layer_count':stats.get('layer_count',stats.get('total',0)),
                'layers':layers,'warnings':warnings,'enhancement':enhancement,'review_required':review_required,
                'elapsed':round(time.monotonic()-start,1),
                'files':{'psd':'result.psd','preview':'preview.png','original':'original.png',
                         'zip':'delivery.zip','report':'report.txt'}}


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('image',type=Path)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    args.directory.mkdir(parents=True,exist_ok=True)
    result=Processor().process(args.image,args.directory,{'scale':3,'crop_ui':True,'split_light':True},
                               lambda p,s: print(f'{p:.0f}% {s}',flush=True))
    print(json.dumps(result,ensure_ascii=False))
