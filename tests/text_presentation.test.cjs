const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const test=require('node:test');

function node(tag='',className='',textContent='') {
  return {tag,className,textContent,children:[],append(...children){this.children.push(...children);},
    setAttribute(){},addEventListener(){}};
}
function allText(item) {return [item.textContent,...item.children.flatMap(allText)].filter(Boolean);}
function ui() {
  const source=fs.readFileSync(path.resolve(__dirname,'../app/static/app.js'),'utf8');
  const functions=source.slice(source.indexOf('function renderCompleted(job)'),source.indexOf('function buildDetails(result)'));
  const scope={element:node,numberOrDash:String,safeArtifactUrl:()=>null,
    state:{previewMode:'result'},els:{resultPanel:node()},renderResult(){},buildDetails:()=>node()};
  vm.createContext(scope);vm.runInContext(functions,scope);
  return scope;
}

test('UI separates preserved background regions from transparent glyphs and native type',()=>{
  const scope=ui();
  const result={detected_text_count:4,text_count:1,text_region_count:1,text_preserved_count:1,
    shared_text_count:1,text_alternative_count:3,layer_count:20,pixel_count:8,shape_count:0,
    text_audit:[{text:'A',mode:'native'},{text:'B',mode:'appearance'},{text:'C',mode:'preserved'},
      {text:'D',mode:'shared'}]};
  const text=allText(scope.buildSummary(result)).join('\n');
  assert.match(text,/1 处透明原字形/);
  assert.match(text,/1 处保真原区域（含背景、需精修）/);
  assert.match(text,/1 处共享字形/);
  assert.match(text,/C：保真原区域（含背景、需精修）/);
  assert.match(text,/3 个默认隐藏的近似改字替代层，未计入直接可改字数量/);
  assert.doesNotMatch(text,/C：透明/);
});

test('preserved region warning is shown even without an OCR review list',()=>{
  const scope=ui();
  scope.renderCompleted({name:'test',result:{width:200,height:100,text_count:0,text_region_count:0,
    text_preserved_count:1,files:{}}});
  const text=allText(scope.els.resultPanel).join('\n');
  assert.match(text,/0 处透明原字形，1 处保真原区域（含背景、需精修）/);
  assert.match(text,/含背景区域移动时会带动背景/);
  assert.match(text,/不能直接改字/);
  assert.doesNotMatch(text,/特殊文字默认显示原字形，可独立移动/);
});
