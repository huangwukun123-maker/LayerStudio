const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const test = require('node:test');
const agPsd = require('ag-psd');
const { createCanvas, loadImage } = require('@napi-rs/canvas');
agPsd.initializeCanvas(createCanvas);

const APP = path.resolve(__dirname, '../app');

function save(directory, filename, width, height, draw) {
  const canvas = createCanvas(width, height);
  draw(canvas.getContext('2d'));
  fs.writeFileSync(path.join(directory, filename), canvas.toBuffer('image/png'));
}

function temporary(t) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'layer-studio-regions-'));
  t.after(() => {
    assert.equal(path.dirname(path.resolve(directory)), path.resolve(os.tmpdir()));
    fs.rmSync(directory, {recursive: true, force: true});
  });
  save(directory, 'background.png', 200, 100, (ctx) => {
    ctx.fillStyle = '#102030'; ctx.fillRect(0, 0, 200, 100);
  });
  return directory;
}

function write(directory, fields) {
  const manifest = {width:200, height:100, original:'background.png', background:'background.png', ...fields};
  const filename = path.join(directory, 'manifest.json');
  fs.writeFileSync(filename, JSON.stringify(manifest));
  const result = spawnSync(process.execPath, [path.join(APP, 'psd_writer.cjs'), filename], {encoding:'utf8'});
  assert.equal(result.status, 0, result.stderr);
  return {
    psd: agPsd.readPsd(fs.readFileSync(path.join(directory, 'result.psd')), {useImageData:true}),
    stats: JSON.parse(fs.readFileSync(path.join(directory, 'psd_stats.json'))),
    summary: JSON.parse(fs.readFileSync(path.join(directory, 'layers.json'))),
  };
}

test('text regions have transparent movable appearance, hidden native alternatives and editable effects', async (t) => {
  const directory = temporary(t);
  save(directory, 'glyph.png', 50, 25, (ctx) => {
    ctx.fillStyle = '#f5bc50'; ctx.fillRect(4, 3, 10, 18); ctx.fillRect(24, 3, 10, 18);
  });
  save(directory, 'shadow.png', 50, 25, (ctx) => {
    ctx.fillStyle = '#125bac'; ctx.fillRect(6, 5, 10, 18); ctx.fillRect(26, 5, 10, 18);
  });
  const {psd, stats, summary} = write(directory, {text_regions:[{
    name:'标题：测试 AB', left:30, top:20,
    rasters:[
      {name:'文字原貌阴影', file:'shadow.png', editableStroke:true},
      {name:'文字原貌字形', file:'glyph.png', editableStroke:true},
    ],
    alternative:{text:'测试 AB', box:[30,20,100,25], color:'#ff00ff', bold:true},
  }]});
  const group = psd.children[5].children[0];
  assert.equal(group.name, '标题：测试 AB');
  assert.equal(group.children.length, 3);
  const appearance = group.children[1];
  assert.equal(appearance.left, 30);
  assert.equal(appearance.top, 20);
  assert.equal(appearance.imageData.data[3], 0, 'original background must not be part of text appearance');
  assert.equal(appearance.effects.stroke[0].enabled, false);
  assert.equal(appearance.effects.solidFill[0].enabled, false);
  assert.equal(appearance.effects.solidFill[0].blendMode, 'normal');
  const alternative = group.children[2];
  assert.equal(alternative.text.text, '测试 AB');
  assert.equal(alternative.hidden, true);
  assert.match(alternative.name, /可改字替代.*近似字体.*默认隐藏/);
  assert.equal(stats.text_count, 0, 'hidden alternatives are not counted as active native type');
  assert.equal(stats.text_alternative_count, 1);
  assert.equal(stats.text_region_count, 1);
  assert.equal(stats.pixel_count, 4);
  assert.equal(stats.layer_count, 12, 'nested groups and their children must all be counted');
  assert.equal(summary[5].children[0].children[2].hidden, true);

  const image = await loadImage(path.join(directory, 'preview.png'));
  const actual = createCanvas(200, 100); actual.getContext('2d').drawImage(image, 0, 0);
  const expected = createCanvas(200, 100); const ctx = expected.getContext('2d');
  ctx.fillStyle = '#102030'; ctx.fillRect(0,0,200,100);
  ctx.drawImage(await loadImage(path.join(directory, 'shadow.png')),30,20);
  ctx.drawImage(await loadImage(path.join(directory, 'glyph.png')),30,20);
  assert.deepEqual(actual.getContext('2d').getImageData(0,0,200,100).data, ctx.getImageData(0,0,200,100).data,
    'hidden native text must not double-draw over the preserved original appearance');
});

test('native alternatives preserve italic shear and tracking in PSD metadata and within the requested bounds', (t) => {
  const directory = temporary(t);
  const {psd,stats} = write(directory, {texts:[
    {name:'隐藏原生文字', text:'ABCD', box:[20,20,160,35], color:'#ffffff', bold:true, hidden:true, shear:-0.3, tracking:8},
  ]});
  const layer = psd.children[5].children[0];
  assert.equal(layer.hidden, true);
  assert.equal(stats.text_count,0);
  assert.equal(layer.text.style.tracking,80);
  assert.equal(layer.text.style.autoKerning,false);
  const matrix = layer.text.transform;
  assert.ok(Math.abs(matrix[2]/matrix[0] + .3)<1e-6, 'native transform must retain the fitted shear');
  const data=layer.imageData.data;
  const occupied=[];
  for(let y=0;y<layer.imageData.height;y++) for(let x=0;x<layer.imageData.width;x++) {
    if(data[(y*layer.imageData.width+x)*4+3]>8) occupied.push([x+layer.left,y+layer.top]);
  }
  for(const [actual,expected] of [
    [Math.min(...occupied.map(p=>p[0])),20], [Math.max(...occupied.map(p=>p[0]))+1,180],
    [Math.min(...occupied.map(p=>p[1])),20], [Math.max(...occupied.map(p=>p[1]))+1,55],
  ]) assert.ok(Math.abs(actual-expected)<=2, `transformed ink bounds ${actual} differ from ${expected}`);
});

test('hidden region children stay hidden as a group and native text statistics inherit parent visibility', async (t) => {
  const directory = temporary(t);
  save(directory, 'glyph.png', 10, 10, (ctx) => {ctx.fillStyle='#ffffff';ctx.fillRect(0,0,10,10);});
  const {psd, stats} = write(directory, {
    texts:[{text:'A', box:[140,10,20,20], color:'#f5bc50', bold:true}],
    text_regions:[{
      name:'隐藏的区域', hidden:true,
      rasters:[{name:'原貌',file:'glyph.png',left:10,top:10}],
      alternative:{text:'AB',box:[10,10,50,20],color:'#ff0000'},
      alternativeHidden:false,
    }],
  });
  const group=psd.children[5].children[1];
  assert.equal(group.hidden,true);
  assert.equal(group.children[1].hidden,false);
  assert.equal(stats.text_count,1,'native child of a hidden group is not visible type');
  assert.equal(stats.text_alternative_count,1);
  const ids=[];
  function visit(layers) { for(const layer of layers) {ids.push(layer.id);if(layer.children)visit(layer.children);} }
  visit(psd.children);
  assert.equal(new Set(ids).size,ids.length,'nested groups and layers need unique Photoshop IDs');
  const preview=await loadImage(path.join(directory,'preview.png'));
  const canvas=createCanvas(200,100);const ctx=canvas.getContext('2d');ctx.drawImage(preview,0,0);
  assert.deepEqual(Array.from(ctx.getImageData(12,12,1,1).data),[16,32,48,255]);
});

test('preserved original regions are counted separately and do not expose glyph-only effects', (t) => {
  const directory=temporary(t);
  save(directory,'original-region.png',50,20,(ctx)=>{ctx.fillStyle='#335599';ctx.fillRect(0,0,50,20);});
  save(directory,'transparent.png',50,20,(ctx)=>{ctx.fillStyle='#ffffff';ctx.fillRect(5,5,10,10);});
  const alternative={text:'AB',box:[10,10,50,20],color:'#ffffff'};
  const {psd,stats,summary}=write(directory,{
    texts:[{text:'C',box:[150,10,20,20],color:'#ffffff'}],
    text_regions:[
      {name:'透明文字',rasters:[{file:'transparent.png',left:10,top:10}],alternative},
      {name:'复杂文字',contains_background:true,rasters:[{file:'original-region.png',left:70,top:10,editableStroke:true}],alternative},
      {name:'共享字形',rasters:[],alternative},
    ],
  });
  assert.equal(stats.text_count,1);
  assert.equal(stats.text_region_count,1,'only transparent appearance groups are counted as extracted glyphs');
  assert.equal(stats.text_preserved_count,1);
  assert.equal(stats.shared_text_count,1);
  assert.equal(stats.text_alternative_count,3);
  assert.equal(stats.text_count+stats.text_region_count+stats.text_preserved_count+stats.shared_text_count,4);
  const group=psd.children[5].children[2];
  assert.match(group.name,/含背景.*需精修/);
  assert.equal(group.children[0].effects,undefined,'outline/color overlay would incorrectly affect the whole preserved region');
  assert.equal(group.children[0].imageData.data[3],255,'preserved backing is honestly opaque');
  assert.equal(group.children[1].hidden,true);
  assert.equal(group.children[1].text.text,'AB');
  assert.equal(summary[5].children[2].contains_background,true);
});
