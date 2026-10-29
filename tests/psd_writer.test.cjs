const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const test = require('node:test');

const agPsd = require('ag-psd');
const { createCanvas, loadImage, GlobalFonts } = require('@napi-rs/canvas');

agPsd.initializeCanvas(createCanvas);
GlobalFonts.registerFromPath('C:/Windows/Fonts/msyh.ttc', 'WriterTestRegular');
GlobalFonts.registerFromPath('C:/Windows/Fonts/msyhbd.ttc', 'WriterTestBold');

const APP_DIR = path.resolve(__dirname, '..', 'app');
const WRITER = path.join(APP_DIR, 'psd_writer.cjs');

test('matched heavy font is used for both editable type and visible glyph pixels', async (t) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'layer-studio-font-'));
  t.after(() => { assert.equal(path.dirname(path.resolve(directory)), path.resolve(os.tmpdir())); fs.rmSync(directory, {recursive:true, force:true}); });
  savePng(path.join(directory,'original.png'),160,120,(ctx)=>{ctx.fillStyle='#eeeeee';ctx.fillRect(0,0,160,120);});
  const manifest={width:160,height:120,original:'original.png',background:'original.png',texts:[
    {text:'安心健康用水',name:'Matched',box:[5,5,150,40],color:'#202020',bold:true,font_key:'LayerStudioSansSC-Black'},
  ]};
  const filename=path.join(directory,'manifest.json');fs.writeFileSync(filename,JSON.stringify(manifest));
  const run=spawnSync(process.execPath,[WRITER,filename],{cwd:APP_DIR,encoding:'utf8'});
  assert.equal(run.status,0,run.stderr);
  const psd=agPsd.readPsd(fs.readFileSync(path.join(directory,'result.psd')),{useImageData:true});
  const matched=findChild(psd.children[5],'Matched');
  assert.equal(matched.text.style.font.name,'LayerStudioSansSC-Black');
  GlobalFonts.registerFromPath(path.join(APP_DIR,'fonts/LayerStudioSansSC-Black.ttf'),'WriterTestBlack');
  const reference=renderReferenceText('安心健康用水',[5,5,150,40],true,'WriterTestBlack');
  assert.ok(alphaMaskIoU(layerInDocument(matched),reference)>.95, 'cached PSD pixels must use the matched font');
  const previous=renderReferenceText('安心健康用水',[5,5,150,40],true);
  assert.ok(alphaMaskIoU(layerInDocument(matched),previous)<.9, 'new font still rendered as the previous YaHei');
});

function savePng(file, width, height, draw) {
  const canvas = createCanvas(width, height);
  draw(canvas.getContext('2d'));
  fs.writeFileSync(file, canvas.toBuffer('image/png'));
}

function findChild(group, name) {
  const child = group.children.find((candidate) => candidate.name === name);
  assert.ok(child, `missing layer ${name}`);
  return child;
}

function alphaBounds(layer) {
  const { width, height, data } = layer.imageData;
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (data[(y * width + x) * 4 + 3] > 8) {
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
      }
    }
  }
  assert.ok(maxX >= minX && maxY >= minY, `layer ${layer.name} has no visible pixels`);
  return {
    left: layer.left + minX,
    top: layer.top + minY,
    right: layer.left + maxX + 1,
    bottom: layer.top + maxY + 1,
  };
}

function rotatedBounds([x, y, width, height], degrees) {
  const radians = degrees * Math.PI / 180;
  const cos = Math.cos(radians);
  const sin = Math.sin(radians);
  const cx = x + width / 2;
  const cy = y + height / 2;
  const points = [
    [x, y], [x + width, y], [x + width, y + height], [x, y + height],
  ].map(([px, py]) => [
    cx + cos * (px - cx) - sin * (py - cy),
    cy + sin * (px - cx) + cos * (py - cy),
  ]);
  return {
    left: Math.min(...points.map(([px]) => px)),
    top: Math.min(...points.map(([, py]) => py)),
    right: Math.max(...points.map(([px]) => px)),
    bottom: Math.max(...points.map(([, py]) => py)),
  };
}

function renderReferenceText(text, box, bold, family = bold ? 'WriterTestBold' : 'WriterTestRegular') {
  const probe = createCanvas(1, 1).getContext('2d');
  probe.font = `100px "${family}"`;
  const advance = probe.measureText(text).width;
  const reference = createCanvas(Math.ceil(advance) + 400, 400);
  const referenceContext = reference.getContext('2d');
  referenceContext.font = `100px "${family}"`;
  referenceContext.fillStyle = '#ffffff';
  referenceContext.fillText(text, 200, 200);
  const referencePixels = referenceContext.getImageData(0, 0, reference.width, reference.height).data;
  let left = reference.width;
  let top = reference.height;
  let right = -1;
  let bottom = -1;
  for (let y = 0; y < reference.height; y += 1) {
    for (let x = 0; x < reference.width; x += 1) {
      if (referencePixels[(y * reference.width + x) * 4 + 3] > 8) {
        left = Math.min(left, x);
        top = Math.min(top, y);
        right = Math.max(right, x + 1);
        bottom = Math.max(bottom, y + 1);
      }
    }
  }
  const [x, y, width, height] = box;
  const output = createCanvas(160, 120);
  const context = output.getContext('2d');
  context.font = `100px "${family}"`;
  context.fillStyle = '#ffffff';
  context.setTransform(
    width / (right - left),
    0,
    0,
    height / (bottom - top),
    x - (left - 200) * width / (right - left),
    y - (top - 200) * height / (bottom - top),
  );
  context.fillText(text, 0, 0);
  return output;
}

function layerInDocument(layer) {
  const layerCanvas = createCanvas(layer.imageData.width, layer.imageData.height);
  layerCanvas.getContext('2d').putImageData(layer.imageData, 0, 0);
  const output = createCanvas(160, 120);
  output.getContext('2d').drawImage(layerCanvas, layer.left, layer.top);
  return output;
}

function alphaMaskIoU(actualCanvas, expectedCanvas) {
  const actual = actualCanvas.getContext('2d').getImageData(0, 0, 160, 120).data;
  const expected = expectedCanvas.getContext('2d').getImageData(0, 0, 160, 120).data;
  let intersection = 0;
  let union = 0;
  for (let index = 3; index < actual.length; index += 4) {
    const actualInk = actual[index] > 8;
    const expectedInk = expected[index] > 8;
    if (actualInk && expectedInk) intersection += 1;
    if (actualInk || expectedInk) union += 1;
  }
  return intersection / union;
}

test('writes full mixed-script pixels, editable PSD semantics, scaled source, and faithful preview', async (t) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'layer-studio-writer-'));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));

  savePng(path.join(directory, 'original.png'), 80, 60, (ctx) => {
    ctx.fillStyle = '#00ff00';
    ctx.fillRect(0, 0, 40, 30);
    ctx.fillStyle = '#0000ff';
    ctx.fillRect(40, 30, 40, 30);
  });
  savePng(path.join(directory, 'background.png'), 160, 120, (ctx) => {
    ctx.fillStyle = '#0c2238';
    ctx.fillRect(0, 0, 160, 120);
  });
  savePng(path.join(directory, 'product.png'), 40, 30, (ctx) => {
    ctx.clearRect(0, 0, 40, 30);
    ctx.fillStyle = 'rgba(240, 40, 20, 0.70)';
    ctx.fillRect(5, 4, 30, 22);
  });
  const originalBefore = fs.readFileSync(path.join(directory, 'original.png'));

  const textBox = [52, 18, 78, 24];
  const rotation = 14;
  const manifest = {
    width: 160,
    height: 120,
    original: 'original.png',
    background: 'background.png',
    rasters: [
      { name: '透明产品', file: 'product.png', group: 'products', left: 24, top: 48 },
    ],
    texts: [
      {
        text: '测试文字',
        name: '可编辑中文',
        box: textBox,
        color: '#f5c542',
        bold: true,
        rotation,
        confidence: 0.96,
        stroke: { color: '#ffffff', width: 2.5 },
      },
      {
        text: 'Gujia顾家',
        name: '混合品牌文字',
        box: [5, 5, 100, 18],
        color: '#ffffff',
        bold: true,
        rotation: 0,
        confidence: 0.99,
      },
      {
        text: '母婴级材质不含双酚A',
        name: '末尾拉丁文字',
        box: [5, 30, 145, 14],
        color: '#ffffff',
        bold: false,
        rotation: 0,
        confidence: 0.99,
      },
    ],
    shapes: [
      { name: '圆角标签', box: [8, 82, 34, 22], color: '#4287f5', radius: 6 },
    ],
    warnings: ['测试警告'],
  };
  const manifestPath = path.join(directory, 'manifest.json');
  fs.writeFileSync(manifestPath, JSON.stringify(manifest), 'utf8');

  const run = spawnSync(process.execPath, [WRITER, manifestPath], {
    cwd: APP_DIR,
    encoding: 'utf8',
  });
  assert.equal(run.status, 0, `writer failed\nstdout: ${run.stdout}\nstderr: ${run.stderr}`);
  assert.deepEqual(fs.readFileSync(path.join(directory, 'original.png')), originalBefore, 'writer modified original.png');

  for (const filename of ['result.psd', 'preview.png', 'layers.json', 'psd_stats.json']) {
    assert.ok(fs.statSync(path.join(directory, filename)).size > 0, `${filename} was not written`);
  }

  const psd = agPsd.readPsd(fs.readFileSync(path.join(directory, 'result.psd')), {
    useImageData: true,
    skipThumbnail: true,
  });
  assert.deepEqual(
    psd.children.map((layer) => layer.name),
    ['00 原图（隐藏）', '01 背景', '02 装饰', '03 产品', '04 光效', '05 文字'],
  );
  assert.equal(psd.children[0].hidden, true);
  const source = findChild(psd.children[0], '原始图像');
  assert.equal(source.imageData.width, 160);
  assert.equal(source.imageData.height, 120);
  const sourceBottomRight = ((110 * source.imageData.width) + 150) * 4;
  assert.deepEqual(Array.from(source.imageData.data.slice(sourceBottomRight, sourceBottomRight + 3)), [0, 0, 255]);

  const product = findChild(psd.children[3], '透明产品');
  assert.equal(product.left, 24);
  assert.equal(product.top, 48);
  assert.equal(product.imageData.data[3], 0, 'transparent corner must retain zero alpha');
  const productCenter = ((15 * product.imageData.width) + 20) * 4;
  assert.ok(product.imageData.data[productCenter + 3] >= 175 && product.imageData.data[productCenter + 3] <= 181);
  assert.equal(product.effects.stroke[0].enabled, false);

  const shape = findChild(psd.children[2], '圆角标签');
  assert.ok(shape.vectorMask && shape.vectorFill, 'shape must remain a native vector layer');
  assert.equal(shape.left, 8);
  assert.equal(shape.top, 82);
  assert.equal(shape.effects.stroke[0].enabled, false);

  const textLayer = findChild(psd.children[5], '可编辑中文');
  assert.equal(textLayer.text.text, '测试文字');
  assert.ok(Math.abs(textLayer.text.transform[1]) > 0.01, 'rotation must survive in native text transform');
  assert.equal(textLayer.effects.stroke[0].enabled, false);
  assert.equal(textLayer.effects.stroke[0].size.value, 2.5);
  assert.deepEqual(textLayer.effects.stroke[0].color, { r: 255, g: 255, b: 255 });

  const actualTextBounds = alphaBounds(textLayer);
  const expectedTextBounds = rotatedBounds(textBox, rotation);
  for (const edge of ['left', 'top', 'right', 'bottom']) {
    assert.ok(
      Math.abs(actualTextBounds[edge] - expectedTextBounds[edge]) <= 3,
      `${edge} placement differs: actual=${actualTextBounds[edge]}, expected=${expectedTextBounds[edge]}`,
    );
  }

  for (const [name, text, box, bold] of [
    ['混合品牌文字', 'Gujia顾家', [5, 5, 100, 18], true],
    ['末尾拉丁文字', '母婴级材质不含双酚A', [5, 30, 145, 14], false],
  ]) {
    const mixedLayer = findChild(psd.children[5], name);
    const similarity = alphaMaskIoU(layerInDocument(mixedLayer), renderReferenceText(text, box, bold));
    assert.ok(similarity > 0.95, `${text} was clipped or distorted; alpha IoU=${similarity.toFixed(3)}`);
  }

  const preview = await loadImage(path.join(directory, 'preview.png'));
  const previewCanvas = createCanvas(160, 120);
  const previewContext = previewCanvas.getContext('2d');
  previewContext.drawImage(preview, 0, 0);
  const pixels = previewContext.getImageData(0, 0, 160, 120).data;
  assert.deepEqual(Array.from(pixels.slice(0, 3)), [12, 34, 56], 'hidden green original leaked into preview');
  const transparentProductCorner = ((48 * 160) + 24) * 4;
  assert.deepEqual(Array.from(pixels.slice(transparentProductCorner, transparentProductCorner + 3)), [12, 34, 56]);
  const productPixel = ((63 * 160) + 44) * 4;
  assert.ok(pixels[productPixel] > pixels[productPixel + 2], 'visible product pixels must be composited');

  const layers = JSON.parse(fs.readFileSync(path.join(directory, 'layers.json'), 'utf8'));
  assert.equal(layers.length, 6);
  assert.equal(layers[5].children[0].kind, 'text');
  const stats = JSON.parse(fs.readFileSync(path.join(directory, 'psd_stats.json'), 'utf8'));
  assert.deepEqual(
    { width: stats.width, height: stats.height, text_count: stats.text_count, shape_count: stats.shape_count, pixel_count: stats.pixel_count },
    { width: 160, height: 120, text_count: 3, shape_count: 1, pixel_count: 3 },
  );
  assert.deepEqual(stats.warnings, ['测试警告']);
});
