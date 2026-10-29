#!/usr/bin/env node
'use strict';

const fs = require('node:fs');
const path = require('node:path');

const agPsd = require('ag-psd');
const { createCanvas, loadImage, Path2D, GlobalFonts } = require('@napi-rs/canvas');

agPsd.initializeCanvas(createCanvas);

const GROUPS = [
  { key: 'source', name: '00 原图（隐藏）', hidden: true },
  { key: 'background', name: '01 背景' },
  { key: 'decor', name: '02 装饰' },
  { key: 'products', name: '03 产品' },
  { key: 'light', name: '04 光效' },
  { key: 'text', name: '05 文字' },
];
const FONT_SIZE = 100;
const MAX_DIMENSION = 30_000;
const FONT_CANDIDATES = {
  regular: [
    ['C:/Windows/Fonts/msyh.ttc', 'LayerStudioRegular', 'MicrosoftYaHei'],
    ['C:/Windows/Fonts/simhei.ttf', 'LayerStudioRegular', 'SimHei'],
  ],
  bold: [
    ['C:/Windows/Fonts/msyhbd.ttc', 'LayerStudioBold', 'MicrosoftYaHei-Bold'],
    ['C:/Windows/Fonts/simhei.ttf', 'LayerStudioBold', 'SimHei'],
  ],
};

function fail(message) {
  throw new Error(message);
}

function finiteNumber(value, label) {
  const number = Number(value);
  if (!Number.isFinite(number)) fail(`${label} must be a finite number`);
  return number;
}

function positiveInteger(value, label) {
  const number = finiteNumber(value, label);
  if (!Number.isInteger(number) || number <= 0) fail(`${label} must be a positive integer`);
  return number;
}

function parseColor(value, label) {
  if (typeof value !== 'string' || !/^#[0-9a-f]{6}$/i.test(value)) {
    fail(`${label} must use #rrggbb format`);
  }
  return {
    r: Number.parseInt(value.slice(1, 3), 16),
    g: Number.parseInt(value.slice(3, 5), 16),
    b: Number.parseInt(value.slice(5, 7), 16),
  };
}

function relativeImagePath(baseDirectory, value, label) {
  if (typeof value !== 'string' || value.length === 0 || path.isAbsolute(value)) {
    fail(`${label} must be a relative image path`);
  }
  const resolved = path.resolve(baseDirectory, value);
  const relative = path.relative(baseDirectory, resolved);
  if (relative.startsWith('..') || path.isAbsolute(relative)) fail(`${label} escapes the manifest directory`);
  if (!fs.statSync(resolved, { throwIfNoEntry: false })?.isFile()) fail(`${label} does not exist: ${value}`);
  return resolved;
}

function validateBox(value, label) {
  if (!Array.isArray(value) || value.length !== 4) fail(`${label} must be [x,y,width,height]`);
  const box = value.map((part, index) => finiteNumber(part, `${label}[${index}]`));
  if (box[2] <= 0 || box[3] <= 0) fail(`${label} width and height must be positive`);
  return box;
}

function imageData(canvas) {
  return canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height);
}

function createStroke(color, width) {
  return {
    present: true,
    showInDialog: true,
    enabled: false,
    position: 'outside',
    fillType: 'color',
    blendMode: 'normal',
    opacity: 1,
    color,
    size: { units: 'Pixels', value: width },
  };
}

function disabledStroke(color = { r: 255, g: 255, b: 255 }, width = 3) {
  return { scale: 1, disabled: false, stroke: [createStroke(color, width)] };
}

function editableAppearanceEffects() {
  return {
    ...disabledStroke(),
    solidFill: [{ present: true, showInDialog: true, enabled: false, blendMode: 'normal',
      color: { r: 255, g: 255, b: 255 }, opacity: 1 }],
  };
}

function registerFont(kind) {
  for (const [filename, canvasFamily, psdName] of FONT_CANDIDATES[kind]) {
    if (fs.existsSync(filename) && GlobalFonts.registerFromPath(filename, canvasFamily)) {
      return { canvasFamily, psdName };
    }
  }
  return {
    canvasFamily: kind === 'bold' ? 'sans-serif' : 'sans-serif',
    psdName: kind === 'bold' ? 'Arial-BoldMT' : 'ArialMT',
  };
}

const FONTS = {
  regular: registerFont('regular'),
  bold: registerFont('bold'),
};
const MATCHED_FONTS = new Map();
function matchedFont(specification) {
  if (!specification.font_key) return specification.bold ? FONTS.bold : FONTS.regular;
  if (MATCHED_FONTS.has(specification.font_key)) return MATCHED_FONTS.get(specification.font_key);
  const directory = path.join(__dirname, 'fonts');
  const catalog = JSON.parse(fs.readFileSync(path.join(directory, 'catalog.json'), 'utf8'));
  const entry = catalog.find((item) => item.key === specification.font_key);
  if (!entry) fail('unknown matched font: ' + specification.font_key);
  const filename = path.join(entry.bundled ? directory : path.join(process.env.WINDIR || 'C:/Windows', 'Fonts'), entry.file);
  const canvasFamily = 'Matched_' + entry.key;
  if (!GlobalFonts.registerFromPath(filename, canvasFamily)) fail('matched font unavailable: ' + entry.label);
  const result = {canvasFamily, psdName: entry.psd_name};
  MATCHED_FONTS.set(entry.key, result);
  return result;
}

async function loadRaster(file, name, left, top, options = {}) {
  const source = await loadImage(file);
  const canvasWidth = options.width ?? source.width;
  const canvasHeight = options.height ?? source.height;
  const canvas = createCanvas(canvasWidth, canvasHeight);
  canvas.getContext('2d').drawImage(source, 0, 0, canvasWidth, canvasHeight);
  const layer = {
    name,
    left: Math.round(left),
    top: Math.round(top),
    imageData: imageData(canvas),
  };
  if (options.hidden) layer.hidden = true;
  if (options.editableStroke) layer.effects = editableAppearanceEffects();
  return { kind: 'pixel', layer, canvas };
}

function roundedRectanglePath(x, y, width, height, radius) {
  const r = Math.max(0, Math.min(radius, width / 2, height / 2));
  const right = x + width;
  const bottom = y + height;
  if (r === 0) return `M ${x} ${y} L ${right} ${y} L ${right} ${bottom} L ${x} ${bottom} Z`;
  const k = 0.5522847498307936;
  return [
    `M ${x + r} ${y}`,
    `L ${right - r} ${y}`,
    `C ${right - r + k * r} ${y} ${right} ${y + r - k * r} ${right} ${y + r}`,
    `L ${right} ${bottom - r}`,
    `C ${right} ${bottom - r + k * r} ${right - r + k * r} ${bottom} ${right - r} ${bottom}`,
    `L ${x + r} ${bottom}`,
    `C ${x + r - k * r} ${bottom} ${x} ${bottom - r + k * r} ${x} ${bottom - r}`,
    `L ${x} ${y + r}`,
    `C ${x} ${y + r - k * r} ${x + r - k * r} ${y} ${x + r} ${y}`,
    'Z',
  ].join(' ');
}

function vectorPathFromSvg(svg) {
  const tokens = svg.match(/[MLCZ]|-?\d+(?:\.\d+)?(?:e[+-]?\d+)?/gi) || [];
  const knots = [];
  let index = 0;
  let closed = false;
  while (index < tokens.length) {
    const operation = tokens[index++].toUpperCase();
    if (operation === 'M' || operation === 'L') {
      const point = [Number(tokens[index++]), Number(tokens[index++])];
      knots.push({ linked: false, points: [...point, ...point, ...point] });
    } else if (operation === 'C') {
      const outgoing = [Number(tokens[index++]), Number(tokens[index++])];
      const incoming = [Number(tokens[index++]), Number(tokens[index++])];
      const point = [Number(tokens[index++]), Number(tokens[index++])];
      knots.at(-1).points.splice(4, 2, ...outgoing);
      knots.push({ linked: false, points: [...incoming, ...point, ...point] });
    } else if (operation === 'Z') {
      closed = true;
    } else {
      fail(`unsupported vector path operation ${operation}`);
    }
  }
  if (knots.length > 1) {
    const first = knots[0].points;
    const last = knots.at(-1).points;
    if (first[2] === last[2] && first[3] === last[3]) {
      first.splice(0, 2, ...last.slice(0, 2));
      knots.pop();
    }
  }
  return { open: !closed, operation: 'combine', fillRule: 'non-zero', knots };
}

function buildShape(specification, index) {
  const name = typeof specification.name === 'string' && specification.name ? specification.name : `形状 ${index + 1}`;
  const [x, y, width, height] = validateBox(specification.box, `shapes[${index}].box`);
  const color = parseColor(specification.color, `shapes[${index}].color`);
  const radius = Math.max(0, finiteNumber(specification.radius ?? 0, `shapes[${index}].radius`));
  const layerLeft = Math.floor(x);
  const layerTop = Math.floor(y);
  const layerRight = Math.ceil(x + width);
  const layerBottom = Math.ceil(y + height);
  const canvas = createCanvas(layerRight - layerLeft, layerBottom - layerTop);
  const context = canvas.getContext('2d');
  const localPath = roundedRectanglePath(x - layerLeft, y - layerTop, width, height, radius);
  context.fillStyle = specification.color;
  context.fill(new Path2D(localPath));
  const documentPath = roundedRectanglePath(x, y, width, height, radius);
  const layer = {
    name,
    left: layerLeft,
    top: layerTop,
    imageData: imageData(canvas),
    vectorFill: { type: 'color', color },
    vectorMask: {
      fillStartsWithAllPixels: false,
      paths: [vectorPathFromSvg(documentPath)],
    },
    effects: disabledStroke(),
  };
  return { kind: 'shape', layer, canvas };
}

function drawSpacedText(context, text, x, y, tracking) {
  if (tracking === 0) {
    context.fillText(text, x, y);
    return;
  }
  let cursor = x;
  for (const character of text) {
    context.fillText(character, cursor, y);
    cursor += context.measureText(character).width + tracking;
  }
}

function measureTextInk(text, font, shear = 0, tracking = 0) {
  const probe = createCanvas(1, 1).getContext('2d');
  probe.font = `${FONT_SIZE}px "${font.canvasFamily}"`;
  const advance = tracking === 0 ? probe.measureText(text).width
    : Array.from(text).reduce((total, character) => total + probe.measureText(character).width + tracking, 0);
  const padding = FONT_SIZE * (2 + Math.abs(shear));
  const canvas = createCanvas(Math.max(1, Math.ceil(advance + padding * 2)), FONT_SIZE * 4);
  const context = canvas.getContext('2d');
  const originX = padding;
  const originY = padding;
  context.font = `${FONT_SIZE}px "${font.canvasFamily}"`;
  context.fillStyle = '#ffffff';
  context.setTransform(1, 0, shear, 1, originX, originY);
  drawSpacedText(context, text, 0, 0, tracking);
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
  let left = canvas.width;
  let top = canvas.height;
  let right = -1;
  let bottom = -1;
  for (let y = 0; y < canvas.height; y += 1) {
    for (let x = 0; x < canvas.width; x += 1) {
      if (pixels[(y * canvas.width + x) * 4 + 3] > 8) {
        left = Math.min(left, x);
        top = Math.min(top, y);
        right = Math.max(right, x + 1);
        bottom = Math.max(bottom, y + 1);
      }
    }
  }
  if (right <= left || bottom <= top) fail('text has no visible ink');
  return {
    left: left - originX,
    top: top - originY,
    right: right - originX,
    bottom: bottom - originY,
  };
}

function composeTextTransform(box, ink, rotationDegrees, shear = 0) {
  const [x, y, width, height] = box;
  const inkWidth = ink.right - ink.left;
  const inkHeight = ink.bottom - ink.top;
  const scaleX = width / inkWidth;
  const scaleY = height / inkHeight;
  const baseX = x - ink.left * scaleX;
  const baseY = y - ink.top * scaleY;
  const radians = rotationDegrees * Math.PI / 180;
  const cos = Math.cos(radians);
  const sin = Math.sin(radians);
  const centerX = x + width / 2;
  const centerY = y + height / 2;
  return [
    cos * scaleX,
    sin * scaleX,
    cos * shear * scaleX - sin * scaleY,
    sin * shear * scaleX + cos * scaleY,
    centerX + cos * (baseX - centerX) - sin * (baseY - centerY),
    centerY + sin * (baseX - centerX) + cos * (baseY - centerY),
  ];
}

function transformedPoint(transform, x, y) {
  return [
    transform[0] * x + transform[2] * y + transform[4],
    transform[1] * x + transform[3] * y + transform[5],
  ];
}

function buildText(specification, index) {
  if (typeof specification.text !== 'string' || specification.text.length === 0) {
    fail(`texts[${index}].text must be a non-empty string`);
  }
  const name = typeof specification.name === 'string' && specification.name ? specification.name : specification.text;
  const box = validateBox(specification.box, `texts[${index}].box`);
  const color = parseColor(specification.color, `texts[${index}].color`);
  const rotation = finiteNumber(specification.rotation ?? 0, `texts[${index}].rotation`);
  const shear = finiteNumber(specification.shear ?? 0, `texts[${index}].shear`);
  const tracking = finiteNumber(specification.tracking ?? 0, `texts[${index}].tracking`);
  const font = matchedFont(specification);
  const ink = measureTextInk(specification.text, font, shear, tracking);
  const transform = composeTextTransform(box, ink, rotation, shear);
  // The measured ink is already sheared. Map those bounds without applying
  // the shear a second time when allocating the layer's cached pixel canvas.
  const boundsTransform = composeTextTransform(box, ink, rotation);
  const inkCorners = [
    [ink.left, ink.top],
    [ink.right, ink.top],
    [ink.right, ink.bottom],
    [ink.left, ink.bottom],
  ].map(([x, y]) => transformedPoint(boundsTransform, x, y));
  const margin = 2;
  const left = Math.floor(Math.min(...inkCorners.map(([x]) => x))) - margin;
  const top = Math.floor(Math.min(...inkCorners.map(([, y]) => y))) - margin;
  const right = Math.ceil(Math.max(...inkCorners.map(([x]) => x))) + margin;
  const bottom = Math.ceil(Math.max(...inkCorners.map(([, y]) => y))) + margin;
  const canvas = createCanvas(Math.max(1, right - left), Math.max(1, bottom - top));
  const context = canvas.getContext('2d');
  context.setTransform(transform[0], transform[1], transform[2], transform[3], transform[4] - left, transform[5] - top);
  context.font = `${FONT_SIZE}px "${font.canvasFamily}"`;
  context.fillStyle = specification.color;
  context.textAlign = 'left';
  context.textBaseline = 'alphabetic';
  drawSpacedText(context, specification.text, 0, 0, tracking);

  const requestedStroke = specification.stroke || {};
  const strokeColor = requestedStroke.color === undefined
    ? { r: 255, g: 255, b: 255 }
    : parseColor(requestedStroke.color, `texts[${index}].stroke.color`);
  const strokeWidth = requestedStroke.width === undefined
    ? 3
    : finiteNumber(requestedStroke.width, `texts[${index}].stroke.width`);
  if (strokeWidth <= 0) fail(`texts[${index}].stroke.width must be positive`);

  const layer = {
    name,
    hidden: specification.hidden === true,
    left,
    top,
    imageData: imageData(canvas),
    text: {
      text: specification.text,
      transform,
      antiAlias: 'smooth',
      orientation: 'horizontal',
      shapeType: 'point',
      style: {
        font: { name: font.psdName },
        fontSize: FONT_SIZE,
        fillColor: color,
        fillFlag: true,
        strokeFlag: false,
        autoKerning: tracking === 0,
        tracking: tracking * 1000 / FONT_SIZE,
      },
      paragraphStyle: { justification: 'left' },
    },
    effects: disabledStroke(strokeColor, strokeWidth),
  };
  return { kind: 'text', layer, canvas };
}

function makeSummary(record) {
  if (record.children) {
    return { name: record.name, kind: 'group', hidden: record.hidden === true,
      ...(record.contains_background ? {contains_background:true} : {}),
      children: record.children.map(makeSummary) };
  }
  return { name: record.layer.name, kind: record.kind, hidden: record.layer.hidden === true };
}

function countSummaries(summaries) {
  return summaries.reduce((total, summary) => total + 1 + countSummaries(summary.children || []), 0);
}

function drawRecord(context, record) {
  if (record.children) {
    if (!record.hidden) record.children.forEach((child) => drawRecord(context, child));
    return;
  }
  if (record.layer.hidden) return;
  context.drawImage(record.canvas, record.layer.left, record.layer.top);
}

function psdLayer(record) {
  if (!record.children) return record.layer;
  return {
    name: record.name,
    hidden: record.hidden === true,
    opened: false,
    children: record.children.map(psdLayer),
  };
}

function countRecords(records, predicate, parentHidden = false) {
  return records.reduce((count, record) => {
    const hidden = parentHidden || (record.children ? record.hidden === true : record.layer.hidden === true);
    return count + (record.children ? countRecords(record.children, predicate, hidden) : Number(predicate(record, hidden)));
  }, 0);
}

async function build(manifestPath) {
  if (!path.isAbsolute(manifestPath)) fail('manifest path must be absolute');
  const manifestDirectory = path.dirname(manifestPath);
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  const width = positiveInteger(manifest.width, 'width');
  const height = positiveInteger(manifest.height, 'height');
  if (width > MAX_DIMENSION || height > MAX_DIMENSION) fail(`canvas exceeds ${MAX_DIMENSION}px PSD limit`);
  const rasters = Array.isArray(manifest.rasters) ? manifest.rasters : [];
  const texts = Array.isArray(manifest.texts) ? manifest.texts : [];
  const textRegions = Array.isArray(manifest.text_regions) ? manifest.text_regions : [];
  const shapes = Array.isArray(manifest.shapes) ? manifest.shapes : [];
  const warnings = Array.isArray(manifest.warnings) ? manifest.warnings.map(String) : [];

  const groups = new Map(GROUPS.map((group) => [group.key, { ...group, children: [] }]));
  groups.get('source').children.push(await loadRaster(
    relativeImagePath(manifestDirectory, manifest.original, 'original'),
    '原始图像',
    0,
    0,
    { width, height },
  ));
  groups.get('background').children.push(await loadRaster(
    relativeImagePath(manifestDirectory, manifest.background, 'background'),
    '重建背景',
    0,
    0,
  ));

  for (let index = 0; index < rasters.length; index += 1) {
    const specification = rasters[index];
    if (!['products', 'decor', 'light'].includes(specification.group)) {
      fail(`rasters[${index}].group must be products, decor, or light`);
    }
    const name = typeof specification.name === 'string' && specification.name ? specification.name : `像素图层 ${index + 1}`;
    const left = finiteNumber(specification.left, `rasters[${index}].left`);
    const top = finiteNumber(specification.top, `rasters[${index}].top`);
    groups.get(specification.group).children.push(await loadRaster(
      relativeImagePath(manifestDirectory, specification.file, `rasters[${index}].file`),
      name,
      left,
      top,
      { hidden: specification.hidden === true, editableStroke: specification.group === 'products' || specification.editableStroke === true },
    ));
  }
  shapes.forEach((specification, index) => groups.get('decor').children.push(buildShape(specification, index)));
  texts.forEach((specification, index) => groups.get('text').children.push(buildText(specification, index)));

  for (let index = 0; index < textRegions.length; index += 1) {
    const specification = textRegions[index];
    const containsBackground = specification.contains_background === true;
    let regionName = specification.name || `文字区域 ${index + 1}`;
    if (containsBackground && !regionName.includes('含背景')) regionName += '｜保真原区域（含背景、需精修）';
    const region = {name: regionName, hidden: specification.hidden === true,
      contains_background: containsBackground, children: []};
    const left = finiteNumber(specification.left ?? 0, `text_regions[${index}].left`);
    const top = finiteNumber(specification.top ?? 0, `text_regions[${index}].top`);
    const appearance = Array.isArray(specification.rasters) ? specification.rasters : [];
    for (let rasterIndex = 0; rasterIndex < appearance.length; rasterIndex += 1) {
      const raster = appearance[rasterIndex];
      const label = `text_regions[${index}].rasters[${rasterIndex}]`;
      region.children.push(await loadRaster(
        relativeImagePath(manifestDirectory, raster.file, `${label}.file`),
        raster.name || (containsBackground ? '保真原区域（含背景、需精修）' : `文字原貌 ${rasterIndex + 1}`),
        finiteNumber(raster.left ?? left, `${label}.left`),
        finiteNumber(raster.top ?? top, `${label}.top`),
        {hidden: raster.hidden === true, editableStroke: !containsBackground && raster.editableStroke !== false},
      ));
    }
    if (specification.alternative) {
      const hidden = specification.alternativeHidden !== false;
      const record = buildText({
        ...specification.alternative,
        name: hidden ? '可改字替代（近似字体，默认隐藏）' : '可改字替代（近似字体）',
        hidden,
      }, texts.length + index);
      record.isAlternative = true;
      region.children.push(record);
    }
    groups.get('text').children.push(region);
  }

  const orderedGroups = GROUPS.map(({ key }) => groups.get(key));
  const children = orderedGroups.map(psdLayer);

  let id = 1;
  function assignIds(layers) {
    for (const layer of layers) {
      layer.id = id++;
      if (layer.children) assignIds(layer.children);
    }
  }
  assignIds(children);

  const preview = createCanvas(width, height);
  const previewContext = preview.getContext('2d');
  for (const group of orderedGroups.slice(1)) {
    for (const record of group.children) drawRecord(previewContext, record);
  }
  const psd = {
    width,
    height,
    imageData: imageData(preview),
    children,
    imageResources: {
      resolutionInfo: {
        horizontalResolution: 72,
        horizontalResolutionUnit: 'PPI',
        widthUnit: 'Pixels',
        verticalResolution: 72,
        verticalResolutionUnit: 'PPI',
        heightUnit: 'Pixels',
      },
    },
  };
  const summaries = orderedGroups.map(makeSummary);
  const statistics = {
    width,
    height,
    text_count: countRecords(orderedGroups, (record, hidden) => record.kind === 'text' && !hidden),
    text_alternative_count: countRecords(orderedGroups, (record) => record.isAlternative === true),
    text_region_count: textRegions.filter((region) => region.rasters?.length && region.contains_background !== true).length,
    text_preserved_count: textRegions.filter((region) => region.rasters?.length && region.contains_background === true).length,
    shared_text_count: textRegions.filter((region) => !region.rasters?.length).length,
    shape_count: shapes.length,
    pixel_count: countRecords(orderedGroups, (record) => record.kind === 'pixel'),
    layer_count: countSummaries(summaries),
    warnings,
  };

  fs.writeFileSync(path.join(manifestDirectory, 'result.psd'), agPsd.writePsdBuffer(psd, {
    generateThumbnail: true,
    invalidateTextLayers: true,
    trimImageData: false,
    noBackground: true,
  }));
  fs.writeFileSync(path.join(manifestDirectory, 'preview.png'), preview.toBuffer('image/png'));
  fs.writeFileSync(path.join(manifestDirectory, 'layers.json'), `${JSON.stringify(summaries, null, 2)}\n`, 'utf8');
  fs.writeFileSync(path.join(manifestDirectory, 'psd_stats.json'), `${JSON.stringify(statistics, null, 2)}\n`, 'utf8');
  return statistics;
}

async function main() {
  if (process.argv.length !== 3) fail('usage: node psd_writer.cjs <absolute-manifest.json>');
  const statistics = await build(process.argv[2]);
  process.stdout.write(`${JSON.stringify(statistics)}\n`);
}

if (require.main === module) {
  main().catch((error) => {
    process.stderr.write(`PSD writer error: ${error.message}\n`);
    process.exitCode = 1;
  });
}

module.exports = { build };
