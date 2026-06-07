// SAM 2 Neuron Segmentation — Professor presentation
// Run with: node build.js
const path = require("path");
const GLOBAL = "/opt/homebrew/lib/node_modules";
const pptxgen = require(path.join(GLOBAL, "pptxgenjs"));
const React = require(path.join(GLOBAL, "react"));
const ReactDOMServer = require(path.join(GLOBAL, "react-dom/server"));
const sharp = require(path.join(GLOBAL, "sharp"));
const FA = require(path.join(GLOBAL, "react-icons/fa"));

// ---------- palette (fluorescence-microscopy inspired) ----------
const C = {
  bg:      "0A1230",   // deep midnight navy (dominant)
  bg2:     "111B40",   // panel navy
  ink:     "0A1230",   // dark text on light
  paper:   "F7F9FC",   // light content background
  card:    "FFFFFF",   // card white
  cardAlt: "EEF2FB",   // subtle card tint
  cyan:    "22D3EE",   // primary accent (electric cyan)
  teal:    "2DD4BF",   // supporting
  violet:  "A855F7",   // sharp accent (magenta-violet)
  amber:   "FBBF24",   // highlight
  muted:   "64748B",   // muted gray
  mutedLt: "94A3B8",   // muted light on dark
  line:    "D8E0EE",   // hairline on light
  good:    "34D399",
};

const W = 13.333, H = 7.5;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";  // built-in 13.333 x 7.5 — maximal PowerPoint/Keynote compatibility
pres.author = "Adi Oizerovich";
pres.title = "Fine-Tuning SAM 2 for Neuron Segmentation";

// ---------- icon helper ----------
function svgFor(Icon, color, size = 256) {
  return ReactDOMServer.renderToStaticMarkup(
    React.createElement(Icon, { color, size: String(size) })
  );
}
async function icon(Icon, color, size = 256) {
  const svg = svgFor(Icon, color, size);
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}
const shadow = () => ({ type: "outer", color: "0A1230", blur: 9, offset: 3, angle: 90, opacity: 0.16 });

// ---------- shared chrome ----------
function dotMotif(slide, x, y, color) {
  // repeated visual motif: a small "neuron node" — filled dot + ring
  slide.addShape(pres.shapes.OVAL, { x: x - 0.02, y: y - 0.02, w: 0.2, h: 0.2, fill: { color }, line: { type: "none" } });
}

let pageNo = 0;
function footer(slide, sectionLabel) {
  pageNo++;
  slide.addText("SAM 2 · Neuron Segmentation", {
    x: 0.55, y: H - 0.42, w: 6, h: 0.3, fontSize: 9, color: C.muted, fontFace: "Calibri", align: "left",
  });
  slide.addText(sectionLabel || "", {
    x: W - 4.55, y: H - 0.42, w: 3.4, h: 0.3, fontSize: 9, color: C.muted, fontFace: "Calibri", align: "right",
  });
  slide.addText(String(pageNo), {
    x: W - 1.0, y: H - 0.42, w: 0.5, h: 0.3, fontSize: 9, color: C.muted, fontFace: "Calibri", align: "right",
  });
}

// content-slide title block (light slides)
function head(slide, kicker, title) {
  slide.background = { color: C.paper };
  slide.addText(kicker.toUpperCase(), {
    x: 0.55, y: 0.42, w: 11, h: 0.3, fontSize: 12, bold: true, color: C.cyan,
    charSpacing: 3, fontFace: "Consolas",
  });
  slide.addText(title, {
    x: 0.52, y: 0.72, w: 12.2, h: 0.85, fontSize: 30, bold: true, color: C.ink,
    fontFace: "Georgia", margin: 0,
  });
  // motif: three nodes top-right
  dotMotif(slide, W - 1.35, 0.66, C.cyan);
  dotMotif(slide, W - 1.0, 0.66, C.violet);
  dotMotif(slide, W - 0.65, 0.66, C.teal);
}

// card helper
function card(slide, x, y, w, h, fill) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h, fill: { color: fill || C.card }, line: { color: C.line, width: 1 }, shadow: shadow(),
  });
}

// dark section divider
function divider(slide, num, title, sub, iconData) {
  slide.background = { color: C.bg };
  // faint big number watermark
  slide.addText(num, {
    x: 8.4, y: 0.2, w: 5, h: 7, fontSize: 320, bold: true, color: C.bg2, fontFace: "Arial Black", align: "right", valign: "middle",
  });
  slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 3.0, w: 0.22, h: 1.5, fill: { color: C.cyan }, line: { type: "none" } });
  slide.addText(`SECTION ${num}`, {
    x: 0.7, y: 2.95, w: 8, h: 0.4, fontSize: 14, bold: true, color: C.cyan, charSpacing: 4, fontFace: "Consolas",
  });
  slide.addText(title, {
    x: 0.66, y: 3.35, w: 9, h: 1.1, fontSize: 44, bold: true, color: "FFFFFF", fontFace: "Georgia", margin: 0,
  });
  if (sub) slide.addText(sub, {
    x: 0.7, y: 4.5, w: 8.2, h: 0.7, fontSize: 15, color: C.mutedLt, fontFace: "Calibri",
  });
  if (iconData) slide.addImage({ data: iconData, x: 0.7, y: 1.55, w: 0.95, h: 0.95 });
}

// ============================================================
(async () => {
  // pre-render icons
  const icBrain   = await icon(FA.FaBrain, "#22D3EE");
  const icMicro   = await icon(FA.FaMicroscope, "#22D3EE");
  const icLayer   = await icon(FA.FaLayerGroup, "#22D3EE");
  const icRocket  = await icon(FA.FaRocket, "#22D3EE");
  const icDraw    = await icon(FA.FaPenNib, "#22D3EE");
  const icFlask   = await icon(FA.FaFlask, "#A855F7");
  const icSitemap = await icon(FA.FaSitemap, "#A855F7");
  const icSync    = await icon(FA.FaSyncAlt, "#A855F7");
  const icBolt    = await icon(FA.FaBolt, "#FBBF24");
  const icCheck   = await icon(FA.FaCheckCircle, "#34D399");
  const icTimes   = await icon(FA.FaTimesCircle, "#F87171");
  const icBook    = await icon(FA.FaBookOpen, "#A855F7");
  const icCubes   = await icon(FA.FaCubes, "#22D3EE");
  const icChart   = await icon(FA.FaChartLine, "#22D3EE");
  const icScale   = await icon(FA.FaBalanceScale, "#A855F7");
  const icDatabase= await icon(FA.FaDatabase, "#22D3EE");
  const icShield  = await icon(FA.FaShieldAlt, "#A855F7");
  const icProject = await icon(FA.FaProjectDiagram, "#22D3EE");
  const icFunc    = await icon(FA.FaSuperscript, "#22D3EE");
  const icWave    = await icon(FA.FaWaveSquare, "#A855F7");

  // ---------------------------------------------------------
  // 1. TITLE
  // ---------------------------------------------------------
  let s = pres.addSlide();
  s.background = { color: C.bg };
  // decorative neuron-network motif: nodes + connecting lines (right side)
  const nodes = [
    [9.7, 1.5], [11.4, 2.2], [10.3, 3.4], [12.0, 4.1], [9.9, 5.2], [11.6, 5.6], [12.6, 3.0],
  ];
  const edges = [[0,1],[0,2],[1,2],[1,6],[2,3],[3,6],[2,4],[3,5],[4,5]];
  edges.forEach(([a,b]) => {
    const [x1,y1] = nodes[a], [x2,y2] = nodes[b];
    s.addShape(pres.shapes.LINE, { x: Math.min(x1,x2)+0.09, y: Math.min(y1,y2)+0.09, w: Math.abs(x2-x1), h: Math.abs(y2-y1),
      line: { color: C.bg2, width: 1.5, beginArrowType: "none", endArrowType: "none" }, flipH: (x2<x1) !== (y2<y1) ? false : false });
  });
  // draw edges properly with flip handling
  nodes.forEach((n, i) => {
    const col = i % 3 === 0 ? C.cyan : (i % 3 === 1 ? C.violet : C.teal);
    s.addShape(pres.shapes.OVAL, { x: n[0], y: n[1], w: 0.18, h: 0.18, fill: { color: col }, line: { type: "none" } });
    s.addShape(pres.shapes.OVAL, { x: n[0]-0.12, y: n[1]-0.12, w: 0.42, h: 0.42, fill: { type: "none" }, line: { color: col, width: 1, transparency: 40 } });
  });

  s.addImage({ data: icBrain, x: 0.7, y: 1.35, w: 0.85, h: 0.85 });
  s.addText("MASTER'S THESIS · COMPUTER VISION FOR MICROSCOPY", {
    x: 0.72, y: 2.35, w: 8.5, h: 0.35, fontSize: 13, bold: true, color: C.cyan, charSpacing: 2, fontFace: "Consolas",
  });
  s.addText("Fine-Tuning SAM 2 for\nNeuron Segmentation", {
    x: 0.66, y: 2.75, w: 8.6, h: 1.9, fontSize: 46, bold: true, color: "FFFFFF", fontFace: "Georgia", lineSpacingMultiple: 1.0, margin: 0,
  });
  s.addText("Parameter-efficient, topology-aware adaptation of a video foundation model for tracking thin, branching neurons in microscopy.", {
    x: 0.7, y: 4.75, w: 8.0, h: 0.9, fontSize: 15, color: C.mutedLt, fontFace: "Calibri", italic: true,
  });
  s.addShape(pres.shapes.LINE, { x: 0.72, y: 5.75, w: 4.2, h: 0, line: { color: C.violet, width: 2 } });
  s.addText([
    { text: "Adi Oizerovich", options: { bold: true, color: "FFFFFF", fontSize: 14, breakLine: true } },
    { text: "Best model: validation IoU 0.8401 · Dice 0.9117", options: { color: C.cyan, fontSize: 12 } },
  ], { x: 0.72, y: 5.9, w: 8, h: 0.8, fontFace: "Calibri" });

  // ---------------------------------------------------------
  // 2. AGENDA
  // ---------------------------------------------------------
  s = pres.addSlide();
  head(s, "Overview", "What this talk covers");
  const agenda = [
    ["01", "Introduction", "The biology problem and the human-in-the-loop labeling workflow.", icMicro],
    ["02", "The napari labeling app", "The interactive GUI scientists use to label and propagate masks.", icDraw],
    ["03", "The fine-tuned model", "Architecture, math, the run-by-run journey, and why we beat base SAM 2.", icBrain],
    ["04", "Next stages", "Where the project goes from here — more data is the next lever.", icRocket],
  ];
  let ay = 1.85;
  agenda.forEach(([n, t, d, ic], i) => {
    card(s, 0.55, ay, 12.2, 1.05);
    s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: ay, w: 0.12, h: 1.05, fill: { color: i === 2 ? C.violet : C.cyan }, line: { type: "none" } });
    s.addText(n, { x: 0.85, y: ay, w: 1.0, h: 1.05, fontSize: 34, bold: true, color: i === 2 ? C.violet : C.cyan, fontFace: "Arial Black", valign: "middle" });
    s.addImage({ data: ic, x: 2.0, y: ay + 0.3, w: 0.45, h: 0.45 });
    s.addText([
      { text: t, options: { bold: true, fontSize: 18, color: C.ink, breakLine: true } },
      { text: d, options: { fontSize: 12.5, color: C.muted } },
    ], { x: 2.7, y: ay + 0.12, w: 9.6, h: 0.85, fontFace: "Calibri", valign: "middle" });
    if (i === 2) s.addText("MAIN", { x: 11.3, y: ay + 0.34, w: 1.2, h: 0.38, fontSize: 11, bold: true, color: "FFFFFF", fill: { color: C.violet }, align: "center", valign: "middle", charSpacing: 1 });
    ay += 1.22;
  });
  footer(s, "Overview");

  // ---------------------------------------------------------
  // 3. SECTION 1 DIVIDER
  // ---------------------------------------------------------
  s = pres.addSlide();
  divider(s, "01", "Introduction", "The problem, and the labeling loop that gives us training data.", icMicro);

  // ---------------------------------------------------------
  // 4. THE PROBLEM
  // ---------------------------------------------------------
  s = pres.addSlide();
  head(s, "Introduction", "The problem: segmenting neurons in microscopy");
  // left: text
  card(s, 0.55, 1.8, 6.1, 4.9);
  s.addImage({ data: icMicro, x: 0.85, y: 2.05, w: 0.5, h: 0.5 });
  s.addText("Why it is hard", { x: 1.5, y: 2.1, w: 4.8, h: 0.45, fontSize: 18, bold: true, color: C.ink, fontFace: "Calibri", valign: "middle" });
  const probs = [
    ["Thin, branching morphology", "Dendrites and axons are 1–2 px wide and tree-like; a single pixel gap breaks a branch."],
    ["Low contrast", "Phase-contrast / fluorescence frames are noisy with weak edges and uneven illumination."],
    ["Temporal tracking", "The same neuron must be followed across many frames as it shifts and deforms."],
    ["Tiny labeled datasets", "Expert pixel-level masks are scarce — we work with ~20 annotated videos."],
  ];
  let py = 2.7;
  probs.forEach(([t, d]) => {
    s.addShape(pres.shapes.OVAL, { x: 0.9, y: py + 0.05, w: 0.13, h: 0.13, fill: { color: C.violet }, line: { type: "none" } });
    s.addText([
      { text: t + " — ", options: { bold: true, color: C.ink } },
      { text: d, options: { color: C.muted } },
    ], { x: 1.18, y: py - 0.06, w: 5.3, h: 0.95, fontSize: 12.5, fontFace: "Calibri", valign: "top" });
    py += 0.98;
  });
  // right: callout + image placeholder
  card(s, 6.85, 1.8, 5.9, 2.25, C.bg);
  s.addText("THE GOAL", { x: 7.15, y: 2.0, w: 5, h: 0.3, fontSize: 11, bold: true, color: C.cyan, charSpacing: 3, fontFace: "Consolas" });
  s.addText("A model that turns a few clicks into accurate, connected, frame-to-frame neuron masks — fast enough to use interactively.", {
    x: 7.15, y: 2.35, w: 5.3, h: 1.5, fontSize: 15, color: "FFFFFF", fontFace: "Calibri", valign: "top",
  });
  // image placeholder
  s.addShape(pres.shapes.RECTANGLE, { x: 6.85, y: 4.25, w: 5.9, h: 2.45, fill: { color: C.cardAlt }, line: { color: C.cyan, width: 1.25, dashType: "dash" } });
  s.addText("[ Visual example: microscopy frame + neuron mask ]", {
    x: 6.95, y: 5.2, w: 5.7, h: 0.5, fontSize: 12, italic: true, color: C.muted, align: "center", fontFace: "Calibri",
  });
  footer(s, "Introduction");

  // ---------------------------------------------------------
  // 5. HUMAN-IN-THE-LOOP WORKFLOW
  // ---------------------------------------------------------
  s = pres.addSlide();
  head(s, "Introduction", "The human-in-the-loop labeling workflow");
  s.addText("Expert effort is the bottleneck. The fine-tuned model does the heavy lifting; the scientist only corrects — and every correction becomes new training data.", {
    x: 0.55, y: 1.7, w: 12.2, h: 0.6, fontSize: 14, color: C.muted, fontFace: "Calibri", italic: true,
  });
  const steps = [
    ["1", "Prompt", "Scientist clicks / draws on one frame to mark a neuron.", icDraw],
    ["2", "Propagate", "The fine-tuned SAM 2 tracks the mask across all frames.", icSync],
    ["3", "Correct", "Scientist fixes the few frames the model got wrong.", icCheck],
    ["4", "Re-train", "Corrected masks feed back into the dataset → better model.", icBrain],
  ];
  const cw = 2.85, gap = 0.32, startX = 0.55, cy = 2.65, ch = 2.7;
  steps.forEach(([n, t, d, ic], i) => {
    const x = startX + i * (cw + gap);
    card(s, x, cy, cw, ch);
    s.addShape(pres.shapes.RECTANGLE, { x, y: cy, w: cw, h: 0.12, fill: { color: i === 3 ? C.violet : C.cyan }, line: { type: "none" } });
    s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: cy + 0.35, w: 0.7, h: 0.7, fill: { color: C.bg }, line: { type: "none" } });
    s.addText(n, { x: x + 0.3, y: cy + 0.35, w: 0.7, h: 0.7, fontSize: 22, bold: true, color: C.cyan, align: "center", valign: "middle", fontFace: "Arial Black" });
    s.addImage({ data: ic, x: x + cw - 0.95, y: cy + 0.4, w: 0.55, h: 0.55 });
    s.addText(t, { x: x + 0.3, y: cy + 1.25, w: cw - 0.6, h: 0.4, fontSize: 17, bold: true, color: C.ink, fontFace: "Calibri" });
    s.addText(d, { x: x + 0.3, y: cy + 1.65, w: cw - 0.6, h: 1.0, fontSize: 12, color: C.muted, fontFace: "Calibri" });
    if (i < 3) s.addText("›", { x: x + cw + 0.01, y: cy + 0.9, w: gap, h: 0.7, fontSize: 30, bold: true, color: C.mutedLt, align: "center", valign: "middle" });
  });
  // loop-back arrow note
  s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: cy + ch + 0.35, w: 12.2, h: 0.72, fill: { color: C.bg } });
  s.addImage({ data: icSync, x: 0.85, y: cy + ch + 0.5, w: 0.42, h: 0.42 });
  s.addText([
    { text: "The virtuous loop:  ", options: { bold: true, color: C.cyan } },
    { text: "a better model means fewer corrections, which means cheaper labeling, which means more data — which means an even better model.", options: { color: "FFFFFF" } },
  ], { x: 1.45, y: cy + ch + 0.36, w: 11.1, h: 0.72, fontSize: 13, fontFace: "Calibri", valign: "middle" });
  footer(s, "Introduction");

  // ---------------------------------------------------------
  // 6. SECTION 2 DIVIDER
  // ---------------------------------------------------------
  s = pres.addSlide();
  divider(s, "02", "The napari Labeling App", "An interactive GUI where the fine-tuned model meets the scientist.", icDraw);

  // ---------------------------------------------------------
  // 7. NAPARI APP
  // ---------------------------------------------------------
  s = pres.addSlide();
  head(s, "The Labeling App", "neurons-labeling: a napari plugin");
  // left features
  const feats = [
    ["Load a video", "Open a microscopy stack as a napari image layer.", icMicro],
    ["Prompt a neuron", "Click points or draw to seed a mask on one frame.", icDraw],
    ["Propagate", "Run the fine-tuned SAM 2 to track the neuron across frames.", icSync],
    ["Correct & export", "Edit masks in-place and export labels for training.", icCheck],
  ];
  let fy = 1.85;
  feats.forEach(([t, d, ic]) => {
    card(s, 0.55, fy, 7.0, 1.05);
    s.addShape(pres.shapes.OVAL, { x: 0.8, y: fy + 0.27, w: 0.5, h: 0.5, fill: { color: C.cardAlt }, line: { type: "none" } });
    s.addImage({ data: ic, x: 0.88, y: fy + 0.35, w: 0.34, h: 0.34 });
    s.addText([
      { text: t, options: { bold: true, fontSize: 15, color: C.ink, breakLine: true } },
      { text: d, options: { fontSize: 11.5, color: C.muted } },
    ], { x: 1.55, y: fy + 0.13, w: 5.85, h: 0.8, fontFace: "Calibri", valign: "middle" });
    fy += 1.16;
  });
  // right: tech note + placeholder
  card(s, 7.75, 1.85, 5.0, 2.05, C.bg);
  s.addImage({ data: icLayer, x: 8.0, y: 2.05, w: 0.45, h: 0.45 });
  s.addText("Smart checkpoint loading", { x: 8.55, y: 2.05, w: 4.1, h: 0.45, fontSize: 14, bold: true, color: "FFFFFF", valign: "middle", fontFace: "Calibri" });
  s.addText("The app auto-detects the LoRA configuration from the checkpoint and applies both adapter passes (main + Hiera last-stage), so the GUI runs the exact fine-tuned model — no manual setup.", {
    x: 8.0, y: 2.55, w: 4.55, h: 1.3, fontSize: 12, color: C.mutedLt, fontFace: "Calibri",
  });
  s.addShape(pres.shapes.RECTANGLE, { x: 7.75, y: 4.1, w: 5.0, h: 2.6, fill: { color: C.cardAlt }, line: { color: C.cyan, width: 1.25, dashType: "dash" } });
  s.addText("[ Screenshot: napari GUI with neuron masks ]", { x: 7.85, y: 5.25, w: 4.8, h: 0.5, fontSize: 12, italic: true, color: C.muted, align: "center", fontFace: "Calibri" });
  footer(s, "The Labeling App");

  // ---------------------------------------------------------
  // 8. SECTION 3 DIVIDER
  // ---------------------------------------------------------
  s = pres.addSlide();
  divider(s, "03", "The Fine-Tuned Model", "Architecture · math · the journey · why we beat base SAM 2.", icBrain);

  // ---------------------------------------------------------
  // 9. SAM 2 ARCHITECTURE
  // ---------------------------------------------------------
  s = pres.addSlide();
  head(s, "The Model", "SAM 2 architecture at a glance");
  s.addText("SAM 2 (Ravi et al., 2024) is a promptable video segmentation foundation model. Four modules turn a prompt on one frame into masks across a whole clip.", {
    x: 0.55, y: 1.65, w: 12.2, h: 0.55, fontSize: 13.5, color: C.muted, fontFace: "Calibri", italic: true,
  });
  const mods = [
    ["Hiera image encoder", "Hierarchical ViT. 1024×1024 frame → 64×64×256 feature map.", "FROZEN", icCubes],
    ["Memory attention", "Conditions the current frame on features & masks from past frames.", "LoRA", icSync],
    ["Mask decoder", "Lightweight transformer that predicts the mask + an IoU score.", "LoRA", icSitemap],
    ["Memory encoder", "Writes each prediction into a memory bank for future frames.", "FROZEN", icDatabase],
  ];
  const mcw = 2.95, mgap = 0.3, mx0 = 0.55, my = 2.45, mh = 3.0;
  mods.forEach(([t, d, tag, ic], i) => {
    const x = mx0 + i * (mcw + mgap);
    card(s, x, my, mcw, mh);
    const tagCol = tag === "LoRA" ? C.violet : C.muted;
    s.addShape(pres.shapes.RECTANGLE, { x, y: my, w: mcw, h: 0.12, fill: { color: tagCol }, line: { type: "none" } });
    s.addImage({ data: ic, x: x + 0.3, y: my + 0.32, w: 0.55, h: 0.55 });
    s.addText(t, { x: x + 0.28, y: my + 1.0, w: mcw - 0.56, h: 0.5, fontSize: 15, bold: true, color: C.ink, fontFace: "Calibri" });
    s.addText(d, { x: x + 0.28, y: my + 1.5, w: mcw - 0.56, h: 0.9, fontSize: 11.5, color: C.muted, fontFace: "Calibri", valign: "top" });
    s.addText(tag, { x: x + 0.28, y: my + mh - 0.45, w: 1.7, h: 0.32, fontSize: 10, bold: true, color: "FFFFFF", fill: { color: tagCol }, align: "center", valign: "middle", charSpacing: 1 });
    if (i < 3) s.addText("→", { x: x + mcw - 0.02, y: my + 1.0, w: mgap, h: 0.6, fontSize: 22, bold: true, color: C.mutedLt, align: "center", valign: "middle" });
  });
  s.addText([
    { text: "Our strategy:  ", options: { bold: true, color: C.violet } },
    { text: "freeze the expensive pretrained encoders; inject small trainable LoRA adapters only where adaptation matters (decoder + memory attention).", options: { color: C.ink } },
  ], { x: 0.55, y: my + mh + 0.25, w: 12.2, h: 0.5, fontSize: 13, fontFace: "Calibri" });
  footer(s, "The Model");

  // ---------------------------------------------------------
  // 10. WHY NOT BASE SAM 2 (domain gap)
  // ---------------------------------------------------------
  s = pres.addSlide();
  head(s, "The Model", "Why not just use Meta's SAM 2?");
  s.addText("Base SAM 2 was trained on natural video (people, cars, animals). It has never seen phase-contrast microscopy — so on neurons it fails in characteristic ways.", {
    x: 0.55, y: 1.65, w: 12.2, h: 0.55, fontSize: 13.5, color: C.muted, italic: true, fontFace: "Calibri",
  });
  // two columns: base fails vs ours
  card(s, 0.55, 2.4, 6.0, 4.0, C.card);
  s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 2.4, w: 6.0, h: 0.62, fill: { color: "F87171" }, line: { type: "none" } });
  s.addImage({ data: icTimes, x: 0.78, y: 2.5, w: 0.42, h: 0.42 });
  s.addText("Base SAM 2 on neurons", { x: 1.3, y: 2.4, w: 5, h: 0.62, fontSize: 16, bold: true, color: "FFFFFF", valign: "middle", fontFace: "Calibri" });
  const bad = [
    "Loses thin dendrites — captures only the bright cell body (soma).",
    "Mask bleeds across weak, low-contrast edges into the background.",
    "Hallucinates objects on noisy texture; unstable across frames.",
    "No notion of connectivity — branches come out fragmented.",
  ];
  let by = 3.2;
  bad.forEach((t) => {
    s.addShape(pres.shapes.OVAL, { x: 0.85, y: by + 0.06, w: 0.12, h: 0.12, fill: { color: "F87171" }, line: { type: "none" } });
    s.addText(t, { x: 1.1, y: by - 0.05, w: 5.25, h: 0.7, fontSize: 12.5, color: C.ink, fontFace: "Calibri" });
    by += 0.78;
  });
  card(s, 6.75, 2.4, 6.0, 4.0, C.card);
  s.addShape(pres.shapes.RECTANGLE, { x: 6.75, y: 2.4, w: 6.0, h: 0.62, fill: { color: C.good }, line: { type: "none" } });
  s.addImage({ data: icCheck, x: 6.98, y: 2.5, w: 0.42, h: 0.42 });
  s.addText("Our fine-tuned model", { x: 7.5, y: 2.4, w: 5, h: 0.62, fontSize: 16, bold: true, color: "FFFFFF", valign: "middle", fontFace: "Calibri" });
  const good = [
    "LoRA adapts features to microscopy while keeping pretrained knowledge.",
    "Topology-aware cbDice loss preserves thin tubular branch structure.",
    "Supervoxel loss penalises fragmentation → connected masks.",
    "Dataset-specific normalization matches the microscopy intensity stats.",
  ];
  let gy = 3.2;
  good.forEach((t) => {
    s.addShape(pres.shapes.OVAL, { x: 7.05, y: gy + 0.06, w: 0.12, h: 0.12, fill: { color: C.good }, line: { type: "none" } });
    s.addText(t, { x: 7.3, y: gy - 0.05, w: 5.25, h: 0.7, fontSize: 12.5, color: C.ink, fontFace: "Calibri" });
    gy += 0.78;
  });
  footer(s, "The Model");

  // ---------------------------------------------------------
  // 11. FOUNDATIONS (papers)
  // ---------------------------------------------------------
  s = pres.addSlide();
  head(s, "The Model", "Foundations: what we build on");
  s.addImage({ data: icBook, x: 0.55, y: 1.62, w: 0.5, h: 0.5 });
  s.addText("Every design choice is grounded in published work.", { x: 1.15, y: 1.62, w: 11, h: 0.5, fontSize: 14, color: C.muted, italic: true, valign: "middle", fontFace: "Calibri" });
  const papers = [
    ["SAM 2", "Ravi et al., 2024", "Promptable video segmentation backbone"],
    ["LoRA", "Hu et al., ICLR 2022", "Low-rank parameter-efficient fine-tuning"],
    ["LoRA+", "Hayou et al., ICML 2024", "Asymmetric A/B learning rates"],
    ["Hiera", "Ryali et al., ICML 2023", "Hierarchical ViT image encoder"],
    ["Focal loss", "Lin et al., ICCV 2017", "Class-imbalance robust loss"],
    ["clDice", "Shit et al., CVPR 2021", "Centerline (skeleton) Dice"],
    ["cbDice", "Shi et al., MICCAI 2024", "Boundary-weighted topology loss"],
    ["AdamW", "Loshchilov & Hutter, 2019", "Decoupled weight-decay optimizer"],
    ["GPT-3", "Brown et al., NeurIPS 2020", "Warmup + cosine LR schedule"],
    ["FlashAttention 1–3", "Dao et al., 2022–24", "IO-aware fast attention kernels"],
  ];
  const pcols = 2, pcw = 6.0, pcgap = 0.2, pch = 0.82, px0 = 0.55, pyy0 = 2.2;
  papers.forEach((p, i) => {
    const col = i % pcols, row = Math.floor(i / pcols);
    const x = px0 + col * (pcw + pcgap);
    const y = pyy0 + row * (pch + 0.08);
    card(s, x, y, pcw, pch);
    s.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.1, h: pch, fill: { color: i % 2 ? C.violet : C.cyan }, line: { type: "none" } });
    s.addText([
      { text: p[0], options: { bold: true, fontSize: 14, color: C.ink } },
      { text: "   " + p[1], options: { fontSize: 10.5, color: C.muted, italic: true } },
    ], { x: x + 0.28, y: y + 0.08, w: pcw - 0.4, h: 0.4, fontFace: "Calibri", valign: "middle" });
    s.addText(p[2], { x: x + 0.28, y: y + 0.44, w: pcw - 0.4, h: 0.36, fontSize: 11, color: C.muted, fontFace: "Calibri" });
  });
  footer(s, "The Model");

  // ---------------------------------------------------------
  // 12. THE MATH — LoRA + LoRA+
  // ---------------------------------------------------------
  s = pres.addSlide();
  head(s, "The Math", "LoRA and LoRA+ : the core equations");
  // LoRA card
  card(s, 0.55, 1.8, 6.05, 4.6, C.bg);
  s.addText("LoRA — low-rank adaptation", { x: 0.85, y: 2.0, w: 5.5, h: 0.4, fontSize: 16, bold: true, color: C.cyan, fontFace: "Calibri" });
  s.addText("Freeze W₀; learn a low-rank update:", { x: 0.85, y: 2.5, w: 5.5, h: 0.35, fontSize: 12.5, color: C.mutedLt, fontFace: "Calibri" });
  s.addText("W = W₀ + ΔW = W₀ + (α / r) · B A", {
    x: 0.85, y: 2.9, w: 5.45, h: 0.6, fontSize: 19, bold: true, color: "FFFFFF", fontFace: "Consolas",
  });
  s.addText([
    { text: "A ∈ ℝ", options: {} }, { text: "r×k", options: { fontSize: 9, superscript: true } },
    { text: "  (Kaiming init),   B ∈ ℝ", options: {} }, { text: "d×r", options: { fontSize: 9, superscript: true } },
    { text: "  (zero init)", options: {} },
  ], { x: 0.85, y: 3.55, w: 5.5, h: 0.35, fontSize: 12.5, color: C.mutedLt, fontFace: "Consolas" });
  const loraPts = [
    "B = 0 at init ⇒ ΔW = 0 ⇒ training starts exactly at the pretrained model.",
    "Rank r ≪ d, so we train only a tiny fraction of the weights.",
    "α / r is a fixed scaling that decouples update size from rank.",
  ];
  let lpy = 4.05;
  loraPts.forEach((t) => {
    s.addShape(pres.shapes.OVAL, { x: 0.9, y: lpy + 0.05, w: 0.11, h: 0.11, fill: { color: C.cyan }, line: { type: "none" } });
    s.addText(t, { x: 1.15, y: lpy - 0.05, w: 5.25, h: 0.65, fontSize: 12, color: "FFFFFF", fontFace: "Calibri" });
    lpy += 0.74;
  });
  // LoRA+ card
  card(s, 6.75, 1.8, 6.0, 4.6, C.bg);
  s.addText("LoRA+ — asymmetric learning rates", { x: 7.05, y: 2.0, w: 5.5, h: 0.4, fontSize: 16, bold: true, color: C.violet, fontFace: "Calibri" });
  s.addText("Because B starts at zero and A does not, they need different learning rates:", { x: 7.05, y: 2.5, w: 5.5, h: 0.6, fontSize: 12.5, color: C.mutedLt, fontFace: "Calibri" });
  s.addText("lr_B = λ · lr_A ,    λ = 16", {
    x: 7.05, y: 3.05, w: 5.4, h: 0.55, fontSize: 19, bold: true, color: "FFFFFF", fontFace: "Consolas",
  });
  const lpPts = [
    "Hayou et al. (ICML 2024) prove a higher LR on B gives feature-learning-optimal updates.",
    "We use lr_A = 1e-4 and lr_B = 1.6e-3; non-LoRA params train at 1e-5.",
    "Result: faster, more stable convergence than a single shared LoRA LR.",
  ];
  let lppy = 3.85;
  lpPts.forEach((t) => {
    s.addShape(pres.shapes.OVAL, { x: 7.1, y: lppy + 0.05, w: 0.11, h: 0.11, fill: { color: C.violet }, line: { type: "none" } });
    s.addText(t, { x: 7.35, y: lppy - 0.05, w: 5.2, h: 0.65, fontSize: 12, color: "FFFFFF", fontFace: "Calibri" });
    lppy += 0.78;
  });
  footer(s, "The Math");

  // ---------------------------------------------------------
  // 13. DUAL-RANK LoRA + trainable params
  // ---------------------------------------------------------
  s = pres.addSlide();
  head(s, "The Model", "Dual-rank LoRA: where we adapt");
  // left: two adapter groups
  card(s, 0.55, 1.85, 7.0, 2.15);
  s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 1.85, w: 0.12, h: 2.15, fill: { color: C.cyan }, line: { type: "none" } });
  s.addText("Main LoRA — rank 16, α = 32", { x: 0.85, y: 2.05, w: 6.5, h: 0.4, fontSize: 16, bold: true, color: C.ink, fontFace: "Calibri" });
  s.addText("68 linear layers across the mask decoder + memory attention. This is where promptable segmentation is re-shaped for neurons. Dropout 0.10.", {
    x: 0.85, y: 2.5, w: 6.4, h: 1.3, fontSize: 12.5, color: C.muted, fontFace: "Calibri",
  });
  card(s, 0.55, 4.15, 7.0, 2.25);
  s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 4.15, w: 0.12, h: 2.25, fill: { color: C.violet }, line: { type: "none" } });
  s.addText("Hiera last-stage LoRA — rank 8, α = 16", { x: 0.85, y: 4.35, w: 6.5, h: 0.4, fontSize: 16, bold: true, color: C.ink, fontFace: "Calibri" });
  s.addText("A second adapter pass on the last 2 blocks (10–11) of the frozen Hiera trunk — the deepest, 768-dim features. Lets the visual representation itself adapt to microscopy without unfreezing the encoder. Smaller rank because high-dim layers give more capacity per rank unit (α/r = 2 kept constant).", {
    x: 0.85, y: 4.8, w: 6.4, h: 1.5, fontSize: 12.5, color: C.muted, fontFace: "Calibri",
  });
  // right: stat callout
  card(s, 7.75, 1.85, 5.0, 4.55, C.bg);
  s.addText("TRAINABLE PARAMETERS", { x: 8.0, y: 2.2, w: 4.5, h: 0.35, fontSize: 12, bold: true, color: C.cyan, charSpacing: 2, fontFace: "Consolas", align: "center" });
  s.addText("13.9%", { x: 7.75, y: 2.6, w: 5.0, h: 1.3, fontSize: 80, bold: true, color: "FFFFFF", align: "center", fontFace: "Arial Black" });
  s.addText("5,557,250  of  39,876,930", { x: 7.75, y: 3.95, w: 5.0, h: 0.4, fontSize: 15, color: C.cyan, align: "center", fontFace: "Consolas" });
  s.addShape(pres.shapes.LINE, { x: 8.5, y: 4.5, w: 3.5, h: 0, line: { color: C.bg2, width: 1 } });
  s.addText("Train ~14% of the model, keep 86% of Meta's pretrained knowledge frozen. That is what makes fine-tuning feasible on ~20 videos without overfitting.", {
    x: 8.0, y: 4.7, w: 4.5, h: 1.5, fontSize: 12.5, color: C.mutedLt, align: "center", fontFace: "Calibri",
  });
  footer(s, "The Model");

  // ---------------------------------------------------------
  // 14. LOSS RECIPE
  // ---------------------------------------------------------
  s = pres.addSlide();
  head(s, "The Math", "A topology-aware loss recipe");
  s.addText("Standard mask losses care about area overlap — they ignore whether a thin branch stays connected. We add skeleton- and connectivity-aware terms.", {
    x: 0.55, y: 1.62, w: 12.2, h: 0.5, fontSize: 13, color: C.muted, italic: true, fontFace: "Calibri",
  });
  s.addText("ℒ = 5·ℒ_focal + 2·ℒ_Dice + 5·ℒ_cbDice + 0.3·ℒ_supervoxel + 1·ℒ_IoU + 1·ℒ_class", {
    x: 0.55, y: 2.15, w: 12.2, h: 0.55, fontSize: 16, bold: true, color: C.ink, align: "center", fontFace: "Consolas",
    fill: { color: C.cardAlt },
  });
  const losses = [
    ["Focal", "5.0", "Pixel classification robust to fg/bg imbalance.", C.cyan],
    ["Dice", "2.0", "Region-level overlap of mask vs. ground truth.", C.cyan],
    ["cbDice", "5.0", "Centerline-boundary Dice — preserves skeleton topology.", C.violet],
    ["Supervoxel", "0.3", "Penalises fragmentation; keeps branches connected.", C.violet],
    ["IoU pred.", "1.0", "Supervises the decoder's self-estimated quality score.", C.muted],
    ["Class", "1.0", "Object-presence term; suppresses hallucinated masks.", C.muted],
  ];
  const lcw = 3.95, lgap = 0.22, lx0 = 0.55, lyy = 2.95, lhh = 1.55;
  losses.forEach((l, i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const x = lx0 + col * (lcw + lgap), y = lyy + row * (lhh + 0.2);
    card(s, x, y, lcw, lhh);
    s.addShape(pres.shapes.RECTANGLE, { x, y, w: lcw, h: 0.1, fill: { color: l[3] }, line: { type: "none" } });
    s.addText(l[0], { x: x + 0.25, y: y + 0.18, w: 2.4, h: 0.45, fontSize: 16, bold: true, color: C.ink, fontFace: "Calibri" });
    s.addText("w=" + l[1], { x: x + lcw - 1.45, y: y + 0.2, w: 1.2, h: 0.4, fontSize: 14, bold: true, color: l[3], align: "right", fontFace: "Consolas" });
    s.addText(l[2], { x: x + 0.25, y: y + 0.68, w: lcw - 0.5, h: 0.8, fontSize: 11.5, color: C.muted, fontFace: "Calibri" });
  });
  s.addText([
    { text: "cbDice math (sketch):  ", options: { bold: true, color: C.violet } },
    { text: "soft-skeletonise the mask, then score skeleton overlap weighted by a Euclidean distance transform — thin structures count more. Topology losses computed at 256×256 (16× faster than 1024, negligible quality loss).", options: { color: C.ink } },
  ], { x: 0.55, y: 6.25, w: 12.2, h: 0.6, fontSize: 11.5, fontFace: "Calibri" });
  footer(s, "The Math");

  // ---------------------------------------------------------
  // 14b. FOCAL & DICE — the pixel/region losses (math)
  // ---------------------------------------------------------
  s = pres.addSlide();
  head(s, "The Math", "Pixel & region losses: Focal + Dice");
  // Focal card
  card(s, 0.55, 1.85, 6.05, 4.55, C.bg);
  s.addImage({ data: icFunc, x: 0.85, y: 2.1, w: 0.45, h: 0.45 });
  s.addText("Focal loss — class imbalance", { x: 1.4, y: 2.1, w: 5, h: 0.45, fontSize: 16, bold: true, color: C.cyan, valign: "middle", fontFace: "Calibri" });
  s.addText("FL(pₜ) = − α (1 − pₜ)^γ · log(pₜ)", { x: 0.85, y: 2.75, w: 5.4, h: 0.5, fontSize: 18, bold: true, color: "FFFFFF", fontFace: "Consolas" });
  s.addText("α = 0.25 ,   γ = 2.0", { x: 0.85, y: 3.25, w: 5.4, h: 0.35, fontSize: 12.5, color: C.cyan, fontFace: "Consolas" });
  const focalPts = [
    "pₜ is the predicted probability of the true class for each pixel.",
    "The (1 − pₜ)^γ term down-weights easy, well-classified background pixels.",
    "Forces learning capacity onto the rare, hard foreground — the thin neuron pixels that dominate the error.",
  ];
  let fpy = 3.8;
  focalPts.forEach((t) => {
    s.addShape(pres.shapes.OVAL, { x: 0.9, y: fpy + 0.05, w: 0.11, h: 0.11, fill: { color: C.cyan }, line: { type: "none" } });
    s.addText(t, { x: 1.15, y: fpy - 0.05, w: 5.25, h: 0.7, fontSize: 12, color: "FFFFFF", fontFace: "Calibri" });
    fpy += 0.8;
  });
  // Dice card
  card(s, 6.75, 1.85, 6.0, 4.55, C.bg);
  s.addImage({ data: icFunc, x: 7.05, y: 2.1, w: 0.45, h: 0.45 });
  s.addText("Dice loss — region overlap", { x: 7.6, y: 2.1, w: 5, h: 0.45, fontSize: 16, bold: true, color: C.violet, valign: "middle", fontFace: "Calibri" });
  s.addText("Dice = 1 − (2 Σ p·g + ε) / (Σ p + Σ g + ε)", { x: 7.05, y: 2.75, w: 5.95, h: 0.5, fontSize: 14, bold: true, color: "FFFFFF", fontFace: "Consolas" });
  s.addText("p = σ(logits) ,   g = ground-truth mask", { x: 7.05, y: 3.25, w: 5.4, h: 0.35, fontSize: 12.5, color: C.violet, fontFace: "Consolas" });
  const dicePts = [
    "A soft, differentiable F1 over the pixel set — directly maximises overlap.",
    "Scale-invariant: insensitive to the foreground/background ratio, so it stays informative on tiny masks.",
    "Complements focal: focal is per-pixel pointwise, Dice is a set-level overlap objective.",
  ];
  let dpy = 3.8;
  dicePts.forEach((t) => {
    s.addShape(pres.shapes.OVAL, { x: 7.1, y: dpy + 0.05, w: 0.11, h: 0.11, fill: { color: C.violet }, line: { type: "none" } });
    s.addText(t, { x: 7.35, y: dpy - 0.05, w: 5.2, h: 0.7, fontSize: 12, color: "FFFFFF", fontFace: "Calibri" });
    dpy += 0.8;
  });
  footer(s, "The Math");

  // ---------------------------------------------------------
  // 14c. TOPOLOGY LOSSES — cbDice & supervoxel (math)
  // ---------------------------------------------------------
  s = pres.addSlide();
  head(s, "The Math", "Topology losses: cbDice & supervoxel");
  s.addText("These are the terms that make a neuron model — they care about connectivity, not just area.", {
    x: 0.55, y: 1.62, w: 12.2, h: 0.45, fontSize: 13, color: C.muted, italic: true, fontFace: "Calibri",
  });
  // cbDice card
  card(s, 0.55, 2.2, 6.05, 4.2, C.card);
  s.addShape(pres.shapes.RECTANGLE, { x: 0.55, y: 2.2, w: 0.12, h: 4.2, fill: { color: C.violet }, line: { type: "none" } });
  s.addText("cbDice  (clDice + boundary weighting)", { x: 0.85, y: 2.4, w: 5.5, h: 0.4, fontSize: 15, bold: true, color: C.ink, fontFace: "Calibri" });
  s.addText("clDice = 2·Tprec·Tsens / (Tprec + Tsens)", { x: 0.85, y: 2.85, w: 5.5, h: 0.4, fontSize: 13.5, bold: true, color: C.violet, fontFace: "Consolas" });
  const cbPts = [
    ["Soft-skeletonise", "iterative min/max pooling (5 iters) gives a differentiable centerline of mask and GT."],
    ["Tprec / Tsens", "skeleton-in-mask and skeleton-in-GT overlaps → a topology precision & sensitivity."],
    ["EDT weighting", "a Euclidean distance transform up-weights thin tubular structures (cbDice over clDice)."],
    ["Why it matters", "a 1-px gap in a dendrite barely moves Dice but snaps the skeleton — cbDice punishes it."],
  ];
  let cby = 3.35;
  cbPts.forEach(([h2, d]) => {
    s.addShape(pres.shapes.OVAL, { x: 0.9, y: cby + 0.05, w: 0.12, h: 0.12, fill: { color: C.violet }, line: { type: "none" } });
    s.addText([
      { text: h2 + " — ", options: { bold: true, color: C.ink } },
      { text: d, options: { color: C.muted } },
    ], { x: 1.18, y: cby - 0.06, w: 5.25, h: 0.7, fontSize: 12, fontFace: "Calibri" });
    cby += 0.74;
  });
  // supervoxel card
  card(s, 6.75, 2.2, 6.0, 4.2, C.card);
  s.addShape(pres.shapes.RECTANGLE, { x: 6.75, y: 2.2, w: 0.12, h: 4.2, fill: { color: C.cyan }, line: { type: "none" } });
  s.addText("Supervoxel  (connectivity preservation)", { x: 7.05, y: 2.4, w: 5.5, h: 0.4, fontSize: 15, bold: true, color: C.ink, fontFace: "Calibri" });
  s.addText("compares connected-component structure of pred vs. GT", { x: 7.05, y: 2.85, w: 5.5, h: 0.4, fontSize: 12.5, bold: true, color: C.cyan, fontFace: "Consolas" });
  const svPts = [
    ["Catches fragmentation", "a gap that clDice tolerates still splits a component — supervoxel sees it."],
    ["Small weight (0.3)", "a gentle structural prior, not a dominant term — it nudges toward single connected masks."],
    ["256² resolution", "connectivity analysis is run at low res: 16× cheaper than 1024², minimal quality loss."],
    ["Honest limit", "at 20 videos it plateaus (~0.05 val) — a data-bound ceiling, not a tuning failure."],
  ];
  let svy = 3.35;
  svPts.forEach(([h2, d]) => {
    s.addShape(pres.shapes.OVAL, { x: 7.1, y: svy + 0.05, w: 0.12, h: 0.12, fill: { color: C.cyan }, line: { type: "none" } });
    s.addText([
      { text: h2 + " — ", options: { bold: true, color: C.ink } },
      { text: d, options: { color: C.muted } },
    ], { x: 7.38, y: svy - 0.06, w: 5.2, h: 0.7, fontSize: 12, fontFace: "Calibri" });
    svy += 0.74;
  });
  footer(s, "The Math");

  // ---------------------------------------------------------
  // 14d. THE TRAINING LOOP
  // ---------------------------------------------------------
  s = pres.addSlide();
  head(s, "Training", "The training loop, step by step");
  s.addText("One step = one batch of 8-frame clips. The model is prompted on frame 0, then must propagate — so it learns genuine tracking, not per-frame segmentation.", {
    x: 0.55, y: 1.62, w: 12.2, h: 0.5, fontSize: 13, color: C.muted, italic: true, fontFace: "Calibri",
  });
  const loopSteps = [
    ["Sample a clip", "Sliding window with clip_stride = 4 over the video's time axis (~17 clips/video)."],
    ["Augment once per clip", "One random spatial + intensity transform applied identically to all 8 frames → temporal consistency; fresh each epoch."],
    ["Condition on frame 0", "num_init_cond_frames = 1: seed the GT mask as a prompt on the first frame only."],
    ["Propagate (Path B)", "track_step runs memory-attention over the memory bank → mask decoder predicts each later frame; memory encoder writes each result back."],
    ["No teacher forcing", "prob_to_use_pt_input = 0: frames 1–7 use the model's own predictions — it must recover from its own errors."],
    ["Weighted multi-frame loss", "the 6 loss terms summed over all frames & objects in the clip."],
    ["Backprop & update", "bf16 autocast forward → backward → clip grad-norm to 1.0 → fused AdamW step (LoRA+ param groups) → LR scheduler step."],
  ];
  const colW2 = 6.0, gap2 = 0.2, x0b = 0.55, y0b = 2.05, rowH2 = 1.04;
  loopSteps.forEach((st, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = x0b + col * (colW2 + gap2), y = y0b + row * (rowH2 + 0.12);
    if (i === 6) { // last item spans full width
      card(s, x0b, y, 12.2, rowH2, C.bg);
      s.addShape(pres.shapes.OVAL, { x: x0b + 0.28, y: y + 0.27, w: 0.5, h: 0.5, fill: { color: C.violet }, line: { type: "none" } });
      s.addText(String(i + 1), { x: x0b + 0.28, y: y + 0.27, w: 0.5, h: 0.5, fontSize: 16, bold: true, color: "FFFFFF", align: "center", valign: "middle", fontFace: "Arial Black" });
      s.addText([
        { text: st[0] + "   ", options: { bold: true, fontSize: 14, color: "FFFFFF" } },
        { text: st[1], options: { fontSize: 12, color: C.mutedLt } },
      ], { x: x0b + 1.0, y: y + 0.12, w: 11.0, h: 0.8, fontFace: "Calibri", valign: "middle" });
      return;
    }
    card(s, x, y, colW2, rowH2);
    s.addShape(pres.shapes.OVAL, { x: x + 0.26, y: y + 0.27, w: 0.5, h: 0.5, fill: { color: C.bg }, line: { type: "none" } });
    s.addText(String(i + 1), { x: x + 0.26, y: y + 0.27, w: 0.5, h: 0.5, fontSize: 16, bold: true, color: C.cyan, align: "center", valign: "middle", fontFace: "Arial Black" });
    s.addText([
      { text: st[0], options: { bold: true, fontSize: 13.5, color: C.ink, breakLine: true } },
      { text: st[1], options: { fontSize: 10.5, color: C.muted } },
    ], { x: x + 0.95, y: y + 0.13, w: colW2 - 1.2, h: 0.82, fontFace: "Calibri", valign: "middle" });
  });
  footer(s, "Training");

  // ---------------------------------------------------------
  // 14e. REGULARIZATION
  // ---------------------------------------------------------
  s = pres.addSlide();
  head(s, "Training", "Regularization: 5.5M params, only 20 videos");
  s.addText("Overfitting is the central risk of fine-tuning on tiny data. Seven overlapping mechanisms keep the model honest.", {
    x: 0.55, y: 1.62, w: 12.2, h: 0.45, fontSize: 13, color: C.muted, italic: true, fontFace: "Calibri",
  });
  const regs = [
    [icScale, "Low-rank constraint", "The update lives in a rank-16 subspace — LoRA itself is a strong structural prior on what can change."],
    [icShield, "Frozen backbone (86%)", "Pretrained features can't drift; only the small adapters move, preserving general knowledge."],
    [icWave, "Dropout 0.10", "Applied inside the LoRA adapters to prevent co-adaptation of the learned directions."],
    [icBolt, "Decoupled weight decay", "AdamW weight decay 0.05 + grad-norm clipping at 1.0 keep updates small and stable."],
    [icProject, "Small batch (2)", "Noisier gradient estimates act as implicit regularization, biasing toward flatter minima."],
    [icSync, "Heavy augmentation", "Flips, ±25° rotation, ±20° shear, scaling, brightness/contrast, blur, noise, elastic — fresh per epoch, effectively enlarging the dataset."],
    [icChart, "Warmup + early stopping", "2-epoch warmup + cosine-to-5% anneal; training halts on stalled validation IoU."],
  ];
  const rcw = 3.95, rgap = 0.22, rx0 = 0.55, ryy = 2.1, rhh = 1.45, rvg = 0.16;
  regs.forEach((r, i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const x = rx0 + col * (rcw + rgap), y = ryy + row * (rhh + rvg);
    card(s, x, y, rcw, rhh);
    s.addShape(pres.shapes.RECTANGLE, { x, y, w: rcw, h: 0.08, fill: { color: i % 2 ? C.violet : C.cyan }, line: { type: "none" } });
    s.addImage({ data: r[0], x: x + 0.26, y: y + 0.24, w: 0.42, h: 0.42 });
    s.addText(r[1], { x: x + 0.82, y: y + 0.2, w: rcw - 1.0, h: 0.5, fontSize: 13, bold: true, color: C.ink, valign: "middle", fontFace: "Calibri" });
    s.addText(r[2], { x: x + 0.26, y: y + 0.72, w: rcw - 0.48, h: 0.66, fontSize: 9.5, color: C.muted, fontFace: "Calibri" });
  });
  // 7th card guidance text in last cell
  s.addText([
    { text: "Together:  ", options: { bold: true, color: C.violet } },
    { text: "a high-capacity model that behaves like a small one — the reason 13.9% trainable params generalise from 20 videos.", options: { color: C.ink } },
  ], { x: 4.72, y: ryy + 2 * (rhh + rvg), w: 8.05, h: rhh, fontSize: 12.5, fontFace: "Calibri", valign: "middle", fill: { color: C.cardAlt } });
  footer(s, "Training");

  // ---------------------------------------------------------
  // 15. SCHEDULE + ENGINEERING
  // ---------------------------------------------------------
  s = pres.addSlide();
  head(s, "Training", "Schedule & systems engineering");
  // left: schedule
  card(s, 0.55, 1.85, 6.05, 4.55);
  s.addImage({ data: icChart, x: 0.85, y: 2.1, w: 0.5, h: 0.5 });
  s.addText("Learning-rate schedule (GPT-3 style)", { x: 1.45, y: 2.1, w: 5, h: 0.5, fontSize: 16, bold: true, color: C.ink, valign: "middle", fontFace: "Calibri" });
  // mini schedule diagram
  const gx = 0.95, gw = 5.2, gtop = 3.0, gh = 1.5;
  s.addShape(pres.shapes.LINE, { x: gx, y: gtop + gh, w: gw, h: 0, line: { color: C.line, width: 1 } });
  // warmup ramp
  s.addShape(pres.shapes.LINE, { x: gx, y: gtop + gh, w: gw * 0.12, h: -gh, line: { color: C.cyan, width: 2.5 } });
  // cosine-ish decay (approx with line segments)
  let prevx = gx + gw * 0.12, prevy = gtop;
  for (let k = 1; k <= 10; k++) {
    const frac = 0.12 + (0.88) * (k / 10);
    const xx = gx + gw * frac;
    const cos = 0.5 * (1 + Math.cos(Math.PI * (k / 10)));
    const yy = gtop + gh * (1 - (0.05 + 0.95 * cos));
    s.addShape(pres.shapes.LINE, { x: Math.min(prevx, xx), y: Math.min(prevy, yy), w: Math.abs(xx - prevx), h: Math.abs(yy - prevy), line: { color: C.violet, width: 2.5 }, flipV: yy < prevy });
    prevx = xx; prevy = yy;
  }
  s.addText("warmup", { x: gx, y: gtop + gh + 0.05, w: 1.1, h: 0.25, fontSize: 9, color: C.cyan, fontFace: "Calibri" });
  s.addText("cosine decay → 5% floor", { x: gx + 1.4, y: gtop + gh + 0.05, w: 3.8, h: 0.25, fontSize: 9, color: C.violet, fontFace: "Calibri" });
  const schedPts = [
    "2-epoch linear warmup (lets zero-init B leave the sharp region).",
    "Cosine decay over 90 epochs to a 5% LR floor (min_lr_ratio = 0.05).",
    "The floor gave a real late gain: 0.8397 @ ep58 → 0.8401 @ ep88.",
  ];
  let scy = 4.8;
  schedPts.forEach((t) => {
    s.addShape(pres.shapes.OVAL, { x: 0.9, y: scy + 0.05, w: 0.11, h: 0.11, fill: { color: C.violet }, line: { type: "none" } });
    s.addText(t, { x: 1.15, y: scy - 0.05, w: 5.2, h: 0.55, fontSize: 12, color: C.ink, fontFace: "Calibri" });
    scy += 0.55;
  });
  // right: B200 engineering
  card(s, 6.75, 1.85, 6.0, 4.55, C.bg);
  s.addImage({ data: icBolt, x: 7.05, y: 2.1, w: 0.5, h: 0.5 });
  s.addText("B200 GPU engineering", { x: 7.65, y: 2.1, w: 5, h: 0.5, fontSize: 16, bold: true, color: "FFFFFF", valign: "middle", fontFace: "Calibri" });
  const eng = [
    "bf16 mixed precision (no GradScaler needed)",
    "channels-last memory format",
    "FlashAttention-3 via cuDNN SDPA",
    "fused AdamW optimizer",
    "pinned memory + deferred .item() GPU syncs",
    "parallel CPU distance transforms for cbDice",
    "full memory-attention path (Path B) for true gradient flow",
    "resume from train_state.pt to chain 4-hour SLURM jobs",
  ];
  let ey = 2.85;
  eng.forEach((t) => {
    s.addShape(pres.shapes.OVAL, { x: 7.1, y: ey + 0.07, w: 0.1, h: 0.1, fill: { color: C.cyan }, line: { type: "none" } });
    s.addText(t, { x: 7.35, y: ey - 0.04, w: 5.2, h: 0.4, fontSize: 12, color: C.mutedLt, fontFace: "Calibri" });
    ey += 0.44;
  });
  footer(s, "Training");

  // ---------------------------------------------------------
  // 16. THE JOURNEY (run-by-run table)
  // ---------------------------------------------------------
  s = pres.addSlide();
  head(s, "The Journey", "Previous tries: a run-by-run history");
  const rows = [
    [{ text: "Run", options: { bold: true, color: "FFFFFF" } }, { text: "Main change", options: { bold: true, color: "FFFFFF" } }, { text: "Data / val", options: { bold: true, color: "FFFFFF" } }, { text: "Path", options: { bold: true, color: "FFFFFF" } }, { text: "Best IoU", options: { bold: true, color: "FFFFFF" } }],
    ["1", "Published recipe (rank 8)", "12 / 2", "A", "0.6951"],
    ["2", "Over-regularised", "12 / 2", "A", "0.7218"],
    ["3", "Roll back reg, cbDice↑, sv off", "12 / 3", "A", "0.7468"],
    ["4", "Val swap + warmup 2 + 5% floor", "12 / 3", "A", "0.7755"],
    ["5", "Path B + obj-score fix + rank 16", "16 / 3", "B", "≈0.8295"],
    ["6", "+ Hiera last-stage LoRA (r8)", "16 / 3", "B", "≈0.8295"],
    [{ text: "7", options: { bold: true } }, { text: "+ 20 videos, 90 ep, 5-vid val", options: { bold: true } }, { text: "20 / 5", options: { bold: true } }, { text: "B", options: { bold: true } }, { text: "0.8401", options: { bold: true } }],
  ];
  const tableRows = rows.map((r, ri) =>
    r.map((cell) => {
      const base = typeof cell === "string" ? { text: cell } : cell;
      const isHead = ri === 0;
      const isBest = ri === rows.length - 1;
      return {
        text: base.text,
        options: Object.assign(
          { fontSize: isHead ? 13 : 12.5, fontFace: "Calibri", valign: "middle",
            color: isHead ? "FFFFFF" : (isBest ? C.ink : C.ink),
            fill: { color: isHead ? C.bg : (isBest ? "D1FAE5" : (ri % 2 ? "FFFFFF" : C.cardAlt)) },
            align: "left", margin: [3, 6, 3, 6] },
          base.options || {}
        ),
      };
    })
  );
  s.addTable(tableRows, {
    x: 0.55, y: 1.9, w: 9.6, colW: [0.8, 4.6, 1.7, 1.0, 1.5], rowH: 0.46,
    border: { pt: 0.5, color: C.line },
  });
  // failed experiments note (right)
  card(s, 10.4, 1.9, 2.35, 3.85, C.bg);
  s.addImage({ data: icTimes, x: 10.65, y: 2.1, w: 0.4, h: 0.4 });
  s.addText("Tried & reverted", { x: 11.1, y: 2.1, w: 1.6, h: 0.4, fontSize: 13, bold: true, color: "FFFFFF", valign: "middle", fontFace: "Calibri" });
  s.addText([
    { text: "Run 8", options: { bold: true, color: "F87171", breakLine: true } },
    { text: "Pumped topology weights → 0.8229. Supervoxel hit a structural ceiling.\n", options: { color: C.mutedLt, fontSize: 11, breakLine: true } },
    { text: "Run 9", options: { bold: true, color: "F87171", breakLine: true } },
    { text: "Topology res 256→384 → 0.8346. Helped but 2.5× slower, still lost.", options: { color: C.mutedLt, fontSize: 11 } },
  ], { x: 10.65, y: 2.65, w: 1.95, h: 3.0, fontSize: 11.5, fontFace: "Calibri", valign: "top" });
  s.addText([
    { text: "Lesson:  ", options: { bold: true, color: C.amber } },
    { text: "at 20 videos, loss / schedule tuning is saturated. More data — not more loss engineering — is the next lever.", options: { color: C.ink } },
  ], { x: 0.55, y: 6.05, w: 9.6, h: 0.6, fontSize: 12.5, fontFace: "Calibri" });
  footer(s, "The Journey");

  // ---------------------------------------------------------
  // 17. CURRENT SETUP (Run 7 spec sheet)
  // ---------------------------------------------------------
  s = pres.addSlide();
  head(s, "Current Setup", "Run 7 — the shipping configuration");
  const spec = [
    ["MODEL", [
      ["Backbone", "facebook/sam2.1-hiera-tiny"],
      ["Adapters", "Dual-rank LoRA (16 / 8)"],
      ["Trainable", "5.56M / 39.88M  (13.9%)"],
      ["Forward path", "Path B (memory attention)"],
    ], C.cyan],
    ["OPTIMIZER", [
      ["lr_A / lr_B", "1e-4 / 1.6e-3  (λ=16)"],
      ["Base lr", "1e-5"],
      ["Weight decay", "0.05"],
      ["Grad clip", "1.0"],
    ], C.violet],
    ["SCHEDULE & DATA", [
      ["Epochs", "90 (warmup 2)"],
      ["LR floor", "5% (cosine)"],
      ["Batch / frames", "2  /  8 per clip"],
      ["Data / val", "20 videos / 5 held out"],
    ], C.teal],
  ];
  const scw = 3.95, sgap = 0.22, sx0 = 0.55, syy = 1.95, shh = 3.6;
  spec.forEach((grp, gi) => {
    const x = sx0 + gi * (scw + sgap);
    card(s, x, syy, scw, shh);
    s.addShape(pres.shapes.RECTANGLE, { x, y: syy, w: scw, h: 0.55, fill: { color: grp[2] }, line: { type: "none" } });
    s.addText(grp[0], { x: x + 0.25, y: syy, w: scw - 0.4, h: 0.55, fontSize: 14, bold: true, color: "FFFFFF", valign: "middle", charSpacing: 1, fontFace: "Consolas" });
    let ry = syy + 0.75;
    grp[1].forEach(([k, v]) => {
      s.addText(k, { x: x + 0.25, y: ry, w: scw - 0.5, h: 0.3, fontSize: 11, color: C.muted, fontFace: "Calibri" });
      s.addText(v, { x: x + 0.25, y: ry + 0.28, w: scw - 0.5, h: 0.35, fontSize: 13, bold: true, color: C.ink, fontFace: "Consolas" });
      ry += 0.72;
    });
  });
  s.addText([
    { text: "Loss weights:  ", options: { bold: true, color: C.violet } },
    { text: "focal 5 · Dice 2 · cbDice 5 · supervoxel 0.3 · IoU 1 · class 1     ", options: { color: C.ink } },
    { text: "Augmentation:  ", options: { bold: true, color: C.violet } },
    { text: "full pixel-level (flips, ±25° rot, ±20% shear, brightness/contrast), fresh per epoch.", options: { color: C.ink } },
  ], { x: 0.55, y: 5.75, w: 12.2, h: 0.7, fontSize: 12.5, fontFace: "Calibri", fill: { color: C.cardAlt }, align: "left", valign: "middle" });
  footer(s, "Current Setup");

  // ---------------------------------------------------------
  // 18. RESULTS
  // ---------------------------------------------------------
  s = pres.addSlide();
  head(s, "Results", "Run 7 performance");
  // big stat callouts
  const stats = [
    ["0.8401", "Validation IoU", "best @ epoch 88", C.cyan],
    ["0.9117", "Validation Dice", "region overlap", C.violet],
    ["+0.145", "IoU vs. baseline", "from 0.6951 (Run 1)", C.good],
  ];
  stats.forEach((st, i) => {
    const x = 0.55 + i * 4.17;
    card(s, x, 1.85, 3.95, 1.9, C.bg);
    s.addText(st[0], { x, y: 1.95, w: 3.95, h: 0.95, fontSize: 50, bold: true, color: st[3], align: "center", fontFace: "Arial Black" });
    s.addText(st[1], { x, y: 2.95, w: 3.95, h: 0.35, fontSize: 14, bold: true, color: "FFFFFF", align: "center", fontFace: "Calibri" });
    s.addText(st[2], { x, y: 3.3, w: 3.95, h: 0.3, fontSize: 11, color: C.mutedLt, align: "center", italic: true, fontFace: "Calibri" });
  });
  // progression chart (shape-drawn — no native chart object, max compatibility)
  card(s, 0.55, 4.0, 7.3, 2.85);
  s.addText("Validation IoU across runs", { x: 0.8, y: 4.15, w: 6, h: 0.35, fontSize: 13, bold: true, color: C.ink, fontFace: "Calibri" });
  {
    const vals = [0.6951, 0.7218, 0.7468, 0.7755, 0.8295, 0.8295, 0.8401];
    const labels = ["R1", "R2", "R3", "R4", "R5", "R6", "R7"];
    const vMin = 0.65, vMax = 0.86;
    const px0 = 1.15, pw = 6.35, pTop = 4.6, pH = 1.75; // plot box
    const xAt = (i) => px0 + pw * (i / (vals.length - 1));
    const yAt = (v) => pTop + pH * (1 - (v - vMin) / (vMax - vMin));
    // gridlines + y labels
    [0.70, 0.75, 0.80, 0.85].forEach((gv) => {
      const gy = yAt(gv);
      s.addShape(pres.shapes.LINE, { x: px0, y: gy, w: pw, h: 0, line: { color: C.line, width: 0.75 } });
      s.addText(gv.toFixed(2), { x: px0 - 0.62, y: gy - 0.12, w: 0.55, h: 0.24, fontSize: 9, color: C.muted, align: "right", fontFace: "Calibri", margin: 0 });
    });
    // line segments
    for (let i = 0; i < vals.length - 1; i++) {
      const x1 = xAt(i), y1 = yAt(vals[i]), x2 = xAt(i + 1), y2 = yAt(vals[i + 1]);
      s.addShape(pres.shapes.LINE, {
        x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
        line: { color: C.cyan, width: 2.75 }, flipV: y2 < y1,
      });
    }
    // dots + x labels
    vals.forEach((v, i) => {
      s.addShape(pres.shapes.OVAL, { x: xAt(i) - 0.07, y: yAt(v) - 0.07, w: 0.14, h: 0.14, fill: { color: i === vals.length - 1 ? C.violet : C.cyan }, line: { color: "FFFFFF", width: 1 } });
      s.addText(labels[i], { x: xAt(i) - 0.3, y: pTop + pH + 0.06, w: 0.6, h: 0.24, fontSize: 9.5, color: C.muted, align: "center", fontFace: "Calibri", margin: 0 });
    });
  }
  // right: breakdown notes + placeholder
  card(s, 8.05, 4.0, 4.7, 2.85, C.bg);
  s.addText("Diagnostics behind the number", { x: 8.3, y: 4.15, w: 4.2, h: 0.35, fontSize: 13, bold: true, color: C.cyan, fontFace: "Calibri" });
  const diag = [
    "Per-video IoU is uniform — no single video lags.",
    "Per-frame IoU breakdown tracks quality over time.",
    "Worst-IoU PNG dumps expose hard cases each epoch.",
    "Visual: branches much sharper than base SAM 2.",
  ];
  let dy = 4.6;
  diag.forEach((t) => {
    s.addShape(pres.shapes.OVAL, { x: 8.35, y: dy + 0.06, w: 0.1, h: 0.1, fill: { color: C.cyan }, line: { type: "none" } });
    s.addText(t, { x: 8.6, y: dy - 0.04, w: 4.0, h: 0.5, fontSize: 11.5, color: C.mutedLt, fontFace: "Calibri" });
    dy += 0.52;
  });
  footer(s, "Results");

  // ---------------------------------------------------------
  // 19. WHY WE WIN (summary)
  // ---------------------------------------------------------
  s = pres.addSlide();
  head(s, "The Model", "Why & how we beat base SAM 2");
  const wins = [
    [icScale, "Adapt, don't forget", "LoRA re-shapes the model for microscopy while 86% of pretrained weights stay frozen — strong features without catastrophic forgetting on a tiny dataset."],
    [icSitemap, "Topology-aware learning", "cbDice (skeleton) + supervoxel (connectivity) losses target the exact failure mode of base SAM 2: broken, fragmented thin branches."],
    [icSync, "Real temporal modelling", "Path B trains the memory-attention adapters with true gradient flow, so the model genuinely tracks neurons frame-to-frame."],
    [icMicro, "Domain-matched inputs", "Dataset-specific normalization + microscopy augmentations align the input distribution with the real data — base SAM 2 never saw any of this."],
  ];
  const wcw = 6.0, wgap = 0.2, wx0 = 0.55, wyy = 1.9, whh = 2.25;
  wins.forEach((w, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = wx0 + col * (wcw + wgap), y = wyy + row * (whh + 0.2);
    card(s, x, y, wcw, whh);
    s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: y + 0.35, w: 0.8, h: 0.8, fill: { color: C.bg }, line: { type: "none" } });
    s.addImage({ data: w[0], x: x + 0.48, y: y + 0.53, w: 0.44, h: 0.44 });
    s.addText(w[1], { x: x + 1.35, y: y + 0.38, w: wcw - 1.6, h: 0.7, fontSize: 17, bold: true, color: C.ink, valign: "middle", fontFace: "Calibri" });
    s.addText(w[2], { x: x + 1.35, y: y + 1.05, w: wcw - 1.6, h: 1.1, fontSize: 12.5, color: C.muted, fontFace: "Calibri" });
  });
  footer(s, "The Model");

  // ---------------------------------------------------------
  // 20. SECTION 4 DIVIDER
  // ---------------------------------------------------------
  s = pres.addSlide();
  divider(s, "04", "Next Stages", "Where the project goes from here.", icRocket);

  // ---------------------------------------------------------
  // 21. NEXT STAGES
  // ---------------------------------------------------------
  s = pres.addSlide();
  head(s, "Next Stages", "The road ahead");
  // hero: data is the lever
  card(s, 0.55, 1.85, 12.2, 1.35, C.bg);
  s.addImage({ data: icDatabase, x: 0.9, y: 2.2, w: 0.65, h: 0.65 });
  s.addText([
    { text: "Priority #1 — more annotated data.  ", options: { bold: true, color: C.cyan } },
    { text: "Runs 8 & 9 proved loss tuning is saturated at 20 videos. Growing the dataset is the single highest-leverage next step, powered by the labeling loop.", options: { color: "FFFFFF" } },
  ], { x: 1.75, y: 1.95, w: 10.8, h: 1.15, fontSize: 14, fontFace: "Calibri", valign: "middle" });
  const next = [
    [icMicro, "Grow to ~30+ videos", "Prioritise hard cases surfaced by the worst-IoU dumps; keep the 5 val videos fixed for comparable numbers."],
    [icSitemap, "Revisit topology losses", "At ~30 videos, re-test the supervoxel weight and higher topology resolution — the structural ceiling may lift with more signal."],
    [icCheck, "Tighten the labeling app", "Faster propagation and smoother correction UX to shrink the cost per labeled video."],
    [icChart, "Robust evaluation", "Expand per-video / per-frame diagnostics and report variance across val splits, not just a single headline IoU."],
  ];
  const ncw = 6.0, ngap = 0.2, nx0 = 0.55, nyy = 3.45, nhh = 1.6;
  next.forEach((n, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = nx0 + col * (ncw + ngap), y = nyy + row * (nhh + 0.18);
    card(s, x, y, ncw, nhh);
    s.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.1, h: nhh, fill: { color: i % 2 ? C.violet : C.cyan }, line: { type: "none" } });
    s.addImage({ data: n[0], x: x + 0.3, y: y + 0.32, w: 0.5, h: 0.5 });
    s.addText(n[1], { x: x + 1.0, y: y + 0.25, w: ncw - 1.2, h: 0.5, fontSize: 15, bold: true, color: C.ink, valign: "middle", fontFace: "Calibri" });
    s.addText(n[2], { x: x + 1.0, y: y + 0.72, w: ncw - 1.2, h: 0.8, fontSize: 11.5, color: C.muted, fontFace: "Calibri" });
  });
  footer(s, "Next Stages");

  // ---------------------------------------------------------
  // 22. CLOSING
  // ---------------------------------------------------------
  s = pres.addSlide();
  s.background = { color: C.bg };
  // motif nodes
  [[1.0, 0.85], [2.15, 1.45], [1.35, 2.1]].forEach((n, i) => {
    const col = [C.cyan, C.violet, C.teal][i];
    s.addShape(pres.shapes.OVAL, { x: n[0], y: n[1], w: 0.16, h: 0.16, fill: { color: col }, line: { type: "none" } });
    s.addShape(pres.shapes.OVAL, { x: n[0] - 0.12, y: n[1] - 0.12, w: 0.4, h: 0.4, fill: { type: "none" }, line: { color: col, width: 1, transparency: 40 } });
  });
  s.addText("Thank you", { x: 0.7, y: 2.7, w: 11, h: 1.1, fontSize: 52, bold: true, color: "FFFFFF", fontFace: "Georgia", margin: 0 });
  s.addText("Fine-tuning SAM 2 turns a general video model into a specialist neuron segmenter — and the labeling loop will keep making it better.", {
    x: 0.72, y: 3.95, w: 8.8, h: 0.9, fontSize: 16, color: C.mutedLt, italic: true, fontFace: "Calibri",
  });
  s.addShape(pres.shapes.LINE, { x: 0.74, y: 4.95, w: 4.0, h: 0, line: { color: C.violet, width: 2 } });
  s.addText([
    { text: "Adi Oizerovich", options: { bold: true, color: "FFFFFF", fontSize: 15, breakLine: true } },
    { text: "Best model — validation IoU 0.8401 · Dice 0.9117  ·  Questions welcome", options: { color: C.cyan, fontSize: 13 } },
  ], { x: 0.74, y: 5.1, w: 9, h: 0.9, fontFace: "Calibri" });

  // ---------------------------------------------------------
  await pres.writeFile({ fileName: "SAM2_Neuron_Segmentation.pptx" });
  console.log("WROTE SAM2_Neuron_Segmentation.pptx");
})();
