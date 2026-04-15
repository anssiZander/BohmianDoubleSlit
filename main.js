const canvas = document.getElementById("c");

if (!navigator.gpu) {
  alert("WebGPU is not available in this browser. Use a recent Chrome or Edge build on localhost or https.");
  throw new Error("WebGPU not available.");
}

const WAVE_FORMAT = "rgba32float";
const DENSITY_FORMAT = "rgba16float";
const UNIFORM_FLOATS = 40;

const params = {
  simScale: 1.0,
  stepsPerFrame: 50,

  hbar: 6.0,
  mass: 1.0,
  p0: 1.5,
  dt: 0.01,

  packetX: 0.30,
  packetY: 0.50,
  packetSigma: 80.0,

  barrierX: 0.55,
  barrierThick: 20.0,
  slitWidth: 25.0,
  slitSep: 60.0,
  V0: 50.0,

  absorbPx: 110.0,
  absorbStrength: 0.25,
  particleKillMargin: 12.0,

  nParticles: 1000,
  rhoMin: 1e-6,
  velClamp: 160.0,

  visGain: 20.0,
  visGamma: 0.5,
  showPhase: 1,

  showParticles: 1,
  dotSize: 10.0,
  dotSigma: 0.28,
  dotGain: 1.0,

  showTrail: 1,
  trailHalfLife: 100.0,
  trailVisGain: 1.0,
  trailVisGamma: 0.5,
  trailStampGain: 0.55,
  trailWidth: 4.0,
  trailBlendMode: 1,

  paletteId: 5,
};

let paused = false;
let device = null;
let context = null;
let presentationFormat = null;

let uniformBuffer = null;
let boundaryRectBuffer = null;

let simW = 0;
let simH = 0;
let densW = 0;
let densH = 0;

let particleCount = Math.floor(params.nParticles);

let waveFlip = 0;
let particleFlip = 0;
let densityFlip = 0;

let waveTextures = [];
let densityTextures = [];
let particleBuffers = [];

let pipelines = {};
let bindGroups = {};

const controls = document.getElementById("controls");
const statsEl = document.getElementById("stats");
const pauseButton = document.getElementById("pause");

function fmt(v) {
  const av = Math.abs(v);
  if (av >= 1000 || (av > 0 && av < 0.01)) return v.toExponential(2);
  return v.toFixed(3).replace(/\.?0+$/, "");
}

function addSlider(key, label, min, max, step, onChange = null) {
  const row = document.createElement("div");
  row.className = "row";

  const lab = document.createElement("label");
  lab.textContent = label;

  const input = document.createElement("input");
  input.type = "range";
  input.min = min;
  input.max = max;
  input.step = step;
  input.value = params[key];

  const val = document.createElement("div");
  val.className = "val";
  val.textContent = fmt(params[key]);

  input.addEventListener("input", () => {
    const next = parseFloat(input.value);
    params[key] = next;
    val.textContent = fmt(next);
  });
  input.addEventListener("change", () => onChange && onChange());

  row.appendChild(lab);
  row.appendChild(input);
  row.appendChild(val);
  controls.appendChild(row);
}

function addToggleInt(key, label) {
  const row = document.createElement("div");
  row.className = "row";

  const lab = document.createElement("label");
  lab.textContent = label;

  const btn = document.createElement("button");
  btn.style.flex = "1";
  btn.textContent = params[key] ? "ON" : "OFF";
  btn.addEventListener("click", () => {
    params[key] = params[key] ? 0 : 1;
    btn.textContent = params[key] ? "ON" : "OFF";
  });

  const val = document.createElement("div");
  val.className = "val";

  row.appendChild(lab);
  row.appendChild(btn);
  row.appendChild(val);
  controls.appendChild(row);
}

function addSectionHeader(label) {
  const header = document.createElement("div");
  header.style.marginTop = "12px";
  header.style.marginBottom = "8px";
  header.style.fontSize = "11px";
  header.style.fontWeight = "700";
  header.style.color = "#aaa";
  header.style.textTransform = "uppercase";
  header.style.letterSpacing = "1px";
  header.textContent = label;
  controls.appendChild(header);
}

function setPaused(next) {
  paused = next;
  pauseButton.textContent = paused ? "Resume" : "Pause";
}

addSlider("stepsPerFrame", "Steps/frame", 1, 100, 1);

addSectionHeader("Physical Parameters");
addSlider("p0", "Momentum p", 0.5, 8.0, 0.1);
addSlider("dt", "dt", 0.005, 0.02, 0.001);
addSlider("packetSigma", "packet sigma", 8.0, 80.0, 1.0);
addSlider("slitWidth", "slit width", 6.0, 40.0, 1.0);
addSlider("slitSep", "slit separation", 18.0, 140.0, 1.0);
addSlider("absorbPx", "absorb boundary", 0.0, 160.0, 1.0);
addSlider("nParticles", "particle count", 1, 3000, 1, () => {
  if (!device) return;
  rebuildParticles();
  createParticleBindGroups();
  writeSimParams();
});

addSectionHeader("Visual Parameters");
addToggleInt("showPhase", "show phase");
addToggleInt("showParticles", "show particles");
addSlider("dotSize", "particle size", 2.0, 16.0, 0.5);
addSlider("dotGain", "particle brightness", 0.1, 3.0, 0.1);

addToggleInt("showTrail", "draw trails");
addSlider("trailHalfLife", "trail half-life", 1.0, 150.0, 1.0);
addSlider("trailVisGain", "trail gain", 0.1, 1.0, 0.1);
addSlider("trailVisGamma", "trail gamma", 0.4, 2.0, 0.05);
addSlider("trailWidth", "trail width (px)", 0.5, 10.0, 0.1);

addSlider("visGain", "wave gain", 0.5, 20.0, 0.5);
addSlider("visGamma", "wave gamma", 0.3, 2.0, 0.05);

document.getElementById("reset").onclick = () => resetAll();
pauseButton.onclick = () => setPaused(!paused);
window.addEventListener("keydown", (e) => {
  const key = e.key.toLowerCase();
  if (key === "r") resetAll();
  if (e.key === " ") {
    e.preventDefault();
    setPaused(!paused);
  }
});

const uiBody = document.getElementById("uibody");
const minBtn = document.getElementById("minui");
let uiMinimized = false;
minBtn.onclick = () => {
  uiMinimized = !uiMinimized;
  uiBody.style.display = uiMinimized ? "none" : "block";
  minBtn.textContent = uiMinimized ? ">" : "v";
};

function resizeCanvas() {
  const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
  const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  const changed = canvas.width !== width || canvas.height !== height;

  if (changed) {
    canvas.width = width;
    canvas.height = height;
    if (device && context) {
      context.configure({
        device,
        format: presentationFormat,
        alphaMode: "opaque",
      });
    }
  }

  return changed;
}

function createTexture(label, width, height, format, usage) {
  const texture = device.createTexture({
    label,
    size: [width, height],
    format,
    usage,
  });
  return { texture, view: texture.createView() };
}

function destroyTextureSet(set) {
  for (const entry of set) {
    if (entry?.texture) entry.texture.destroy();
  }
}

function destroyBufferSet(set) {
  for (const buffer of set) {
    if (buffer) buffer.destroy();
  }
}

function randn() {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function computeBarrierOpacity() {
  const E = (params.p0 * params.p0) / (2 * params.mass);
  if (params.V0 <= 0) return 0.0;
  if (E >= params.V0) return 0.20;
  const kappa = Math.sqrt(2 * params.mass * (params.V0 - E)) / params.hbar;
  const T = Math.exp(-2 * kappa * params.barrierThick);
  return Math.min(1.0, Math.max(0.20, 1.0 - T));
}

function writeSimParams() {
  if (!device || !uniformBuffer) return;

  const values = new Float32Array(UNIFORM_FLOATS);
  values[0] = simW;
  values[1] = simH;
  values[2] = canvas.width;
  values[3] = canvas.height;

  values[4] = params.hbar;
  values[5] = params.mass;
  values[6] = params.p0;
  values[7] = params.dt;

  values[8] = params.packetX;
  values[9] = params.packetY;
  values[10] = params.packetSigma;
  values[11] = params.barrierX;

  values[12] = params.barrierThick;
  values[13] = params.slitWidth;
  values[14] = params.slitSep;
  values[15] = params.V0;

  values[16] = params.absorbPx;
  values[17] = params.absorbStrength;
  values[18] = params.particleKillMargin;
  values[19] = params.rhoMin;

  values[20] = params.velClamp;
  values[21] = params.visGain;
  values[22] = params.visGamma;
  values[23] = params.showPhase;

  values[24] = params.showParticles;
  values[25] = params.dotSize;
  values[26] = params.dotSigma;
  values[27] = params.dotGain;

  values[28] = params.showTrail;
  values[29] = params.trailHalfLife;
  values[30] = params.trailVisGain;
  values[31] = params.trailVisGamma;

  values[32] = params.trailStampGain;
  values[33] = params.trailWidth;
  values[34] = params.trailBlendMode;
  values[35] = params.paletteId;

  values[36] = Math.floor(params.stepsPerFrame);
  values[37] = particleCount;
  values[38] = computeBarrierOpacity();
  values[39] = 0.0;

  device.queue.writeBuffer(uniformBuffer, 0, values);
}

function updateBoundaryRects() {
  if (!device || !boundaryRectBuffer || !simW || !simH) return;

  const base = params.absorbPx + params.particleKillMargin;
  const absDistX = 1.5 * base;
  const absDistY = 1.0 * base;
  const freezeDistX = 1.5 * absDistX;
  const freezeDistXLeft = freezeDistX * 1.20;
  const freezeDistY = 1.5 * absDistY;

  const scaleX = canvas.width / simW;
  const scaleY = canvas.height / simH;
  const leftBoundaryX = freezeDistXLeft * scaleX;
  const rightBoundaryX = (simW - freezeDistX) * scaleX;
  const topBoundaryY = freezeDistY * scaleY;
  const bottomBoundaryY = (simH - freezeDistY) * scaleY;
  const thickness = 2.0;

  const canvasToNDCX = (px) => (px * 2 / canvas.width) - 1;
  const canvasToNDCY = (py) => 1 - (py * 2 / canvas.height);

  const rects = new Float32Array([
    canvasToNDCX(leftBoundaryX), -1, canvasToNDCX(leftBoundaryX + thickness), 1,
    canvasToNDCX(rightBoundaryX - thickness), -1, canvasToNDCX(rightBoundaryX), 1,
    -1, canvasToNDCY(topBoundaryY), 1, canvasToNDCY(topBoundaryY + thickness),
    -1, canvasToNDCY(bottomBoundaryY - thickness), 1, canvasToNDCY(bottomBoundaryY),
  ]);

  device.queue.writeBuffer(boundaryRectBuffer, 0, rects);
}

function rebuildParticles() {
  if (!device || !simW || !simH) return;

  destroyBufferSet(particleBuffers);
  particleBuffers = [];
  particleFlip = 0;
  particleCount = Math.max(1, Math.floor(params.nParticles));

  const data = new Float32Array(particleCount * 4);
  const sigma1D = params.packetSigma / Math.sqrt(2);
  const x0 = params.packetX * simW;
  const y0 = params.packetY * simH;

  for (let i = 0; i < particleCount; i++) {
    let x = x0 + randn() * sigma1D;
    let y = y0 + randn() * sigma1D;
    x = Math.max(0, Math.min(simW - 1, x));
    y = Math.max(0, Math.min(simH - 1, y));

    const base = i * 4;
    data[base + 0] = x;
    data[base + 1] = y;
    data[base + 2] = 1.0;
    data[base + 3] = 0.0;
  }

  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const src = device.createBuffer({ label: "particles-a", size: data.byteLength, usage });
  const dst = device.createBuffer({ label: "particles-b", size: data.byteLength, usage });
  device.queue.writeBuffer(src, 0, data);
  device.queue.writeBuffer(dst, 0, data);
  particleBuffers = [src, dst];
}

const simCommonWGSL = String.raw`
struct SimParams {
  simRes: vec2f,
  canvasRes: vec2f,
  hbar: f32,
  mass: f32,
  p0: f32,
  dt: f32,
  packetX: f32,
  packetY: f32,
  packetSigma: f32,
  barrierX: f32,
  barrierThick: f32,
  slitWidth: f32,
  slitSep: f32,
  V0: f32,
  absorbPx: f32,
  absorbStrength: f32,
  particleKillMargin: f32,
  rhoMin: f32,
  velClamp: f32,
  visGain: f32,
  visGamma: f32,
  showPhase: f32,
  showParticles: f32,
  dotSize: f32,
  dotSigma: f32,
  dotGain: f32,
  showTrail: f32,
  trailHalfLife: f32,
  trailVisGain: f32,
  trailVisGamma: f32,
  trailStampGain: f32,
  trailWidth: f32,
  trailBlendMode: f32,
  paletteId: f32,
  stepsPerFrame: f32,
  nParticles: f32,
  barrierOpacity: f32,
  _pad0: f32,
};

@group(0) @binding(0) var<uniform> params: SimParams;

const PI: f32 = 3.141592653589793;
const TAU: f32 = 6.283185307179586;
const LN2: f32 = 0.6931471805599453;

fn band(x: f32, c: f32, halfW: f32, feather: f32) -> f32 {
  return smoothstep(c - halfW - feather, c - halfW, x) *
         (1.0 - smoothstep(c + halfW, c + halfW + feather, x));
}

fn simSizeU() -> vec2u {
  return vec2u(u32(params.simRes.x), u32(params.simRes.y));
}

fn simSizeI() -> vec2i {
  return vec2i(i32(params.simRes.x), i32(params.simRes.y));
}

fn canvasSizeU() -> vec2u {
  return vec2u(u32(params.canvasRes.x), u32(params.canvasRes.y));
}

fn barrierWallMaskAtPx(xPx: vec2f) -> f32 {
  let bx = params.barrierX * params.simRes.x;
  let slab = band(xPx.x, bx, 0.5 * params.barrierThick, 1.0);

  let y0 = 0.5 * params.simRes.y;
  let s = 0.5 * params.slitSep;
  let hw = 0.5 * params.slitWidth;

  let slit1 = band(xPx.y, y0 - s, hw, 1.0);
  let slit2 = band(xPx.y, y0 + s, hw, 1.0);
  let slits = clamp(slit1 + slit2, 0.0, 1.0);

  return slab * (1.0 - slits);
}

fn barrierPotentialPx(xPx: vec2f) -> f32 {
  return params.V0 * barrierWallMaskAtPx(xPx);
}

fn absorbW(xPx: vec2f) -> f32 {
  if (params.absorbPx <= 0.0) {
    return 0.0;
  }

  let leftFactor = 1.2;
  let dx = min(xPx.x * leftFactor, params.simRes.x - 1.0 - xPx.x);
  let dy = min(xPx.y, params.simRes.y - 1.0 - xPx.y);
  let d = min(dx, dy);
  let t = clamp((params.absorbPx - d) / max(params.absorbPx, 1.0), 0.0, 1.0);
  let profile = t * t * t;
  return params.absorbStrength * profile;
}

fn kineticEnergy() -> f32 {
  return 0.5 * params.p0 * params.p0 / params.mass;
}

fn cis(a: f32) -> vec2f {
  return vec2f(cos(a), sin(a));
}

fn schrodingerRHS(psi: vec2f, lapPsi: vec2f, V: f32) -> vec2f {
  let cLap = params.hbar / (2.0 * params.mass);
  let cV = V / params.hbar;
  return vec2f(
    -cLap * lapPsi.y + cV * psi.y,
     cLap * lapPsi.x - cV * psi.x
  );
}

fn initialPacketAtPx(xPx: vec2f, t: f32) -> vec2f {
  let x0 = vec2f(params.packetX, params.packetY) * params.simRes;
  let d = xPx - x0;
  let sigma2 = max(params.packetSigma * params.packetSigma, 1e-6);
  let amp = exp(-dot(d, d) / (2.0 * sigma2));
  let k = params.p0 / params.hbar;
  let phaseSpace = k * d.x;
  let phaseTime = -kineticEnergy() * t / params.hbar;
  return amp * cis(phaseSpace + phaseTime);
}
`;

const paletteWGSL = String.raw`
struct PaletteData {
  a: vec3f,
  b: vec3f,
  c: vec3f,
  d: vec3f,
};

fn palette(t: f32, a: vec3f, b: vec3f, c: vec3f, d: vec3f) -> vec3f {
  return a + b * cos(TAU * (c * t + d));
}

fn wavePalette(id: u32) -> PaletteData {
  switch id {
    case 0u: { return PaletteData(vec3f(0.08, 0.07, 0.12), vec3f(0.55, 0.50, 0.70), vec3f(1.0, 1.0, 1.0), vec3f(0.00, 0.15, 0.35)); }
    case 1u: { return PaletteData(vec3f(0.06, 0.02, 0.10), vec3f(0.85, 0.35, 0.95), vec3f(1.0, 1.0, 1.0), vec3f(0.00, 0.10, 0.25)); }
    case 2u: { return PaletteData(vec3f(0.22, 0.32, 0.28), vec3f(0.40, 0.45, 0.35), vec3f(1.0, 1.0, 1.0), vec3f(0.15, 0.55, 0.75)); }
    case 3u: { return PaletteData(vec3f(0.10, 0.02, 0.02), vec3f(0.90, 0.45, 0.20), vec3f(1.0, 1.0, 1.0), vec3f(0.00, 0.08, 0.20)); }
    case 4u: { return PaletteData(vec3f(0.02, 0.05, 0.08), vec3f(0.40, 0.70, 0.85), vec3f(1.0, 1.0, 1.0), vec3f(0.10, 0.30, 0.55)); }
    case 5u: { return PaletteData(vec3f(0.10, 0.02, 0.12), vec3f(0.75, 0.15, 0.90), vec3f(1.0, 1.0, 1.0), vec3f(0.00, 0.10, 0.30)); }
    case 6u: { return PaletteData(vec3f(0.05, 0.15, 0.15), vec3f(0.20, 0.80, 0.60), vec3f(1.0, 1.0, 1.0), vec3f(0.10, 0.40, 0.20)); }
    case 7u: { return PaletteData(vec3f(0.15, 0.05, 0.00), vec3f(0.95, 0.50, 0.10), vec3f(1.0, 1.0, 1.0), vec3f(0.00, 0.05, 0.15)); }
    case 8u: { return PaletteData(vec3f(0.20, 0.15, 0.10), vec3f(0.60, 0.50, 0.30), vec3f(1.0, 1.0, 1.0), vec3f(0.10, 0.30, 0.25)); }
    case 9u: { return PaletteData(vec3f(0.02, 0.02, 0.02), vec3f(0.00, 0.80, 0.80), vec3f(1.0, 1.0, 1.0), vec3f(0.90, 0.10, 0.90)); }
    default: { return PaletteData(vec3f(0.75, 0.70, 0.80), vec3f(0.60, 0.85, 0.70), vec3f(1.0, 1.0, 1.0), vec3f(0.10, 0.20, 0.30)); }
  }
}

fn particlePalette(id: u32) -> PaletteData {
  switch id {
    case 0u: { return PaletteData(vec3f(0.05, 0.03, 0.08), vec3f(0.85, 0.65, 0.95), vec3f(1.0, 1.0, 1.0), vec3f(0.00, 0.20, 0.55)); }
    case 1u: { return PaletteData(vec3f(0.02, 0.01, 0.05), vec3f(1.00, 0.35, 1.00), vec3f(1.0, 1.0, 1.0), vec3f(0.05, 0.10, 0.75)); }
    case 2u: { return PaletteData(vec3f(0.10, 0.18, 0.14), vec3f(0.70, 0.90, 0.55), vec3f(1.0, 1.0, 1.0), vec3f(0.15, 0.45, 0.75)); }
    case 3u: { return PaletteData(vec3f(0.08, 0.02, 0.01), vec3f(1.00, 0.65, 0.25), vec3f(1.0, 1.0, 1.0), vec3f(0.05, 0.15, 0.30)); }
    case 4u: { return PaletteData(vec3f(0.02, 0.06, 0.10), vec3f(0.65, 0.95, 1.00), vec3f(1.0, 1.0, 1.0), vec3f(0.10, 0.30, 0.60)); }
    case 5u: { return PaletteData(vec3f(0.08, 0.06, 0.02), vec3f(1.00, 0.90, 0.40), vec3f(1.0, 1.0, 1.0), vec3f(0.08, 0.18, 0.28)); }
    case 6u: { return PaletteData(vec3f(0.03, 0.07, 0.03), vec3f(0.50, 1.00, 0.65), vec3f(1.0, 1.0, 1.0), vec3f(0.10, 0.35, 0.55)); }
    case 7u: { return PaletteData(vec3f(0.07, 0.05, 0.02), vec3f(1.00, 0.85, 0.20), vec3f(1.0, 1.0, 1.0), vec3f(0.00, 0.10, 0.20)); }
    case 8u: { return PaletteData(vec3f(0.07, 0.02, 0.04), vec3f(1.00, 0.55, 0.30), vec3f(1.0, 1.0, 1.0), vec3f(0.05, 0.25, 0.45)); }
    default: { return PaletteData(vec3f(0.02, 0.03, 0.08), vec3f(0.35, 1.00, 1.00), vec3f(1.0, 1.0, 1.0), vec3f(0.05, 0.35, 0.55)); }
  }
}

fn complementColor(id: u32) -> vec3f {
  switch id {
    case 0u: { return vec3f(0.92, 0.93, 0.88); }
    case 1u: { return vec3f(0.10, 0.60, 0.10); }
    case 2u: { return vec3f(0.80, 0.60, 0.55); }
    case 3u: { return vec3f(0.10, 0.60, 0.80); }
    case 4u: { return vec3f(0.80, 0.30, 0.15); }
    case 5u: { return vec3f(0.20, 0.80, 0.30); }
    case 6u: { return vec3f(0.85, 0.25, 0.25); }
    case 7u: { return vec3f(0.10, 0.10, 0.80); }
    case 8u: { return vec3f(0.40, 0.50, 0.70); }
    case 9u: { return vec3f(0.90, 0.90, 0.10); }
    default: { return vec3f(0.40, 0.40, 0.60); }
  }
}
`;

const fullscreenVertexWGSL = String.raw`
struct FullscreenOut {
  @builtin(position) position: vec4f,
};

@vertex
fn main(@builtin(vertex_index) vertexIndex: u32) -> FullscreenOut {
  var positions = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f(3.0, -1.0),
    vec2f(-1.0, 3.0)
  );

  var out: FullscreenOut;
  out.position = vec4f(positions[vertexIndex], 0.0, 1.0);
  return out;
}
`;

const waveInitWGSL = simCommonWGSL + String.raw`
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let sim = simSizeU();
  if (gid.x >= sim.x || gid.y >= sim.y) {
    return;
  }

  let xPx = vec2f(f32(gid.x), f32(gid.y)) + vec2f(0.5, 0.5);
  let psi0 = initialPacketAtPx(xPx, 0.0);

  let psiE = initialPacketAtPx(xPx + vec2f(1.0, 0.0), 0.0);
  let psiW = initialPacketAtPx(xPx + vec2f(-1.0, 0.0), 0.0);
  let psiN = initialPacketAtPx(xPx + vec2f(0.0, 1.0), 0.0);
  let psiS = initialPacketAtPx(xPx + vec2f(0.0, -1.0), 0.0);
  let lap0 = psiE + psiW + psiN + psiS - 4.0 * psi0;

  let V = barrierPotentialPx(xPx);
  var rhs0 = schrodingerRHS(psi0, lap0, V);
  rhs0 += -(absorbW(xPx) / params.hbar) * psi0;

  let psiPrev = psi0 - params.dt * rhs0;
  textureStore(outputTex, vec2i(gid.xy), vec4f(psi0, psiPrev));
}
`;

const waveStepWGSL = simCommonWGSL + String.raw`
@group(0) @binding(1) var stateTex: texture_2d<f32>;
@group(0) @binding(2) var outputTex: texture_storage_2d<rgba32float, write>;

fn fetchPsi(q: vec2i) -> vec2f {
  let sim = simSizeI();
  if (q.x < 0 || q.y < 0 || q.x >= sim.x || q.y >= sim.y) {
    return vec2f(0.0, 0.0);
  }
  return textureLoad(stateTex, q, 0).xy;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let sim = simSizeU();
  if (gid.x >= sim.x || gid.y >= sim.y) {
    return;
  }

  let p = vec2i(gid.xy);
  let s = textureLoad(stateTex, p, 0);
  let psi = s.xy;
  let psiPrev = s.zw;

  let psiE = fetchPsi(p + vec2i(1, 0));
  let psiW = fetchPsi(p + vec2i(-1, 0));
  let psiN = fetchPsi(p + vec2i(0, 1));
  let psiS = fetchPsi(p + vec2i(0, -1));
  let lapPsi = psiE + psiW + psiN + psiS - 4.0 * psi;

  let xPx = vec2f(f32(p.x), f32(p.y));
  let V = barrierPotentialPx(xPx);
  var rhs = schrodingerRHS(psi, lapPsi, V);
  rhs += -(absorbW(xPx) / params.hbar) * psi;

  let psiNext = psiPrev + 2.0 * params.dt * rhs;
  textureStore(outputTex, p, vec4f(psiNext, psi));
}
`;

const particleUpdateWGSL = simCommonWGSL + String.raw`
struct ParticleState {
  pos: vec2f,
  mode: f32,
  pad: f32,
};

struct BoundaryAction {
  freeze: bool,
  frozenPos: vec2f,
};

@group(0) @binding(1) var waveTex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> particlesIn: array<ParticleState>;
@group(0) @binding(3) var<storage, read_write> particlesOut: array<ParticleState>;

fn boundaryAction(xPx: vec2f) -> BoundaryAction {
  let base = params.absorbPx + params.particleKillMargin;
  let absDistX = 1.5 * base;
  let absDistY = 1.0 * base;
  let freezeDistX = 1.5 * absDistX;
  let freezeDistXLeft = freezeDistX * 1.20;
  let freezeDistY = 1.5 * absDistY;

  let w = params.simRes.x - 1.0;
  let h = params.simRes.y - 1.0;

  if (xPx.x < freezeDistXLeft) {
    return BoundaryAction(true, vec2f(freezeDistXLeft, clamp(xPx.y, 0.0, h)));
  }
  if (xPx.x > (w - freezeDistX)) {
    return BoundaryAction(true, vec2f(w - freezeDistX, clamp(xPx.y, 0.0, h)));
  }
  if (xPx.y < freezeDistY) {
    return BoundaryAction(true, vec2f(clamp(xPx.x, 0.0, w), freezeDistY));
  }
  if (xPx.y > (h - freezeDistY)) {
    return BoundaryAction(true, vec2f(clamp(xPx.x, 0.0, w), h - freezeDistY));
  }

  return BoundaryAction(false, xPx);
}

fn samplePsiBilinear(xPx: vec2f) -> vec2f {
  let maxX = params.simRes - vec2f(1.0001, 1.0001);
  let x = clamp(xPx, vec2f(0.0, 0.0), maxX);
  let x0 = floor(x);
  let f = x - x0;

  let p00 = vec2i(x0);
  let p10 = min(p00 + vec2i(1, 0), simSizeI() - vec2i(1, 1));
  let p01 = min(p00 + vec2i(0, 1), simSizeI() - vec2i(1, 1));
  let p11 = min(p00 + vec2i(1, 1), simSizeI() - vec2i(1, 1));

  let a = textureLoad(waveTex, p00, 0).xy;
  let b = textureLoad(waveTex, p10, 0).xy;
  let c = textureLoad(waveTex, p01, 0).xy;
  let d = textureLoad(waveTex, p11, 0).xy;

  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

fn bohmVelocity(xPx: vec2f) -> vec2f {
  let psi = samplePsiBilinear(xPx);
  let psiE = samplePsiBilinear(xPx + vec2f(1.0, 0.0));
  let psiW = samplePsiBilinear(xPx + vec2f(-1.0, 0.0));
  let psiN = samplePsiBilinear(xPx + vec2f(0.0, 1.0));
  let psiS = samplePsiBilinear(xPx + vec2f(0.0, -1.0));

  let dpsidx = 0.5 * (psiE - psiW);
  let dpsidy = 0.5 * (psiN - psiS);

  let rho = dot(psi, psi);
  let rhoEff = max(rho, params.rhoMin);
  let a = psi.x;
  let b = psi.y;

  let jx = (params.hbar / params.mass) * (a * dpsidx.y - b * dpsidx.x);
  let jy = (params.hbar / params.mass) * (a * dpsidy.y - b * dpsidy.x);

  var v = vec2f(jx, jy) / rhoEff;
  let sp = length(v);
  if (sp > params.velClamp) {
    v *= params.velClamp / sp;
  }
  return v;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let index = gid.x;
  let count = u32(params.nParticles);
  if (index >= count) {
    return;
  }

  let particle = particlesIn[index];
  let x = particle.pos;
  let mode = particle.mode;

  if (mode < 0.5 || mode > 1.5) {
    particlesOut[index] = particle;
    return;
  }

  let act0 = boundaryAction(x);
  if (barrierWallMaskAtPx(x) > 0.5) {
    particlesOut[index] = ParticleState(vec2f(-10.0, -10.0), 0.0, 0.0);
    return;
  }
  if (act0.freeze) {
    particlesOut[index] = ParticleState(act0.frozenPos, 2.0, 0.0);
    return;
  }

  let v1 = bohmVelocity(x);
  let xm = clamp(x + 0.5 * params.dt * v1, vec2f(0.0, 0.0), params.simRes - vec2f(1.0, 1.0));
  let actM = boundaryAction(xm);
  if (actM.freeze) {
    particlesOut[index] = ParticleState(actM.frozenPos, 2.0, 0.0);
    return;
  }

  let v2 = bohmVelocity(xm);
  let xn = x + params.dt * v2;
  let actN = boundaryAction(xn);
  if (actN.freeze) {
    particlesOut[index] = ParticleState(actN.frozenPos, 2.0, 0.0);
    return;
  }

  particlesOut[index] = ParticleState(xn, 1.0, 0.0);
}
`;

const waveRenderWGSL = simCommonWGSL + paletteWGSL + String.raw`
@group(0) @binding(1) var waveTex: texture_2d<f32>;

struct FragIn {
  @builtin(position) position: vec4f,
};

@fragment
fn main(in: FragIn) -> @location(0) vec4f {
  let canvas = canvasSizeU();
  let sim = simSizeU();
  let pixel = min(vec2u(in.position.xy), canvas - vec2u(1u, 1u));
  let uv = (vec2f(pixel) + vec2f(0.5, 0.5)) / params.canvasRes;
  let simCoord = min(vec2u(uv * params.simRes), sim - vec2u(1u, 1u));
  let psi = textureLoad(waveTex, vec2i(simCoord), 0).xy;
  let rho = dot(psi, psi);

  var intensity = 1.0 - exp(-params.visGain * rho);
  intensity = pow(clamp(intensity, 0.0, 1.0), params.visGamma);

  var paletteId = u32(params.paletteId);
  if (paletteId == 5u && params.showPhase < 0.5) {
    paletteId = 2u;
  }

  let pal = wavePalette(paletteId);
  var color = vec3f(0.0, 0.0, 0.0);

  if (params.showPhase >= 0.5) {
    let phase = atan2(psi.y, psi.x);
    let t = fract((phase + PI) / TAU);
    color = palette(t, pal.a, pal.b, pal.c, pal.d) * intensity;
  } else {
    color = palette(intensity, pal.a, pal.b, pal.c, pal.d) * (0.15 + 0.85 * intensity);
  }

  let wall = barrierWallMaskAtPx(uv * params.simRes);
  let wallColor = vec3f(0.20, 0.28, 0.35);
  let wallAlpha = wall * (0.10 + 0.80 * clamp(params.barrierOpacity, 0.0, 1.0));
  color = mix(color, wallColor, wallAlpha);

  return vec4f(color, 1.0);
}
`;

const densityStepWGSL = simCommonWGSL + String.raw`
@group(0) @binding(1) var prevTex: texture_2d<f32>;

struct FragIn {
  @builtin(position) position: vec4f,
};

@fragment
fn main(in: FragIn) -> @location(0) vec4f {
  let canvas = canvasSizeU();
  let pixel = min(vec2u(in.position.xy), canvas - vec2u(1u, 1u));
  let prev = max(textureLoad(prevTex, vec2i(pixel), 0), vec4f(0.0, 0.0, 0.0, 0.0));

  var fade = 0.0;
  if (params.trailHalfLife > 0.0) {
    let dtTotal = params.dt * params.stepsPerFrame;
    fade = exp(-LN2 * (dtTotal / params.trailHalfLife));
  }

  return prev * fade;
}
`;

const densityRenderWGSL = simCommonWGSL + String.raw`
@group(0) @binding(1) var densityTex: texture_2d<f32>;

struct FragIn {
  @builtin(position) position: vec4f,
};

@fragment
fn main(in: FragIn) -> @location(0) vec4f {
  let canvas = canvasSizeU();
  let pixel = min(vec2u(in.position.xy), canvas - vec2u(1u, 1u));
  let dacc = max(textureLoad(densityTex, vec2i(pixel), 0), vec4f(0.0, 0.0, 0.0, 0.0));
  var value = max(max(dacc.x, dacc.y), dacc.z);
  value = 1.0 - exp(-params.trailVisGain * value);
  value = pow(clamp(value, 0.0, 1.0), params.trailVisGamma);

  let color = vec3f(1.0, 1.0, 0.0);
  let mode = u32(params.trailBlendMode);

  if (mode == 1u) {
    return vec4f(color * value, 1.0 - value);
  }
  if (mode == 2u) {
    return vec4f(color * value, 1.0);
  }
  return vec4f(color, value);
}
`;

const spriteVertexCommonWGSL = simCommonWGSL + String.raw`
struct ParticleState {
  pos: vec2f,
  mode: f32,
  pad: f32,
};

struct SpriteOut {
  @builtin(position) position: vec4f,
  @location(0) localPos: vec2f,
  @location(1) mode: f32,
};

fn spriteCorner(index: u32) -> vec2f {
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0),
    vec2f(1.0, -1.0),
    vec2f(1.0, 1.0),
    vec2f(-1.0, -1.0),
    vec2f(1.0, 1.0),
    vec2f(-1.0, 1.0)
  );
  return corners[index];
}

fn particleCenterNDC(pos: vec2f) -> vec2f {
  let canvasPos = pos / params.simRes * params.canvasRes;
  return vec2f(
    canvasPos.x / params.canvasRes.x * 2.0 - 1.0,
    1.0 - canvasPos.y / params.canvasRes.y * 2.0
  );
}
`;

const particleRenderVertexWGSL = spriteVertexCommonWGSL + String.raw`
@group(0) @binding(1) var<storage, read> particles: array<ParticleState>;

@vertex
fn main(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32,
) -> SpriteOut {
  let particle = particles[instanceIndex];
  let corner = spriteCorner(vertexIndex);
  let halfSize = max(0.5 * params.dotSize, 0.5);
  let offset = vec2f(
    corner.x * halfSize * 2.0 / params.canvasRes.x,
    -corner.y * halfSize * 2.0 / params.canvasRes.y
  );

  var out: SpriteOut;
  out.position = vec4f(particleCenterNDC(particle.pos) + offset, 0.0, 1.0);
  out.localPos = corner * 0.5;
  out.mode = particle.mode;
  return out;
}
`;

const particleRenderFragmentWGSL = simCommonWGSL + paletteWGSL + String.raw`
struct SpriteIn {
  @location(0) localPos: vec2f,
  @location(1) mode: f32,
};

@fragment
fn main(in: SpriteIn) -> @location(0) vec4f {
  if (in.mode < 0.5) {
    discard;
  }

  let r = length(in.localPos);
  if (r > 0.5) {
    discard;
  }

  let edge = smoothstep(0.5, 0.42, r);
  let sigma = max(params.dotSigma, 1e-4);
  let blur = exp(-(r * r) / sigma);
  let alpha = clamp(params.dotGain * blur * edge, 0.0, 0.85);

  let pal = particlePalette(u32(params.paletteId));
  let color = palette(0.85, pal.a, pal.b, pal.c, pal.d);
  return vec4f(color, alpha);
}
`;

const trailStampVertexWGSL = spriteVertexCommonWGSL + String.raw`
@group(0) @binding(1) var<storage, read> particles: array<ParticleState>;

@vertex
fn main(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32,
) -> SpriteOut {
  let particle = particles[instanceIndex];
  let corner = spriteCorner(vertexIndex);
  let size = max(select(params.dotSize, params.trailWidth, params.trailWidth > 0.0), 0.5);
  let halfSize = 0.5 * size;
  let offset = vec2f(
    corner.x * halfSize * 2.0 / params.canvasRes.x,
    -corner.y * halfSize * 2.0 / params.canvasRes.y
  );

  var out: SpriteOut;
  out.position = vec4f(particleCenterNDC(particle.pos) + offset, 0.0, 1.0);
  out.localPos = corner * 0.5;
  out.mode = particle.mode;
  return out;
}
`;

const trailStampFragmentWGSL = simCommonWGSL + String.raw`
struct SpriteIn {
  @location(0) localPos: vec2f,
  @location(1) mode: f32,
};

@fragment
fn main(in: SpriteIn) -> @location(0) vec4f {
  if (in.mode < 0.5) {
    discard;
  }

  let r = length(in.localPos);
  if (r > 0.5) {
    discard;
  }

  let edge = smoothstep(0.5, 0.42, r);
  let sigma = max(params.dotSigma, 1e-4);
  let blur = exp(-(r * r) / sigma);
  let value = clamp(params.dotGain * params.trailStampGain * blur * edge, 0.0, 1.0);
  return vec4f(value, value, value, value);
}
`;

const boundaryWGSL = simCommonWGSL + paletteWGSL + String.raw`
@group(0) @binding(1) var<storage, read> rects: array<vec4f, 4>;

struct BoundaryOut {
  @builtin(position) position: vec4f,
  @location(0) color: vec4f,
};

fn rectCorner(index: u32) -> vec2f {
  var corners = array<vec2f, 6>(
    vec2f(0.0, 0.0),
    vec2f(1.0, 0.0),
    vec2f(1.0, 1.0),
    vec2f(0.0, 0.0),
    vec2f(1.0, 1.0),
    vec2f(0.0, 1.0)
  );
  return corners[index];
}

@vertex
fn vs(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32,
) -> BoundaryOut {
  let rect = rects[instanceIndex];
  let corner = rectCorner(vertexIndex);
  let pos = vec2f(
    mix(rect.x, rect.z, corner.x),
    mix(rect.y, rect.w, corner.y)
  );

  var out: BoundaryOut;
  out.position = vec4f(pos, 0.0, 1.0);
  out.color = vec4f(complementColor(u32(params.paletteId)), 0.15);
  return out;
}

@fragment
fn fs(in: BoundaryOut) -> @location(0) vec4f {
  return in.color;
}
`;

function createRenderPipeline(label, vertexCode, fragmentCode, format, blend = undefined) {
  const target = blend ? { format, blend } : { format };
  return device.createRenderPipeline({
    label,
    layout: "auto",
    vertex: {
      module: device.createShaderModule({ code: vertexCode }),
      entryPoint: "main",
    },
    fragment: {
      module: device.createShaderModule({ code: fragmentCode }),
      entryPoint: "main",
      targets: [target],
    },
    primitive: {
      topology: "triangle-list",
    },
  });
}

function buildPipelines() {
  pipelines.waveInit = device.createComputePipeline({
    label: "wave-init",
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: waveInitWGSL }),
      entryPoint: "main",
    },
  });

  pipelines.waveStep = device.createComputePipeline({
    label: "wave-step",
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: waveStepWGSL }),
      entryPoint: "main",
    },
  });

  pipelines.particleUpdate = device.createComputePipeline({
    label: "particle-update",
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: particleUpdateWGSL }),
      entryPoint: "main",
    },
  });

  pipelines.waveRender = createRenderPipeline(
    "wave-render",
    fullscreenVertexWGSL,
    waveRenderWGSL,
    presentationFormat,
  );

  pipelines.densityStep = createRenderPipeline(
    "density-step",
    fullscreenVertexWGSL,
    densityStepWGSL,
    DENSITY_FORMAT,
  );

  const additiveBlend = {
    color: { operation: "add", srcFactor: "src-alpha", dstFactor: "one" },
    alpha: { operation: "add", srcFactor: "one", dstFactor: "one" },
  };

  const screenBlend = {
    color: { operation: "add", srcFactor: "src-alpha", dstFactor: "one-minus-src" },
    alpha: { operation: "add", srcFactor: "one", dstFactor: "one-minus-src-alpha" },
  };

  const denseBlend = {
    color: { operation: "add", srcFactor: "one", dstFactor: "one" },
    alpha: { operation: "add", srcFactor: "one", dstFactor: "one" },
  };

  pipelines.densityRenderAdd = createRenderPipeline(
    "density-render-add",
    fullscreenVertexWGSL,
    densityRenderWGSL,
    presentationFormat,
    additiveBlend,
  );

  pipelines.densityRenderScreen = createRenderPipeline(
    "density-render-screen",
    fullscreenVertexWGSL,
    densityRenderWGSL,
    presentationFormat,
    screenBlend,
  );

  pipelines.densityRenderDense = createRenderPipeline(
    "density-render-dense",
    fullscreenVertexWGSL,
    densityRenderWGSL,
    presentationFormat,
    denseBlend,
  );

  pipelines.particleRender = createRenderPipeline(
    "particle-render",
    particleRenderVertexWGSL,
    particleRenderFragmentWGSL,
    presentationFormat,
    additiveBlend,
  );

  pipelines.trailStamp = createRenderPipeline(
    "trail-stamp",
    trailStampVertexWGSL,
    trailStampFragmentWGSL,
    DENSITY_FORMAT,
    denseBlend,
  );

  pipelines.boundary = device.createRenderPipeline({
    label: "boundary-render",
    layout: "auto",
    vertex: {
      module: device.createShaderModule({ code: boundaryWGSL }),
      entryPoint: "vs",
    },
    fragment: {
      module: device.createShaderModule({ code: boundaryWGSL }),
      entryPoint: "fs",
      targets: [{
        format: presentationFormat,
        blend: {
          color: { operation: "add", srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
          alpha: { operation: "add", srcFactor: "one", dstFactor: "one-minus-src-alpha" },
        },
      }],
    },
    primitive: {
      topology: "triangle-list",
    },
  });
}

function createWaveBindGroups() {
  bindGroups.waveInit = waveTextures.map((entry) => device.createBindGroup({
    layout: pipelines.waveInit.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: entry.view },
    ],
  }));

  bindGroups.waveStep = [
    device.createBindGroup({
      layout: pipelines.waveStep.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: waveTextures[0].view },
        { binding: 2, resource: waveTextures[1].view },
      ],
    }),
    device.createBindGroup({
      layout: pipelines.waveStep.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: waveTextures[1].view },
        { binding: 2, resource: waveTextures[0].view },
      ],
    }),
  ];

  bindGroups.waveRender = waveTextures.map((entry) => device.createBindGroup({
    layout: pipelines.waveRender.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: entry.view },
    ],
  }));
}

function createDensityBindGroups() {
  bindGroups.densityStep = [
    device.createBindGroup({
      layout: pipelines.densityStep.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: densityTextures[0].view },
      ],
    }),
    device.createBindGroup({
      layout: pipelines.densityStep.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: densityTextures[1].view },
      ],
    }),
  ];

  bindGroups.densityRenderAdd = densityTextures.map((entry) => device.createBindGroup({
    layout: pipelines.densityRenderAdd.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: entry.view },
    ],
  }));

  bindGroups.densityRenderScreen = densityTextures.map((entry) => device.createBindGroup({
    layout: pipelines.densityRenderScreen.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: entry.view },
    ],
  }));

  bindGroups.densityRenderDense = densityTextures.map((entry) => device.createBindGroup({
    layout: pipelines.densityRenderDense.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: entry.view },
    ],
  }));
}

function createParticleBindGroups() {
  if (!particleBuffers.length) return;

  bindGroups.particleUpdate = [
    device.createBindGroup({
      layout: pipelines.particleUpdate.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: waveTextures[1].view },
        { binding: 2, resource: { buffer: particleBuffers[0] } },
        { binding: 3, resource: { buffer: particleBuffers[1] } },
      ],
    }),
    device.createBindGroup({
      layout: pipelines.particleUpdate.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: waveTextures[0].view },
        { binding: 2, resource: { buffer: particleBuffers[1] } },
        { binding: 3, resource: { buffer: particleBuffers[0] } },
      ],
    }),
  ];

  bindGroups.particleRender = particleBuffers.map((buffer) => device.createBindGroup({
    layout: pipelines.particleRender.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer } },
    ],
  }));

  bindGroups.trailStamp = particleBuffers.map((buffer) => device.createBindGroup({
    layout: pipelines.trailStamp.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer } },
    ],
  }));
}

function createBoundaryBindGroup() {
  bindGroups.boundary = device.createBindGroup({
    layout: pipelines.boundary.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: boundaryRectBuffer } },
    ],
  });
}

function rebuildBindGroups() {
  createWaveBindGroups();
  createDensityBindGroups();
  createParticleBindGroups();
  createBoundaryBindGroup();
}

function clearDensityTextures(encoder) {
  for (const entry of densityTextures) {
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: entry.view,
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });
    pass.end();
  }
}

function resetWaveAndDensity() {
  if (!device) return;

  waveFlip = 0;
  densityFlip = 0;
  writeSimParams();

  const encoder = device.createCommandEncoder({ label: "reset-sim" });
  const groupsX = Math.ceil(simW / 8);
  const groupsY = Math.ceil(simH / 8);

  const computePass = encoder.beginComputePass();
  computePass.setPipeline(pipelines.waveInit);
  computePass.setBindGroup(0, bindGroups.waveInit[0]);
  computePass.dispatchWorkgroups(groupsX, groupsY);
  computePass.setBindGroup(0, bindGroups.waveInit[1]);
  computePass.dispatchWorkgroups(groupsX, groupsY);
  computePass.end();

  clearDensityTextures(encoder);
  device.queue.submit([encoder.finish()]);
}

function rebuildSimulation() {
  if (!device) return;

  resizeCanvas();

  simW = Math.max(64, Math.floor(canvas.width * params.simScale));
  simH = Math.max(64, Math.floor(canvas.height * params.simScale));
  densW = canvas.width;
  densH = canvas.height;

  destroyTextureSet(waveTextures);
  destroyTextureSet(densityTextures);

  waveTextures = [
    createTexture("wave-a", simW, simH, WAVE_FORMAT, GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING),
    createTexture("wave-b", simW, simH, WAVE_FORMAT, GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING),
  ];

  densityTextures = [
    createTexture("density-a", densW, densH, DENSITY_FORMAT, GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT),
    createTexture("density-b", densW, densH, DENSITY_FORMAT, GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT),
  ];

  rebuildParticles();
  rebuildBindGroups();
  resetWaveAndDensity();
  updateStats();
}

function resetAll() {
  if (!device) return;
  rebuildParticles();
  createParticleBindGroups();
  resetWaveAndDensity();
}

function updateStats() {
  statsEl.innerHTML = [
    "<b>Backend:</b> WebGPU",
    `<b>Canvas:</b> ${canvas.width} x ${canvas.height}`,
    `<b>Sim grid:</b> ${simW} x ${simH}`,
    `<b>Particles:</b> ${particleCount}`,
    `<b>Steps/frame:</b> ${Math.floor(params.stepsPerFrame)}`,
    `<b>Status:</b> ${paused ? "Paused" : "Running"}`,
  ].join("<br>");
}

function encodeSimulation(encoder) {
  const steps = Math.max(1, Math.floor(params.stepsPerFrame));
  const waveGroupsX = Math.ceil(simW / 8);
  const waveGroupsY = Math.ceil(simH / 8);
  const particleGroups = Math.ceil(particleCount / 64);

  const computePass = encoder.beginComputePass({ label: "simulate" });
  for (let i = 0; i < steps; i++) {
    computePass.setPipeline(pipelines.waveStep);
    computePass.setBindGroup(0, bindGroups.waveStep[waveFlip]);
    computePass.dispatchWorkgroups(waveGroupsX, waveGroupsY);
    waveFlip = 1 - waveFlip;

    computePass.setPipeline(pipelines.particleUpdate);
    computePass.setBindGroup(0, bindGroups.particleUpdate[particleFlip]);
    computePass.dispatchWorkgroups(particleGroups);
    particleFlip = 1 - particleFlip;
  }
  computePass.end();

  const dstIndex = 1 - densityFlip;
  const densityPass = encoder.beginRenderPass({
    label: "density-step-and-stamp",
    colorAttachments: [{
      view: densityTextures[dstIndex].view,
      clearValue: { r: 0, g: 0, b: 0, a: 0 },
      loadOp: "clear",
      storeOp: "store",
    }],
  });

  densityPass.setPipeline(pipelines.densityStep);
  densityPass.setBindGroup(0, bindGroups.densityStep[densityFlip]);
  densityPass.draw(3);

  densityPass.setPipeline(pipelines.trailStamp);
  densityPass.setBindGroup(0, bindGroups.trailStamp[particleFlip]);
  densityPass.draw(6, particleCount);
  densityPass.end();

  densityFlip = dstIndex;
}

function densityOverlayPipeline() {
  const mode = Math.max(0, Math.min(2, Math.round(params.trailBlendMode)));
  if (mode === 1) return [pipelines.densityRenderScreen, bindGroups.densityRenderScreen];
  if (mode === 2) return [pipelines.densityRenderDense, bindGroups.densityRenderDense];
  return [pipelines.densityRenderAdd, bindGroups.densityRenderAdd];
}

function encodeRender(encoder) {
  const currentTexture = context.getCurrentTexture();
  const view = currentTexture.createView();

  const pass = encoder.beginRenderPass({
    label: "render-frame",
    colorAttachments: [{
      view,
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
      loadOp: "clear",
      storeOp: "store",
    }],
  });

  pass.setPipeline(pipelines.waveRender);
  pass.setBindGroup(0, bindGroups.waveRender[waveFlip]);
  pass.draw(3);

  if (params.showTrail) {
    const [pipeline, groups] = densityOverlayPipeline();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, groups[densityFlip]);
    pass.draw(3);
  }

  pass.setPipeline(pipelines.boundary);
  pass.setBindGroup(0, bindGroups.boundary);
  pass.draw(6, 4);

  if (params.showParticles) {
    pass.setPipeline(pipelines.particleRender);
    pass.setBindGroup(0, bindGroups.particleRender[particleFlip]);
    pass.draw(6, particleCount);
  }

  pass.end();
}

async function initWebGPU() {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("Unable to acquire a WebGPU adapter.");

  device = await adapter.requestDevice();
  context = canvas.getContext("webgpu");
  if (!context) throw new Error("Unable to acquire a WebGPU canvas context.");
  presentationFormat = navigator.gpu.getPreferredCanvasFormat();

  resizeCanvas();
  context.configure({
    device,
    format: presentationFormat,
    alphaMode: "opaque",
  });

  uniformBuffer = device.createBuffer({
    label: "sim-uniforms",
    size: UNIFORM_FLOATS * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  boundaryRectBuffer = device.createBuffer({
    label: "boundary-rects",
    size: 16 * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  buildPipelines();
  rebuildSimulation();
}

function frame() {
  if (resizeCanvas()) rebuildSimulation();

  writeSimParams();
  updateBoundaryRects();

  const encoder = device.createCommandEncoder({ label: "frame" });
  if (!paused) encodeSimulation(encoder);
  encodeRender(encoder);
  device.queue.submit([encoder.finish()]);

  updateStats();
  requestAnimationFrame(frame);
}

window.addEventListener("resize", () => {
  if (device && resizeCanvas()) rebuildSimulation();
});

initWebGPU()
  .then(() => {
    requestAnimationFrame(frame);
  })
  .catch((err) => {
    console.error(err);
    alert(String(err));
  });
