#version 300 es
precision highp float;
precision highp sampler2D;

uniform sampler2D uState; // RG=psi_n, BA=psi_{n-1}
uniform ivec2 uSimRes;

uniform float uHBAR;
uniform float uMass;
uniform float uP0;
uniform float uDT;

uniform float uBarrierXFrac;
uniform float uBarrierThickPx;
uniform float uSlitWidthPx;
uniform float uSlitSepPx;
uniform float uV0;

uniform float uAbsorbPx;        // thickness of absorbing layer (px)
uniform float uAbsorbStrength;  // CAP strength (energy units)

out vec4 fragColor;

float band(float x, float c, float halfW, float feather){
  return smoothstep(c-halfW-feather, c-halfW, x) *
         (1.0 - smoothstep(c+halfW, c+halfW+feather, x));
}

float barrierPotentialPx(vec2 xPx){
  float bx = uBarrierXFrac * float(uSimRes.x);
  float slab = band(xPx.x, bx, 0.5*uBarrierThickPx, 1.0);

  float y0 = 0.5 * float(uSimRes.y);
  float s  = 0.5 * uSlitSepPx;
  float hw = 0.5 * uSlitWidthPx;

  float slit1 = band(xPx.y, y0 - s, hw, 1.0);
  float slit2 = band(xPx.y, y0 + s, hw, 1.0);
  float slits = clamp(slit1 + slit2, 0.0, 1.0);

  float wall = slab * (1.0 - slits);
  return uV0 * wall;
}

// Complex absorbing potential W(x) near edges (energy units).
float absorbW(vec2 xPx){
  if(uAbsorbPx <= 0.0) return 0.0;

  // widen left-side absorbing region by 20%
  float leftFactor = 1.20;
  float dx = min(xPx.x * leftFactor, float(uSimRes.x) - 1.0 - xPx.x);
  float dy = min(xPx.y, float(uSimRes.y) - 1.0 - xPx.y);
  float d  = min(dx, dy);

  float t = clamp((uAbsorbPx - d) / max(uAbsorbPx, 1.0), 0.0, 1.0);
  
  // Smooth cubic profile for gradual absorption
  float profile = t * t * t;
  
  return uAbsorbStrength * profile;
}

// Fetch with "Dirichlet outside = 0" (reduces edge reflection vs clamping).
vec2 fetchPsi(ivec2 q){
  if(q.x < 0 || q.y < 0 || q.x >= uSimRes.x || q.y >= uSimRes.y) return vec2(0.0);
  return texelFetch(uState, q, 0).rg;
}

// RHS for real-imag:
vec2 schrodingerRHS(vec2 psi, vec2 lapPsi, float V){
  // ∂ψ/∂t = i*(ħ/2m)∇²ψ - i*(V/ħ)ψ
  float cLap = uHBAR / (2.0*uMass);
  float cV   = V / uHBAR;
  return vec2(-cLap*lapPsi.y + cV*psi.y,
               cLap*lapPsi.x - cV*psi.x);
}

void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);

  vec4 s = texelFetch(uState, p, 0);
  vec2 psi     = s.rg;
  vec2 psiPrev = s.ba;

  // Laplacian with outside=0
  vec2 psiE = fetchPsi(p + ivec2( 1, 0));
  vec2 psiW = fetchPsi(p + ivec2(-1, 0));
  vec2 psiN = fetchPsi(p + ivec2( 0, 1));
  vec2 psiS = fetchPsi(p + ivec2( 0,-1));
  vec2 lapPsi = (psiE + psiW + psiN + psiS - 4.0*psi);

  vec2 xPx = vec2(p);
  float V = barrierPotentialPx(xPx);

  // Base Schr RHS
  vec2 rhs = schrodingerRHS(psi, lapPsi, V);

  // CAP damping term: V -> V - i W  =>  ψ_t ...  - (W/ħ) ψ
  float W = absorbW(xPx);
  rhs += -(W / uHBAR) * psi;

  // Leapfrog
  vec2 psiNext = psiPrev + 2.0 * uDT * rhs;

  fragColor = vec4(psiNext, psi);
}