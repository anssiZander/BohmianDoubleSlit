#version 300 es
precision highp float;

in float vAlive;
in float vParticleId;
out vec4 fragColor;

uniform float uDotSigma;
uniform float uDotGain;
uniform float uStampGain;

void main(){
  if(vAlive < 0.5) discard;

  vec2 p = gl_PointCoord - vec2(0.5);
  float r = length(p);

  // Strict circle mask
  if(r > 0.5) discard;

  // Soft edge so stamps don't look pixelly
  float edge = smoothstep(0.5, 0.42, r);

  float s = max(uDotSigma, 1e-4);
  float blur = exp(-(r*r) / s);

  float a = uDotGain * uStampGain * blur * edge;
  a = clamp(a, 0.0, 1.0);

  // constant yellowish color for trails (old theme)
  vec3 particleColor = vec3(1.0, 1.0, 0.0);
  fragColor = vec4(particleColor, a);
}