# DoubleSlit2.0

Standalone copy of the full-applet Bohmian double-slit WebGL2 simulation from qontic-dev/notebooks/double-slit-webgl.

The original full-applet.html is the project index.html. The simulation-speed.js helper is included locally.

Each new run holds the initial wave and particles still for 0.5 seconds before motion begins. This is a visual pause, with no trajectory precomputation or hidden physics steps. Manual pause also freezes this opening interval; returning from a paused or hidden tab does not accumulate time to catch up. Playback still advances the existing fixed number of physics steps per rendered frame.

Particle dots and trails use the visual style from qontic-dev/notebooks/delayed-choice: bright cores, soft halos, alpha-blended dots, and colored trails with the same exposure and fading settings. The wave equation and guiding laws are unchanged. Wave initialization, evolution, and particle sampling use consistent pixel-center coordinates, keeping the symmetric wave, slit geometry, and histogram aligned. Top and bottom wave absorption is symmetric about that same midpoint.

The barrier is rendered as brushed titanium with beveled edges, polished slit rims, and subtle engraved details. Its solid face stays visible over the wave and trails. Its thickness and openings follow the simulation geometry; the material changes only the rendering.

The wave's phase view uses the red–blue mapping from `Bohmian Free Packet`: blue at +π/2, red at −π/2, and magenta at ±π. ClassicalLimit's purple density ramp is blended underneath with its default phase weight, `(1 - 0.3)^0.7` (about 78% phase, 22% density). This keeps the density visible where the phase colors fade near zero; the whole blend still fades with density. The circular legend shows the combined colors at full reference brightness and sits beyond the histogram. It appears in the live view and recordings when **show phase** is on. The density-only palette is unchanged.

Adjust `PHASE_AMPLITUDE_CUTOFF` near the top of `shaders/wave_render.frag` to hide faint wave tails in both phase and density views, including the purple density underneath phase colors. It uses raw amplitude `|ψ|` (the initial Gaussian peak is 1), not probability density `|ψ|²` or display brightness. The current value `0.01` hides the wave below amplitude 0.01 and smoothly restores full visibility by 0.02. Higher values hide more faint wave detail; `0` disables the cutoff. Reload after editing. The same cutoff applies to recordings.

The **up/down** and **right/left** buttons independently color particles by their starting position relative to the Gaussian's center. Both are on by default:

| Starting quadrant | Color |
| --- | --- |
| Upper left | Original yellow |
| Upper right | Coral |
| Lower left | Cyan |
| Lower right | Violet |

With only **up/down** enabled, upper particles are yellow and lower particles are cyan. With only **right/left** enabled, left particles are yellow and right particles are coral. With both off, all particles and trails are yellow. The shading stays the same in every mode.

Each particle and its trail keep their starting classification, including after crossing the center or reaching a boundary. Either button recolors the full trail history immediately, without resetting the simulation or clearing trails. All four starting quadrants continue to be recorded while either division is off.

The third button, **radial**, overrides both directional settings with four concentric bands around the starting Gaussian center: yellow, cyan, coral, and violet from inside outward. Particles are sorted by their initial distance and divided into four equally populated groups (differing by at most one when the count is not divisible by four). The bands follow the sampled Gaussian rather than using equal radial widths. Turning radial off restores the directional settings. Dots, existing trails, histogram hits, and recordings all follow the selected coloring without restarting the run.

## Stage and offscreen absorption

The **Wide** view shows the full stage height; **Close** returns to the tighter framing. Both shift the scene left by about one fifth of the viewport, or farther when needed to hide the left-edge detections and reserve space for the histogram. The histogram always extends rightward from the right detector, clear of the incoming trails. Switching views or using the mouse wheel preserves the current run, and recordings use the same framing.

The wave grid extends above and below the displayed stage. **Offscreen space** controls the extra grid cells on each side (128–512, default 256). A 24-cell free-propagation gap separates the visible stage from a gradual absorbing layer in each extension. The vertical damping strength scales with the packet's kinetic energy to reduce reflections of slower waves. The upper/lower particle detectors are also outside the stage. The packet, slits, dots, trails, and histogram all use the same visible coordinates, so adding absorption space does not stretch the scene.

More offscreen space allows gentler absorption over a longer distance and increases graphics work. Changing it restarts the run. The grid spacing and integration step are unchanged. The left/right absorption and right detector remain at their existing horizontal positions.

## Right detector histogram

A fixed histogram sits beside the right detector and follows its position when zooming the view. It records each right-edge detection within the visible stage once into 64 vertical bins. Left, top, bottom, and offscreen-extension detections are excluded. Offscreen hits are never clamped into the outermost visible bins. The horizontal scale shows actual hits per bin; the bars are not smoothed. Their stacked colors follow the up/down and right/left buttons without changing counts.

The bins have an exact boundary at the vertical midpoint, with 32 bins on each side and a subtle center divider. Colors continue to represent starting positions: a trajectory that crosses the midpoint can still contribute to a mixed-color bin. There is no total-hit counter.

The **right histogram** button hides or shows the display while counting continues. Ensemble resets clear the histogram. In which-slit mode, completed runs and **Next particle (R)** preserve all earlier hits; **Clear histogram** clears the counts without restarting the current particle. Changing the physical experiment, rebuilding the grid, or leaving which-slit mode clears the series. Readback is limited to 10 updates per second during ensemble playback, with a final update on pause; single-particle hits are captured before the next launch.

## Which slit mode

**Which slit** switches to successive single-particle runs. After each particle finishes, its final position is held for half a second, then a fresh run begins with the usual half-second opening hold. Hits accumulate into one histogram. **Next particle (R)** launches another sample immediately while keeping the counts. A reflected or timed-out particle produces no invented right-detector hit. Turning the mode off restores the previous ensemble count.

Starting positions are deliberately biased toward passage through a slit. Each launch chooses either aperture with equal probability and draws from a broad Gaussian proposal that includes the inward-going routes. The earlier tiny interval aimed at each aperture's center mostly selected outward-going paths, making two narrow separated bands. The proposal estimates a starting position by undoing free Gaussian spreading and the selected Pauli spin rotation; narrow openings use particles farther forward in the packet. Only the initial particle sample changes. The initial wave and guiding law stay the same, and the program does not pre-run or select among simulated trajectories. This is still a demonstration sample, not an unbiased Born-distributed ensemble or guaranteed transmission for every setting. The accumulated histogram describes these biased, conditionally localized runs.

The result is determined by the particle's actual forward exit through an opening, followed by a crossing of a detector plane farther downstream. At the default settings, that plane is 35 grid cells beyond the back face and its one-cell feather. The two small brackets mark the plane; the selected slit lights up. Detection is checked every eight physics steps after approaching the wall so an entire rendered frame cannot delay the trigger. It cannot trigger inside the barrier or from a particle that never crossed an opening.

Detection prepares a finite Gaussian centered transversely on the selected slit, at the particle's current longitudinal position. Its carrier momentum is the incident forward momentum with zero mean transverse momentum. The particle keeps its actual position and follows the evolving conditional wave; it is never resampled or moved to a prescribed detector bin. The default amplitude widths are 10 grid cells longitudinally and at least 14 transversely. Slower incident particles use wider packets to reduce the relative momentum spread: at momentum 0.5 the nominal widths are 20 by about 24 cells. The detector moves with the width to leave 3.5 longitudinal widths of clearance behind the packet, and the longitudinal width is limited by the available flight distance to reserve space before the screen too. Less than 0.00004% of an unmodified Gaussian probability then lies behind the rear face at preparation. The width and clearance settings are in `WhichSlitMeasurement.detector()` in `which-slit.js`.

The transverse width can increase further when the detected particle is already well away from the selected slit's axis. Its offset is limited to 1.5 amplitude widths, with a tighter limit for low forward momentum. This avoids placing the actual particle in a vanishing Gaussian tail, where a steep density gradient would produce rapid transverse spreading and, with Pauli guidance, strong backward motion. The preparation keeps the slit-centered packet and the original particle position, while allowing a gentler continuation. This is a finite-width conditional preparation, not the exact diffraction field of a hard-edged aperture.

The prepared Gaussian is shaped to respect the nearby solid barrier. A smooth envelope is zero throughout the wall and its potential feather, and rises to full amplitude over a distance of at most 2 grid cells, limited to half the slit width. The clearance keeps this correction in a negligible boundary tail. The envelope is applied to the new physical wave state during initialization. Its subsequent propagation uses the existing barrier and wave equation. A second-order backward time history, calculated with the same Hamiltonian and absorption, also suppresses leapfrog startup oscillations. The display cutoff is unchanged. Particles can still reflect or miss the visible screen, depending on their actual trajectory; only real visible hits enter the histogram.

The conditioning lasts at most four simulation time units, shortened at high momentum to finish within approximately six cells of forward travel. At the default settings it takes about one ninth of a second at 60 Hz. During that interval, the old state and Gaussian state evolve separately with the same barrier and absorbers. A phase-aligned, overlap-normalized mixture uses a quintic ramp with zero first and second derivatives at its endpoints. Both stored time levels use the coefficient for their own time, avoiding an inconsistent leapfrog restart. The particle and trail use this conditional wave throughout. After the ramp, ordinary wave evolution continues from the resulting Gaussian state; the old branch is discarded.

This is an illustrative conditional Gaussian preparation, not a microscopic simulation of detector entanglement or a stochastic continuous-measurement equation. The longitudinal localization also contains arrival information. The mixture's normalization uses the overlap at the start of detection; the existing boundary absorbers continue to remove probability. Its biased launches and illustrative conditioning are not a quantitative Born-rule measurement model. In an ideal which-path experiment the two conditional probabilities add without a two-slit interference term; depending on widths and separation, that sum can still have two broad peaks. See [Feynman, Quantum Behavior, section 1–6](https://www.feynmanlectures.caltech.edu/III_01.html#Ch1-S6) and [Jacobs and Steck, A Straightforward Introduction to Continuous Quantum Measurement](https://arxiv.org/abs/quant-ph/0611067). Colored phase bands in one outgoing packet are not themselves interference fringes in its probability density.

The recorder includes the detector markers and accumulated histogram, and repeats single-particle runs just like the live view. Recording from the current time copies any transition already in progress, the delay before the next run, and the random-generator state for future launches. Recording from the beginning of the current run starts with the hits accumulated before that run, avoiding a duplicate count of its own hit. Resetting, switching modes, or completing the transition releases the temporary wave buffers.

## Video export

The **Video export** controls use the recorder from ClassicalLimit: 30 fps, 1920 or 2560 pixels wide (2K by default), 1–120 seconds, and **Beginning** or **Current time**. The height follows the current stage's aspect ratio. The recording includes the wave, barrier, particles, trails, and right histogram; the main controls and Theory panel are excluded. The histogram is included even if hidden in the live view.

Each frame is rendered at twice the output dimensions before downsampling and encoding. MP4 is preferred, with WebM as a fallback. The local encoder bundle needs no external service. The result downloads automatically, with a **Save again** link afterward. Chrome or Edge with WebCodecs is required.

An isolated rendering copy keeps the existing physical grid and initial particle sample. **Current time** also copies the current wave, particles, trails, and detector counts. **Beginning** keeps the 0.5-second opening hold. Video time advances by frame numbers, so slow rendering does not drop frames or speed up the physics. Finishing or cancelling preserves the live run, including when the window is resized during export.

## Run locally

From this folder, run:

```powershell
python -m http.server 8000
```

Then open http://localhost:8000 in a desktop browser supporting WebGL2 and floating-point render targets. Serve over HTTP because the app uses JavaScript modules and fetches shader files.

## Included

- index.html: original full-applet page
- main.js: simulation
- simulation-speed.js: required shared helper
- right-detector-histogram.js: hit counting and histogram display
- export.js and recorder-ui.js: video rendering and recording controls adapted from ClassicalLimit
- which-slit.js: slit-crossing detection and finite-duration Gaussian conditioning
- vendor/: ClassicalLimit's Mediabunny 1.56.0 bundle, license, and source notice
- shaders/: simulation and rendering shaders
- LICENSE: original MIT license and attribution

The notebook page, explanatory images, notebook metadata, and unused recording script are excluded.
