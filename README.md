# Bohmian Double Slit

WebGL2 Bohmian mechanics double-slit simulation with two selectable guiding laws.

- `Schrodinger`: `v = j / rho`
- `Pauli spin-1/2 (+z)`: `v = j / rho + (hbar / (2 m rho)) * (d_y rho, -d_x rho)`

The wave field is still advanced with the same scalar Schrodinger stepper. In the Pauli mode this corresponds to a factorized spinor with fixed spin-up along `+z` and no magnetic field, so only the trajectory law changes.

## Run

Serve the repository root with any static file server and open `index.html`.

```powershell
py -m http.server
```

Then open `http://localhost:8000/`.

## Controls

- `guiding law` switches between the two trajectory laws and resets the run.
- `Reset` restarts the wave, particles, and trails.
- `Pause` stops time stepping.
- `R` resets the simulation.
