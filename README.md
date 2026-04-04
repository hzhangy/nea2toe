# nea2toe – N.E.A. Framework: The Continuous Zig-Zag and the Meta-Theoretical Closure

This repository hosts Papers VII and VIII of the N.E.A. (Network Emergent Architecture) series, completing the foundational phase of the framework.

## Contents

- `paper_VII.tex` / `paper_VII.pdf` – Paper VII: *Anisotropy Stretches Space, Isotropy Compresses Space*  
  Establishes the universal topological law and unifies physics, chemistry, biology, economics and politics under a single continuous zig‑zag.
- `paper_VIII.tex` / `paper_VIII.pdf` – Paper VIII: *The Completion of the Meta-Theoretical Framework*  
  Audits the observer’s cognitive horizon, introduces the observer rent, and proposes the \((M,N)\) saturation conjecture.
- `code/` – All numerical experiment scripts:
  - `nea_anisotropy_evolution.py` – Core graph experiment (Table 4)
  - `nea_fractal_34_scaling.py` – Third‑order fractal scaling (Kleiber’s law)
  - `nea_full_spectrum_auditor.py` – Cross‑layer audits (chemistry, biology, economics, politics)
  - `nea_meta_observer_limit.py` – Meta‑observer audit for Paper VIII
- `figures/` – Contains `nea_fractal_34_scaling.png` (Figure 1 of Paper VII)

## Requirements

- Python 3.9+ with `numpy`, `scipy`, `matplotlib`
- LaTeX (e.g., `pdflatex`) for compiling the `.tex` files

## Running the Code

All scripts are self‑contained. For example:

```bash
cd code
python nea_anisotropy_evolution.py
python nea_fractal_34_scaling.py
python nea_full_spectrum_auditor.py
python nea_meta_observer_limit.py
