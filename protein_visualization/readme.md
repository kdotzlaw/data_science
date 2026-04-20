# Protein Structure Viewer

A single-page Dash app for exploring 3D protein structures in the browser. Upload a local `.pdb` / `.cif` file or fetch any entry from the [RCSB Protein Data Bank](https://www.rcsb.org/) by ID, then inspect the structure interactively with `dash-bio`'s `Molecule3dViewer`.

## Features

- **Load structures two ways** — drag-and-drop a local `.pdb` or `.cif` file, or type a 4-character PDB ID (e.g. `1CRN`, `4HHB`) and fetch directly from RCSB.
- **Visualization controls** — switch between cartoon / stick / sphere rendering, recolor by atom / residue / chain / residue type, and change the background color.
- **Ligand handling** — toggle HETATM records (heme groups, cofactors, waters) on/off and render them independently of the main protein style.
- **Selection tools** — filter by chain, highlight a residue range (`10-50` or `10,15,20`), or click any residue in the sequence view to focus it in the 3D viewer.
- **Sequence view** — one-letter amino acid sequence rendered above the viewer, grouped by chain, with hover tooltips showing chain + residue number.
- **Structure stats panel** (expandable) — per-chain residue/atom/HETATM counts, an amino acid composition bar chart, and a Ramachandran plot computed from the backbone dihedral angles.

## Installation

```bash
pip install -r requirements.txt
pip install "dash-bio>=1.0.2" --no-deps
```

`dash-bio` is installed with `--no-deps` because its `ParmEd` dependency has no Windows wheel and requires MSVC to build from source. A local stub ([parmed.py](parmed.py)) satisfies the import without pulling in the real package — the viewer does not need `ParmEd`'s functionality. All other transitive dependencies are pinned explicitly in [requirements.txt](requirements.txt).

Python 3.8–3.11 is recommended. On 3.12+, some transitive deps may lack wheels.

## Running

```bash
python protein_dashboard.py
```

Then open http://127.0.0.1:8050 in a browser.

## Usage

1. **Load a structure.** Either drag a `.pdb`/`.cif` file onto the upload area, or type a PDB ID (e.g. `1CRN`) and click **Fetch from RCSB**.
2. **Adjust rendering.** Use the Style / Color scheme / Background controls to change how the structure looks.
3. **Explore.** Pick chains from the chain filter, type a residue range, or click residues in the sequence strip to highlight them in the 3D viewer.
4. **Inspect details.** Expand the **Structure details** panel for chain breakdown, composition, and Ramachandran plot.

Quick test IDs: `1CRN` (crambin, 327 atoms, 1 chain) for a fast sanity check, `4HHB` (hemoglobin, ~4800 atoms, 4 chains, heme groups) for multi-chain + HETATM behavior.

## Project structure

- [protein_dashboard.py](protein_dashboard.py) — Dash app: layout + callbacks.
- [shared_utils.py](shared_utils.py) — PDB/CIF parsing, RCSB fetch, style/selection/stats helpers.
- [parmed.py](parmed.py) — local stub so `dash-bio` imports cleanly on Windows without building ParmEd.
- [requirements.txt](requirements.txt) — pinned dependencies.
- [protein_visualization_plan.md](protein_visualization_plan.md) — original implementation plan and notes on possible expansions.
