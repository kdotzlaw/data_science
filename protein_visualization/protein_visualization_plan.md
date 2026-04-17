# Protein Visualization Dashboard — Implementation Plan

## Context

The user is building a small standalone Dash app that accepts a user-uploaded PDB file and renders the protein structure with [dash-bio](https://dash.plotly.com/dash-bio)'s `Molecule3dViewer`. The target file [protein_dashboard.py](protein_dashboard.py) currently exists as an empty stub. No dash-bio code exists anywhere in the repo yet, and `dash-bio` is not installed.

The new app should mirror the conventions already established in [../Covid-Test/src/covid_dashboard.py](../Covid-Test/src/covid_dashboard.py): a single-file Dash app using `Dash(__name__)`, `html.Div` + `dcc` layout, `@app.callback` wiring, and `app.run(debug=True)` under `__main__`. Reusable helpers live in a sibling `shared_utils.py`, matching [../Covid-Test/shared_utils.py](../Covid-Test/shared_utils.py).

Outcome: a user drags a `.pdb` file onto an upload area, the structure renders in 3D, and dropdown controls let them switch visualization style, color scheme, and background color without re-uploading.

## File structure

Create under [protein_visualization/](./):

- `protein_dashboard.py` — Dash app entry point (fill the empty stub)
- `shared_utils.py` — PDB decoding + parsing helpers
- `requirements.txt` — pinned dependencies
- `sample_pdbs/` — optional local folder for manual testing (user downloads `1crn.pdb`, `4hhb.pdb` from RCSB; not committed)

No `src/` subfolder — this project is small enough to stay flat, matching the empty-stub location the user already chose.

## Dependencies — `requirements.txt`

```
dash>=2.14,<3.0
dash-bio>=1.0.2
pandas
```

Notes:
- Pin `dash<3.0` — `dash-bio` has not fully caught up to Dash 3.x.
- `dash-bio` installs cleanly on Python 3.8–3.11. On 3.12+, transitive deps (`ParmEd`, `periodictable`) sometimes lack wheels; fall back to Python 3.11 if install fails.

## App layout (IDs are load-bearing for callbacks)

Wrap everything in one centered container: `html.Div(..., style={'maxWidth': '1100px', 'margin': 'auto', 'padding': '20px'})` — matches the Covid dashboard.

Components, in order:

- `html.H1('Protein Structure Viewer', style={'textAlign': 'center'})`
- `dcc.Upload(id='pdb-upload', accept='.pdb,chemical/x-pdb', multiple=False, children=html.Div(['Drag and drop or ', html.A('select a .pdb file')]))` with dashed-border styling
- `html.Div(id='upload-status')` — success summary (green) or error (red)
- Controls row (`html.Div` flex):
  - `dcc.Dropdown(id='style-dropdown', options=[cartoon, stick, sphere], value='cartoon', clearable=False)`
  - `dcc.Dropdown(id='color-dropdown', options=[atom, residue, chain, residue_type], value='chain', clearable=False)`
  - `dcc.Input(id='bg-color', type='text', value='#FFFFFF', debounce=True)` — plain hex input (skip `dash_daq` for now)
- `dcc.Store(id='pdb-store')` — holds `{'modelData': {...}, 'filename': str}`
- `Molecule3dViewer(id='mol-viewer', modelData={'atoms': [], 'bonds': []}, styles=[], backgroundColor='#FFFFFF', backgroundOpacity=1.0, selectionType='atom', style={'height': '600px'})`
- `html.Div(id='info-panel')` — atom / chain / residue counts + filename

## Callbacks — split for performance

**Callback A — parse on upload** (runs once per file):

```
Output('pdb-store', 'data')
Output('upload-status', 'children')
Input('pdb-upload', 'contents')
State('pdb-upload', 'filename')
```

Delegates to `parse_uploaded_pdb(contents, filename)` in `shared_utils.py`. On error, returns `(dash.no_update, error_msg)` so the last good structure stays rendered.

**Callback B — restyle** (runs on every control change):

```
Output('mol-viewer', 'modelData')
Output('mol-viewer', 'styles')
Output('mol-viewer', 'backgroundColor')
Output('info-panel', 'children')
Input('pdb-store', 'data')
Input('style-dropdown', 'value')
Input('color-dropdown', 'value')
Input('bg-color', 'value')
```

Raises `PreventUpdate` when the store is empty. Calls `dash_bio.utils.create_mol3d_style(model_data['atoms'], visualization_type=style, color_element=color)` to rebuild the `styles` list.

**Why split:** `PdbParser` is the slow path (reads + parses thousands of atoms). Dropdowns should only rebuild the cheap `styles` list, not re-parse. Parsed `modelData` lives in `dcc.Store` so callback B never touches the upload payload.

## `shared_utils.py` — helper signatures

```python
def parse_uploaded_pdb(contents: str, filename: str) -> tuple[dict | None, str | None]:
    """Decode a dcc.Upload payload and run PdbParser. Return (modelData, error_msg)."""

def summarize_model(model_data: dict, filename: str) -> str:
    """Return 'Loaded 1crn.pdb — 327 atoms, 1 chain, 46 residues'."""
```

`parse_uploaded_pdb` steps:
1. `contents is None` → return `(None, None)` (initial render, not an error).
2. `filename.lower().endswith('.pdb')` — else `(None, 'File must have a .pdb extension')`.
3. Split contents on first `','`, base64-decode, UTF-8 decode with `errors='replace'`.
4. Wrap in `io.StringIO`, pass to `dash_bio.utils.pdb_parser.PdbParser(...).mol3d_data()`. If the installed version rejects file-like objects, fall back to `tempfile.NamedTemporaryFile(suffix='.pdb', delete=False)`, parse, then `os.unlink`.
5. Empty atoms list → `(None, 'PDB contains no ATOM/HETATM records')`.
6. Else `(model_data, None)`.

## Error handling

All failure modes surface in the `upload-status` div — never crash the callback:

- `None` contents → no-op
- Wrong extension → "File must have a .pdb extension"
- Base64 / UTF-8 failure → "Could not decode file — is this a text PDB?"
- `PdbParser` raises → catch broadly (its errors aren't well-typed), log to stderr, show `"Failed to parse PDB: {type(e).__name__}"`
- Empty atoms → "PDB contains no ATOM/HETATM records"

Callback B uses `PreventUpdate` when the store is empty so the viewer keeps its last good state.

## Critical files to modify

- [protein_dashboard.py](protein_dashboard.py) — fill empty stub with app
- [shared_utils.py](shared_utils.py) — new
- [requirements.txt](requirements.txt) — new

Reference (do not modify):
- [../Covid-Test/src/covid_dashboard.py](../Covid-Test/src/covid_dashboard.py) — layout + callback conventions to mirror
- [../Covid-Test/shared_utils.py](../Covid-Test/shared_utils.py) — helper-module conventions

## Verification

1. `cd protein_visualization && pip install -r requirements.txt`
2. `python protein_dashboard.py` — expect `Dash is running on http://127.0.0.1:8050/`
3. Browser → `localhost:8050`. Empty viewer, no status message.
4. Download `1CRN.pdb` from `https://files.rcsb.org/download/1CRN.pdb` (327 atoms, 1 chain — fast sanity check). Drag onto upload zone. Status shows "Loaded 1crn.pdb — 327 atoms, 1 chain, 46 residues". Cartoon ribbon renders.
5. Flip style dropdown: cartoon → stick → sphere. Viewer updates without re-upload (confirms callback split).
6. Flip color dropdown: atom → residue → chain → residue_type.
7. Type `#222222` into bg-color, tab out — background darkens.
8. Repeat with `4HHB.pdb` (hemoglobin, ~4800 atoms, 4 chains) to confirm multi-chain coloring.
9. Negative tests:
   - Upload a `.py` file → "File must have a .pdb extension".
   - Rename a non-PDB text file to `fake.pdb` and upload → "PDB contains no ATOM/HETATM records" or "Failed to parse PDB: ...".

---

# Expansion Plans

Six independent expansions, ordered roughly by effort. Each can ship on its own; they compose cleanly because they all read from the existing `pdb-store` and layer new components/callbacks onto the current app. No breaking changes to [shared_utils.py](shared_utils.py)'s public API are needed — only additive functions.

Shared conventions that every expansion follows:

- New controls go into their own row (`html.Div(style=CONTROLS_ROW_STYLE)`) rather than cramming the existing one.
- New data-producing callbacks write to `pdb-store` (or a sibling store) using `dash.callback_context` to coexist with [protein_dashboard.py:117](protein_dashboard.py#L117) `handle_upload`.
- Helpers live in [shared_utils.py](shared_utils.py); only add a new module if the function exceeds ~50 lines.
- Errors surface to the existing `upload-status` div — no new status regions unless the expansion truly needs one.

## Expansion 1 — Fetch by PDB ID

**Goal:** let users type a 4-char PDB ID (e.g. `1CRN`) and fetch from RCSB instead of downloading first.

**UI additions** (new row above the upload area):
- `dcc.Input(id='pdb-id-input', placeholder='e.g. 1CRN', debounce=True, maxLength=4)`
- `html.Button('Fetch', id='fetch-pdb-btn', n_clicks=0)`

**Callback changes:**
- Merge upload + fetch into one callback that writes `pdb-store`, using `dash.ctx.triggered_id` to pick the source. Inputs: `Input('pdb-upload', 'contents')`, `Input('fetch-pdb-btn', 'n_clicks')`. State: `State('pdb-upload', 'filename')`, `State('pdb-id-input', 'value')`.
- Single `handle_load` function replaces the existing `handle_upload` at [protein_dashboard.py:117](protein_dashboard.py#L117).

**New helper in [shared_utils.py](shared_utils.py):**
```python
def fetch_pdb_by_id(pdb_id: str) -> tuple[dict | None, str | None]:
    """GET https://files.rcsb.org/download/{ID}.pdb, parse, return (modelData, error)."""
```
- Validate with `re.fullmatch(r'[1-9][0-9A-Za-z]{3}', pdb_id)` (RCSB's format).
- `requests.get(..., timeout=10)`; 404 → "No PDB entry for {id}"; timeout → "RCSB request timed out".
- Reuse `_parse_pdb_text` on the response body.

**Dependencies:** add `requests` to [requirements.txt](requirements.txt) (already transitively installed via GEOparse; make it explicit).

**Verification:** type `1CRN`, click Fetch, viewer renders crambin; type `XXXX`, see 404 error; with airplane mode on, see timeout.

**Effort:** ~30 LOC.

## Expansion 2 — Residue / chain selector

**Goal:** highlight specific chains or a residue range via `Molecule3dViewer.selectedAtomIds`.

**UI additions** (new controls row):
- `dcc.Dropdown(id='chain-filter', multi=True, placeholder='All chains')` — options populated dynamically from the loaded structure.
- `dcc.Input(id='residue-range', type='text', placeholder='e.g. 10-50 or 10,15,20', debounce=True)`

**New callbacks:**
- Callback C: `Input('pdb-store', 'data')` → `Output('chain-filter', 'options')`. Derives options from `sorted({a['chain'] for a in atoms})`.
- Callback D: `Input('pdb-store', 'data')` + `Input('chain-filter', 'value')` + `Input('residue-range', 'value')` → `Output('mol-viewer', 'selectedAtomIds')`.

**New helper in [shared_utils.py](shared_utils.py):**
```python
def select_atom_indices(atoms: list, chains: list | None, residue_spec: str | None) -> list[int]:
    """Return positional indices into `atoms` matching the filters."""
```
- Parse `residue_spec`: accept `'10-50'`, `'10,15,20'`, or `'10-50,70'` → set of `residue_index` values.
- Empty selection → return `[]` (viewer shows nothing selected, all atoms still rendered).

**Risks:** `selectedAtomIds` uses positional indices into the flattened atom list — confirm against [Molecule3dViewer](https://dash.plotly.com/dash-bio/molecule3dviewer) docs. If it uses atom `serial` instead, swap accordingly.

**Verification:** load 4HHB, pick chains `['A', 'C']`, verify only those two chains highlight; type `1-10` in range input, verify first 10 residues highlight.

**Effort:** ~40 LOC.

## Expansion 3 — Sequence view

**Goal:** render the one-letter amino acid sequence above the 3D view; clicking a residue focuses it in the viewer.

**UI additions:**
- `html.Div(id='sequence-view', style={'fontFamily': 'monospace', 'letterSpacing': '2px', 'marginBottom': '15px', 'overflowWrap': 'break-word'})` above the `mol-viewer` container.
- Each residue is a clickable `html.Span` with a pattern-matching ID: `{'type': 'residue-span', 'chain': chain, 'index': residue_index}`. Chain labels as `html.Strong` between runs.

**New callbacks:**
- Callback E: `Input('pdb-store', 'data')` → `Output('sequence-view', 'children')`. Calls `build_sequence(atoms)`.
- Callback F (pattern-matching): `Input({'type': 'residue-span', 'chain': ALL, 'index': ALL}, 'n_clicks')` → `Output('mol-viewer', 'selectedAtomIds')`. Uses `dash.ctx.triggered_id` to find which residue was clicked.

**New helpers in [shared_utils.py](shared_utils.py):**
```python
AA_THREE_TO_ONE = {'ALA': 'A', 'ARG': 'R', ..., 'VAL': 'V'}  # 20 standard + 'X' fallback

def build_sequence(atoms: list) -> list[tuple[str, int, str]]:
    """Return [(chain, residue_index, one_letter_code), ...] in order."""
```
- Dedup on `(chain, residue_index)`, preserve insertion order from atoms.
- Unknown residues → `'X'`.

**Edge cases:** Callback F may conflict with Expansion 2's Callback D (both write `selectedAtomIds`). Resolve by merging: one combined callback that reads chain-filter + range input + last-clicked residue.

**Verification:** load 1CRN, see `TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN`; click `C` at position 3, viewer highlights just Cys3.

**Effort:** ~60 LOC.

## Expansion 4 — Structure stats panel

**Goal:** replace the thin info line with an expandable details panel showing chain breakdown, residue composition, and (optional) Ramachandran plot.

**UI changes:**
- Replace `html.Div(id='info-panel')` at [protein_dashboard.py:113](protein_dashboard.py#L113) with:
  ```python
  html.Details([
      html.Summary(id='info-summary'),   # one-line headline
      html.Div(id='stats-body'),         # tables + charts
  ], open=False)
  ```

**New callback:**
- Callback G: `Input('pdb-store', 'data')` → `Output('info-summary', 'children')` + `Output('stats-body', 'children')`.

**New helpers in [shared_utils.py](shared_utils.py) or a new `stats.py`:**
```python
def compute_chain_table(atoms) -> list[dict]:
    """Rows: {'chain': 'A', 'residues': 46, 'atoms': 327}."""

def compute_composition(atoms) -> dict[str, int]:
    """{'ALA': 12, 'ARG': 8, ...} — counts by residue type."""

def compute_ramachandran(atoms) -> tuple[list[float], list[float]]:
    """Return (phi, psi) lists in degrees by iterating N/CA/C atom triplets per chain."""
```

**Rendering:**
- Chain table: `html.Table` built from `compute_chain_table`.
- Composition: `plotly.express.bar(composition)` in a `dcc.Graph`.
- Ramachandran: `plotly.express.scatter(x=phi, y=psi, range_x=[-180,180], range_y=[-180,180])` — cheap for proteins up to ~10k residues.

**Dependencies:** already have `plotly` via `dash`; no new installs.

**Risks:** Ramachandran math uses NumPy dihedral angle calculation — ~20 LOC that's easy to get wrong. Gate it behind a checkbox so a broken calc doesn't block the rest of the panel.

**Verification:** load 4HHB → table shows 4 chains × ~140 residues each; composition bar shows ~12% Ala/Lys/Leu; Ramachandran plot shows the characteristic two-cluster pattern.

**Effort:** ~80 LOC (Ramachandran ~40 of those).

## Expansion 5 — HETATM / ligand toggle

**Goal:** show or hide non-protein atoms (ligands, waters, cofactors) independently of the main style.

**Parser change** in [shared_utils.py](shared_utils.py#L73):
- Add one field: `'is_hetatm': record == 'HETATM'` to each atom dict (one-line change in `_parse_pdb_text`).

**UI addition** (slot into existing controls row):
- `dcc.Checklist(id='hetatm-toggle', options=[{'label': 'Ligands & HETATM', 'value': 'show'}], value=['show'])`
- `dcc.Dropdown(id='hetatm-style', options=[{'label':'Stick','value':'stick'},{'label':'Sphere','value':'sphere'}], value='stick')`

**Callback change:**
- Add `Input('hetatm-toggle', 'value')` and `Input('hetatm-style', 'value')` to the existing `restyle` callback at [protein_dashboard.py:143](protein_dashboard.py#L143).
- Extend `create_mol3d_style` signature: `create_mol3d_style(atoms, visualization_type, color_element, show_hetatm=True, hetatm_visualization='stick')`. When `show_hetatm=False` and `atom['is_hetatm']` is True, emit `{'visualization_type': 'hidden', 'color': '#000000'}` (3dmol.js honors `hidden`). When True, override `visualization_type` with `hetatm_visualization` for those atoms.

**Verification:** load 4HHB (has heme groups, HETATM), uncheck → hemes disappear; re-check and pick sphere → hemes render as sphere while the protein stays cartoon.

**Effort:** ~20 LOC.

## Expansion 6 — mmCIF support

**Goal:** accept `.cif` files (RCSB's modern default format) alongside `.pdb`.

**Parser addition** in [shared_utils.py](shared_utils.py):
```python
def _parse_cif_text(text: str) -> dict:
    """Parse the _atom_site loop from an mmCIF file. Same output shape as _parse_pdb_text."""
```
- Find `loop_` block whose first header line starts with `_atom_site.`.
- Collect header names in order (each `_atom_site.<field>` line).
- Parse whitespace-separated data rows until the next `loop_` / `#` / `data_` boundary.
- Map fields: `label_atom_id` → name, `type_symbol` → element, `label_comp_id` → residue_name, `label_asym_id` → chain, `label_seq_id` → residue_num, `Cartn_x/y/z` → positions, `group_PDB` → is_hetatm.

**Dispatch in `parse_uploaded_pdb`:**
- Rename the function to `parse_uploaded_structure` (keep a thin `parse_uploaded_pdb = parse_uploaded_structure` alias for backwards compat).
- Branch on `filename.lower().endswith('.cif')` → `_parse_cif_text`; else `.pdb` → `_parse_pdb_text`.
- Update error message: "File must have a .pdb or .cif extension".

**UI change** in [protein_dashboard.py:52](protein_dashboard.py#L52):
- `dcc.Upload(accept='.pdb,.cif,chemical/x-pdb,chemical/x-cif', ...)`

**Fetch integration (if combined with Expansion 1):**
- Add a radio `dcc.RadioItems(id='fetch-format', options=['pdb', 'cif'], value='pdb', inline=True)` next to the fetch button. RCSB URL: `https://files.rcsb.org/download/{id}.{fmt}`.

**Edge cases:**
- Quoted CIF values (e.g. `'CA 2+'`) — need to handle single-quoted tokens in the tokenizer.
- Multi-block CIF files — take the first `data_` block only for v1.
- Very large CIFs (100k+ atoms) — reuse the same parser; browser rendering is the real bottleneck.

**Verification:** download `1CRN.cif` from RCSB, drag-drop, confirm same render as the `.pdb` version. Side-by-side atom counts should match exactly.

**Effort:** ~100 LOC (tokenizer + loop parser).

---

## Suggested rollout order

If implementing more than one, this order minimises rework:

1. **Expansion 1 (Fetch by PDB ID)** — removes the friction of manual downloads; benefits every other expansion during testing.
2. **Expansion 5 (HETATM toggle)** — trivial parser change; unblocks testing with realistic structures like hemoglobin.
3. **Expansion 2 (Residue/chain selector)** — establishes the `selectedAtomIds` pattern that Expansion 3 reuses.
4. **Expansion 3 (Sequence view)** — depends on #2's selection plumbing.
5. **Expansion 4 (Stats panel)** — independent; purely additive.
6. **Expansion 6 (mmCIF)** — independent; can slot in at any point.
