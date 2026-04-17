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
