"""Stub for `parmed` to satisfy dash_bio.utils.pdb_parser's top-level import.

dash_bio/__init__.py eagerly imports dash_bio.utils, which imports
pdb_parser, which does `import parmed as pmd`. parmed itself has no
Windows wheel and requires MSVC to build from source. Since this project
uses its own pure-Python PDB parser (see shared_utils.py) and never
instantiates dash_bio's PdbParser, we only need the `import parmed`
statement to succeed — no real API is ever called.

Placed in the project directory so `python protein_dashboard.py` puts
it on sys.path[0] and resolves it before the site-packages lookup.
"""


def _unavailable(*_args, **_kwargs):
    raise RuntimeError(
        'parmed is stubbed out in this project — dash_bio.utils.PdbParser '
        'is unavailable. Use shared_utils.parse_uploaded_pdb instead.'
    )


load_file = _unavailable
download_PDB = _unavailable
