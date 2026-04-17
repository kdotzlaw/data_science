import base64
import os
import sys
import tempfile
import traceback

from dash_bio.utils.pdb_parser import PdbParser


def parse_uploaded_pdb(contents, filename):
    """Decode a dcc.Upload payload and run PdbParser. Return (modelData, error_msg)."""
    if contents is None:
        return None, None

    if not filename or not filename.lower().endswith('.pdb'):
        return None, 'File must have a .pdb extension'

    try:
        _, b64 = contents.split(',', 1)
        decoded = base64.b64decode(b64).decode('utf-8', errors='replace')
    except Exception:
        return None, 'Could not decode file — is this a text PDB?'

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.pdb', delete=False, encoding='utf-8'
        ) as tmp:
            tmp.write(decoded)
            tmp_path = tmp.name

        parser = PdbParser(tmp_path)
        model_data = parser.mol3d_data()
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        return None, f'Failed to parse PDB: {type(exc).__name__}'
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    if not model_data or not model_data.get('atoms'):
        return None, 'PDB contains no ATOM/HETATM records'

    return model_data, None


def summarize_model(model_data, filename):
    """Return 'Loaded 1crn.pdb — 327 atoms, 1 chain, 46 residues'."""
    atoms = model_data.get('atoms', [])
    chains = {a.get('chain') for a in atoms if a.get('chain') is not None}
    residues = {
        (a.get('chain'), a.get('residue_index'))
        for a in atoms
        if a.get('residue_index') is not None
    }

    chain_word = 'chain' if len(chains) == 1 else 'chains'
    residue_word = 'residue' if len(residues) == 1 else 'residues'

    return (
        f'Loaded {filename} — {len(atoms)} atoms, '
        f'{len(chains)} {chain_word}, {len(residues)} {residue_word}'
    )
