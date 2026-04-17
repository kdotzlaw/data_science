import base64
import math
import re
import sys
import traceback

import requests


_CHAIN_PALETTE = [
    '#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00',
    '#FFD92F', '#A65628', '#F781BF', '#66C2A5', '#8DA0CB',
]

_ELEMENT_COLORS = {
    'H': '#FFFFFF', 'C': '#909090', 'N': '#3050F8', 'O': '#FF0D0D',
    'S': '#FFFF30', 'P': '#FF8000', 'F': '#90E050', 'CL': '#1FF01F',
    'BR': '#A62929', 'I': '#940094', 'FE': '#E06633', 'ZN': '#7D80B0',
    'CA': '#3DFF00', 'MG': '#8AFF00', 'K': '#8F40D4', 'NA': '#AB5CF2',
}

_RESIDUE_TYPE_COLORS = {
    'ALA': '#C8C8C8', 'VAL': '#C8C8C8', 'LEU': '#C8C8C8',
    'ILE': '#C8C8C8', 'MET': '#C8C8C8', 'PHE': '#C8C8C8',
    'TRP': '#C8C8C8', 'PRO': '#C8C8C8',
    'SER': '#00DCDC', 'THR': '#00DCDC', 'CYS': '#00DCDC',
    'ASN': '#00DCDC', 'GLN': '#00DCDC', 'TYR': '#00DCDC', 'GLY': '#00DCDC',
    'LYS': '#145AFF', 'ARG': '#145AFF', 'HIS': '#145AFF',
    'ASP': '#E60A0A', 'GLU': '#E60A0A',
}

AA_THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}


def _parse_pdb_text(text):
    atoms = []
    chain_order = []
    residue_seen = {}

    for line in text.splitlines():
        record = line[:6]
        if record not in ('ATOM  ', 'HETATM'):
            continue
        if len(line) < 54:
            continue

        try:
            serial = int(line[6:11])
            name = line[12:16].strip()
            alt_loc = line[16:17].strip()
            if alt_loc and alt_loc not in ('A', '1'):
                continue
            residue_name = line[17:20].strip()
            chain = line[21:22].strip() or 'A'
            try:
                residue_num = int(line[22:26])
            except ValueError:
                residue_num = 0
            icode = line[26:27].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            element = line[76:78].strip() if len(line) >= 78 else ''
            if not element:
                element = re.sub(r'[0-9]', '', name)[:2].strip()
        except (ValueError, IndexError):
            continue

        if chain not in chain_order:
            chain_order.append(chain)
        chain_index = chain_order.index(chain)

        residue_key = (chain, residue_num, icode)
        if residue_key not in residue_seen:
            residue_seen[residue_key] = len(residue_seen)
        residue_index = residue_seen[residue_key]

        atoms.append({
            'serial': serial,
            'name': name,
            'element': element,
            'positions': [x, y, z],
            'residue_index': residue_index,
            'residue_num': residue_num,
            'residue_name': residue_name,
            'chain': chain,
            'chain_index': chain_index,
            'is_hetatm': record == 'HETATM',
        })

    return {'atoms': atoms, 'bonds': []}


def _tokenize_cif_row(line):
    tokens = []
    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if c in ' \t':
            i += 1
            continue
        if c in ("'", '"'):
            quote = c
            j = i + 1
            while j < n and line[j] != quote:
                j += 1
            tokens.append(line[i + 1:j])
            i = j + 1
        else:
            j = i
            while j < n and line[j] not in ' \t':
                j += 1
            tokens.append(line[i:j])
            i = j
    return tokens


def _parse_cif_text(text):
    """Parse the first _atom_site loop out of an mmCIF file."""
    lines = text.splitlines()
    n = len(lines)
    headers = []
    data_start = n

    i = 0
    while i < n:
        stripped = lines[i].strip()
        if stripped == 'loop_':
            j = i + 1
            block_headers = []
            while j < n and lines[j].strip().startswith('_'):
                block_headers.append(lines[j].strip())
                j += 1
            if block_headers and block_headers[0].startswith('_atom_site.'):
                headers = [h[len('_atom_site.'):] for h in block_headers]
                data_start = j
                break
        i += 1

    if not headers:
        return {'atoms': [], 'bonds': []}

    idx_of = {h: i for i, h in enumerate(headers)}

    def get(row, *names, default=''):
        for name in names:
            k = idx_of.get(name)
            if k is not None and k < len(row):
                val = row[k]
                if val not in ('.', '?'):
                    return val
        return default

    atoms = []
    chain_order = []
    residue_seen = {}
    serial_counter = 0

    i = data_start
    while i < n:
        stripped = lines[i].strip()
        if not stripped or stripped.startswith('#') or stripped == 'loop_' or stripped.startswith('data_'):
            break
        if stripped.startswith('_'):
            break

        row = _tokenize_cif_row(stripped)
        if len(row) < len(headers):
            i += 1
            continue

        try:
            group = get(row, 'group_PDB')
            name = get(row, 'label_atom_id', 'auth_atom_id')
            element = get(row, 'type_symbol')
            residue_name = get(row, 'label_comp_id', 'auth_comp_id')
            chain = get(row, 'auth_asym_id', 'label_asym_id') or 'A'
            raw_rnum = get(row, 'auth_seq_id', 'label_seq_id', default='0')
            try:
                residue_num = int(raw_rnum)
            except ValueError:
                residue_num = 0
            x = float(get(row, 'Cartn_x'))
            y = float(get(row, 'Cartn_y'))
            z = float(get(row, 'Cartn_z'))
            raw_serial = get(row, 'id')
            try:
                serial = int(raw_serial)
            except ValueError:
                serial_counter += 1
                serial = serial_counter
        except (ValueError, IndexError):
            i += 1
            continue

        if chain not in chain_order:
            chain_order.append(chain)
        chain_index = chain_order.index(chain)

        residue_key = (chain, residue_num)
        if residue_key not in residue_seen:
            residue_seen[residue_key] = len(residue_seen)
        residue_index = residue_seen[residue_key]

        atoms.append({
            'serial': serial,
            'name': name,
            'element': element,
            'positions': [x, y, z],
            'residue_index': residue_index,
            'residue_num': residue_num,
            'residue_name': residue_name,
            'chain': chain,
            'chain_index': chain_index,
            'is_hetatm': group == 'HETATM',
        })
        i += 1

    return {'atoms': atoms, 'bonds': []}


def parse_uploaded_structure(contents, filename):
    """Decode an upload, dispatch by extension, parse. Return (modelData, error_msg)."""
    if contents is None:
        return None, None

    if not filename:
        return None, 'File must have a .pdb or .cif extension'

    lower = filename.lower()
    if lower.endswith('.pdb'):
        parser = _parse_pdb_text
    elif lower.endswith('.cif'):
        parser = _parse_cif_text
    else:
        return None, 'File must have a .pdb or .cif extension'

    try:
        _, b64 = contents.split(',', 1)
        decoded = base64.b64decode(b64).decode('utf-8', errors='replace')
    except Exception:
        return None, 'Could not decode file — is this a text structure file?'

    try:
        model_data = parser(decoded)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        return None, f'Failed to parse structure: {type(exc).__name__}'

    if not model_data['atoms']:
        return None, 'No ATOM/HETATM records found'

    return model_data, None


parse_uploaded_pdb = parse_uploaded_structure


def fetch_pdb_by_id(pdb_id, fmt='pdb'):
    """Fetch a structure from RCSB. Return (modelData, filename, error_msg)."""
    if not pdb_id:
        return None, None, 'Enter a PDB ID'

    pdb_id = pdb_id.strip().upper()
    if not re.fullmatch(r'[1-9][0-9A-Z]{3}', pdb_id):
        return None, None, 'PDB ID must be 4 chars (digit + 3 alphanumerics)'

    if fmt not in ('pdb', 'cif'):
        fmt = 'pdb'

    url = f'https://files.rcsb.org/download/{pdb_id}.{fmt}'
    try:
        resp = requests.get(url, timeout=10)
    except requests.Timeout:
        return None, None, 'RCSB request timed out'
    except requests.RequestException as exc:
        return None, None, f'Network error: {type(exc).__name__}'

    if resp.status_code == 404:
        return None, None, f'No PDB entry for {pdb_id}'
    if resp.status_code != 200:
        return None, None, f'RCSB returned HTTP {resp.status_code}'

    parser = _parse_cif_text if fmt == 'cif' else _parse_pdb_text
    try:
        model_data = parser(resp.text)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        return None, None, f'Failed to parse response: {type(exc).__name__}'

    if not model_data['atoms']:
        return None, None, 'Fetched file had no ATOM records'

    return model_data, f'{pdb_id}.{fmt}', None


def summarize_model(model_data, filename):
    """Return 'Loaded 1crn.pdb — 327 atoms, 1 chain, 46 residues'."""
    atoms = model_data.get('atoms', [])
    chains = {a.get('chain') for a in atoms if a.get('chain')}
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


def create_mol3d_style(
    atoms,
    visualization_type='cartoon',
    color_element='chain',
    show_hetatm=True,
    hetatm_visualization='stick',
):
    """Generate per-atom styles. HETATM atoms can be hidden or styled independently."""
    chain_cache = {}
    residue_cache = {}

    def chain_color(chain):
        if chain not in chain_cache:
            chain_cache[chain] = _CHAIN_PALETTE[len(chain_cache) % len(_CHAIN_PALETTE)]
        return chain_cache[chain]

    def residue_color(residue_name, residue_index):
        key = (residue_name, residue_index)
        if key not in residue_cache:
            residue_cache[key] = _CHAIN_PALETTE[len(residue_cache) % len(_CHAIN_PALETTE)]
        return residue_cache[key]

    styles = []
    for atom in atoms:
        is_het = atom.get('is_hetatm', False)

        if is_het and not show_hetatm:
            styles.append({'visualization_type': 'hidden', 'color': '#000000'})
            continue

        if is_het:
            color = _ELEMENT_COLORS.get(atom.get('element', '').upper(), '#909090')
        elif color_element == 'atom':
            color = _ELEMENT_COLORS.get(atom.get('element', '').upper(), '#909090')
        elif color_element == 'residue':
            color = residue_color(atom.get('residue_name'), atom.get('residue_index'))
        elif color_element == 'residue_type':
            color = _RESIDUE_TYPE_COLORS.get(atom.get('residue_name'), '#C8C8C8')
        else:
            color = chain_color(atom.get('chain', 'A'))

        vtype = hetatm_visualization if is_het else visualization_type

        styles.append({'color': color, 'visualization_type': vtype})
    return styles


def _parse_residue_spec(spec):
    """Parse '10-50,70,100-120' into a set of integers."""
    if not spec:
        return set()
    result = set()
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            try:
                a, b = part.split('-', 1)
                result.update(range(int(a), int(b) + 1))
            except ValueError:
                continue
        else:
            try:
                result.add(int(part))
            except ValueError:
                continue
    return result


def select_atom_indices(atoms, chains=None, residue_spec=None):
    """Return positional indices into atoms matching the filters.

    Empty chains AND empty spec → empty selection (nothing highlighted).
    residue_spec uses PDB residue sequence numbers (residue_num), not positional.
    """
    chain_set = set(chains) if chains else None
    residue_set = _parse_residue_spec(residue_spec) if residue_spec else None

    if not chain_set and not residue_set:
        return []

    indices = []
    for i, atom in enumerate(atoms):
        if chain_set and atom.get('chain') not in chain_set:
            continue
        if residue_set and atom.get('residue_num') not in residue_set:
            continue
        indices.append(i)
    return indices


def build_sequence(atoms):
    """Return [(chain, residue_index, one_letter, residue_num), ...] in order."""
    seen = set()
    result = []
    for atom in atoms:
        if atom.get('is_hetatm'):
            continue
        chain = atom.get('chain')
        ri = atom.get('residue_index')
        key = (chain, ri)
        if key in seen:
            continue
        seen.add(key)
        rn = atom.get('residue_name', '')
        code = AA_THREE_TO_ONE.get(rn, 'X')
        result.append((chain, ri, code, atom.get('residue_num')))
    return result


def compute_chain_table(atoms):
    """Return [{'chain': 'A', 'atoms': 327, 'residues': 46}, ...] sorted by chain."""
    by_chain = {}
    for a in atoms:
        ch = a.get('chain')
        if ch not in by_chain:
            by_chain[ch] = {'atoms': 0, 'residues': set(), 'hetatms': 0}
        by_chain[ch]['atoms'] += 1
        if a.get('is_hetatm'):
            by_chain[ch]['hetatms'] += 1
        by_chain[ch]['residues'].add(a.get('residue_index'))
    return [
        {
            'chain': ch,
            'atoms': d['atoms'],
            'residues': len(d['residues']),
            'hetatms': d['hetatms'],
        }
        for ch, d in sorted(by_chain.items())
    ]


def compute_composition(atoms):
    """Return {'ALA': 12, 'ARG': 8, ...} by residue type (protein atoms only)."""
    counts = {}
    seen = set()
    for a in atoms:
        if a.get('is_hetatm'):
            continue
        key = (a.get('chain'), a.get('residue_index'))
        if key in seen:
            continue
        seen.add(key)
        rn = a.get('residue_name', '')
        if not rn:
            continue
        counts[rn] = counts.get(rn, 0) + 1
    return counts


def _dihedral(p1, p2, p3, p4):
    def sub(a, b):
        return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]

    def cross(a, b):
        return [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]

    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def normalize(a):
        n = math.sqrt(dot(a, a))
        return [0.0, 0.0, 0.0] if n == 0 else [a[0] / n, a[1] / n, a[2] / n]

    b1 = sub(p2, p1)
    b2 = sub(p3, p2)
    b3 = sub(p4, p3)

    n1 = cross(b1, b2)
    n2 = cross(b2, b3)
    m1 = cross(n1, normalize(b2))

    x = dot(n1, n2)
    y = dot(m1, n2)

    return math.degrees(math.atan2(y, x))


def compute_ramachandran(atoms):
    """Return (phi_degrees, psi_degrees) lists, one entry per residue with valid phi & psi."""
    by_chain = {}
    for a in atoms:
        if a.get('is_hetatm'):
            continue
        if a.get('name') not in ('N', 'CA', 'C'):
            continue
        chain = a.get('chain')
        ri = a.get('residue_index')
        if chain not in by_chain:
            by_chain[chain] = {}
        if ri not in by_chain[chain]:
            by_chain[chain][ri] = {
                'residue_num': a.get('residue_num'),
                'atoms': {},
            }
        by_chain[chain][ri]['atoms'][a['name']] = a['positions']

    phi_list = []
    psi_list = []

    for chain, residues in by_chain.items():
        sorted_ris = sorted(residues.keys())
        for idx in range(1, len(sorted_ris) - 1):
            ri_prev = sorted_ris[idx - 1]
            ri_curr = sorted_ris[idx]
            ri_next = sorted_ris[idx + 1]
            d_prev = residues[ri_prev]
            d_curr = residues[ri_curr]
            d_next = residues[ri_next]

            rn_prev = d_prev.get('residue_num')
            rn_curr = d_curr.get('residue_num')
            rn_next = d_next.get('residue_num')
            if None in (rn_prev, rn_curr, rn_next):
                continue
            if rn_curr - rn_prev != 1 or rn_next - rn_curr != 1:
                continue

            try:
                phi = _dihedral(
                    d_prev['atoms']['C'],
                    d_curr['atoms']['N'],
                    d_curr['atoms']['CA'],
                    d_curr['atoms']['C'],
                )
                psi = _dihedral(
                    d_curr['atoms']['N'],
                    d_curr['atoms']['CA'],
                    d_curr['atoms']['C'],
                    d_next['atoms']['N'],
                )
            except KeyError:
                continue

            phi_list.append(phi)
            psi_list.append(psi)

    return phi_list, psi_list
