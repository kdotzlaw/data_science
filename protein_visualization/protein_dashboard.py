import dash
import dash_bio
import plotly.express as px
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html
from dash.exceptions import PreventUpdate

from shared_utils import (
    AA_THREE_TO_ONE,
    build_sequence,
    compute_chain_table,
    compute_composition,
    compute_ramachandran,
    create_mol3d_style,
    fetch_pdb_by_id,
    parse_uploaded_structure,
    select_atom_indices,
    summarize_model,
)


EMPTY_MODEL = {'atoms': [], 'bonds': []}

STYLE_OPTIONS = [
    {'label': 'Cartoon', 'value': 'cartoon'},
    {'label': 'Stick', 'value': 'stick'},
    {'label': 'Sphere', 'value': 'sphere'},
]

COLOR_OPTIONS = [
    {'label': 'Atom', 'value': 'atom'},
    {'label': 'Residue', 'value': 'residue'},
    {'label': 'Chain', 'value': 'chain'},
    {'label': 'Residue type', 'value': 'residue_type'},
]

HETATM_STYLE_OPTIONS = [
    {'label': 'Stick', 'value': 'stick'},
    {'label': 'Sphere', 'value': 'sphere'},
]

FETCH_FORMAT_OPTIONS = [
    {'label': ' PDB', 'value': 'pdb'},
    {'label': ' CIF', 'value': 'cif'},
]

UPLOAD_STYLE = {
    'width': '100%',
    'padding': '30px',
    'borderWidth': '2px',
    'borderStyle': 'dashed',
    'borderRadius': '8px',
    'borderColor': '#999',
    'textAlign': 'center',
    'marginBottom': '15px',
    'cursor': 'pointer',
}

CONTROLS_ROW_STYLE = {
    'display': 'flex',
    'gap': '15px',
    'alignItems': 'flex-end',
    'marginBottom': '15px',
    'flexWrap': 'wrap',
}

CONTROL_ITEM_STYLE = {'minWidth': '160px', 'flex': '1'}

SEQUENCE_VIEW_STYLE = {
    'fontFamily': 'monospace',
    'fontSize': '14px',
    'letterSpacing': '1px',
    'padding': '10px',
    'background': '#fafafa',
    'border': '1px solid #eee',
    'borderRadius': '4px',
    'marginBottom': '15px',
    'maxHeight': '180px',
    'overflowY': 'auto',
    'overflowWrap': 'break-word',
}

RESIDUE_SPAN_STYLE = {
    'cursor': 'pointer',
    'padding': '1px 2px',
    'borderRadius': '2px',
}

STATUS_BASE_STYLE = {'minHeight': '22px', 'marginBottom': '10px'}

app = Dash(__name__)

app.layout = html.Div([
    html.H1('Protein Structure Viewer', style={'textAlign': 'center'}),

    html.Div([
        dcc.Input(
            id='pdb-id-input',
            type='text',
            placeholder='e.g. 1CRN',
            debounce=True,
            maxLength=4,
            style={'width': '120px', 'height': '34px', 'padding': '0 8px'},
        ),
        dcc.RadioItems(
            id='fetch-format',
            options=FETCH_FORMAT_OPTIONS,
            value='pdb',
            inline=True,
            style={'margin': '0 4px'},
        ),
        html.Button(
            'Fetch from RCSB',
            id='fetch-pdb-btn',
            n_clicks=0,
            style={'height': '34px', 'padding': '0 14px', 'cursor': 'pointer'},
        ),
    ], style={'display': 'flex', 'alignItems': 'center', 'gap': '10px', 'marginBottom': '15px'}),

    dcc.Upload(
        id='pdb-upload',
        accept='.pdb,.cif,chemical/x-pdb,chemical/x-cif',
        multiple=False,
        children=html.Div([
            'Drag and drop or ',
            html.A('select a .pdb / .cif file'),
        ]),
        style=UPLOAD_STYLE,
    ),

    html.Div(id='upload-status', style=STATUS_BASE_STYLE),

    html.Div([
        html.Div([
            html.Label('Style', style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='style-dropdown',
                options=STYLE_OPTIONS,
                value='cartoon',
                clearable=False,
            ),
        ], style=CONTROL_ITEM_STYLE),
        html.Div([
            html.Label('Color scheme', style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='color-dropdown',
                options=COLOR_OPTIONS,
                value='chain',
                clearable=False,
            ),
        ], style=CONTROL_ITEM_STYLE),
        html.Div([
            html.Label('Background', style={'fontWeight': 'bold'}),
            dcc.Input(
                id='bg-color',
                type='text',
                value='#FFFFFF',
                debounce=True,
                style={'width': '100%', 'height': '36px'},
            ),
        ], style=CONTROL_ITEM_STYLE),
        html.Div([
            html.Label('Ligands / HETATM', style={'fontWeight': 'bold'}),
            dcc.Checklist(
                id='hetatm-toggle',
                options=[{'label': ' Show', 'value': 'show'}],
                value=['show'],
                style={'marginTop': '8px'},
            ),
        ], style=CONTROL_ITEM_STYLE),
        html.Div([
            html.Label('Ligand style', style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='hetatm-style',
                options=HETATM_STYLE_OPTIONS,
                value='stick',
                clearable=False,
            ),
        ], style=CONTROL_ITEM_STYLE),
    ], style=CONTROLS_ROW_STYLE),

    html.Div([
        html.Div([
            html.Label('Chain filter', style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='chain-filter',
                multi=True,
                placeholder='All chains',
            ),
        ], style={**CONTROL_ITEM_STYLE, 'flex': '2'}),
        html.Div([
            html.Label('Residue range', style={'fontWeight': 'bold'}),
            dcc.Input(
                id='residue-range',
                type='text',
                placeholder='e.g. 10-50 or 10,15,20',
                debounce=True,
                style={'width': '100%', 'height': '36px'},
            ),
        ], style=CONTROL_ITEM_STYLE),
    ], style=CONTROLS_ROW_STYLE),

    dcc.Store(id='pdb-store'),

    html.Div(id='sequence-view', style=SEQUENCE_VIEW_STYLE),

    html.Div(
        dash_bio.Molecule3dViewer(
            id='mol-viewer',
            modelData=EMPTY_MODEL,
            styles=[],
            backgroundColor='#FFFFFF',
            backgroundOpacity=1.0,
            selectionType='atom',
            selectedAtomIds=[],
            style={'height': '600px'},
        ),
        style={
            'border': '1px solid #ddd',
            'borderRadius': '4px',
            'marginBottom': '15px',
        },
    ),

    html.Details([
        html.Summary(id='info-summary', style={
            'cursor': 'pointer',
            'padding': '8px 12px',
            'background': '#f0f0f0',
            'borderRadius': '4px',
        }),
        html.Div(id='stats-body', style={'padding': '15px 8px'}),
    ], open=False, style={'marginBottom': '15px'}),

], style={'maxWidth': '1100px', 'margin': 'auto', 'padding': '20px'})


@app.callback(
    Output('pdb-store', 'data'),
    Output('upload-status', 'children'),
    Output('upload-status', 'style'),
    Input('pdb-upload', 'contents'),
    Input('fetch-pdb-btn', 'n_clicks'),
    State('pdb-upload', 'filename'),
    State('pdb-id-input', 'value'),
    State('fetch-format', 'value'),
    prevent_initial_call=True,
)
def handle_load(contents, fetch_clicks, upload_filename, pdb_id, fmt):
    trig = ctx.triggered_id

    if trig == 'fetch-pdb-btn':
        if not fetch_clicks:
            raise PreventUpdate
        model_data, filename, error = fetch_pdb_by_id(pdb_id, fmt or 'pdb')
        if error:
            return dash.no_update, error, {**STATUS_BASE_STYLE, 'color': 'crimson'}
        summary = summarize_model(model_data, filename)
        return (
            {'modelData': model_data, 'filename': filename},
            summary,
            {**STATUS_BASE_STYLE, 'color': 'seagreen'},
        )

    if trig == 'pdb-upload':
        if contents is None:
            raise PreventUpdate
        model_data, error = parse_uploaded_structure(contents, upload_filename)
        if error:
            return dash.no_update, error, {**STATUS_BASE_STYLE, 'color': 'crimson'}
        summary = summarize_model(model_data, upload_filename)
        return (
            {'modelData': model_data, 'filename': upload_filename},
            summary,
            {**STATUS_BASE_STYLE, 'color': 'seagreen'},
        )

    raise PreventUpdate


@app.callback(
    Output('chain-filter', 'options'),
    Output('chain-filter', 'value'),
    Input('pdb-store', 'data'),
)
def populate_chains(store):
    if not store or not store.get('modelData'):
        return [], []
    atoms = store['modelData']['atoms']
    chains = sorted({a.get('chain') for a in atoms if a.get('chain')})
    return [{'label': c, 'value': c} for c in chains], []


@app.callback(
    Output('mol-viewer', 'modelData'),
    Output('mol-viewer', 'styles'),
    Output('mol-viewer', 'backgroundColor'),
    Input('pdb-store', 'data'),
    Input('style-dropdown', 'value'),
    Input('color-dropdown', 'value'),
    Input('bg-color', 'value'),
    Input('hetatm-toggle', 'value'),
    Input('hetatm-style', 'value'),
)
def restyle(store, style, color, bg, hetatm_toggle, hetatm_vis):
    if not store or not store.get('modelData'):
        raise PreventUpdate

    atoms = store['modelData']['atoms']
    show_het = 'show' in (hetatm_toggle or [])

    styles = create_mol3d_style(
        atoms,
        visualization_type=style,
        color_element=color,
        show_hetatm=show_het,
        hetatm_visualization=hetatm_vis or 'stick',
    )

    return store['modelData'], styles, bg or '#FFFFFF'


@app.callback(
    Output('mol-viewer', 'selectedAtomIds'),
    Input('pdb-store', 'data'),
    Input('chain-filter', 'value'),
    Input('residue-range', 'value'),
    Input({'type': 'residue-span', 'chain': ALL, 'index': ALL}, 'n_clicks'),
)
def update_selection(store, chains, residue_range, _span_clicks):
    if not store or not store.get('modelData'):
        return []

    atoms = store['modelData']['atoms']
    trig = ctx.triggered_id

    if isinstance(trig, dict) and trig.get('type') == 'residue-span':
        triggered_value = ctx.triggered[0]['value'] if ctx.triggered else None
        if triggered_value:
            chain = trig.get('chain')
            ri = trig.get('index')
            return [
                i for i, a in enumerate(atoms)
                if a.get('chain') == chain and a.get('residue_index') == ri
            ]

    return select_atom_indices(atoms, chains=chains, residue_spec=residue_range)


@app.callback(
    Output('sequence-view', 'children'),
    Input('pdb-store', 'data'),
)
def render_sequence(store):
    if not store or not store.get('modelData'):
        return html.Em('Load a structure to see its sequence', style={'color': '#999'})

    atoms = store['modelData']['atoms']
    seq = build_sequence(atoms)
    if not seq:
        return html.Em('No protein residues found', style={'color': '#999'})

    children = []
    last_chain = None
    for chain, residue_index, code, residue_num in seq:
        if chain != last_chain:
            if last_chain is not None:
                children.append(html.Br())
            children.append(html.Strong(
                f'{chain} ',
                style={'marginRight': '6px', 'color': '#555'},
            ))
            last_chain = chain

        title = f'Chain {chain} · {code}'
        if residue_num:
            title += f' · residue {residue_num}'
        children.append(html.Span(
            code,
            id={'type': 'residue-span', 'chain': chain, 'index': residue_index},
            title=title,
            style=RESIDUE_SPAN_STYLE,
            n_clicks=0,
        ))

    return children


@app.callback(
    Output('info-summary', 'children'),
    Output('stats-body', 'children'),
    Input('pdb-store', 'data'),
)
def render_stats(store):
    if not store or not store.get('modelData'):
        return 'Structure details (load a file or fetch an ID)', ''

    atoms = store['modelData']['atoms']
    filename = store.get('filename', 'structure')

    summary = summarize_model(store['modelData'], filename)

    chain_rows = compute_chain_table(atoms)
    composition = compute_composition(atoms)

    table = html.Table(
        [
            html.Thead(html.Tr([
                html.Th('Chain', style={'textAlign': 'left', 'padding': '4px 12px'}),
                html.Th('Residues', style={'textAlign': 'right', 'padding': '4px 12px'}),
                html.Th('Atoms', style={'textAlign': 'right', 'padding': '4px 12px'}),
                html.Th('HETATM', style={'textAlign': 'right', 'padding': '4px 12px'}),
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(row['chain'], style={'padding': '4px 12px'}),
                    html.Td(row['residues'], style={'padding': '4px 12px', 'textAlign': 'right'}),
                    html.Td(row['atoms'], style={'padding': '4px 12px', 'textAlign': 'right'}),
                    html.Td(row['hetatms'], style={'padding': '4px 12px', 'textAlign': 'right'}),
                ])
                for row in chain_rows
            ]),
        ],
        style={'borderCollapse': 'collapse', 'marginBottom': '20px'},
    )

    body_children = [html.H4('Chains', style={'marginTop': 0}), table]

    if composition:
        items = sorted(composition.items(), key=lambda kv: -kv[1])
        comp_fig = px.bar(
            x=[k for k, _ in items],
            y=[v for _, v in items],
            labels={'x': 'Residue', 'y': 'Count'},
        )
        comp_fig.update_layout(
            height=300,
            margin=dict(t=20, b=40, l=40, r=10),
            plot_bgcolor='white',
        )
        body_children.extend([
            html.H4('Amino acid composition'),
            dcc.Graph(figure=comp_fig, config={'displayModeBar': False}),
        ])

    phi, psi = compute_ramachandran(atoms)
    if phi and psi:
        rama_fig = px.scatter(
            x=phi, y=psi,
            labels={'x': 'φ (degrees)', 'y': 'ψ (degrees)'},
        )
        rama_fig.update_traces(marker=dict(size=4, opacity=0.6))
        rama_fig.update_layout(
            height=450,
            xaxis=dict(range=[-180, 180], dtick=60, zeroline=True),
            yaxis=dict(range=[-180, 180], dtick=60, zeroline=True),
            margin=dict(t=20, b=40, l=40, r=10),
            plot_bgcolor='white',
        )
        body_children.extend([
            html.H4('Ramachandran plot'),
            dcc.Graph(figure=rama_fig, config={'displayModeBar': False}),
        ])

    return summary, body_children


if __name__ == '__main__':
    app.run(debug=True)
