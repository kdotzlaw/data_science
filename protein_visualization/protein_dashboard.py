import dash
import dash_bio
from dash import Dash, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate
from dash_bio.utils import create_mol3d_style

from shared_utils import parse_uploaded_pdb, summarize_model


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
    'alignItems': 'center',
    'marginBottom': '15px',
    'flexWrap': 'wrap',
}

CONTROL_ITEM_STYLE = {'minWidth': '180px', 'flex': '1'}

app = Dash(__name__)

app.layout = html.Div([
    html.H1('Protein Structure Viewer', style={'textAlign': 'center'}),

    dcc.Upload(
        id='pdb-upload',
        accept='.pdb,chemical/x-pdb',
        multiple=False,
        children=html.Div([
            'Drag and drop or ',
            html.A('select a .pdb file'),
        ]),
        style=UPLOAD_STYLE,
    ),

    html.Div(id='upload-status', style={'minHeight': '22px', 'marginBottom': '10px'}),

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
    ], style=CONTROLS_ROW_STYLE),

    dcc.Store(id='pdb-store'),

    html.Div(
        dash_bio.Molecule3dViewer(
            id='mol-viewer',
            modelData=EMPTY_MODEL,
            styles=[],
            backgroundColor='#FFFFFF',
            backgroundOpacity=1.0,
            selectionType='atom',
            style={'height': '600px'},
        ),
        style={
            'border': '1px solid #ddd',
            'borderRadius': '4px',
            'marginBottom': '15px',
        },
    ),

    html.Div(id='info-panel', style={'color': '#555'}),
], style={'maxWidth': '1100px', 'margin': 'auto', 'padding': '20px'})


@app.callback(
    Output('pdb-store', 'data'),
    Output('upload-status', 'children'),
    Output('upload-status', 'style'),
    Input('pdb-upload', 'contents'),
    State('pdb-upload', 'filename'),
)
def handle_upload(contents, filename):
    base_style = {'minHeight': '22px', 'marginBottom': '10px'}

    if contents is None:
        return dash.no_update, '', base_style

    model_data, error = parse_uploaded_pdb(contents, filename)

    if error:
        return dash.no_update, error, {**base_style, 'color': 'crimson'}

    summary = summarize_model(model_data, filename)
    return (
        {'modelData': model_data, 'filename': filename},
        summary,
        {**base_style, 'color': 'seagreen'},
    )


@app.callback(
    Output('mol-viewer', 'modelData'),
    Output('mol-viewer', 'styles'),
    Output('mol-viewer', 'backgroundColor'),
    Output('info-panel', 'children'),
    Input('pdb-store', 'data'),
    Input('style-dropdown', 'value'),
    Input('color-dropdown', 'value'),
    Input('bg-color', 'value'),
)
def restyle(store, style, color, bg):
    if not store or not store.get('modelData'):
        raise PreventUpdate

    model_data = store['modelData']
    atoms = model_data.get('atoms', [])
    filename = store.get('filename', '')

    styles = create_mol3d_style(
        atoms,
        visualization_type=style,
        color_element=color,
    )

    chains = sorted({a.get('chain') for a in atoms if a.get('chain')})
    residues = {
        (a.get('chain'), a.get('residue_index'))
        for a in atoms
        if a.get('residue_index') is not None
    }

    info = html.Div([
        html.Strong(filename or 'structure'),
        html.Span(
            f' · {len(atoms)} atoms · {len(chains)} chain(s)'
            f' · {len(residues)} residue(s)'
            + (f' · chains: {", ".join(chains)}' if chains else '')
        ),
    ])

    return model_data, styles, bg or '#FFFFFF', info


if __name__ == '__main__':
    app.run(debug=True)
