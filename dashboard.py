# ============================================================
#  ML Dashboard Pro v2.0 — Professional ML Platform
#  Run: python dashboard.py  →  http://127.0.0.1:8050
# ============================================================

# ── AUTO-INSTALL ─────────────────────────────────────────────
import sys, subprocess
REQUIRED = ['pandas', 'openpyxl', 'plotly', 'dash', 'scikit-learn', 'numpy', 'scipy']
for _pkg in REQUIRED:
    _iname = 'sklearn' if _pkg == 'scikit-learn' else _pkg.replace('-','_')
    try:
        __import__(_iname)
    except ImportError:
        print(f"Installing {_pkg}...")
        subprocess.check_call([sys.executable,'-m','pip','install',_pkg,'--quiet'],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("All packages ready.")

# ── IMPORTS ──────────────────────────────────────────────────
import warnings; warnings.filterwarnings('ignore')
import os, uuid, pickle, base64, io
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               RandomForestRegressor, GradientBoostingRegressor, IsolationForest)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    r2_score, mean_absolute_error, mean_squared_error,
)

import dash
from dash import dcc, html, dash_table, Input, Output, State, callback_context, ALL, MATCH
import dash.exceptions

# ─────────────────────────────────────────────────────────────
#  SERVER-SIDE CACHE
# ─────────────────────────────────────────────────────────────
CACHE_DIR = Path(__file__).resolve().parent / '.cache'
CACHE_DIR.mkdir(exist_ok=True)
DEFAULT_FILE = Path(__file__).resolve().parent / 'placement_predict_50k_Dataset.xlsx'

def cache_save(sid, key, obj):
    with open(CACHE_DIR / f'{sid}_{key}.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol=4)

def cache_load(sid, key):
    p = CACHE_DIR / f'{sid}_{key}.pkl'
    if not p.exists(): return None
    with open(p, 'rb') as f: return pickle.load(f)

def load_default():
    if DEFAULT_FILE.exists():
        return pd.read_excel(DEFAULT_FILE)
    return None

# ─────────────────────────────────────────────────────────────
#  THEME
# ─────────────────────────────────────────────────────────────
C = {
    'bg':     '#07071a',
    'card':   '#0d0d2b',
    'card2':  '#12123a',
    'panel':  '#181850',
    'a1':     '#8b5cf6',   # violet
    'a2':     '#06b6d4',   # cyan
    'a3':     '#f59e0b',   # amber
    'a4':     '#10b981',   # emerald
    'red':    '#ef4444',
    'text':   '#e2e8f0',
    'sub':    '#94a3b8',
    'border': '#1e1e5a',
}

BASE_LAYOUT = dict(
    paper_bgcolor=C['card'],
    plot_bgcolor=C['card2'],
    font=dict(color=C['text'], family='Inter, sans-serif', size=12),
    colorway=[C['a1'], C['a2'], C['a3'], C['a4'], C['red'], '#a78bfa', '#34d399', '#fb923c'],
    xaxis=dict(gridcolor=C['border'], zerolinecolor=C['border'], color=C['sub']),
    yaxis=dict(gridcolor=C['border'], zerolinecolor=C['border'], color=C['sub']),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=C['text'])),
    hoverlabel=dict(bgcolor=C['card2'], font_color=C['text'], bordercolor=C['border']),
    margin=dict(t=50, l=10, r=10, b=10),
)

def atheme(fig):
    fig.update_layout(**BASE_LAYOUT)
    return fig

# ─────────────────────────────────────────────────────────────
#  UI HELPERS
# ─────────────────────────────────────────────────────────────
def card(content, title=None, min_w='260px', flex=1, accent=None):
    accent = accent or C['border']
    kids   = []
    if title:
        kids.append(html.Div(title, style={
            'color': C['a2'], 'fontWeight': '700', 'fontSize': '11px',
            'letterSpacing': '1.5px', 'textTransform': 'uppercase', 'marginBottom': '14px',
        }))
    kids += content if isinstance(content, list) else [content]
    return html.Div(kids, style={
        'background': f'linear-gradient(145deg,{C["card"]},{C["card2"]})',
        'border': f'1px solid {accent}',
        'borderRadius': '18px', 'padding': '22px',
        'flex': str(flex), 'minWidth': min_w,
        'boxShadow': '0 8px 32px rgba(0,0,0,.6),inset 0 1px 0 rgba(255,255,255,.04)',
    })

def stat_box(label_, value, color):
    return html.Div([
        html.Div(str(value), style={'fontSize':'26px','fontWeight':'900','color':color,'lineHeight':'1.1'}),
        html.Div(label_, style={'color':C['sub'],'fontSize':'11px','marginTop':'5px',
                                'fontWeight':'600','textTransform':'uppercase','letterSpacing':'.5px'}),
    ], style={
        'background': f'linear-gradient(135deg,{C["card"]},{C["card2"]})',
        'border': f'1px solid {color}33', 'borderLeft': f'4px solid {color}',
        'borderRadius': '14px', 'padding': '16px 20px', 'minWidth': '140px',
        'boxShadow': f'0 4px 20px {color}22',
    })

def sec(text):
    return html.Div(text, style={
        'color': C['a2'], 'fontWeight': '800', 'fontSize': '20px',
        'marginBottom': '20px', 'paddingBottom': '10px',
        'borderBottom': f'2px solid {C["a1"]}33',
    })

def lbl(text):
    return html.Div(text, style={
        'color': C['sub'], 'fontSize': '11px', 'fontWeight': '700',
        'textTransform': 'uppercase', 'letterSpacing': '.5px',
        'marginBottom': '6px', 'marginTop': '14px',
    })

def btn(text, id_, color=None):
    c = color or C['a1']
    return html.Button(text, id=id_, n_clicks=0, style={
        'background': f'linear-gradient(135deg,{c},{C["a2"]})',
        'color': 'white', 'border': 'none', 'borderRadius': '10px',
        'padding': '11px 26px', 'cursor': 'pointer', 'fontWeight': '700',
        'fontSize': '13px', 'fontFamily': 'Inter, sans-serif',
        'boxShadow': f'0 4px 18px {c}44', 'letterSpacing': '.3px',
    })

def divr():
    return html.Hr(style={'border':'none','borderTop':f'1px solid {C["border"]}','margin':'18px 0'})

def spin(children, id_=''):
    return dcc.Loading(children=children, type='circle', color=C['a1'],
                       id=f'spin-{id_}' if id_ else None)

DD_STYLE = {'color': '#111'}

# ─────────────────────────────────────────────────────────────
#  NAV TABS DEFINITION
# ─────────────────────────────────────────────────────────────
TABS = [
    ('t-data',    'Load Data',       '📁'),
    ('t-prep',    'Preprocessing',   '⚙️'),
    ('t-eda',     'Interactive EDA', '🔍'),
    ('t-models',  'Model Training',  '🤖'),
    ('t-predict', 'Prediction Hub',  '🎯'),
]

# ─────────────────────────────────────────────────────────────
#  APP
# ─────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    title='ML Dashboard Pro',
)
server = app.server   # expose Flask server if needed

app.layout = html.Div([

    # lightweight client stores
    dcc.Store(id='sid'),          # session ID (string)
    dcc.Store(id='col-meta'),     # {columns, cat_cols, num_cols}
    dcc.Store(id='proc-ok'),      # bool
    dcc.Store(id='train-ok'),     # bool
    dcc.Store(id='active-tab', data='t-data'),

    # downloads
    dcc.Download(id='dl-batch'),

    # ── NAVBAR ───────────────────────────────────────────────
    html.Div([

        # Logo
        html.Div([
            html.Div('ML', style={
                'background': f'linear-gradient(135deg,{C["a1"]},{C["a2"]})',
                'color': 'white', 'fontWeight': '900', 'fontSize': '15px',
                'padding': '5px 9px', 'borderRadius': '8px',
                'boxShadow': f'0 4px 14px {C["a1"]}55',
            }),
            html.Div([
                html.Div('Dashboard Pro', style={
                    'fontWeight': '800', 'fontSize': '16px',
                    'background': f'linear-gradient(90deg,{C["a1"]},{C["a2"]})',
                    '-webkit-background-clip': 'text',
                    '-webkit-text-fill-color': 'transparent',
                    'backgroundClip': 'text',
                }),
                html.Div('Professional ML Platform v2.0', style={
                    'color': C['sub'], 'fontSize': '10px', 'marginTop': '-1px',
                }),
            ]),
        ], style={'display':'flex','alignItems':'center','gap':'9px'}),

        # Tabs
        html.Div([
            html.Button(
                [html.Span(icon+' '), html.Span(label_)],
                id=f'nav-{tid}', n_clicks=0,
                style={
                    'background':'transparent','color':C['sub'],
                    'border':f'1px solid {C["border"]}','borderRadius':'10px',
                    'padding':'8px 14px','cursor':'pointer','fontWeight':'600',
                    'fontSize':'12px','fontFamily':'Inter,sans-serif',
                    'whiteSpace':'nowrap','display':'flex','alignItems':'center','gap':'3px',
                }
            ) for tid, label_, icon in TABS
        ], style={'display':'flex','gap':'6px','flexWrap':'wrap'}),

        # Session badge
        html.Div(id='sid-badge', style={
            'color':C['sub'],'fontSize':'10px','background':C['card2'],
            'borderRadius':'7px','padding':'5px 10px',
            'border':f'1px solid {C["border"]}','whiteSpace':'nowrap',
        }),

    ], style={
        'display':'flex','justifyContent':'space-between','alignItems':'center',
        'padding':'12px 26px',
        'background': f'linear-gradient(180deg,{C["card"]},rgba(13,13,43,.96))',
        'borderBottom':f'1px solid {C["border"]}',
        'position':'sticky','top':'0','zIndex':'300',
        'backdropFilter':'blur(12px)',
        'boxShadow':'0 4px 30px rgba(0,0,0,.6)',
    }),

    # ── PAGE ─────────────────────────────────────────────────
    html.Div(id='page', style={
        'padding':'24px 28px','minHeight':'90vh',
        'maxWidth':'1600px','margin':'0 auto',
    }),

], style={
    'background': C['bg'], 'minHeight':'100vh',
    'fontFamily':"'Inter','Segoe UI',sans-serif", 'color':C['text'],
})

# ─────────────────────────────────────────────────────────────
#  SESSION (clientside)
# ─────────────────────────────────────────────────────────────
app.clientside_callback(
    """
    function(data) {
        var k = 'ml_dash_sid';
        var s = sessionStorage.getItem(k);
        if (!s) {
            s = 'S' + Math.random().toString(36).substr(2,9) + Date.now();
            sessionStorage.setItem(k, s);
        }
        return s;
    }
    """,
    Output('sid', 'data'),
    Input('sid', 'data'),
)

@app.callback(Output('sid-badge','children'), Input('sid','data'))
def badge(sid):
    if not sid: return 'Session: ...'
    return f'SID: {str(sid)[:14]}...'

# ─────────────────────────────────────────────────────────────
#  TAB SWITCHING
# ─────────────────────────────────────────────────────────────
@app.callback(
    Output('active-tab','data'),
    [Input(f'nav-{tid}','n_clicks') for tid,_,_ in TABS],
    prevent_initial_call=True,
)
def switch_tab(*args):
    ctx = callback_context
    if not ctx.triggered: return 't-data'
    bid = ctx.triggered[0]['prop_id'].split('.')[0]
    for tid,_,_ in TABS:
        if f'nav-{tid}' == bid: return tid
    return 't-data'

@app.callback(
    [Output(f'nav-{tid}','style') for tid,_,_ in TABS],
    Input('active-tab','data'),
)
def style_nav(active):
    out = []
    for tid,_,_ in TABS:
        if tid == active:
            out.append({'background':f'linear-gradient(135deg,{C["a1"]}cc,{C["a2"]}cc)',
                        'color':'white','border':'none','borderRadius':'10px',
                        'padding':'8px 14px','cursor':'pointer','fontWeight':'700',
                        'fontSize':'12px','fontFamily':'Inter,sans-serif',
                        'boxShadow':f'0 4px 14px {C["a1"]}55',
                        'whiteSpace':'nowrap','display':'flex','alignItems':'center','gap':'3px'})
        else:
            out.append({'background':'transparent','color':C['sub'],
                        'border':f'1px solid {C["border"]}','borderRadius':'10px',
                        'padding':'8px 14px','cursor':'pointer','fontWeight':'600',
                        'fontSize':'12px','fontFamily':'Inter,sans-serif',
                        'whiteSpace':'nowrap','display':'flex','alignItems':'center','gap':'3px'})
    return out

@app.callback(Output('page','children'), Input('active-tab','data'))
def router(tab):
    pages = {'t-data':pg_data,'t-prep':pg_prep,'t-eda':pg_eda,
             't-models':pg_models,'t-predict':pg_predict}
    return pages.get(tab, pg_data)()

# ══════════════════════════════════════════════════════════════
#  PAGE 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════
def pg_data():
    return html.Div([
        sec('📁  Data Ingestion'),

        card([
            dcc.Upload(id='upload',
                children=html.Div([
                    html.Div('📂', style={'fontSize':'52px','marginBottom':'10px'}),
                    html.Div('Drag & Drop or Click to Upload', style={'fontWeight':'700','fontSize':'16px','color':C['text']}),
                    html.Div('CSV  •  Excel (.xlsx / .xls)', style={'color':C['sub'],'fontSize':'13px','marginTop':'5px'}),
                    html.Div('Default fallback: placement_predict_50k_Dataset.xlsx',
                             style={'color':C['a3'],'fontSize':'11px','marginTop':'4px','fontWeight':'600'}),
                ], style={'textAlign':'center','padding':'28px 0'}),
                style={
                    'border':f'2px dashed {C["a1"]}','borderRadius':'16px','cursor':'pointer',
                    'background':f'linear-gradient(135deg,{C["a1"]}07,{C["a2"]}07)',
                },
                accept='.csv,.xlsx,.xls', multiple=False,
            ),
            html.Div(id='up-status',style={'marginTop':'10px','textAlign':'center','fontSize':'13px'}),
        ], title='Upload Dataset', min_w='100%'),

        html.Div(style={'height':'18px'}),

        html.Div([
            btn('Load Default Dataset',  'btn-default', color=C['a4']),
            html.Div(style={'width':'10px'}),
            btn('Use Uploaded File',     'btn-upload',  color=C['a1']),
        ], style={'display':'flex','marginBottom':'22px'}),

        spin(html.Div(id='data-out'), 'data'),
    ])

@app.callback(Output('up-status','children'), Input('upload','filename'))
def up_status(fname):
    if fname:
        return html.Span(f'File ready: {fname}', style={'color':C['a4'],'fontWeight':'600'})
    return html.Span('No file uploaded — will use default', style={'color':C['sub']})

def _summary_ui(df):
    cat_c = df.select_dtypes(include=['object','string','category']).columns.tolist()
    num_c = df.select_dtypes(include='number').columns.tolist()
    miss  = df.isnull().sum()
    miss_pct = (miss / len(df) * 100).round(2)
    info_df  = pd.DataFrame({'Column':miss.index,'Type':df.dtypes.values.astype(str),
                              'Missing':miss.values,'Missing%':miss_pct.values})
    miss_nonzero = info_df[info_df['Missing']>0].sort_values('Missing',ascending=False)

    fig_miss = go.Figure()
    if len(miss_nonzero):
        fig_miss = px.bar(miss_nonzero, x='Column', y='Missing%', title='Missing Values (%)',
                          color='Missing%', color_continuous_scale=['#10b981','#ef4444'])
        fig_miss = atheme(fig_miss); fig_miss.update_coloraxes(showscale=False)
        fig_miss.update_xaxes(tickangle=-40)
    else:
        fig_miss.add_annotation(text='No missing values!', xref='paper',yref='paper',
                                x=.5,y=.5,showarrow=False,font=dict(size=18,color=C['a4']))
        fig_miss = atheme(fig_miss); fig_miss.update_layout(title='Missing Values')

    desc = df.describe(include='all').round(3).reset_index().rename(columns={'index':'Stat'})

    return html.Div([
        html.Div([
            stat_box('Rows',        f'{len(df):,}',          C['a1']),
            stat_box('Columns',     str(len(df.columns)),    C['a2']),
            stat_box('Numeric',     str(len(num_c)),         C['a3']),
            stat_box('Categorical', str(len(cat_c)),         C['a4']),
            stat_box('Missing',     f'{int(miss.sum()):,}',  C['red']),
            stat_box('Duplicates',  f'{int(df.duplicated().sum()):,}', C['sub']),
        ], style={'display':'flex','gap':'12px','flexWrap':'wrap','marginBottom':'22px'}),

        card(dcc.Graph(figure=fig_miss,config={'displayModeBar':False}),
             title='Missing Values', min_w='100%'),

        html.Div(style={'height':'18px'}),

        card([
            html.Div('Column Info',style={'color':C['a2'],'fontWeight':'700','fontSize':'11px',
                                          'letterSpacing':'1.5px','textTransform':'uppercase','marginBottom':'12px'}),
            dash_table.DataTable(
                data=info_df.to_dict('records'),
                columns=[{'name':c,'id':c} for c in info_df.columns],
                page_size=12, sort_action='native', filter_action='native',
                style_table={'overflowX':'auto'},
                style_cell={'backgroundColor':C['card2'],'color':C['text'],
                            'border':f'1px solid {C["border"]}',
                            'padding':'9px 13px','fontFamily':'Inter,sans-serif','fontSize':'12px','textAlign':'center'},
                style_header={'backgroundColor':C['a1'],'color':'white','fontWeight':'700','fontSize':'12px'},
                style_data_conditional=[
                    {'if':{'row_index':'odd'},'backgroundColor':C['card']},
                    {'if':{'filter_query':'{Missing} > 0'},'color':C['a3']},
                ],
            ),
        ], min_w='100%'),

        html.Div(style={'height':'18px'}),

        card([
            html.Div('Statistical Summary',style={'color':C['a2'],'fontWeight':'700','fontSize':'11px',
                                                   'letterSpacing':'1.5px','textTransform':'uppercase','marginBottom':'12px'}),
            dash_table.DataTable(
                data=desc.to_dict('records'),
                columns=[{'name':c,'id':c} for c in desc.columns],
                page_size=10,
                style_table={'overflowX':'auto'},
                style_cell={'backgroundColor':C['card2'],'color':C['text'],
                            'border':f'1px solid {C["border"]}',
                            'padding':'9px 13px','fontFamily':'Inter,sans-serif','fontSize':'12px','textAlign':'center'},
                style_header={'backgroundColor':C['a2'],'color':'white','fontWeight':'700','fontSize':'12px'},
            ),
        ], min_w='100%'),
    ])

@app.callback(
    [Output('data-out','children'), Output('col-meta','data')],
    [Input('sid','data'), Input('btn-default','n_clicks'), Input('btn-upload','n_clicks')],
    [State('upload','contents'), State('upload','filename')],
)
def load_data(sid, n1, n2, contents, fname):
    ctx = callback_context
    if not ctx.triggered or not sid: raise dash.exceptions.PreventUpdate
    trig = ctx.triggered[0]['prop_id'].split('.')[0]

    df = None
    # A newly created browser session immediately receives the default dataset.
    if trig in ('sid', 'btn-default'):
        df = load_default()
        if df is None:
            return html.Div('Default file not found!', style={'color':C['red']}), dash.no_update
    else:
        if not contents:
            return html.Div('No file uploaded.', style={'color':C['red']}), dash.no_update
        _, cs = contents.split(',')
        raw = base64.b64decode(cs)
        try:
            df = pd.read_csv(io.StringIO(raw.decode('utf-8'))) if str(fname).endswith('.csv') else pd.read_excel(io.BytesIO(raw))
        except Exception as e:
            return html.Div(f'Parse error: {e}', style={'color':C['red']}), dash.no_update

    cache_save(sid, 'raw', df)
    cat_c = df.select_dtypes(include=['object','string','category']).columns.tolist()
    num_c = df.select_dtypes(include='number').columns.tolist()
    meta  = {'columns':list(df.columns),'cat_cols':cat_c,'num_cols':num_c,'n':len(df)}
    return _summary_ui(df), meta

# ══════════════════════════════════════════════════════════════
#  PAGE 2 — PREPROCESSING
# ══════════════════════════════════════════════════════════════
def pg_prep():
    return html.Div([
        sec('⚙️  Preprocessing Configurator'),
        html.Div(id='prep-area'),
    ])

@app.callback(Output('prep-area','children'), Input('col-meta','data'))
def draw_prep(meta):
    if not meta:
        return html.Div('Load a dataset first (Tab 1).', style={'color':C['sub'],'textAlign':'center','padding':'40px'})
    cols = meta['columns']
    tgt_candidates = [c for c in ['PlacementStatus','Salary Package','target','label'] if c in cols]
    default_tgt = tgt_candidates[0] if tgt_candidates else cols[-1]
    col_opts = [{'label':c,'value':c} for c in cols]

    return html.Div([
        html.Div([
            # ── Config panel ─────────────────────────────────
            card([
                lbl('Target Column'),
                dcc.Dropdown(id='cfg-tgt', options=col_opts, value=default_tgt,
                             clearable=False, style=DD_STYLE),

                lbl('Task Type'),
                dcc.RadioItems(id='cfg-task',
                    options=[{'label':'  Classification','value':'clf'},
                             {'label':'  Regression','value':'reg'}],
                    value='clf', inline=True,
                    labelStyle={'color':C['text'],'marginRight':'18px','fontSize':'13px'}),

                lbl('Columns to Drop'),
                dcc.Checklist(id='cfg-drop',
                    options=col_opts,
                    value=[c for c in ['StudentID','IsAnomaly'] if c in cols],
                    style={'maxHeight':'140px','overflowY':'auto','background':C['card2'],
                           'padding':'10px','borderRadius':'8px','fontSize':'12px'},
                    labelStyle={'color':C['text'],'display':'block','marginBottom':'4px'}),

                lbl('Missing Value Strategy'),
                dcc.Dropdown(id='cfg-impute', options=[
                    {'label':'Median (numeric) + Mode (cat)','value':'median'},
                    {'label':'Mean (numeric) + Mode (cat)', 'value':'mean'},
                    {'label':'Most Frequent (all)',          'value':'most_frequent'},
                    {'label':'Constant  (0 / "missing")',   'value':'constant'},
                ], value='median', clearable=False, style=DD_STYLE),

                lbl('Feature Scaling'),
                dcc.Dropdown(id='cfg-scaler', options=[
                    {'label':'StandardScaler — Z-score (recommended)','value':'std'},
                    {'label':'MinMaxScaler — 0–1 range',              'value':'mm'},
                    {'label':'RobustScaler — outlier-robust',         'value':'rob'},
                    {'label':'None',                                   'value':'none'},
                ], value='std', clearable=False, style=DD_STYLE),

                lbl('Test Size (%)'),
                dcc.Slider(id='cfg-split', min=10, max=50, step=5, value=20,
                           marks={i:f'{i}%' for i in range(10,55,10)},
                           tooltip={'placement':'bottom','always_visible':False}),

                divr(),
                dcc.Checklist(id='cfg-anom',
                    options=[{'label':'  Detect & remove anomalies (Isolation Forest, 5% contamination)','value':'y'}],
                    value=[],
                    labelStyle={'color':C['a3'],'fontSize':'13px','fontWeight':'600'}),

                html.Div(style={'height':'14px'}),
                btn('Apply Pipeline', 'btn-prep'),
            ], title='Configuration', min_w='320px', flex='0.85'),

            # ── Output panel ─────────────────────────────────
            html.Div([spin(html.Div(id='prep-out'),'prep')],
                     style={'flex':'2.2','minWidth':'380px'}),
        ], style={'display':'flex','gap':'20px','flexWrap':'wrap'}),
    ])

@app.callback(
    [Output('prep-out','children'), Output('proc-ok','data')],
    Input('btn-prep','n_clicks'),
    [State('cfg-tgt','value'), State('cfg-task','value'),
     State('cfg-drop','value'), State('cfg-impute','value'),
     State('cfg-scaler','value'), State('cfg-split','value'),
     State('cfg-anom','value'), State('sid','data')],
    prevent_initial_call=True,
)
def apply_prep(n, tgt, task, drop_c, impute, scaler_key, split_pct, anom_opt, sid):
    if not n or not sid: raise dash.exceptions.PreventUpdate
    df_raw = cache_load(sid,'raw')
    if df_raw is None:
        return html.Div('Load a dataset first!', style={'color':C['red']}), False

    drop_c = drop_c or []
    df = df_raw.drop(columns=[c for c in drop_c if c in df_raw.columns], errors='ignore')
    if tgt not in df.columns:
        return html.Div(f'Target "{tgt}" not found!', style={'color':C['red']}), False

    log = []
    y_raw = df[tgt].copy()
    X_df  = df.drop(columns=[tgt])

    cat_c  = X_df.select_dtypes(include=['object','string','category']).columns.tolist()
    num_c  = [c for c in X_df.select_dtypes(include='number').columns
               if pd.api.types.is_numeric_dtype(X_df[c])]

    # Step 1 — Impute
    miss_b = int(X_df.isnull().sum().sum())
    num_imp = SimpleImputer(strategy=impute if impute in ['mean','median','most_frequent'] else 'median')
    cat_imp = SimpleImputer(strategy='most_frequent' if impute != 'constant' else 'constant',
                            fill_value='missing')
    if num_c:  X_df[num_c] = num_imp.fit_transform(X_df[num_c])
    if cat_c:  X_df[cat_c] = cat_imp.fit_transform(X_df[cat_c].astype(str))
    if y_raw.isnull().any():
        yi = SimpleImputer(strategy='most_frequent')
        y_raw = pd.Series(yi.fit_transform(y_raw.values.reshape(-1,1)).ravel())
    miss_a = int(X_df.isnull().sum().sum())
    log.append({'Step':'1. Imputation','Status':'Done','Detail':f'{miss_b} missing -> {miss_a}'})

    # Step 2 — Encode
    le_map = {}
    for col in cat_c:
        le = LabelEncoder(); X_df[col] = le.fit_transform(X_df[col].astype(str)); le_map[col] = le
    log.append({'Step':'2. Label Encoding','Status':'Done','Detail':f'{len(cat_c)} cat cols encoded'})

    # Encode target
    le_tgt, tgt_cls = None, None
    if task == 'clf':
        le_tgt = LabelEncoder(); y = le_tgt.fit_transform(y_raw.astype(str)); tgt_cls = list(le_tgt.classes_)
    else:
        y = pd.to_numeric(y_raw, errors='coerce').fillna(0).values

    # Step 3 — Anomaly
    remove_anom = 'y' in (anom_opt or [])
    anom_labels = None
    if remove_anom:
        iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
        anom_labels = iso.fit_predict(X_df.values)
        n_anom = int((anom_labels == -1).sum())
        mask = anom_labels == 1
        X_clean, y_clean = X_df[mask], y[mask]
        log.append({'Step':'3. Anomaly Removal','Status':'Done',
                    'Detail':f'{n_anom} anomalies removed ({n_anom/len(X_df)*100:.1f}%)'})
    else:
        X_clean, y_clean = X_df, y
        log.append({'Step':'3. Anomaly Detection','Status':'Skipped','Detail':'Not applied'})

    cache_save(sid,'X_raw_enc', X_df)
    cache_save(sid,'anom_labels', anom_labels)

    # Step 4 — Scale
    feat_names = list(X_clean.columns)
    X_arr = X_clean.values.astype(float)
    if scaler_key == 'std':    scaler = StandardScaler(); X_s = scaler.fit_transform(X_arr)
    elif scaler_key == 'mm':   scaler = MinMaxScaler();   X_s = scaler.fit_transform(X_arr)
    elif scaler_key == 'rob':  scaler = RobustScaler();   X_s = scaler.fit_transform(X_arr)
    else:                       scaler = None;              X_s = X_arr
    log.append({'Step':'4. Scaling','Status':'Done','Detail':scaler_key if scaler_key!='none' else 'No scaling'})

    # Step 5 — Split
    ts = split_pct / 100
    stratify = y_clean if task=='clf' and len(np.unique(y_clean)) < 50 else None
    X_tr,X_te,y_tr,y_te = train_test_split(X_s,y_clean,test_size=ts,random_state=42,stratify=stratify)
    log.append({'Step':'5. Train/Test Split','Status':'Done',
                'Detail':f'{len(X_tr):,} train / {len(X_te):,} test ({split_pct}% test)'})

    proc = {'X_tr':X_tr,'X_te':X_te,'y_tr':y_tr,'y_te':y_te,
            'feat':feat_names,'cat_c':cat_c,'num_c':num_c,
            'le_map':le_map,'le_tgt':le_tgt,'tgt_cls':tgt_cls,
            'scaler':scaler,'task':task,'target':tgt,
            'num_imp':num_imp,'cat_imp':cat_imp,'df_raw':df_raw}
    cache_save(sid,'proc',proc)

    # Build UI
    log_tbl = dash_table.DataTable(
        data=log, columns=[{'name':c,'id':c} for c in ['Step','Status','Detail']],
        style_cell={'backgroundColor':C['card2'],'color':C['text'],
                    'border':f'1px solid {C["border"]}',
                    'padding':'11px 15px','fontFamily':'Inter,sans-serif','fontSize':'13px','textAlign':'left'},
        style_header={'backgroundColor':C['a1'],'color':'white','fontWeight':'700','fontSize':'12px'},
        style_data_conditional=[
            {'if':{'filter_query':'{Status} = Done'},   'color':C['a4']},
            {'if':{'filter_query':'{Status} = Skipped'},'color':C['a3']},
        ],
    )
    return html.Div([
        html.Div([
            stat_box('Train',    f'{len(X_tr):,}',      C['a1']),
            stat_box('Test',     f'{len(X_te):,}',      C['a2']),
            stat_box('Features', str(len(feat_names)),  C['a3']),
            stat_box('Task',     task.upper(),           C['a4']),
        ], style={'display':'flex','gap':'12px','flexWrap':'wrap','marginBottom':'20px'}),
        card(log_tbl, title='Pipeline Steps', min_w='100%'),
        html.Div('Preprocessing complete — proceed to EDA, Model Training & Prediction!',
                 style={'color':C['a4'],'fontWeight':'700','fontSize':'14px',
                        'marginTop':'16px','textAlign':'center'}),
    ]), True

# ══════════════════════════════════════════════════════════════
#  PAGE 3 — INTERACTIVE EDA
# ══════════════════════════════════════════════════════════════
def pg_eda():
    return html.Div([
        sec('🔍  Interactive EDA Hub'),
        html.Div(id='eda-area'),
    ])

@app.callback(Output('eda-area','children'), Input('col-meta','data'))
def draw_eda(meta):
    if not meta:
        return html.Div('Load a dataset to unlock EDA.',
                        style={'color':C['sub'],'textAlign':'center','padding':'40px'})
    cols     = meta['columns']
    num_cols = meta['num_cols']
    col_opts = [{'label':c,'value':c} for c in cols]
    num_opts = [{'label':c,'value':c} for c in num_cols]
    dflt_x   = num_cols[0] if num_cols else cols[0] if cols else None
    dflt_y   = num_cols[1] if len(num_cols)>1 else None

    return html.Div([
        html.Div([
            # Controls
            card([
                lbl('Chart Type'),
                dcc.Dropdown(id='eda-type', options=[
                    {'label':'Histogram / Distribution','value':'hist'},
                    {'label':'Scatter Plot (2D)',       'value':'scatter'},
                    {'label':'Box Plot',                'value':'box'},
                    {'label':'Bar Chart (Categorical)', 'value':'bar'},
                    {'label':'Correlation Heatmap',     'value':'heat'},
                ], value='hist', clearable=False, style=DD_STYLE),

                lbl('X-Axis Column'),
                dcc.Dropdown(id='eda-x', options=col_opts, value=dflt_x, style=DD_STYLE),

                lbl('Y-Axis (Scatter / Box)'),
                dcc.Dropdown(id='eda-y', options=num_opts, value=dflt_y, style=DD_STYLE),

                lbl('Color / Split By'),
                dcc.Dropdown(id='eda-color', options=[{'label':'(none)','value':''}]+col_opts,
                             value='', clearable=False, style=DD_STYLE),

                html.Div(style={'height':'12px'}),
                btn('Render Chart', 'btn-eda', color=C['a2']),
            ], title='Chart Builder', min_w='270px', flex='0.65'),

            html.Div([spin(html.Div(id='eda-chart'),'eda-chart')],
                     style={'flex':'2.5','minWidth':'380px'}),
        ], style={'display':'flex','gap':'20px','flexWrap':'wrap','marginBottom':'24px'}),

        # Anomaly visualization (post-preprocessing)
        spin(html.Div(id='eda-anom'),'eda-anom'),
    ])

@app.callback(
    Output('eda-chart','children'),
    Input('btn-eda','n_clicks'),
    [State('eda-type','value'), State('eda-x','value'),
     State('eda-y','value'), State('eda-color','value'), State('sid','data')],
    prevent_initial_call=True,
)
def eda_chart(n, ctype, cx, cy, ccolor, sid):
    if not n or not sid: raise dash.exceptions.PreventUpdate
    df = cache_load(sid,'raw')
    if df is None: return html.Div('Load data first.',style={'color':C['red']})
    col = ccolor if ccolor and ccolor in df.columns else None
    try:
        if ctype == 'hist':
            if not cx or cx not in df.columns: return html.Div('Pick a column.',style={'color':C['sub']})
            fig = px.histogram(df, x=cx, color=col, barmode='overlay', opacity=.75,
                               marginal='box', title=f'Distribution: {cx}',
                               color_discrete_sequence=[C['a1'],C['a2'],C['a3']])
        elif ctype == 'scatter':
            if not cx or not cy or cx not in df.columns or cy not in df.columns:
                return html.Div('Select X and Y columns.',style={'color':C['sub']})
            s = df.sample(min(3000,len(df)),random_state=42)
            fig = px.scatter(s, x=cx, y=cy, color=col, opacity=.65,
                             title=f'Scatter: {cx} vs {cy}',
                             color_discrete_sequence=[C['a1'],C['a2'],C['a3']])
        elif ctype == 'box':
            if not cx or cx not in df.columns: return html.Div('Select a column.',style={'color':C['sub']})
            fig = px.box(df, x=col, y=cx, color=col, title=f'Box Plot: {cx}',
                         color_discrete_sequence=[C['a1'],C['a2'],C['a3'],C['a4'],C['red']])
        elif ctype == 'bar':
            if not cx or cx not in df.columns: return html.Div('Select a column.',style={'color':C['sub']})
            vc = df[cx].value_counts().head(20)
            fig = px.bar(x=vc.index.astype(str), y=vc.values, title=f'Value Counts: {cx}',
                         color=vc.values, color_continuous_scale=['#7c3aed','#06b6d4'])
            fig.update_coloraxes(showscale=False)
        elif ctype == 'heat':
            nd = df.select_dtypes(include='number')
            if len(nd.columns) < 2: return html.Div('Need 2+ numeric columns.',style={'color':C['sub']})
            corr = nd.corr()
            fig = px.imshow(corr, color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                            title='Correlation Heatmap', text_auto='.2f')
            fig.update_traces(textfont_size=9)
        else:
            return html.Div('Unknown type.',style={'color':C['sub']})
        fig = atheme(fig)
        return card(dcc.Graph(figure=fig, config={'displayModeBar':False},
                              style={'height':'480px'}), min_w='100%')
    except Exception as e:
        return html.Div(f'Chart error: {e}', style={'color':C['red']})

@app.callback(
    Output('eda-anom','children'),
    Input('proc-ok','data'),
    [State('sid','data'), State('col-meta','data')],
)
def eda_anom(ok, sid, meta):
    if not ok or not sid: return html.Div()
    X_enc = cache_load(sid,'X_raw_enc')
    alabs = cache_load(sid,'anom_labels')
    if X_enc is None: return html.Div()
    try:
        pca  = PCA(n_components=2, random_state=42)
        xy   = pca.fit_transform(X_enc.values)
        pdf  = pd.DataFrame(xy, columns=['PC1','PC2'])
        var  = pca.explained_variance_ratio_[:2].sum()*100
        n_anom = 0
        if alabs is not None:
            pdf['Type'] = ['Anomaly' if l==-1 else 'Normal' for l in alabs]
            n_anom = int((alabs==-1).sum())
            cmap = {'Normal':C['a4'],'Anomaly':C['red']}
            fig = px.scatter(pdf.sample(min(6000,len(pdf)),random_state=42),
                             x='PC1',y='PC2',color='Type',color_discrete_map=cmap,
                             opacity=.6, title='Anomaly Detection — PCA 2D Projection (Isolation Forest)')
        else:
            fig = px.scatter(pdf.sample(min(6000,len(pdf)),random_state=42),
                             x='PC1',y='PC2',opacity=.5,title='PCA 2D Projection of Feature Space',
                             color_discrete_sequence=[C['a1']])
        fig = atheme(fig)
        return html.Div([
            divr(),
            html.Div('Anomaly Detection Visualization', style={
                'color':C['a2'],'fontWeight':'800','fontSize':'18px',
                'marginBottom':'16px','paddingBottom':'8px','borderBottom':f'2px solid {C["a1"]}33',
            }),
            html.Div([
                stat_box('Anomalies', f'{n_anom:,}',              C['red']),
                stat_box('Normal',    f'{len(X_enc)-n_anom:,}',   C['a4']),
                stat_box('PCA Var.',  f'{var:.1f}%',              C['a2']),
                stat_box('Features',  str(len(X_enc.columns)),    C['a3']),
            ], style={'display':'flex','gap':'12px','flexWrap':'wrap','marginBottom':'20px'}),
            card(dcc.Graph(figure=fig,config={'displayModeBar':False},style={'height':'460px'}),
                 min_w='100%'),
        ])
    except Exception as e:
        return html.Div(f'Anomaly viz error: {e}', style={'color':C['red']})

# ══════════════════════════════════════════════════════════════
#  PAGE 4 — MODEL TRAINING
# ══════════════════════════════════════════════════════════════
def pg_models():
    return html.Div([
        sec('🤖  Model Training & Evaluation'),
        html.Div(id='models-area'),
    ])

@app.callback(Output('models-area','children'),
              Input('proc-ok','data'), State('sid','data'))
def draw_models(ok, sid):
    if not ok or not sid:
        return html.Div('Complete preprocessing first (Tab 2).',
                        style={'color':C['sub'],'textAlign':'center','padding':'40px'})
    proc = cache_load(sid,'proc')
    if not proc: return html.Div('Preprocessing data missing.',style={'color':C['sub']})
    task = proc['task']
    clf_opts = [{'label':'Logistic Regression','value':'lr'},
                {'label':'Random Forest',      'value':'rf'},
                {'label':'Gradient Boosting',  'value':'gb'},
                {'label':'Decision Tree',      'value':'dt'}]
    reg_opts = [{'label':'Linear Regression',        'value':'linreg'},
                {'label':'Ridge Regression',         'value':'ridge'},
                {'label':'Random Forest Regressor',  'value':'rfr'},
                {'label':'Gradient Boost Regressor', 'value':'gbr'}]
    m_opts   = clf_opts if task=='clf' else reg_opts
    m_dflt   = ['rf','lr'] if task=='clf' else ['rfr','linreg']

    return html.Div([
        html.Div([
            card([
                lbl('Models to Train'),
                dcc.Checklist(id='mdl-sel', options=m_opts, value=m_dflt,
                              labelStyle={'color':C['text'],'display':'block',
                                          'marginBottom':'8px','fontSize':'13px'}),
                divr(),
                html.Div('Hyperparameters', style={'color':C['a3'],'fontWeight':'700',
                                                    'fontSize':'11px','letterSpacing':'1px',
                                                    'textTransform':'uppercase'}),
                lbl('n_estimators  (RF / GB)'),
                dcc.Slider(id='hp-est', min=50,max=500,step=50,value=100,
                           marks={i:str(i) for i in [50,100,200,300,500]},
                           tooltip={'placement':'bottom','always_visible':False}),
                lbl('max_depth  (RF / GB / DT)'),
                dcc.Slider(id='hp-dep', min=2,max=20,step=1,value=8,
                           marks={i:str(i) for i in [2,5,8,12,16,20]},
                           tooltip={'placement':'bottom','always_visible':False}),
                lbl('learning_rate  (GB)'),
                dcc.Slider(id='hp-lr', min=0.01,max=0.5,step=0.01,value=0.1,
                           marks={0.01:'0.01',0.1:'0.1',0.3:'0.3',0.5:'0.5'},
                           tooltip={'placement':'bottom','always_visible':False}),
                lbl('C  (Logistic Regression)'),
                dcc.Slider(id='hp-c', min=0.01,max=10,step=0.1,value=1.0,
                           marks={0.01:'0.01',1:'1',5:'5',10:'10'},
                           tooltip={'placement':'bottom','always_visible':False}),
                html.Div(style={'height':'14px'}),
                btn('Train Models', 'btn-train'),
            ], title='Model Config', min_w='300px', flex='0.8'),

            html.Div([spin(html.Div(id='train-out'),'train')],
                     style={'flex':'2.5','minWidth':'380px'}),
        ], style={'display':'flex','gap':'20px','flexWrap':'wrap'}),
    ])

FRIENDLY = {'lr':'Logistic Reg.','rf':'Random Forest','gb':'Grad. Boost','dt':'Dec. Tree',
            'linreg':'Linear Reg.','ridge':'Ridge','rfr':'RF Regressor','gbr':'GB Regressor'}
CLRS = [C['a1'],C['a2'],C['a3'],C['a4'],C['red']]

@app.callback(
    [Output('train-out','children'), Output('train-ok','data')],
    Input('btn-train','n_clicks'),
    [State('mdl-sel','value'), State('hp-est','value'), State('hp-dep','value'),
     State('hp-lr','value'), State('hp-c','value'), State('sid','data')],
    prevent_initial_call=True,
)
def train(n, keys, n_est, depth, lr, c_v, sid):
    if not n or not sid: raise dash.exceptions.PreventUpdate
    proc = cache_load(sid,'proc')
    if not proc: return html.Div('Run preprocessing first!',style={'color':C['red']}), False
    keys = keys or []
    if not keys: return html.Div('Select at least one model.',style={'color':C['a3']}), False

    X_tr,X_te = proc['X_tr'], proc['X_te']
    y_tr,y_te = proc['y_tr'], proc['y_te']
    task, feat = proc['task'], proc['feat']
    tgt_cls   = proc.get('tgt_cls',['0','1'])

    clf_reg = {
        'lr':    LogisticRegression(C=c_v,max_iter=500,random_state=42),
        'rf':    RandomForestClassifier(n_estimators=n_est,max_depth=depth,random_state=42,n_jobs=-1),
        'gb':    GradientBoostingClassifier(n_estimators=n_est,max_depth=depth,learning_rate=lr,random_state=42),
        'dt':    DecisionTreeClassifier(max_depth=depth,random_state=42),
        'linreg':LinearRegression(),
        'ridge': Ridge(alpha=1.0),
        'rfr':   RandomForestRegressor(n_estimators=n_est,max_depth=depth,random_state=42,n_jobs=-1),
        'gbr':   GradientBoostingRegressor(n_estimators=n_est,max_depth=depth,learning_rate=lr,random_state=42),
    }

    res = {}
    for k in keys:
        if k not in clf_reg: continue
        m = clf_reg[k]; m.fit(X_tr, y_tr); yp = m.predict(X_te)
        if task=='clf':
            yprob = m.predict_proba(X_te)[:,1] if hasattr(m,'predict_proba') else None
            res[k] = {'model':m,'acc':accuracy_score(y_te,yp),
                      'f1':f1_score(y_te,yp,average='weighted',zero_division=0),
                      'prec':precision_score(y_te,yp,average='weighted',zero_division=0),
                      'rec':recall_score(y_te,yp,average='weighted',zero_division=0),
                      'cm':confusion_matrix(y_te,yp),'yprob':yprob,'yp':yp,
                      'report':classification_report(y_te,yp,output_dict=True,zero_division=0)}
        else:
            res[k] = {'model':m,'r2':r2_score(y_te,yp),
                      'mae':mean_absolute_error(y_te,yp),
                      'mse':mean_squared_error(y_te,yp),
                      'rmse':np.sqrt(mean_squared_error(y_te,yp)),'yp':yp}
    cache_save(sid,'models',res)

    out = _clf_results(res,y_te,feat,tgt_cls) if task=='clf' else _reg_results(res,y_te,feat)
    return out, True

def _clf_results(res, y_te, feat, tgt_cls):
    ks   = list(res.keys())
    # Metrics comparison grouped bar
    fig_m = go.Figure()
    for i,met in enumerate(['acc','f1','prec','rec']):
        labels = ['Accuracy','F1','Precision','Recall']
        fig_m.add_trace(go.Bar(name=labels[i],
            x=[FRIENDLY.get(k,k) for k in ks],
            y=[res[k][met]*100 for k in ks],
            marker_color=CLRS[i], opacity=.85,
            text=[f'{res[k][met]*100:.1f}%' for k in ks], textposition='outside'))
    fig_m.update_layout(**BASE_LAYOUT,title='Metrics Comparison',barmode='group',
                        yaxis_range=[0,115],yaxis_title='Score (%)')

    # ROC
    fig_roc = go.Figure()
    for i,k in enumerate(ks):
        if res[k]['yprob'] is not None:
            try:
                fpr,tpr,_ = roc_curve(y_te,res[k]['yprob'])
                fig_roc.add_trace(go.Scatter(x=fpr,y=tpr,mode='lines',
                    name=f'{FRIENDLY.get(k,k)} (AUC={auc(fpr,tpr):.3f})',
                    line=dict(width=2.5,color=CLRS[i%len(CLRS)])))
            except: pass
    fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',name='Random',
                                  line=dict(dash='dot',color=C['sub'],width=1.5)))
    fig_roc.update_layout(**BASE_LAYOUT,title='ROC Curves',
                          xaxis_title='FPR',yaxis_title='TPR')

    # PR curve
    fig_pr = go.Figure()
    for i,k in enumerate(ks):
        if res[k]['yprob'] is not None:
            try:
                p,r,_ = precision_recall_curve(y_te,res[k]['yprob'])
                fig_pr.add_trace(go.Scatter(x=r,y=p,mode='lines',
                    name=FRIENDLY.get(k,k), line=dict(width=2,color=CLRS[i%len(CLRS)])))
            except: pass
    fig_pr.update_layout(**BASE_LAYOUT,title='Precision-Recall Curve',
                         xaxis_title='Recall',yaxis_title='Precision')

    # Confusion matrix (best)
    best_k = max(res,key=lambda k:res[k]['acc'])
    cm     = res[best_k]['cm']
    cls_l  = tgt_cls if len(tgt_cls)==cm.shape[0] else [str(i) for i in range(cm.shape[0])]
    fig_cm = px.imshow(cm,text_auto=True,x=cls_l,y=cls_l,
                       title=f'Confusion Matrix — {FRIENDLY.get(best_k,best_k)}',
                       color_continuous_scale='Purples',labels=dict(x='Predicted',y='Actual'))
    fig_cm = atheme(fig_cm); fig_cm.update_traces(textfont_size=16)

    # Feature importance
    fi_cards = []
    for k in ks:
        m = res[k]['model']
        if hasattr(m,'feature_importances_') and len(feat)==len(m.feature_importances_):
            fi = pd.Series(m.feature_importances_,index=feat).sort_values(ascending=False).head(15)
            fig_fi = px.bar(x=fi.values,y=fi.index,orientation='h',
                            title=f'Feature Importance — {FRIENDLY.get(k,k)}',
                            color=fi.values,color_continuous_scale=['#7c3aed','#f59e0b'])
            fig_fi = atheme(fig_fi); fig_fi.update_coloraxes(showscale=False)
            fig_fi.update_layout(yaxis={'categoryorder':'total ascending'})
            fi_cards.append(card(dcc.Graph(figure=fig_fi,config={'displayModeBar':False},
                                            style={'height':'420px'}),
                                  title=f'Feat. Importance — {FRIENDLY.get(k,k)}'))

    badge = html.Div([
        html.Span('Best: ',style={'color':C['sub']}),
        html.Span(FRIENDLY.get(best_k,best_k),
                  style={'color':C['a3'],'fontWeight':'900','fontSize':'20px'}),
        html.Span(f'  {res[best_k]["acc"]*100:.2f}% accuracy',
                  style={'color':C['a4'],'fontWeight':'700','fontSize':'17px','marginLeft':'8px'}),
    ], style={'background':C['card2'],'border':f'2px solid {C["a3"]}44',
              'borderLeft':f'4px solid {C["a3"]}','borderRadius':'14px',
              'padding':'15px 22px','marginBottom':'22px'})

    return html.Div([
        badge,
        card(dcc.Graph(figure=fig_m,config={'displayModeBar':False},style={'height':'390px'}),
             title='Metrics Comparison',min_w='100%'),
        html.Div(style={'height':'18px'}),
        html.Div([
            card(dcc.Graph(figure=fig_roc,config={'displayModeBar':False},style={'height':'380px'}),title='ROC Curves'),
            card(dcc.Graph(figure=fig_pr, config={'displayModeBar':False},style={'height':'380px'}),title='Precision-Recall'),
        ],style={'display':'flex','gap':'18px','flexWrap':'wrap','marginBottom':'18px'}),
        card(dcc.Graph(figure=fig_cm,config={'displayModeBar':False},style={'height':'380px'}),
             title='Confusion Matrix',min_w='100%'),
        html.Div(style={'height':'18px'}),
        html.Div(fi_cards,style={'display':'flex','gap':'18px','flexWrap':'wrap'}),
    ])

def _reg_results(res, y_te, feat):
    ks = list(res.keys())
    rows = [{'Model':FRIENDLY.get(k,k),'R²':f"{res[k]['r2']:.4f}",
             'MAE':f"{res[k]['mae']:.4f}",'MSE':f"{res[k]['mse']:.4f}",
             'RMSE':f"{res[k]['rmse']:.4f}"} for k in ks]
    tbl  = dash_table.DataTable(data=rows,columns=[{'name':c,'id':c} for c in ['Model','R²','MAE','MSE','RMSE']],
        style_cell={'backgroundColor':C['card2'],'color':C['text'],'border':f'1px solid {C["border"]}',
                    'padding':'12px 14px','fontFamily':'Inter,sans-serif','fontSize':'13px','textAlign':'center'},
        style_header={'backgroundColor':C['a1'],'color':'white','fontWeight':'700','fontSize':'13px'})

    best_k  = max(ks,key=lambda k:res[k]['r2'])
    yp_best = res[best_k]['yp']
    idx     = np.random.choice(len(y_te),min(2000,len(y_te)),replace=False)
    fig_pa  = go.Figure()
    fig_pa.add_trace(go.Scatter(x=y_te[idx],y=yp_best[idx],mode='markers',
                                 opacity=.55,marker=dict(color=C['a1'],size=5),name='Points'))
    mn,mx   = min(y_te.min(),yp_best.min()),max(y_te.max(),yp_best.max())
    fig_pa.add_trace(go.Scatter(x=[mn,mx],y=[mn,mx],mode='lines',name='Perfect',
                                 line=dict(dash='dash',color=C['a4'],width=2)))
    fig_pa.update_layout(**BASE_LAYOUT,title=f'Predicted vs Actual — {FRIENDLY.get(best_k,best_k)}',
                         xaxis_title='Actual',yaxis_title='Predicted')

    resid = y_te[idx]-yp_best[idx]
    fig_r = px.histogram(x=resid,nbins=50,title='Residual Distribution',
                          color_discrete_sequence=[C['a2']])
    fig_r = atheme(fig_r)

    fi_cards = []
    for k in ks:
        m = res[k]['model']
        if hasattr(m,'feature_importances_') and len(feat)==len(m.feature_importances_):
            fi = pd.Series(m.feature_importances_,index=feat).sort_values(ascending=False).head(15)
            fig_fi = px.bar(x=fi.values,y=fi.index,orientation='h',
                            title=f'Feature Importance — {FRIENDLY.get(k,k)}',
                            color=fi.values,color_continuous_scale=['#7c3aed','#f59e0b'])
            fig_fi = atheme(fig_fi); fig_fi.update_coloraxes(showscale=False)
            fig_fi.update_layout(yaxis={'categoryorder':'total ascending'})
            fi_cards.append(card(dcc.Graph(figure=fig_fi,config={'displayModeBar':False},
                                            style={'height':'420px'}),
                                  title=f'Feat. Importance — {FRIENDLY.get(k,k)}'))
    return html.Div([
        card(tbl,title='Regression Metrics',min_w='100%'),
        html.Div(style={'height':'18px'}),
        html.Div([
            card(dcc.Graph(figure=fig_pa,config={'displayModeBar':False},style={'height':'380px'}),title='Predicted vs Actual'),
            card(dcc.Graph(figure=fig_r, config={'displayModeBar':False},style={'height':'380px'}),title='Residual Distribution'),
        ],style={'display':'flex','gap':'18px','flexWrap':'wrap','marginBottom':'18px'}),
        html.Div(fi_cards,style={'display':'flex','gap':'18px','flexWrap':'wrap'}),
    ])

# ══════════════════════════════════════════════════════════════
#  PAGE 5 — PREDICTION HUB
# ══════════════════════════════════════════════════════════════
def pg_predict():
    return html.Div([
        sec('Prediction Hub'),
        html.Div(id='pred-area'),
    ])

@app.callback(Output('pred-area','children'),
              Input('train-ok','data'), State('sid','data'))
def draw_predict(ok, sid):
    if not ok or not sid:
        return html.Div('Train models first (Tab 4).',
                        style={'color':C['sub'],'textAlign':'center','padding':'40px'})
    proc   = cache_load(sid,'proc')
    models = cache_load(sid,'models')
    if not proc or not models: return html.Div('Data missing.',style={'color':C['red']})

    feat     = proc['feat']
    cat_c    = proc['cat_c']
    task     = proc['task']
    df_raw   = proc['df_raw']
    m_keys   = list(models.keys())
    m_opts   = [{'label':FRIENDLY.get(k,k),'value':k} for k in m_keys]

    # Build an input for every active feature so the form always matches the
    # trained pipeline, including datasets with more than the default columns.
    inputs = []
    for col in feat:
        if col in cat_c and col in df_raw.columns:
            opts = sorted(df_raw[col].dropna().unique().tolist())
            ctrl = dcc.Dropdown(id={'type':'pi','col':col},
                                options=[{'label':str(v),'value':str(v)} for v in opts],
                                value=str(opts[0]),clearable=False,style=DD_STYLE)
        elif col in df_raw.columns:
            try:
                mn = float(df_raw[col].min()); mx = float(df_raw[col].max())
                md = float(df_raw[col].median())
                st = round((mx-mn)/100,4) if mx!=mn else 1
                ctrl = dcc.Slider(id={'type':'pi','col':col},min=round(mn,2),max=round(mx,2),
                                  step=st,value=round(md,2),marks=None,
                                  tooltip={'placement':'bottom','always_visible':True})
            except:
                ctrl = dcc.Input(id={'type':'pi','col':col},value='0',type='text',
                                 style={'background':C['card2'],'color':C['text'],
                                        'border':f'1px solid {C["border"]}',
                                        'borderRadius':'8px','padding':'8px','width':'100%'})
        else:
            ctrl = dcc.Input(id={'type':'pi','col':col},value='0',type='text',
                             style={'background':C['card2'],'color':C['text'],
                                    'border':f'1px solid {C["border"]}',
                                    'borderRadius':'8px','padding':'8px','width':'100%'})
        inputs.append(html.Div([
            html.Label(col,style={'color':C['sub'],'fontSize':'11px','fontWeight':'700',
                                   'display':'block','marginBottom':'5px',
                                   'textTransform':'uppercase','letterSpacing':'.5px'}),
            ctrl,
        ],style={'marginBottom':'15px'}))

    return html.Div([
        html.Div([
            # ── LIVE ─────────────────────────────────────────
            card([
                lbl('Select Model'),
                dcc.Dropdown(id='live-mdl',options=m_opts,value=m_keys[0],
                             clearable=False,style=DD_STYLE),
                divr(),
                html.Div('Input Features',style={'color':C['a2'],'fontWeight':'700','fontSize':'11px',
                                                  'letterSpacing':'1.5px','textTransform':'uppercase','marginBottom':'14px'}),
                *inputs,
                html.Div(style={'height':'8px'}),
                btn('Predict','btn-live'),
            ],title='Live Prediction',min_w='300px',flex='0.9'),

            card([
                html.Div('Result',style={'color':C['a2'],'fontWeight':'700','fontSize':'11px',
                                          'letterSpacing':'1.5px','textTransform':'uppercase','marginBottom':'16px'}),
                spin(html.Div(id='live-out',children=[
                    html.Div('Fill inputs and click Predict',style={'color':C['sub'],'textAlign':'center','padding':'50px 0','fontSize':'14px'})
                ]),'live-out'),
            ],min_w='300px',flex='1.4'),
        ],style={'display':'flex','gap':'20px','flexWrap':'wrap','marginBottom':'28px'}),

        # ── BATCH ────────────────────────────────────────────
        card([
            html.Div('Batch Inference',style={'color':C['a2'],'fontWeight':'700','fontSize':'11px',
                                               'letterSpacing':'1.5px','textTransform':'uppercase','marginBottom':'12px'}),
            html.Div('Upload a CSV/Excel with the same feature columns. Predictions + confidence are appended and downloadable.',
                     style={'color':C['sub'],'fontSize':'13px','marginBottom':'14px'}),
            dcc.Upload(id='batch-up',
                children=html.Div(['Upload Batch File (CSV / Excel) — drag & drop or click'],
                                   style={'fontWeight':'600','fontSize':'13px'}),
                style={'border':f'2px dashed {C["a2"]}','borderRadius':'12px','padding':'18px',
                       'textAlign':'center','cursor':'pointer','background':f'{C["a2"]}08','marginBottom':'14px'},
                accept='.csv,.xlsx,.xls'),
            lbl('Model for Batch'),
            dcc.Dropdown(id='batch-mdl',options=m_opts,value=m_keys[0],
                         clearable=False,style={**DD_STYLE,'marginBottom':'14px'}),
            btn('Run Batch Predictions','btn-batch',color=C['a3']),
            spin(html.Div(id='batch-out',style={'marginTop':'16px'}),'batch-out'),
        ],min_w='100%'),
    ])

@app.callback(
    Output('live-out','children'),
    Input('btn-live','n_clicks'),
    [State({'type':'pi','col':ALL},'value'),
     State({'type':'pi','col':ALL},'id'),
     State('live-mdl','value'), State('sid','data')],
    prevent_initial_call=True,
)
def live_pred(n, vals, ids, mk, sid):
    if not n or not sid: raise dash.exceptions.PreventUpdate
    proc   = cache_load(sid,'proc')
    models = cache_load(sid,'models')
    if not proc or not models or mk not in models:
        return html.Div('Data missing.',style={'color':C['red']})

    feat,cat_c,num_c  = proc['feat'],proc['cat_c'],proc['num_c']
    le_map,scaler,task = proc['le_map'],proc['scaler'],proc['task']
    tgt_cls            = proc.get('tgt_cls',['0','1'])
    mdl                = models[mk]['model']

    cvm = {d['col']:v for d,v in zip(ids,vals)}
    row = {}
    for col in feat:
        v = cvm.get(col,None)
        if col in cat_c:
            le = le_map.get(col)
            try: row[col] = int(le.transform([str(v)])[0]) if le and v is not None else 0
            except: row[col] = 0
        else:
            try: row[col] = float(v) if v is not None else 0.0
            except: row[col] = 0.0

    inp = pd.DataFrame([[row[c] for c in feat]], columns=feat)
    # A slider normally supplies all numeric values, but preserve the configured
    # missing-value policy for programmatic input and incomplete forms as well.
    if num_c:
        inp[num_c] = proc['num_imp'].transform(inp[num_c])
    # The scaler was fitted on the complete encoded feature matrix, so it must
    # receive the same ordered set of columns at inference time. Keep its float
    # output as an array instead of assigning it into integer DataFrame columns.
    X_inp = scaler.transform(inp.values) if scaler is not None else inp.values

    pred  = mdl.predict(X_inp)[0]
    proba = mdl.predict_proba(X_inp)[0] if hasattr(mdl,'predict_proba') else None

    if task == 'clf':
        lbl_s = tgt_cls[int(pred)] if int(pred)<len(tgt_cls) else str(pred)
        conf  = float(proba[int(pred)])*100 if proba is not None else None
        pos   = any(k in lbl_s.lower() for k in ['1','yes','placed','hired','selected'])
        col   = C['a4'] if pos else C['red']

        return html.Div([
            html.Div('PREDICTED CLASS',style={'color':C['sub'],'fontSize':'11px','fontWeight':'700',
                                               'textTransform':'uppercase','letterSpacing':'1px','marginBottom':'8px'}),
            html.Div(lbl_s,style={'fontSize':'38px','fontWeight':'900','color':col,
                                   'marginBottom':'6px','textShadow':f'0 0 30px {col}88'}),
            html.Div(f'Confidence: {conf:.1f}%' if conf else '',
                     style={'color':C['text'],'fontSize':'15px','fontWeight':'600','marginBottom':'8px'}),
            html.Div([html.Div(style={'width':f'{conf:.1f}%' if conf else '0%','height':'10px',
                                       'background':f'linear-gradient(90deg,{C["a1"]},{col})',
                                       'borderRadius':'5px','boxShadow':f'0 0 14px {col}77'})],
                     style={'background':C['card2'],'borderRadius':'5px','overflow':'hidden','marginBottom':'20px'}),
            html.Div([
                html.Div('Class Probabilities',style={'color':C['sub'],'fontSize':'11px',
                                                       'fontWeight':'700','textTransform':'uppercase',
                                                       'letterSpacing':'1px','marginBottom':'8px'}),
                *([html.Div([html.Span(f'{cls}: ',style={'color':C['text'],'fontWeight':'600'}),
                              html.Span(f'{p*100:.1f}%',style={'color':C['a2']})],
                             style={'fontSize':'13px','marginBottom':'4px'})
                   for cls,p in zip(tgt_cls,proba)] if proba is not None else []),
            ],style={'background':C['card2'],'borderRadius':'10px','padding':'12px 16px','marginBottom':'12px'}),
            html.Div([html.Span('Model: ',style={'color':C['sub'],'fontSize':'11px'}),
                      html.Span(FRIENDLY.get(mk,mk),style={'color':C['a3'],'fontWeight':'700'})]),
        ])
    else:
        return html.Div([
            html.Div('PREDICTED VALUE',style={'color':C['sub'],'fontSize':'11px','fontWeight':'700',
                                               'textTransform':'uppercase','letterSpacing':'1px','marginBottom':'8px'}),
            html.Div(f'{pred:.4f}',style={'fontSize':'44px','fontWeight':'900','color':C['a2'],
                                           'marginBottom':'12px','textShadow':f'0 0 30px {C["a2"]}88'}),
            html.Div([html.Span('Model: ',style={'color':C['sub'],'fontSize':'11px'}),
                      html.Span(FRIENDLY.get(mk,mk),style={'color':C['a3'],'fontWeight':'700'})]),
        ])

@app.callback(
    [Output('batch-out','children'), Output('dl-batch','data')],
    Input('btn-batch','n_clicks'),
    [State('batch-up','contents'), State('batch-up','filename'),
     State('batch-mdl','value'), State('sid','data')],
    prevent_initial_call=True,
)
def batch_pred(n, contents, fname, mk, sid):
    if not n or not sid: raise dash.exceptions.PreventUpdate
    proc   = cache_load(sid,'proc')
    models = cache_load(sid,'models')
    if not proc or not models:
        return html.Div('Train models first!',style={'color':C['red']}), dash.no_update
    if not contents:
        return html.Div('Upload a batch file first.',style={'color':C['a3']}), dash.no_update

    _,cs = contents.split(','); raw = base64.b64decode(cs)
    try:
        bdf = pd.read_csv(io.StringIO(raw.decode('utf-8'))) if str(fname).endswith('.csv') else pd.read_excel(io.BytesIO(raw))
    except Exception as e:
        return html.Div(f'Parse error: {e}',style={'color':C['red']}), dash.no_update

    feat,cat_c,num_c = proc['feat'],proc['cat_c'],proc['num_c']
    le_map,scaler,task = proc['le_map'],proc['scaler'],proc['task']
    tgt_cls = proc.get('tgt_cls',['0','1'])
    mdl     = models[mk]['model']

    bp = bdf.copy()
    for col in feat:
        if col not in bp.columns:
            bp[col] = np.nan if col in num_c else 'missing'
    if num_c:
        bp[num_c] = proc['num_imp'].transform(bp[num_c])
    for col in cat_c:
        if col in bp.columns:
            le = le_map.get(col)
            if le:
                bp[col] = bp[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0)

    Xb = bp[feat].fillna(0).values.astype(float)
    if scaler is not None:
        Xb = scaler.transform(Xb)

    preds = mdl.predict(Xb)
    if task=='clf':
        bdf['Predicted_Class'] = [tgt_cls[int(p)] if int(p)<len(tgt_cls) else str(p) for p in preds]
        if hasattr(mdl,'predict_proba'):
            bdf['Confidence_%'] = (mdl.predict_proba(Xb).max(axis=1)*100).round(2)
    else:
        bdf['Predicted_Value'] = preds.round(4)

    preview = bdf.head(8)
    tbl = dash_table.DataTable(
        data=preview.to_dict('records'),
        columns=[{'name':c,'id':c} for c in preview.columns],
        style_table={'overflowX':'auto'},
        style_cell={'backgroundColor':C['card2'],'color':C['text'],
                    'border':f'1px solid {C["border"]}',
                    'padding':'9px','fontSize':'12px','textAlign':'center',
                    'fontFamily':'Inter,sans-serif'},
        style_header={'backgroundColor':C['a1'],'color':'white','fontWeight':'700'},page_size=8)

    dl = dict(content=bdf.to_csv(index=False),filename='batch_predictions.csv',type='text/csv')
    out = html.Div([
        html.Div([stat_box('Rows',f'{len(bdf):,}',C['a1']),
                  stat_box('Predictions',f'{len(preds):,}',C['a2'])],
                 style={'display':'flex','gap':'12px','marginBottom':'14px'}),
        html.Div('Preview (first 8 rows):',style={'color':C['sub'],'fontSize':'12px',
                                                    'fontWeight':'600','marginBottom':'8px'}),
        tbl,
        html.Div('Download triggered automatically!',style={'color':C['a4'],'fontSize':'12px','marginTop':'10px','fontWeight':'600'}),
    ])
    return out, dl

# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  ML Dashboard Pro v2.0")
    print("  Open: http://127.0.0.1:8050")
    print("=" * 55)
    app.run(debug=True, host='127.0.0.1', port=8050)
