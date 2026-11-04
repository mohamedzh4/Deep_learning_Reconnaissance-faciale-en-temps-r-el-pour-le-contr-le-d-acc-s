"""
Face Access Control — Streamlit Dashboard
Run: streamlit run app.py
"""
import base64
import sqlite3
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import yaml

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import ROOT, load_config
from src.gallery import Gallery
from src.pipeline import AccessPipeline, draw_overlay

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FaceAccess",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🛡</text></svg>",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# SVG Icon library  (Lucide — MIT licence)
# ─────────────────────────────────────────────────────────────────────────────
_SVG: dict[str, str] = {
    "shield-check": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="CLRX" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg>',
    "scan-face":    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="CLRX" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><path d="M9 9h.01"/><path d="M15 9h.01"/></svg>',
    "layout":       '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="CLRX" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="7" height="9" x="3" y="3" rx="1"/><rect width="7" height="5" x="3" y="16" rx="1"/><rect width="7" height="9" x="14" y="12" rx="1"/><rect width="7" height="5" x="14" y="3" rx="1"/></svg>',
    "users":        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="CLRX" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
    "video":        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="CLRX" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m22 8-6 4 6 4V8z"/><rect width="14" height="12" x="2" y="6" rx="2" ry="2"/></svg>',
    "file-text":    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="CLRX" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/><line x1="16" x2="8" y1="13" y2="13"/><line x1="16" x2="8" y1="17" y2="17"/><line x1="10" x2="8" y1="9" y2="9"/></svg>',
    "settings":     '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="CLRX" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>',
    "user":         '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="CLRX" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>',
    "trash":        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="CLRX" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/><line x1="10" x2="10" y1="11" y2="17"/><line x1="14" x2="14" y1="11" y2="17"/></svg>',
    "camera":       '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="CLRX" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/></svg>',
    "check-circle": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="CLRX" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
    "x-circle":     '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="CLRX" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/></svg>',
    "lock":         '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="CLRX" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>',
    "unlock":       '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="CLRX" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 9.9-1"/></svg>',
    "download":     '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="CLRX" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/></svg>',
    "plus-circle":  '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="CLRX" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M8 12h8"/><path d="M12 8v8"/></svg>',
    "activity":     '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="CLRX" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
    "bar-chart":    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="CLRX" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" x2="18" y1="20" y2="10"/><line x1="12" x2="12" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="16"/></svg>',
    "cpu":          '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="CLRX" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2"/></svg>',
}

_NAV_PAGES = [
    ("Dashboard",    "layout"),
    ("Gallery",      "users"),
    ("Live Control", "video"),
    ("Audit Log",    "file-text"),
    ("Settings",     "settings"),
]


def _ico(name: str, size: int = 18, color: str = "#8b949e") -> str:
    """Return inline SVG for the given icon."""
    raw = _SVG.get(name, _SVG["scan-face"]).replace("CLRX", color)
    return raw.replace(
        "<svg ",
        f'<svg width="{size}" height="{size}" style="vertical-align:middle;display:inline-block;flex-shrink:0" ',
        1,
    )


def _b64_uri(name: str, color: str = "%238b949e") -> str:
    """Return CSS data-URI for a given icon (URL-encoded color)."""
    raw = _SVG.get(name, "").replace("CLRX", color.replace("%23", "#"))
    return "data:image/svg+xml;base64," + base64.b64encode(raw.encode()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# Build nav-icon CSS  (base64 SVG per nav label via nth-child)
# ─────────────────────────────────────────────────────────────────────────────
_nav_icon_css = ""
for _idx, (_lbl, _iname) in enumerate(_NAV_PAGES, 1):
    _uri = _b64_uri(_iname, "#8b949e")
    _nav_icon_css += f"""
    [data-testid="stRadio"] > div > label:nth-child({_idx})::before {{
        background-image: url("{_uri}");
    }}"""

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [data-testid="stAppViewContainer"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0d1117;
    color: #c9d1d9;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: #161b22;
    border-right: 1px solid #21262d;
}}
[data-testid="stSidebar"] > div:first-child {{ padding-top: 0 !important; }}

/* ── Sidebar logo block ── */
.sidebar-logo {{
    display: flex; align-items: center; gap: 12px;
    padding: 20px 20px 16px; border-bottom: 1px solid #21262d;
    margin-bottom: 10px;
}}
.sl-name {{ font-size: 1.05rem; font-weight: 700; color: #e6edf3; letter-spacing: -.3px; }}
.sl-sub  {{ font-size: .68rem; color: #58a6ff; text-transform: uppercase; letter-spacing: .8px; margin-top: 2px; }}

/* ── Sidebar nav (radio) ── */
[data-testid="stRadio"] {{ margin: 0 !important; }}
[data-testid="stRadio"] > div {{ display: flex; flex-direction: column; gap: 2px; padding: 0 10px; }}

[data-testid="stRadio"] > div > label {{
    display: flex !important; align-items: center;
    padding: 10px 14px 10px 42px !important;
    border-radius: 8px !important; cursor: pointer;
    color: #8b949e !important; font-size: .9rem; font-weight: 500;
    transition: background .15s, color .15s; position: relative;
    border: none !important; background: transparent !important;
}}
[data-testid="stRadio"] > div > label::before {{
    content: '';
    position: absolute; left: 14px; top: 50%; transform: translateY(-50%);
    width: 18px; height: 18px;
    background-repeat: no-repeat; background-position: center; background-size: 18px 18px;
}}
{_nav_icon_css}
/* Hide the radio dot */
[data-testid="stRadio"] > div > label > div:first-child {{ display: none !important; }}
[data-testid="stRadio"] > div > label:hover {{
    background: #21262d !important; color: #e6edf3 !important;
}}
/* Active nav item — Streamlit adds aria-checked or a checked class */
[data-testid="stRadio"] > div > label[data-baseweb="radio"]:has(input:checked),
[data-testid="stRadio"] > div > label:has(input:checked) {{
    background: #21262d !important; color: #58a6ff !important;
    border-left: 2px solid #58a6ff !important; padding-left: 40px !important;
}}

/* ── Sidebar footer stats ── */
.sb-stat {{ padding: 6px 20px 20px; }}
.sb-stat .metric {{ display:flex; justify-content:space-between; align-items:center;
    padding: 8px 12px; background:#1c2128; border-radius:8px; margin-top:6px; }}
.sb-stat .metric-label {{ font-size:.78rem; color:#8b949e; }}
.sb-stat .metric-val   {{ font-size:.9rem; font-weight:700; color:#e6edf3; }}

/* ── Page header ── */
.page-header {{
    display: flex; align-items: center; gap: 14px;
    padding-bottom: 20px; border-bottom: 1px solid #21262d; margin-bottom: 24px;
}}
.ph-icon {{
    background: #21262d; border: 1px solid #30363d;
    border-radius: 10px; padding: 10px; display: flex;
}}
.ph-title {{ font-size: 1.5rem; font-weight: 700; color: #e6edf3; margin: 0; line-height: 1.2; }}
.ph-sub   {{ font-size: .82rem; color: #8b949e; margin: 4px 0 0; }}

/* ── KPI Cards ── */
.kpi-row {{ display: flex; gap: 12px; margin-bottom: 24px; flex-wrap: wrap; }}
.kpi-card {{
    flex: 1; min-width: 140px;
    background: #161b22; border: 1px solid #21262d; border-radius: 12px;
    padding: 18px 20px; display: flex; flex-direction: column; gap: 8px;
    transition: border-color .2s;
}}
.kpi-card:hover {{ border-color: #30363d; }}
.kpi-value {{ font-size: 1.9rem; font-weight: 700; line-height: 1; }}
.kpi-label {{ font-size: .78rem; color: #8b949e; font-weight: 500; text-transform: uppercase; letter-spacing: .5px; }}

/* ── Section title ── */
.sect {{
    font-size: .78rem; font-weight: 600; color: #8b949e;
    text-transform: uppercase; letter-spacing: .7px;
    display: flex; align-items: center; gap: 8px; margin: 20px 0 12px;
}}
.sect::after {{ content: ''; flex: 1; height: 1px; background: #21262d; }}

/* ── Person card ── */
.person-card {{
    background: #161b22; border: 1px solid #21262d; border-radius: 12px;
    padding: 20px 14px; text-align: center; margin-bottom: 10px;
    transition: border-color .2s, transform .2s;
}}
.person-card:hover {{ border-color: #58a6ff; transform: translateY(-2px); }}
.p-avatar {{
    width: 56px; height: 56px; background: #21262d; border-radius: 50%;
    display: flex; align-items: center; justify-content: center; margin: 0 auto 12px;
}}
.p-name {{ font-size: .95rem; font-weight: 600; color: #e6edf3; }}

/* ── Badges ── */
.badge {{ display:inline-flex; align-items:center; gap:5px;
    border-radius:6px; padding:3px 10px; font-size:.78rem; font-weight:600; }}
.badge-granted {{ background:#1a4731; color:#3fb950; }}
.badge-denied  {{ background:#3d1515; color:#f85149; }}

/* ── Animated dots ── */
.dot-live {{ display:inline-block; width:9px; height:9px; background:#3fb950; border-radius:50%;
    margin-right:6px; animation:blink 1.4s ease-in-out infinite; }}
.dot-rec  {{ display:inline-block; width:9px; height:9px; background:#f85149; border-radius:50%;
    margin-right:6px; animation:blink .8s ease-in-out infinite; }}
@keyframes blink {{ 0%,100%{{opacity:1}} 50%{{opacity:.2}} }}

/* ── Status pill ── */
.status-pill {{
    display:inline-flex; align-items:center; gap:8px;
    background:#161b22; border:1px solid #21262d; border-radius:8px;
    padding:8px 14px; font-size:.88rem; font-weight:500; margin-bottom:12px;
}}

/* ── Capture counter ── */
.cap-pill {{ background:#1a4731; color:#3fb950; border-radius:8px;
    padding:8px 16px; font-weight:700; font-size:.95rem; display:inline-block; }}

/* ── Buttons ── */
.stButton > button {{
    border-radius: 8px !important; font-weight: 600 !important;
    font-size: .88rem !important; border: 1px solid #30363d !important;
    transition: all .15s !important;
}}
.stButton > button[kind="primary"] {{
    background: #1f6feb !important; border-color: #388bfd !important; color: #fff !important;
}}
.stButton > button[kind="primary"]:hover {{ background: #388bfd !important; }}
.stButton > button:not([kind="primary"]):hover {{ background: #21262d !important; color: #e6edf3 !important; }}

/* ── Metrics ── */
[data-testid="stMetric"] {{
    background: #161b22; border: 1px solid #21262d;
    border-radius: 10px; padding: 14px !important;
}}
[data-testid="stMetricLabel"] {{ color: #8b949e !important; font-size: .78rem !important; text-transform: uppercase; }}
[data-testid="stMetricValue"] {{ color: #e6edf3 !important; }}
[data-testid="stMetricDelta"] {{ font-size: .8rem !important; }}

/* ── DataFrames ── */
.stDataFrame {{ border: 1px solid #21262d !important; border-radius: 10px !important; overflow: hidden; }}
[data-testid="stDataFrame"] {{ border-radius: 10px; }}

/* ── Expander ── */
[data-testid="stExpander"] {{
    border: 1px solid #21262d !important; border-radius: 10px !important;
    background: #161b22 !important; overflow: hidden;
}}
[data-testid="stExpander"] summary {{
    font-weight: 600 !important; color: #e6edf3 !important;
    padding: 14px 16px !important;
}}

/* ── Inputs ── */
input[type="text"], input[type="number"], textarea {{
    background: #0d1117 !important; border: 1px solid #30363d !important;
    color: #e6edf3 !important; border-radius: 8px !important;
}}
input:focus, textarea:focus {{ border-color: #58a6ff !important; box-shadow: 0 0 0 3px rgba(88,166,255,.15) !important; }}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: #161b22; }}
::-webkit-scrollbar-thumb {{ background: #30363d; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: #484f58; }}

/* ── Progress bar ── */
.stProgress > div > div {{ background: #58a6ff !important; border-radius: 4px; }}
.stProgress > div {{ background: #21262d !important; border-radius: 4px; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────
CFG_PATH = ROOT / "config.yaml"


def fresh_cfg():
    with open(CFG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def gal_path(cfg):  return ROOT / cfg["paths"]["gallery"]
def log_path(cfg):  return ROOT / cfg["paths"]["log_db"]
def load_gal(cfg):  return Gallery.load(gal_path(cfg))


@st.cache_resource
def load_engine(model_name: str, use_gpu: bool):
    from src.face_engine import FaceEngine
    return FaceEngine(fresh_cfg())


def read_events(cfg, limit: int = 500) -> pd.DataFrame:
    p = log_path(cfg)
    if not p.exists():
        return pd.DataFrame(columns=["ts", "event", "name", "score", "note"])
    con = sqlite3.connect(str(p))
    df  = pd.read_sql(
        f"SELECT ts, event, name, score, note FROM events ORDER BY ts DESC LIMIT {limit}", con
    )
    con.close()
    return df


def today_stats(df: pd.DataFrame):
    today = time.strftime("%Y-%m-%d")
    t = df[df["ts"].str.startswith(today)]
    return int((t["event"] == "GRANTED").sum()), int((t["event"] == "DENIED").sum())


def page_header(icon_name: str, title: str, subtitle: str = ""):
    sub = f'<p class="ph-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div class="page-header">
        <div class="ph-icon">{_ico(icon_name, 28, "#58a6ff")}</div>
        <div><h1 class="ph-title">{title}</h1>{sub}</div>
    </div>""", unsafe_allow_html=True)


def kpi_card(col, icon_name: str, value, label: str, color: str = "#58a6ff"):
    col.markdown(f"""
    <div class="kpi-card">
        {_ico(icon_name, 20, color)}
        <div class="kpi-value" style="color:{color}">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>""", unsafe_allow_html=True)


def show_image(ph, arr):
    ph.image(arr, use_column_width=True)


def _draw_enroll(frame, faces, n_cap, left, dur):
    out = frame.copy()
    good = len(faces) == 1 and faces[0].det_score >= 0.70
    for f in faces:
        x1, y1, x2, y2 = f.bbox.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 80) if good else (60, 100, 220), 2)
    bar_w  = out.shape[1]
    filled = int(bar_w * max(0, left / dur))
    cv2.rectangle(out, (0, 0), (bar_w, 7), (30, 30, 30), -1)
    cv2.rectangle(out, (0, 0), (filled, 7), (0, 180, 80), -1)
    hint  = f"Captures: {n_cap}" if good else ("1 face required" if len(faces) != 1 else "Low quality")
    cv2.putText(out, hint, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, .65, (230, 230, 230), 2)
    cv2.putText(out, f"{left:.1f}s", (bar_w - 80, 36), cv2.FONT_HERSHEY_SIMPLEX, .7, (88, 166, 255), 2)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
cfg     = fresh_cfg()
gallery = load_gal(cfg)

with st.sidebar:
    # Logo
    st.markdown(f"""
    <div class="sidebar-logo">
        {_ico("scan-face", 36, "#58a6ff")}
        <div>
            <div class="sl-name">FaceAccess</div>
            <div class="sl-sub">Deep Learning Security</div>
        </div>
    </div>""", unsafe_allow_html=True)

    page = st.radio(
        "nav",
        [lbl for lbl, _ in _NAV_PAGES],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Quick stats
    df_all        = read_events(cfg)
    g_today, d_today = today_stats(df_all)
    c1, c2 = st.columns(2)
    c1.metric("Users",  len(gallery))
    c2.metric("Today",  g_today + d_today)
    if len(gallery) == 0:
        st.warning("No users enrolled yet.")
    else:
        st.markdown(
            f'<div style="padding:8px 0 4px; font-size:.8rem; color:#3fb950;">'
            f'{_ico("check-circle", 14, "#3fb950")} {len(gallery)} user(s) active</div>',
            unsafe_allow_html=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE — DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    page_header("layout", "Dashboard", "Real-time overview of the access control system")

    k1, k2, k3, k4 = st.columns(4)
    kpi_card(k1, "users",       len(gallery),         "Enrolled Users", "#58a6ff")
    kpi_card(k2, "check-circle", g_today,             "Granted Today",  "#3fb950")
    kpi_card(k3, "x-circle",    d_today,              "Denied Today",   "#f85149")
    kpi_card(k4, "activity",    "Active" if len(gallery) > 0 else "Idle", "System Status", "#d2a8ff")

    st.markdown('<div class="sect">Recent Events</div>', unsafe_allow_html=True)
    df_recent = read_events(cfg, 30)
    if df_recent.empty:
        st.info("No events yet. Start Live Control to begin logging.")
    else:
        d = df_recent.copy()
        d["score"] = d["score"].apply(lambda x: f"{float(x):.3f}")
        d["event"] = d["event"].apply(lambda e: "GRANTED" if e == "GRANTED" else "DENIED")
        st.dataframe(d.reset_index(drop=True), height=280)

    if not df_recent.empty and len(df_recent) >= 3:
        st.markdown('<div class="sect">Events History</div>', unsafe_allow_html=True)
        c = df_recent.copy()
        c["granted"] = (c["event"] == "GRANTED").astype(int)
        c["denied"]  = (c["event"] == "DENIED").astype(int)
        st.bar_chart(c.set_index("ts")[["granted", "denied"]])


# ═════════════════════════════════════════════════════════════════════════════
# PAGE — GALLERY
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Gallery":
    page_header("users", "User Gallery", "Manage enrolled identities")

    # ── Enroll new person ─────────────────────────────────────────────────────
    with st.expander("Add New Person — Video Enrollment", expanded=len(gallery) == 0):
        st.markdown(f"""
        <div style="display:flex;gap:10px;align-items:flex-start;padding:4px 0 12px">
            {_ico("camera", 20, "#58a6ff")}
            <div style="font-size:.88rem; color:#8b949e; line-height:1.6">
                Enter the person's name, choose a duration, then click <strong style="color:#e6edf3">Start Recording</strong>.
                The webcam opens automatically — vary your head angle and expression.
                Frames are captured every 0.5 s whenever a good face is detected.
            </div>
        </div>""", unsafe_allow_html=True)

        c1, c2, c3 = st.columns([3, 2, 1])
        enroll_name = c1.text_input("Full Name", placeholder="e.g. Ahmed Benali", key="enroll_name")
        enroll_dur  = c2.slider("Duration (seconds)", 5, 60, 20, key="enroll_dur")
        enroll_cam  = c3.number_input("Camera", 0, 10, cfg["camera"]["index"], key="enroll_cam")

        if "enroll_running" not in st.session_state:
            st.session_state.enroll_running = False

        bc1, bc2 = st.columns(2)
        start_enroll  = bc1.button("Start Recording", type="primary",
                                    disabled=not enroll_name.strip())
        cancel_enroll = bc2.button("Cancel", disabled=not st.session_state.enroll_running)

        if cancel_enroll:
            st.session_state.enroll_running = False
        if start_enroll and enroll_name.strip():
            st.session_state.enroll_running = True

        # ── Recording loop ───────────────────────────────────────────────────
        if st.session_state.enroll_running:
            st.markdown(
                f'<div class="status-pill"><span class="dot-rec"></span>'
                f'<strong>RECORDING</strong>&nbsp; — look at the camera, vary your head angle</div>',
                unsafe_allow_html=True,
            )
            prog   = st.progress(0.0)
            st_txt = st.empty()
            fr_ph  = st.empty()

            engine = load_engine(cfg["model"]["name"], cfg["model"]["use_gpu"])
            cap    = cv2.VideoCapture(int(enroll_cam))

            if not cap.isOpened():
                st.error(f"Cannot open camera {int(enroll_cam)}.")
                st.session_state.enroll_running = False
                st.stop()

            embeddings, last_cap = [], 0.0
            t0 = time.time()
            try:
                while True:
                    elapsed = time.time() - t0
                    left    = enroll_dur - elapsed
                    if left <= 0 or not st.session_state.enroll_running:
                        break
                    ok, frame = cap.read()
                    if not ok:
                        break
                    faces = engine.analyze(frame)
                    good  = len(faces) == 1 and faces[0].det_score >= 0.70
                    if good and (time.time() - last_cap) >= 0.5:
                        embeddings.append(faces[0].embedding)
                        last_cap = time.time()
                    annotated = _draw_enroll(frame, faces, len(embeddings), left, enroll_dur)
                    show_image(fr_ph, cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
                    prog.progress(min(elapsed / enroll_dur, 1.0))
                    st_txt.markdown(
                        f'<span class="cap-pill">{_ico("camera", 16, "#3fb950")} '
                        f'  {len(embeddings)} frames captured</span>'
                        f'  <span style="color:#8b949e; font-size:.85rem; margin-left:10px">'
                        f'{left:.1f} s remaining</span>',
                        unsafe_allow_html=True,
                    )
                    time.sleep(0.03)
            finally:
                cap.release()
                cv2.destroyAllWindows()

            fr_ph.empty()
            prog.progress(1.0)
            st.session_state.enroll_running = False

            if not embeddings:
                st.error("No face detected. Try again with better lighting or move closer.")
            else:
                g = load_gal(cfg)
                g.add(enroll_name.strip(), embeddings)
                g.save(gal_path(cfg))
                st.success(f"'{enroll_name.strip()}' enrolled with {len(embeddings)} captures.")
                st.balloons()
                time.sleep(1)
                st.rerun()

    st.markdown("---")

    # ── Gallery grid ─────────────────────────────────────────────────────────
    gallery = load_gal(cfg)
    if len(gallery) == 0:
        st.info("No users enrolled. Use the form above.")
    else:
        st.markdown(
            f'<div class="sect">{_ico("users", 14, "#8b949e")}  {len(gallery)} enrolled person(s)</div>',
            unsafe_allow_html=True,
        )
        COLS = 4
        for row_names in [gallery.names[i:i + COLS] for i in range(0, len(gallery.names), COLS)]:
            cols = st.columns(COLS)
            for col, name in zip(cols, row_names):
                with col:
                    st.markdown(f"""
                    <div class="person-card">
                        <div class="p-avatar">{_ico("user", 28, "#58a6ff")}</div>
                        <div class="p-name">{name}</div>
                    </div>""", unsafe_allow_html=True)
                    if st.button("Remove", key=f"del_{name}"):
                        g = load_gal(cfg)
                        g.remove(name)
                        g.save(gal_path(cfg))
                        st.success(f"'{name}' removed.")
                        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE — LIVE CONTROL
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Live Control":
    page_header("video", "Live Control", "Real-time face recognition access control")

    gallery = load_gal(cfg)
    if len(gallery) == 0:
        st.warning("Gallery is empty. Enroll at least one person in Gallery first.")
        st.stop()

    col_v, col_p = st.columns([3, 1])
    with col_p:
        st.markdown('<div class="sect">Controls</div>', unsafe_allow_html=True)
        cam_idx    = st.number_input("Camera index", 0, 10, cfg["camera"]["index"], key="live_cam")
        threshold  = st.slider("Similarity threshold", 0.10, 0.90, float(cfg["matching"]["threshold"]), 0.01, key="live_thr")
        proc_every = st.slider("Process every N frames", 1, 10, cfg["runtime"]["process_every"], key="live_pe")
        st.markdown('<div class="sect">Live Stats</div>', unsafe_allow_html=True)
        fps_ph   = st.empty()
        door_ph  = st.empty()
        name_ph  = st.empty()
        score_ph = st.empty()
        event_ph = st.empty()

    with col_v:
        status_ph = st.empty()
        frame_ph  = st.empty()

    bc1, bc2 = st.columns(2)
    start_btn = bc1.button("Start", type="primary")
    stop_btn  = bc2.button("Stop")

    if "live_on" not in st.session_state:
        st.session_state.live_on = False
    if start_btn:
        st.session_state.live_on = True
    if stop_btn:
        st.session_state.live_on = False

    if not st.session_state.live_on:
        frame_ph.markdown(
            f'<div style="background:#161b22;border:1px solid #21262d;border-radius:12px;'
            f'padding:60px;text-align:center;color:#484f58">'
            f'{_ico("video", 48, "#484f58")}<br><br>'
            f'<span style="font-size:1rem">Press <strong style="color:#e6edf3">Start</strong> '
            f'to begin real-time recognition</span></div>',
            unsafe_allow_html=True,
        )
        st.stop()

    cfg["camera"]["index"]          = int(cam_idx)
    cfg["matching"]["threshold"]    = float(threshold)
    cfg["runtime"]["process_every"] = int(proc_every)

    try:
        pipe = AccessPipeline(cfg)
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        st.session_state.live_on = False
        st.stop()

    cap = cv2.VideoCapture(int(cam_idx))
    if not cap.isOpened():
        st.error(f"Cannot open camera {int(cam_idx)}.")
        st.session_state.live_on = False
        st.stop()

    fps, t_prev, n, last_res = 0.0, time.perf_counter(), 0, None
    status_ph.markdown(
        f'<div class="status-pill"><span class="dot-live"></span>'
        f'<strong>LIVE</strong>&nbsp; — press Stop to end</div>',
        unsafe_allow_html=True,
    )

    try:
        while st.session_state.live_on:
            ok, frame = cap.read()
            if not ok:
                break
            if n % int(proc_every) == 0:
                last_res = pipe.process(frame)
            annotated = draw_overlay(frame, last_res, pipe.door.is_open, fps)
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev, n = now, n + 1
            show_image(frame_ph, cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            fps_ph.metric("FPS",  f"{fps:.1f}")
            door_ph.metric("Door", "OPEN" if pipe.door.is_open else "LOCKED")
            if last_res:
                name_ph.metric("Person", last_res.name or "Unknown")
                score_ph.metric("Score",  f"{last_res.score:.3f}")
                if last_res.event:
                    cls = "granted" if last_res.event == "GRANTED" else "denied"
                    ico = "check-circle" if cls == "granted" else "x-circle"
                    clr = "#3fb950"     if cls == "granted" else "#f85149"
                    event_ph.markdown(
                        f'<div class="badge badge-{cls}">'
                        f'{_ico(ico, 14, clr)} {last_res.event}</div>',
                        unsafe_allow_html=True,
                    )
            time.sleep(0.01)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            pipe.audit.close()
        except Exception:
            pass

    st.session_state.live_on = False
    status_ph.info("Stream stopped.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE — AUDIT LOG
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Audit Log":
    page_header("file-text", "Audit Log", "Complete history of access events")

    df = read_events(cfg, 1000)
    if df.empty:
        st.info("No events recorded. Run Live Control to start logging.")
        st.stop()

    fc1, fc2, fc3 = st.columns(3)
    ev_filter   = fc1.multiselect("Event", ["GRANTED", "DENIED"], default=["GRANTED", "DENIED"])
    name_opts   = sorted(df["name"].dropna().unique().tolist())
    name_filter = fc2.multiselect("Person", name_opts)
    date_filter = fc3.text_input("Date (YYYY-MM-DD)", placeholder="2026-11-04")

    flt = df[df["event"].isin(ev_filter)]
    if name_filter:
        flt = flt[flt["name"].isin(name_filter)]
    if date_filter.strip():
        flt = flt[flt["ts"].str.startswith(date_filter.strip())]

    k1, k2, k3 = st.columns(3)
    kpi_card(k1, "bar-chart",   len(flt),                                 "Total Events", "#58a6ff")
    kpi_card(k2, "check-circle", int((flt["event"] == "GRANTED").sum()),  "Granted",      "#3fb950")
    kpi_card(k3, "x-circle",    int((flt["event"] == "DENIED").sum()),    "Denied",       "#f85149")

    st.markdown('<div class="sect">Events</div>', unsafe_allow_html=True)
    d = flt.copy()
    d["score"] = d["score"].apply(lambda x: f"{float(x):.3f}")
    st.dataframe(d.reset_index(drop=True), height=380)

    csv = flt.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"{_ico('download', 15, '#8b949e')} Export CSV",
        csv, file_name="audit_log.csv", mime="text/csv"
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE — SETTINGS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Settings":
    page_header("settings", "Settings", "Edit system configuration — saved to config.yaml")

    with st.form("settings_form"):
        st.markdown(f'<div class="sect">{_ico("camera", 14, "#8b949e")} Camera</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        s_cam_idx    = c1.number_input("Camera index", 0, 10, cfg["camera"]["index"])
        s_cam_w      = c2.number_input("Width (px)",  320, 1920, cfg["camera"]["width"],  step=16)
        s_cam_h      = c3.number_input("Height (px)", 240, 1080, cfg["camera"]["height"], step=16)

        st.markdown(f'<div class="sect">{_ico("cpu", 14, "#8b949e")} Model</div>', unsafe_allow_html=True)
        c4, c5, c6 = st.columns(3)
        s_model  = c4.selectbox("Model", ["buffalo_s", "buffalo_l"],
                                index=0 if cfg["model"]["name"] == "buffalo_s" else 1)
        s_gpu    = c5.checkbox("Use GPU  (onnxruntime-gpu + CUDA)", value=bool(cfg["model"]["use_gpu"]))
        s_det    = c6.select_slider("Det. input size", [160, 320, 640], value=cfg["model"]["det_size"][0])

        st.markdown(f'<div class="sect">{_ico("activity", 14, "#8b949e")} Matching & Voting</div>', unsafe_allow_html=True)
        c7, c8, c9 = st.columns(3)
        s_thr   = c7.slider("Similarity threshold",  0.10, 0.90, float(cfg["matching"]["threshold"]), 0.01)
        s_win   = c8.slider("Vote window (frames)",  1, 20, cfg["decision"]["window"])
        s_votes = c9.slider("Min votes to decide",   1, s_win, min(cfg["decision"]["min_votes"], s_win))

        c10, c11, c12 = st.columns(3)
        s_cool  = c10.slider("Cooldown (s)",               1,  60, cfg["decision"]["cooldown_s"])
        s_door  = c11.slider("Door open duration (s)",     1,  30, cfg["door"]["open_seconds"])
        s_every = c12.slider("Process every N frames",     1,  10, cfg["runtime"]["process_every"])

        s_ntpl = st.slider("Enrollment templates per person", 1, 30, cfg["enroll"]["n_templates"])

        saved = st.form_submit_button("Save Settings", type="primary")

    if saved:
        new = {
            "camera":   {"index": int(s_cam_idx), "width": int(s_cam_w), "height": int(s_cam_h)},
            "model":    {"name": s_model, "det_size": [s_det, s_det], "use_gpu": bool(s_gpu)},
            "matching": {"threshold": float(s_thr)},
            "decision": {"window": int(s_win), "min_votes": int(s_votes), "cooldown_s": int(s_cool)},
            "door":     {"open_seconds": int(s_door)},
            "runtime":  {"process_every": int(s_every)},
            "enroll":   {"n_templates": int(s_ntpl)},
            "paths":    cfg["paths"],
        }
        with open(CFG_PATH, "w", encoding="utf-8") as f:
            yaml.dump(new, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        st.cache_resource.clear()
        st.success("Settings saved. Reload the page to apply model changes.")
