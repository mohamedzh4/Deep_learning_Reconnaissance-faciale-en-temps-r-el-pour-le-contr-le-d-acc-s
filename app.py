"""
Face Access Control — Streamlit Dashboard
Run: streamlit run app.py
"""
import sqlite3
import sys
import time
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import yaml
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import ROOT, load_config
from src.gallery import Gallery
from src.pipeline import AccessPipeline, draw_overlay

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Face Access Control",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS  — dark modern card design
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0d1117;
    color: #c9d1d9;
}
[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }

/* ── metric cards ── */
.kpi-card {
    background: linear-gradient(145deg,#161b22,#1f2937);
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 22px 18px;
    text-align: center;
    margin-bottom: 10px;
}
.kpi-card .kpi-value { font-size: 2.4rem; font-weight: 700; margin: 0; }
.kpi-card .kpi-label { font-size: 0.82rem; color: #8b949e; margin: 4px 0 0; }

/* ── person card ── */
.person-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 14px;
    text-align: center;
    margin-bottom: 10px;
    transition: border-color .2s, transform .2s;
}
.person-card:hover { border-color: #58a6ff; transform: translateY(-3px); }
.person-card .avatar { font-size: 2.8rem; line-height: 1; }
.person-card .p-name { font-size: 1rem; font-weight: 600; color: #e6edf3; margin-top: 8px; }

/* ── event badges ── */
.badge-granted { background:#1a4731; color:#3fb950; border-radius:20px; padding:3px 10px; font-size:.78rem; font-weight:600; }
.badge-denied  { background:#3d1515; color:#f85149; border-radius:20px; padding:3px 10px; font-size:.78rem; font-weight:600; }

/* ── live dot ── */
.live-dot {
    display:inline-block; width:11px; height:11px;
    background:#3fb950; border-radius:50%; margin-right:7px;
    animation: blink 1.4s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.25} }

/* ── section header ── */
.section-title { font-size:1.35rem; font-weight:700; color:#e6edf3; margin:18px 0 10px; }

/* ── scrollable log ── */
.log-box {
    background:#0d1117; border:1px solid #30363d;
    border-radius:10px; padding:10px 14px;
    max-height:400px; overflow-y:auto;
    font-family: monospace; font-size:.82rem;
}
.log-row { padding:4px 0; border-bottom:1px solid #21262d; }

/* ── buttons ── */
.stButton > button { border-radius:8px; font-weight:600; }

/* ── inputs ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background:#161b22; border:1px solid #30363d;
    color:#e6edf3; border-radius:8px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
CFG_PATH = ROOT / "config.yaml"


@st.cache_resource
def _load_cfg():
    return load_config()


def fresh_cfg():
    """Always read from disk (used after settings save)."""
    with open(CFG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def gallery_path(cfg):
    return ROOT / cfg["paths"]["gallery"]


def log_path(cfg):
    return ROOT / cfg["paths"]["log_db"]


def load_gallery(cfg):
    return Gallery.load(gallery_path(cfg))


@st.cache_resource
def load_engine(model_name: str, use_gpu: bool):
    """Cache the InsightFace engine — slow to init."""
    from src.face_engine import FaceEngine
    cfg = fresh_cfg()
    return FaceEngine(cfg)


def read_events(cfg, limit=500) -> pd.DataFrame:
    p = log_path(cfg)
    if not p.exists():
        return pd.DataFrame(columns=["ts", "event", "name", "score", "note"])
    con = sqlite3.connect(str(p))
    df = pd.read_sql(
        f"SELECT ts, event, name, score, note FROM events ORDER BY ts DESC LIMIT {limit}", con
    )
    con.close()
    return df


def today_stats(df: pd.DataFrame):
    today = time.strftime("%Y-%m-%d")
    t = df[df["ts"].str.startswith(today)]
    return int((t["event"] == "GRANTED").sum()), int((t["event"] == "DENIED").sum())


def kpi(col, value, label, color="#58a6ff"):
    col.markdown(
        f'<div class="kpi-card">'
        f'<p class="kpi-value" style="color:{color}">{value}</p>'
        f'<p class="kpi-label">{label}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────
cfg = fresh_cfg()
gallery = load_gallery(cfg)

with st.sidebar:
    st.markdown("## 🔐 Face Access Control")
    st.caption("Deep Learning · ArcFace · InsightFace")
    st.markdown("---")
    page = st.radio(
        "nav",
        ["🏠  Dashboard", "👥  Gallery", "🎥  Live Control", "📋  Audit Log", "⚙️  Settings"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    c1, c2 = st.columns(2)
    c1.metric("Users", len(gallery))
    df_all = read_events(cfg)
    g_today, d_today = today_stats(df_all)
    c2.metric("Events today", g_today + d_today)
    if len(gallery) == 0:
        st.warning("No users enrolled yet.")
    else:
        st.success(f"{len(gallery)} user(s) active")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE ① — DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠  Dashboard":
    st.markdown("# 🏠 Dashboard")

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, len(gallery), "Enrolled Users", "#58a6ff")
    kpi(k2, g_today, "Granted Today", "#3fb950")
    kpi(k3, d_today, "Denied Today", "#f85149")
    system_ok = "🟢 Ready" if len(gallery) > 0 else "🔴 No Users"
    kpi(k4, system_ok, "System Status", "#d2a8ff")

    st.markdown("---")

    # Recent events table
    st.markdown('<p class="section-title">📊 Recent Events</p>', unsafe_allow_html=True)
    df_all = read_events(cfg, limit=30)
    if df_all.empty:
        st.info("No events yet. Run Live Control to start logging.")
    else:
        display = df_all.copy()
        display["score"] = display["score"].apply(lambda x: f"{float(x):.3f}")
        display["event"] = display["event"].apply(
            lambda e: f"✅ GRANTED" if e == "GRANTED" else "❌ DENIED"
        )
        st.dataframe(display, use_container_width=True, hide_index=True)

    # Mini chart — events per hour
    if not df_all.empty and len(df_all) >= 3:
        st.markdown('<p class="section-title">📈 Events (last 50)</p>', unsafe_allow_html=True)
        chart_df = df_all.copy()
        chart_df["granted"] = (chart_df["event"].str.contains("GRANTED")).astype(int)
        chart_df["denied"]  = (chart_df["event"].str.contains("DENIED")).astype(int)
        chart_df = chart_df[["ts", "granted", "denied"]].set_index("ts")
        st.bar_chart(chart_df[["granted", "denied"]])


# ═════════════════════════════════════════════════════════════════════════════
# PAGE ② — GALLERY
# ═════════════════════════════════════════════════════════════════════════════
elif page == "👥  Gallery":
    st.markdown("# 👥 Enrolled Users Gallery")

    # ── Add new person form ──────────────────────────────────────────────────
    with st.expander("➕  Add New Person", expanded=len(gallery) == 0):
        name_input = st.text_input(
            "Person Name", placeholder="e.g. Ahmed, Fatima, Mohamed…", key="enroll_name"
        )

        st.markdown("**📸 Webcam snapshots** — take several photos (vary pose / expression)")
        img_data = st.camera_input("Take a photo", key="cam_snap")

        col_add, col_clear = st.columns([2, 1])
        add_btn   = col_add.button("➕ Add this photo to buffer", type="secondary")
        enroll_btn = st.button("✅ Enroll now", type="primary", use_container_width=True)

        # Accumulate embeddings across multiple snapshots
        if "enroll_buffer" not in st.session_state:
            st.session_state.enroll_buffer = []
        if "enroll_name_prev" not in st.session_state:
            st.session_state.enroll_name_prev = ""

        # Reset buffer if name changed
        if name_input != st.session_state.enroll_name_prev:
            st.session_state.enroll_buffer = []
            st.session_state.enroll_name_prev = name_input

        if add_btn and img_data and name_input.strip():
            img_pil  = Image.open(BytesIO(img_data.getvalue())).convert("RGB")
            img_bgr  = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            engine   = load_engine(cfg["model"]["name"], cfg["model"]["use_gpu"])
            emb      = engine.embed(img_bgr)
            if emb is not None:
                st.session_state.enroll_buffer.append(emb)
                st.success(f"Face captured ({len(st.session_state.enroll_buffer)} so far). Take more photos for better accuracy.")
            else:
                st.error("❌ No face detected. Try better lighting or move closer.")

        if len(st.session_state.enroll_buffer) > 0:
            st.info(f"Buffer: **{len(st.session_state.enroll_buffer)}** photo(s) for «{name_input}»")

        if enroll_btn:
            if not name_input.strip():
                st.warning("⚠️ Enter a name first.")
            elif len(st.session_state.enroll_buffer) == 0:
                st.warning("⚠️ Capture at least one photo first.")
            else:
                g = load_gallery(cfg)
                g.add(name_input.strip(), st.session_state.enroll_buffer)
                g.save(gallery_path(cfg))
                st.session_state.enroll_buffer = []
                st.session_state.enroll_name_prev = ""
                st.success(f"✅ «{name_input.strip()}» enrolled with {len(st.session_state.enroll_buffer or [1])} embedding(s)!")
                st.rerun()

    st.markdown("---")

    # ── Display gallery grid ──────────────────────────────────────────────────
    gallery = load_gallery(cfg)
    if len(gallery) == 0:
        st.info("👤 No users enrolled yet. Use the form above to add someone.")
    else:
        st.markdown(f'<p class="section-title">👤 {len(gallery)} enrolled person(s)</p>', unsafe_allow_html=True)
        COLS = 4
        rows = [gallery.names[i : i + COLS] for i in range(0, len(gallery.names), COLS)]
        for row in rows:
            cols = st.columns(COLS)
            for col, name in zip(cols, row):
                with col:
                    st.markdown(
                        f'<div class="person-card">'
                        f'<div class="avatar">🧑</div>'
                        f'<div class="p-name">{name}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if st.button("🗑️ Remove", key=f"del_{name}", use_container_width=True):
                        g = load_gallery(cfg)
                        g.remove(name)
                        g.save(gallery_path(cfg))
                        st.success(f"«{name}» removed.")
                        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE ③ — LIVE CONTROL
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🎥  Live Control":
    st.markdown("# 🎥 Real-time Access Control")

    gallery = load_gallery(cfg)
    if len(gallery) == 0:
        st.warning("⚠️ Gallery is empty! Go to **Gallery** and enroll at least one person first.")
        st.stop()

    # ── Layout ───────────────────────────────────────────────────────────────
    col_video, col_panel = st.columns([3, 1])

    with col_panel:
        st.markdown("### ⚙️ Controls")
        cam_idx    = st.number_input("Camera index", 0, 10, cfg["camera"]["index"], key="live_cam")
        threshold  = st.slider("Threshold", 0.10, 0.90, float(cfg["matching"]["threshold"]), 0.01, key="live_thr")
        proc_every = st.slider("Process every N frames", 1, 10, cfg["runtime"]["process_every"], key="live_pe")
        st.markdown("---")
        st.markdown("### 📊 Live Stats")
        fps_ph   = st.empty()
        door_ph  = st.empty()
        name_ph  = st.empty()
        score_ph = st.empty()
        event_ph = st.empty()

    with col_video:
        status_ph = st.empty()
        frame_ph  = st.empty()

    # ── Start / Stop ──────────────────────────────────────────────────────────
    bc1, bc2 = st.columns(2)
    start_btn = bc1.button("▶  Start", type="primary",   use_container_width=True)
    stop_btn  = bc2.button("⏹  Stop",  use_container_width=True)

    if "live_on" not in st.session_state:
        st.session_state.live_on = False
    if start_btn:
        st.session_state.live_on = True
    if stop_btn:
        st.session_state.live_on = False

    if not st.session_state.live_on:
        frame_ph.info("👆 Press **▶ Start** to begin real-time recognition.")
        st.stop()

    # ── Live loop ─────────────────────────────────────────────────────────────
    # Patch config with UI values
    cfg["camera"]["index"]       = int(cam_idx)
    cfg["matching"]["threshold"] = float(threshold)
    cfg["runtime"]["process_every"] = int(proc_every)

    try:
        pipe = AccessPipeline(cfg)
    except Exception as e:
        st.error(f"Pipeline init error: {e}")
        st.session_state.live_on = False
        st.stop()

    cap = cv2.VideoCapture(int(cam_idx))
    if not cap.isOpened():
        st.error(f"Cannot open camera {int(cam_idx)}. Try another index in Controls.")
        st.session_state.live_on = False
        st.stop()

    fps, t_prev, n = 0.0, time.perf_counter(), 0
    last_res = None

    status_ph.markdown(
        '<span class="live-dot"></span> **LIVE** — press ⏹ Stop to end',
        unsafe_allow_html=True,
    )

    try:
        while st.session_state.live_on:
            ok, frame = cap.read()
            if not ok:
                st.error("Camera read error.")
                break

            # Run pipeline every N frames
            if n % proc_every == 0:
                last_res = pipe.process(frame)

            annotated = draw_overlay(frame, last_res, pipe.door.is_open, fps)
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now
            n += 1

            # Display frame
            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_ph.image(frame_rgb, use_container_width=True)

            # Side panel stats
            fps_ph.metric("FPS", f"{fps:.1f}")
            door_ph.metric("Door", "🟢 OPEN" if pipe.door.is_open else "🔴 LOCKED")
            if last_res:
                name_ph.metric("Person", last_res.name or "Unknown")
                score_ph.metric("Score", f"{last_res.score:.3f}")
                if last_res.event:
                    color = "green" if last_res.event == "GRANTED" else "red"
                    event_ph.markdown(
                        f'<span class="badge-{"granted" if last_res.event=="GRANTED" else "denied"}">'
                        f'{last_res.event}</span>',
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
    status_ph.info("⏹ Stream stopped.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE ④ — AUDIT LOG
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📋  Audit Log":
    st.markdown("# 📋 Audit Log")

    df = read_events(cfg, limit=1000)

    if df.empty:
        st.info("No events recorded yet. Run Live Control to generate events.")
        st.stop()

    # ── Filters ──────────────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns(3)
    event_filter = fc1.multiselect(
        "Event type", ["GRANTED", "DENIED"], default=["GRANTED", "DENIED"]
    )
    name_opts  = sorted(df["name"].dropna().unique().tolist())
    name_filter = fc2.multiselect("Person", name_opts)
    date_filter = fc3.text_input("Date (YYYY-MM-DD)", placeholder="e.g. 2026-11-04")

    filtered = df[df["event"].isin(event_filter)]
    if name_filter:
        filtered = filtered[filtered["name"].isin(name_filter)]
    if date_filter.strip():
        filtered = filtered[filtered["ts"].str.startswith(date_filter.strip())]

    # ── KPIs ─────────────────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    kpi(m1, len(filtered), "Total Events", "#58a6ff")
    kpi(m2, int((filtered["event"] == "GRANTED").sum()), "Granted", "#3fb950")
    kpi(m3, int((filtered["event"] == "DENIED").sum()), "Denied",  "#f85149")

    st.markdown("---")

    # ── Table ─────────────────────────────────────────────────────────────────
    display = filtered.copy()
    display["score"] = display["score"].apply(lambda x: f"{float(x):.3f}")
    st.dataframe(display, use_container_width=True, hide_index=True, height=400)

    # ── Export ────────────────────────────────────────────────────────────────
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️  Export to CSV", csv, file_name="audit_log.csv", mime="text/csv"
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE ⑤ — SETTINGS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "⚙️  Settings":
    st.markdown("# ⚙️ Settings")
    st.caption("Changes are saved to `config.yaml`. Restart the app after changing model/GPU settings.")

    with st.form("settings_form"):
        st.markdown("### 📷 Camera")
        s_cam_idx    = st.number_input("Camera index",    0, 10, cfg["camera"]["index"])
        s_cam_width  = st.number_input("Width (px)",  320, 1920, cfg["camera"]["width"],  step=16)
        s_cam_height = st.number_input("Height (px)", 240, 1080, cfg["camera"]["height"], step=16)

        st.markdown("### 🧠 Model")
        s_model = st.selectbox(
            "Model",
            ["buffalo_s", "buffalo_l"],
            index=0 if cfg["model"]["name"] == "buffalo_s" else 1,
            help="buffalo_s = fast (CPU OK) | buffalo_l = accurate (GPU recommended)",
        )
        s_gpu = st.toggle("Use GPU  (requires onnxruntime-gpu + CUDA)", value=bool(cfg["model"]["use_gpu"]))
        s_det = st.select_slider(
            "Detection input size", [160, 320, 640], value=cfg["model"]["det_size"][0]
        )

        st.markdown("### 🎯 Matching")
        s_threshold = st.slider(
            "Similarity threshold", 0.10, 0.90, float(cfg["matching"]["threshold"]), 0.01
        )

        st.markdown("### 🗳️ Decision Voting")
        s_window    = st.slider("Vote window (frames)", 1, 20, cfg["decision"]["window"])
        s_min_votes = st.slider(
            "Min votes to decide", 1, s_window, min(cfg["decision"]["min_votes"], s_window)
        )
        s_cooldown  = st.slider("Cooldown (s)", 1, 60, cfg["decision"]["cooldown_s"])

        st.markdown("### 🚪 Door")
        s_door_sec = st.slider("Door open duration (s)", 1, 30, cfg["door"]["open_seconds"])

        st.markdown("### ⚡ Runtime")
        s_proc_every  = st.slider("Process every N frames", 1, 10, cfg["runtime"]["process_every"])
        s_n_templates = st.slider(
            "Enrollment templates per person", 1, 30, cfg["enroll"]["n_templates"]
        )

        saved = st.form_submit_button("💾  Save Settings", type="primary", use_container_width=True)

    if saved:
        new_cfg = {
            "camera":   {"index": int(s_cam_idx), "width": int(s_cam_width), "height": int(s_cam_height)},
            "model":    {"name": s_model, "det_size": [s_det, s_det], "use_gpu": bool(s_gpu)},
            "matching": {"threshold": float(s_threshold)},
            "decision": {"window": int(s_window), "min_votes": int(s_min_votes), "cooldown_s": int(s_cooldown)},
            "door":     {"open_seconds": int(s_door_sec)},
            "runtime":  {"process_every": int(s_proc_every)},
            "enroll":   {"n_templates": int(s_n_templates)},
            "paths":    cfg["paths"],
        }
        with open(CFG_PATH, "w", encoding="utf-8") as f:
            yaml.dump(new_cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        st.cache_resource.clear()
        st.success("✅ Settings saved! Reload the page to apply.")
