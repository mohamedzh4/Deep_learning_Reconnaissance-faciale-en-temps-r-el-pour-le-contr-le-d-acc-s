"""
Face Access Control — Streamlit Dashboard
Run: streamlit run app.py
"""
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
# Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Face Access Control",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Compatibility helper — works on old AND new Streamlit versions
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as _st
_st_version = tuple(int(x) for x in _st.__version__.split(".")[:2])

def show_image(placeholder, img_array, **kwargs):
    """Display an image; use the right kwarg for the installed Streamlit version."""
    if _st_version >= (1, 25):
        placeholder.image(img_array, use_container_width=True)
    else:
        placeholder.image(img_array, use_column_width=True)


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

/* ── recording dot ── */
.rec-dot {
    display:inline-block; width:11px; height:11px;
    background:#f85149; border-radius:50%; margin-right:7px;
    animation: blink 0.8s ease-in-out infinite;
}

/* ── section header ── */
.section-title { font-size:1.35rem; font-weight:700; color:#e6edf3; margin:18px 0 10px; }

/* ── capture counter badge ── */
.cap-badge {
    display:inline-block;
    background:#1a4731; color:#3fb950;
    border-radius:20px; padding:4px 14px;
    font-size:.9rem; font-weight:700;
}

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


def fresh_cfg():
    """Always read from disk."""
    with open(CFG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_gallery_path(cfg):
    return ROOT / cfg["paths"]["gallery"]


def get_log_path(cfg):
    return ROOT / cfg["paths"]["log_db"]


def load_gallery(cfg):
    return Gallery.load(get_gallery_path(cfg))


@st.cache_resource
def load_engine(model_name: str, use_gpu: bool):
    """Cache the InsightFace engine — slow to init."""
    from src.face_engine import FaceEngine
    cfg = fresh_cfg()
    return FaceEngine(cfg)


def read_events(cfg, limit=500) -> pd.DataFrame:
    p = get_log_path(cfg)
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


def _draw_enroll_frame(frame, faces, n_captured, duration_left, min_det=0.70):
    """Draw enrollment overlay on frame."""
    out = frame.copy()
    good_face = len(faces) == 1 and faces[0].det_score >= min_det
    for f in faces:
        x1, y1, x2, y2 = f.bbox.astype(int)
        color = (0, 220, 80) if good_face else (0, 80, 220)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

    # Countdown bar
    bar_w = out.shape[1]
    filled = int(bar_w * max(0, duration_left / 20.0))
    cv2.rectangle(out, (0, 0), (bar_w, 6), (40, 40, 40), -1)
    cv2.rectangle(out, (0, 0), (filled, 6), (0, 180, 80), -1)

    # Text overlay
    status = f"✓ {n_captured} captured" if good_face else ("Move closer / 1 face only" if len(faces) != 1 else "Low quality")
    t_txt = f"{duration_left:.1f}s"
    cv2.putText(out, status, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(out, t_txt, (bar_w - 90, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 220, 255), 2)
    return out


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
            lambda e: "✅ GRANTED" if e == "GRANTED" else "❌ DENIED"
        )
        st.dataframe(display, hide_index=True)

    # Mini chart
    if not df_all.empty and len(df_all) >= 3:
        st.markdown('<p class="section-title">📈 Events history</p>', unsafe_allow_html=True)
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

    # ── Add new person ───────────────────────────────────────────────────────
    with st.expander("➕  Add New Person — Video Enrollment", expanded=len(gallery) == 0):

        st.markdown("""
        **How it works:**
        1. Enter the person's name
        2. Set the recording duration (default 20 s)
        3. Click **🎥 Start Recording** — the webcam opens automatically
        4. Look at the camera, vary your head angle / expression
        5. The system auto-captures frames whenever a good face is detected
        6. At the end it enrolls you automatically ✅
        """)

        col_name, col_dur, col_cam = st.columns([3, 2, 1])
        enroll_name = col_name.text_input(
            "Person Name", placeholder="e.g. Ahmed, Fatima…", key="enroll_name"
        )
        enroll_dur = col_dur.slider("Duration (seconds)", 5, 60, 20, key="enroll_dur")
        enroll_cam = col_cam.number_input("Camera", 0, 10, cfg["camera"]["index"], key="enroll_cam")

        # Session state flags
        if "enroll_running" not in st.session_state:
            st.session_state.enroll_running = False
        if "enroll_done" not in st.session_state:
            st.session_state.enroll_done = False

        col_start, col_cancel = st.columns(2)
        start_enroll  = col_start.button("🎥 Start Recording", type="primary",
                                          use_container_width=False if _st_version < (1, 25) else True,
                                          disabled=not enroll_name.strip())
        cancel_enroll = col_cancel.button("✖ Cancel", disabled=not st.session_state.enroll_running)

        if cancel_enroll:
            st.session_state.enroll_running = False
            st.session_state.enroll_done    = False

        if start_enroll and enroll_name.strip():
            st.session_state.enroll_running = True
            st.session_state.enroll_done    = False

        # ── Recording loop ───────────────────────────────────────────────────
        if st.session_state.enroll_running:
            st.markdown(
                '<span class="rec-dot"></span> **RECORDING** — look at the camera, move your head slowly',
                unsafe_allow_html=True,
            )

            prog_bar    = st.progress(0.0)
            status_txt  = st.empty()
            frame_ph    = st.empty()
            counter_ph  = st.empty()

            engine = load_engine(cfg["model"]["name"], cfg["model"]["use_gpu"])
            cap    = cv2.VideoCapture(int(enroll_cam))

            if not cap.isOpened():
                st.error(f"Cannot open camera {int(enroll_cam)}. Try a different index.")
                st.session_state.enroll_running = False
                st.stop()

            embeddings: list = []
            last_capture  = 0.0
            capture_every = 0.5          # capture at most 1 frame every 0.5 s
            t_start       = time.time()

            try:
                while True:
                    elapsed  = time.time() - t_start
                    left     = enroll_dur - elapsed
                    progress = min(elapsed / enroll_dur, 1.0)

                    if left <= 0 or not st.session_state.enroll_running:
                        break

                    ok, frame = cap.read()
                    if not ok:
                        st.error("Camera read error.")
                        break

                    faces = engine.analyze(frame)
                    good  = (len(faces) == 1 and faces[0].det_score >= 0.70)

                    # Auto-capture
                    if good and (time.time() - last_capture) >= capture_every:
                        embeddings.append(faces[0].embedding)
                        last_capture = time.time()

                    # Draw overlay
                    annotated = _draw_enroll_frame(frame, faces, len(embeddings), left)
                    frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    show_image(frame_ph, frame_rgb)

                    prog_bar.progress(progress)
                    status_txt.markdown(
                        f'<span class="cap-badge">📸 {len(embeddings)} frames captured</span>'
                        f'  &nbsp; ⏱ **{left:.1f} s** remaining',
                        unsafe_allow_html=True,
                    )

                    time.sleep(0.03)

            finally:
                cap.release()
                cv2.destroyAllWindows()

            # ── Enroll ──────────────────────────────────────────────────────
            frame_ph.empty()
            prog_bar.progress(1.0)
            st.session_state.enroll_running = False

            if len(embeddings) == 0:
                st.error("❌ No face detected during recording. Try again with better lighting.")
            else:
                g = load_gallery(cfg)
                g.add(enroll_name.strip(), embeddings)
                g.save(get_gallery_path(cfg))
                st.session_state.enroll_done = True
                st.success(
                    f"✅ **«{enroll_name.strip()}»** enrolled successfully "
                    f"with **{len(embeddings)}** captures!"
                )
                st.balloons()
                time.sleep(1.5)
                st.rerun()

    if st.session_state.get("enroll_done"):
        st.session_state.enroll_done = False

    st.markdown("---")

    # ── Gallery grid ─────────────────────────────────────────────────────────
    gallery = load_gallery(cfg)
    if len(gallery) == 0:
        st.info("👤 No users enrolled yet. Use the form above to add someone.")
    else:
        st.markdown(
            f'<p class="section-title">👤 {len(gallery)} enrolled person(s)</p>',
            unsafe_allow_html=True,
        )
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
                    if st.button("🗑️ Remove", key=f"del_{name}"):
                        g = load_gallery(cfg)
                        g.remove(name)
                        g.save(get_gallery_path(cfg))
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
    start_btn = bc1.button("▶  Start", type="primary")
    stop_btn  = bc2.button("⏹  Stop")

    if "live_on" not in st.session_state:
        st.session_state.live_on = False
    if start_btn:
        st.session_state.live_on = True
    if stop_btn:
        st.session_state.live_on = False

    if not st.session_state.live_on:
        frame_ph.info("👆 Press **▶ Start** to begin real-time recognition.")
        st.stop()

    # ── Build pipeline ────────────────────────────────────────────────────────
    cfg["camera"]["index"]          = int(cam_idx)
    cfg["matching"]["threshold"]    = float(threshold)
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

    # ── Live loop ─────────────────────────────────────────────────────────────
    try:
        while st.session_state.live_on:
            ok, frame = cap.read()
            if not ok:
                st.error("Camera read error.")
                break

            if n % int(proc_every) == 0:
                last_res = pipe.process(frame)

            annotated = draw_overlay(frame, last_res, pipe.door.is_open, fps)
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now
            n += 1

            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            show_image(frame_ph, frame_rgb)   # ← uses correct kwarg for Streamlit version

            fps_ph.metric("FPS", f"{fps:.1f}")
            door_ph.metric("Door", "🟢 OPEN" if pipe.door.is_open else "🔴 LOCKED")
            if last_res:
                name_ph.metric("Person", last_res.name or "Unknown")
                score_ph.metric("Score", f"{last_res.score:.3f}")
                if last_res.event:
                    cls = "granted" if last_res.event == "GRANTED" else "denied"
                    event_ph.markdown(
                        f'<span class="badge-{cls}">{last_res.event}</span>',
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

    # Filters
    fc1, fc2, fc3 = st.columns(3)
    event_filter = fc1.multiselect(
        "Event type", ["GRANTED", "DENIED"], default=["GRANTED", "DENIED"]
    )
    name_opts   = sorted(df["name"].dropna().unique().tolist())
    name_filter = fc2.multiselect("Person", name_opts)
    date_filter = fc3.text_input("Date (YYYY-MM-DD)", placeholder="e.g. 2026-11-04")

    filtered = df[df["event"].isin(event_filter)]
    if name_filter:
        filtered = filtered[filtered["name"].isin(name_filter)]
    if date_filter.strip():
        filtered = filtered[filtered["ts"].str.startswith(date_filter.strip())]

    m1, m2, m3 = st.columns(3)
    kpi(m1, len(filtered), "Total Events", "#58a6ff")
    kpi(m2, int((filtered["event"] == "GRANTED").sum()), "Granted", "#3fb950")
    kpi(m3, int((filtered["event"] == "DENIED").sum()), "Denied",  "#f85149")

    st.markdown("---")

    display = filtered.copy()
    display["score"] = display["score"].apply(lambda x: f"{float(x):.3f}")
    st.dataframe(display, hide_index=True, height=400)

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️  Export to CSV", csv, file_name="audit_log.csv", mime="text/csv")


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
        s_gpu = st.toggle(
            "Use GPU  (requires onnxruntime-gpu + CUDA)",
            value=bool(cfg["model"]["use_gpu"]),
        )
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
        s_cooldown = st.slider("Cooldown (s)", 1, 60, cfg["decision"]["cooldown_s"])

        st.markdown("### 🚪 Door")
        s_door_sec = st.slider("Door open duration (s)", 1, 30, cfg["door"]["open_seconds"])

        st.markdown("### ⚡ Runtime")
        s_proc_every  = st.slider("Process every N frames", 1, 10, cfg["runtime"]["process_every"])
        s_n_templates = st.slider(
            "Enrollment templates per person", 1, 30, cfg["enroll"]["n_templates"]
        )

        saved = st.form_submit_button("💾  Save Settings", type="primary")

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
