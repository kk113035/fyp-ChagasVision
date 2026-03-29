import streamlit as st
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="enable_nested_tensor")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import json
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Sri Lanka timezone (UTC+5:30)
SL_TZ = timezone(timedelta(hours=5, minutes=30))

def now_sl():
    """Return current time in Sri Lanka timezone."""
    return datetime.now(SL_TZ)

from config import (
    MODELS_DIR, LEAD_NAMES, SAMPLING_RATE, CHAGAS_PATTERNS, ModelConfig,
)
from model import build_model
from preprocessing import ECGPreprocessor
from xai import ComprehensiveXAI

# CONFIG

st.set_page_config(page_title="ChagasVision", page_icon="🫀", layout="wide")

for key, val in [("authenticated", False), ("page", "home"),
                 ("username", ""), ("full_name", ""), ("role", ""),
                 ("show_disclaimer", True)]:
    if key not in st.session_state:
        st.session_state[key] = val

# CSS


st.markdown("""
    <style>
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
section[data-testid="stSidebar"] {
    background: #0a0f1a !important; color: #e2e8f0; font-family: 'DM Sans', sans-serif;
}
[data-testid="stSidebar"] > div { background: #0f1629 !important; }
h1, h2, h3, h4, h5 { color: #f1f5f9 !important; }
p, li, span { color: #cbd5e1; }
label { color: #94a3b8 !important; }

input, textarea, select, [data-baseweb="select"],
.stTextInput > div > div, .stNumberInput > div > div,
.stSelectbox > div > div {
    background: #1e293b !important; color: #e2e8f0 !important; border-color: #334155 !important;
}

.stButton > button { border-radius: 8px !important; font-weight: 600 !important; transition: all 0.2s !important; }
.stButton > button[kind="primary"] { background: linear-gradient(135deg, #2563eb, #3b82f6) !important; border: none !important; }
.stButton > button[kind="primary"]:hover { background: linear-gradient(135deg, #1d4ed8, #2563eb) !important; transform: translateY(-1px) !important; box-shadow: 0 4px 15px rgba(37,99,235,0.3) !important; }

.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1a2744 40%, #0f172a 100%);
    border: 1px solid #1e3a5f; border-radius: 16px; padding: 2.8rem 2.2rem;
    margin-bottom: 1.5rem; position: relative; overflow: hidden;
}
.hero::before { content: ''; position: absolute; top: -50%; right: -20%; width: 500px; height: 500px; background: radial-gradient(circle, rgba(59,130,246,0.12) 0%, transparent 70%); border-radius: 50%; }
.hero::after { content: ''; position: absolute; bottom: -40%; left: -10%; width: 400px; height: 400px; background: radial-gradient(circle, rgba(239,68,68,0.06) 0%, transparent 70%); border-radius: 50%; }
.hero h1 { font-size: 2.4rem; font-weight: 800; margin: 0; letter-spacing: -0.5px; color: #f1f5f9 !important; position: relative; z-index: 1; }
.hero .acc { color: #3b82f6; }
.hero p { color: #94a3b8; margin: 0.6rem 0 0 0; max-width: 620px; font-size: 1.05rem; position: relative; z-index: 1; line-height: 1.6; }

.card { background: #111827; border: 1px solid #1e3a5f; border-radius: 12px; padding: 1.4rem; margin: 0.5rem 0; }
.res-pos { background: linear-gradient(135deg, #1c0a0a, #2d1111); border-left: 5px solid #dc2626; padding: 1.3rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 0 20px rgba(220,38,38,0.08); }
.res-neg { background: linear-gradient(135deg, #051a0e, #0b2e1a); border-left: 5px solid #16a34a; padding: 1.3rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 0 20px rgba(22,163,74,0.08); }
.res-bor { background: linear-gradient(135deg, #1a1500, #332b00); border-left: 5px solid #ca8a04; padding: 1.3rem; border-radius: 10px; margin: 1rem 0; }
.shdr { font-size: 1rem; font-weight: 700; color: #e2e8f0 !important; border-bottom: 1px solid #1e3a5f; padding-bottom: 0.4rem; margin: 1.5rem 0 0.8rem 0; }

.upbox { background: #111827; border: 2px dashed #1e3a5f; border-radius: 14px; padding: 2.5rem; text-align: center; margin: 0.5rem 0; }
.upbox .ic { font-size: 3rem; margin-bottom: 0.5rem; }
.upbox h4 { color: #e2e8f0 !important; margin: 0.3rem 0; }
.upbox p { color: #64748b; font-size: 0.9rem; }
.instr-box { background: #111827; border: 1px solid #1e3a5f; border-radius: 10px; padding: 1rem 1.2rem; font-size: 0.88rem; color: #94a3b8; }
.instr-box ol { padding-left: 1.2rem; margin: 0.5rem 0; }
.instr-box li { margin: 0.4rem 0; }
.instr-box b { color: #e2e8f0; }
.disc { background: #0c1a33; border: 1px solid #1e40af; border-radius: 8px; padding: 0.9rem; font-size: 0.82rem; color: #93c5fd; margin-top: 1rem; margin-bottom: 1rem; }

.stTabs [data-baseweb="tab-list"] { gap: 4px; background: transparent; }
.stTabs [data-baseweb="tab"] { background: #1e293b !important; border-radius: 8px !important; color: #94a3b8 !important; padding: 0.5rem 1rem !important; }
.stTabs [aria-selected="true"] { background: #2563eb !important; color: #ffffff !important; }
.streamlit-expanderHeader { background: #111827 !important; color: #e2e8f0 !important; }
.streamlit-expanderContent { background: #0f172a !important; }
[data-testid="stMetricValue"] { color: #e2e8f0 !important; }
[data-testid="stMetricLabel"] { color: #64748b !important; }

.stButton > button:focus-visible, input:focus-visible, select:focus-visible { outline: 2px solid #3b82f6 !important; outline-offset: 2px !important; }

@media (max-width: 768px) {
    .hero { padding: 1.5rem 1rem; } .hero h1 { font-size: 1.6rem; }
}

#MainMenu, footer { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="stHeader"] { background: #0a0f1a !important; }
</style>
""", unsafe_allow_html=True)


# DATABASE

DB_PATH = Path("chagasvision.db")

def init_db():
    conn = sqlite3.connect(str(DB_PATH)); c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
    if c.fetchone():
        c.execute("PRAGMA table_info(users)")
        cols = [row[1] for row in c.fetchall()]
        if "doctor_id" not in cols:
            c.execute("DROP TABLE users")
            c.execute("DROP TABLE IF EXISTS scans")
            c.execute("DROP TABLE IF EXISTS login_log")
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE, password_hash TEXT, full_name TEXT,
        doctor_id TEXT DEFAULT '', role TEXT DEFAULT 'clinician',
        created_at TEXT DEFAULT '')""")
    c.execute("""CREATE TABLE IF NOT EXISTS scans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user TEXT, patient_id TEXT, age INT, sex TEXT,
        probability REAL, prediction TEXT, threshold REAL,
        confidence TEXT, agreement REAL, top_leads TEXT,
        scan_date TEXT DEFAULT '')""")
    c.execute("""CREATE TABLE IF NOT EXISTS login_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT, role TEXT, action TEXT,
        ip_info TEXT DEFAULT '', timestamp TEXT DEFAULT '')""")
    c.execute("SELECT COUNT(*) FROM users")
    if c.fetchone()[0] == 0:
        for u, p, n, did, r in [
            ("adminTest1", "admin123@", "System Administrator", "EMP-001", "admin"),
            ("clinicianTest1", "clinic123@", "Dr. Ajith Peris", "DOC-001", "clinician"),
        ]:
            c.execute("INSERT INTO users (username,password_hash,full_name,doctor_id,role,created_at) VALUES (?,?,?,?,?,?)",
                      (u, hashlib.sha256(p.encode()).hexdigest(), n, did, r, now_sl().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit(); conn.close()

def verify_user(u, p):
    conn = sqlite3.connect(str(DB_PATH)); c = conn.cursor()
    c.execute("SELECT full_name, role, doctor_id FROM users WHERE username=? AND password_hash=?",
              (u, hashlib.sha256(p.encode()).hexdigest()))
    r = c.fetchone(); conn.close(); return r

def register_user(u, p, n, doctor_id, role):
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute("INSERT INTO users (username,password_hash,full_name,doctor_id,role,created_at) VALUES (?,?,?,?,?,?)",
                     (u, hashlib.sha256(p.encode()).hexdigest(), n, doctor_id, role, now_sl().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit(); conn.close(); return True
    except: conn.close(); return False

def log_login(username, role, action):
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("INSERT INTO login_log (username, role, action, timestamp) VALUES (?,?,?,?)",
                 (username, role, action, now_sl().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit(); conn.close()

def get_login_log(limit=100):
    conn = sqlite3.connect(str(DB_PATH)); c = conn.cursor()
    c.execute("SELECT * FROM login_log ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = c.fetchall(); cols = [d[0] for d in c.description]; conn.close()
    return [dict(zip(cols, r)) for r in rows]

def get_all_users():
    conn = sqlite3.connect(str(DB_PATH)); c = conn.cursor()
    c.execute("SELECT id, username, full_name, doctor_id, role, created_at FROM users ORDER BY created_at DESC")
    rows = c.fetchall(); cols = [d[0] for d in c.description]; conn.close()
    return [dict(zip(cols, r)) for r in rows]

def update_user_full(user_id, username, full_name, doctor_id, role, new_password=None):
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute("UPDATE users SET username=?, full_name=?, doctor_id=?, role=? WHERE id=?",
                     (username, full_name, doctor_id, role, user_id))
        if new_password:
            conn.execute("UPDATE users SET password_hash=? WHERE id=?",
                         (hashlib.sha256(new_password.encode()).hexdigest(), user_id))
        conn.commit(); conn.close(); return True
    except: conn.close(); return False

def delete_user(user_id):
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit(); conn.close()

def save_scan(**kw):
    kw["scan_date"] = now_sl().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("INSERT INTO scans (user,patient_id,age,sex,probability,prediction,threshold,confidence,agreement,top_leads,scan_date) VALUES (:user,:patient_id,:age,:sex,:prob,:prediction,:threshold,:confidence,:agreement,:top_leads,:scan_date)", kw)
    conn.commit(); conn.close()

def get_scans(user=None, limit=50):
    conn = sqlite3.connect(str(DB_PATH)); c = conn.cursor()
    if user: c.execute("SELECT * FROM scans WHERE user=? ORDER BY scan_date DESC LIMIT ?", (user, limit))
    else: c.execute("SELECT * FROM scans ORDER BY scan_date DESC LIMIT ?", (limit,))
    rows = c.fetchall(); cols = [d[0] for d in c.description]; conn.close()
    return [dict(zip(cols, r)) for r in rows]


# MODEL

@st.cache_resource
def load_ensemble():
    for edir in [MODELS_DIR/"ensemble_v2_improved", MODELS_DIR/"ensemble"]:
        if (edir/"ensemble_config.json").exists():
            cfg = json.load(open(edir/"ensemble_config.json")); break
    else: return None, None, 0.5, "error"
    models = []
    for i in range(1, cfg["n_folds"]+1):
        p = edir/f"fold_{i}"/"best_model.pth"
        if not p.exists(): continue
        try:
            m = build_model(dropout_override=0.0)
            m.load_state_dict(torch.load(p, map_location="cpu", weights_only=False)["model_state_dict"])
            m.eval(); models.append(m)
        except: pass
    if not models: return None, None, 0.5, "error"
    res = json.load(open(edir/"ensemble_results.json")) if (edir/"ensemble_results.json").exists() else {}
    return models, res, cfg.get("optimal_threshold", 0.5), "ok"


# NAVBAR

def navbar():
    auth = st.session_state["authenticated"]
    role = st.session_state.get("role", "")
    if auth and role == "clinician":
        cols = st.columns([3, 1, 1, 1, 2])
        with cols[0]: st.markdown("### 🫀 ChagasVision - Clinician")
        with cols[1]:
            if st.button("Home", use_container_width=True): st.session_state["page"]="home"; st.rerun()
        with cols[2]:
            if st.button("Scanner", use_container_width=True): st.session_state["page"]="scanner"; st.rerun()
        with cols[3]:
            if st.button("History", use_container_width=True): st.session_state["page"]="history"; st.rerun()
        with cols[4]:
            if st.button(f"Logout ({st.session_state['full_name']})", use_container_width=True):
                log_login(st.session_state["username"], "clinician", "logout")
                for k in ["authenticated","username","full_name","role"]:
                    st.session_state[k] = "" if k != "authenticated" else False
                st.session_state["page"]="home"; st.rerun()
    elif auth and role == "admin":
        cols = st.columns([3, 1, 1, 1, 1, 2])
        with cols[0]: st.markdown("### 🫀 ChagasVision — Administrator")
        with cols[1]:
            if st.button("Home", use_container_width=True): st.session_state["page"]="home"; st.rerun()
        with cols[2]:
            if st.button("Users", use_container_width=True): st.session_state["page"]="manage_users"; st.rerun()
        with cols[3]:
            if st.button("Login Log", use_container_width=True): st.session_state["page"]="login_log"; st.rerun()
        with cols[4]:
            if st.button("All Scans", use_container_width=True): st.session_state["page"]="all_scans"; st.rerun()
        with cols[5]:
            if st.button(f"Logout ({st.session_state['full_name']})", use_container_width=True):
                log_login(st.session_state["username"], "admin", "logout")
                for k in ["authenticated","username","full_name","role"]:
                    st.session_state[k] = "" if k != "authenticated" else False
                st.session_state["page"]="home"; st.rerun()
    else:
        cols = st.columns([3, 1, 1])
        with cols[0]: st.markdown("### 🫀 ChagasVision")
        with cols[1]:
            if st.button("Home", use_container_width=True): st.session_state["page"]="home"; st.rerun()
        with cols[2]:
            if st.button("Login", use_container_width=True): st.session_state["page"]="login"; st.rerun()
    st.markdown("---")


# PLOTS

BG = "#0a0f1a"; BG2 = "#111827"; FG = "#e2e8f0"; GR = "#1e3a5f"; TX2 = "#64748b"

def plot_probability_gauge(prob, threshold):
    fig, ax = plt.subplots(figsize=(10, 1.5), facecolor=BG2); ax.set_facecolor(BG2)
    ax.imshow(np.linspace(0,1,300).reshape(1,-1), aspect="auto", cmap="RdYlGn_r", extent=[0,100,0,1], alpha=0.15)
    c = "#dc2626" if prob >= threshold else "#10b981"
    ax.axvline(prob*100, color=c, lw=6, solid_capstyle="round")
    ax.axvline(threshold*100, color=FG, lw=1.5, ls="--", alpha=0.4)
    ax.text(prob*100, 1.2, f"{prob*100:.1f}%", ha="center", fontsize=13, fontweight="bold", color=c)
    ax.text(threshold*100, -0.3, f"Threshold {threshold*100:.0f}%", ha="center", fontsize=8, color=TX2)
    ax.set_xlim(0,100); ax.set_ylim(0,1); ax.set_yticks([]); ax.set_xticks([])
    for s in ax.spines.values(): s.set_visible(False)
    plt.tight_layout(); return fig

def plot_model_agreement(preds, threshold):
    fig, ax = plt.subplots(figsize=(10, 3.5), facecolor=BG2); ax.set_facecolor(BG2)
    x = np.arange(len(preds)); colors = ["#dc2626" if p >= threshold else "#10b981" for p in preds]
    bars = ax.bar(x, preds, color=colors, alpha=0.9, width=0.5, edgecolor=BG2, linewidth=2)
    ax.axhline(threshold, color=FG, ls="--", lw=1, alpha=0.3)
    ax.axhline(np.mean(preds), color="#3b82f6", lw=2, label=f"Mean: {np.mean(preds):.3f}")
    ax.set_xticks(x); ax.set_xticklabels([f"M{i+1}" for i in range(len(preds))], fontweight="600", color=FG)
    ax.set_ylim(0, 1); ax.set_ylabel("Probability", color=FG, fontweight="600"); ax.tick_params(colors=FG)
    for b, p in zip(bars, preds):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.03, f"{p:.3f}", ha="center", fontsize=10, fontweight="bold", color=FG)
    ax.legend(fontsize=9, facecolor=BG2, edgecolor=GR, labelcolor=FG)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    for sp in ["bottom","left"]: ax.spines[sp].set_color(GR)
    plt.tight_layout(); return fig

def plot_lead_importance(lead_imp):
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG2); ax.set_facecolor(BG2)
    si = sorted(lead_imp.items(), key=lambda x: x[1], reverse=True)
    leads = [x[0] for x in si]; vals = [x[1]*100 for x in si]
    chl = {"V1","V2","V6","I","aVL","II"}
    colors = ["#dc2626" if l in chl else "#3b82f6" for l in leads]
    ax.barh(leads, vals, color=colors, alpha=0.85, height=0.6)
    ax.set_xlabel("Importance (%)", color=FG, fontweight="600"); ax.invert_yaxis(); ax.tick_params(colors=FG)
    for i, v in enumerate(vals):
        ax.text(v+0.3, i, f"{v:.1f}%", va="center", fontsize=9, fontweight="600", color=FG)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#dc2626",label="Chagas-relevant"),Patch(color="#3b82f6",label="Other")],
              fontsize=9, loc="lower right", facecolor=BG2, edgecolor=GR, labelcolor=FG)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    for sp in ["bottom","left"]: ax.spines[sp].set_color(GR)
    plt.tight_layout(); return fig

def plot_ecg_gradcam(signal, lead_imp, grad_cam=None):
    """IMPROVED Grad-CAM ECG Overlay. Uses smooth colour-mapped highlighting
    instead of pixel-by-pixel axvspan. Each lead gets attention intensity
    proportional to its XAI importance * Grad-CAM temporal activation."""
    fig = plt.figure(figsize=(16, 14), facecolor=BG)
    t = np.arange(signal.shape[1]) / SAMPLING_RATE

    # Upsample Grad-CAM to signal length for smooth overlay
    cam_up = None
    if grad_cam is not None and len(grad_cam) > 0:
        cam_up = np.interp(np.linspace(0, 1, signal.shape[1]),
                           np.linspace(0, 1, len(grad_cam)), grad_cam)
        cam_up = (cam_up - cam_up.min()) / (cam_up.max() - cam_up.min() + 1e-8)

    mx = max(lead_imp.values()) if lead_imp else 1
    mn = min(lead_imp.values()) if lead_imp else 0
    chl = {"V1", "V2", "V6", "I", "aVL", "II"}

    for i, name in enumerate(LEAD_NAMES):
        ax = fig.add_subplot(6, 2, i + 1); ax.set_facecolor(BG2)
        ni = (lead_imp.get(name, 0) - mn) / (mx - mn + 1e-8)  # normalised importance

        # ECG line colour: red-ish for important leads, grey for less important
        lc = plt.cm.Reds(ni * 0.6 + 0.2) if ni > 0.5 else "#64748b"
        ax.plot(t, signal[i], color=lc, lw=0.9, zorder=2)

        # Smooth Grad-CAM highlighting using fill_between with alpha
        if cam_up is not None and ni > 0.2:
            # Create smooth attention band: signal ± scaled by cam attention
            sig_range = signal[i].max() - signal[i].min()
            if sig_range > 0:
                alpha_vals = cam_up * ni * 0.5  # scale by lead importance
                # Use fill_between with per-point alpha approximation
                # Split into high/medium/low attention segments
                for thresh_val, alpha_base, color in [(0.6, 0.35, "#dc2626"), (0.3, 0.15, "#f97316")]:
                    mask = cam_up > thresh_val
                    if mask.any():
                        ax.fill_between(t, signal[i].min(), signal[i].max(),
                                        where=mask, alpha=alpha_base * ni,
                                        color=color, lw=0, zorder=1)

        # Lead label
        lbl_c = "#dc2626" if name in chl else FG
        ax.set_ylabel(name, fontsize=10, fontweight="bold", color=lbl_c)
        ax.text(0.98, 0.9, f"{lead_imp.get(name, 0) * 100:.1f}%", transform=ax.transAxes,
                fontsize=8, ha="right", va="top",
                color="#dc2626" if ni > 0.5 else TX2,
                fontweight="bold" if ni > 0.5 else "normal")
        ax.grid(True, alpha=0.08, color=GR); ax.set_xlim(0, t[-1])
        ax.tick_params(labelsize=7, colors=FG)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        for sp in ["bottom", "left"]: ax.spines[sp].set_color(GR)
        if i >= 10: ax.set_xlabel("Time (s)", fontsize=9, color=FG)

    fig.suptitle("12-Lead ECG — Grad-CAM Attention Overlay", fontsize=14, fontweight="800", color=FG, y=0.98)
    fig.text(0.99, 0.005, "Red/Orange = AI attention regions  •  Red labels = Chagas-relevant leads", fontsize=8, ha="right", color=TX2)
    plt.tight_layout(rect=[0, 0.015, 1, 0.97]); return fig

def plot_gradcam_detail(cam, sig):
    if cam is None or len(cam) == 0: return None
    t = np.arange(len(sig)) / SAMPLING_RATE
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 4.5), sharex=True, gridspec_kw={"height_ratios": [1, 3]}, facecolor=BG2)
    a1.imshow(cam.reshape(1, -1), aspect="auto", cmap="hot", interpolation="bilinear", extent=[0, t[-1], 0, 1])
    a1.set_yticks([]); a1.set_facecolor(BG2)
    a1.set_title("Grad-CAM Temporal Attention (conv2 layer)", fontsize=11, fontweight="700", color=FG)
    cn = np.interp(np.linspace(0, 1, len(sig)), np.linspace(0, 1, len(cam)), cam)
    cn = (cn - cn.min()) / (cn.max() - cn.min() + 1e-8)
    a2.set_facecolor(BG2); a2.plot(t, sig, color="#94a3b8", lw=1)
    a2.fill_between(t, sig.min(), sig, alpha=cn * 0.4, color="#dc2626")
    a2.set_xlabel("Time (s)", color=FG); a2.set_ylabel("Lead II", color=FG); a2.set_xlim(0, t[-1]); a2.tick_params(colors=FG)
    for ax in [a1, a2]:
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        for sp in ["bottom", "left"]: ax.spines[sp].set_color(GR)
    plt.tight_layout(); return fig

def plot_temporal_occlusion(temp_occ, stride=32):
    if temp_occ is None or len(temp_occ) == 0: return None
    fig, ax = plt.subplots(figsize=(14, 2.5), facecolor=BG2); ax.set_facecolor(BG2)
    t = np.arange(len(temp_occ)) * stride / SAMPLING_RATE
    ax.fill_between(t, 0, temp_occ, color="#3b82f6", alpha=0.6); ax.plot(t, temp_occ, color="#60a5fa", lw=1.5)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(temp_occ, height=np.percentile(temp_occ, 75), distance=3)
    if len(peaks) > 0:
        ax.scatter(t[peaks], temp_occ[peaks], color="#ef4444", s=40, zorder=5, label="Attention peaks")
        ax.legend(fontsize=8, facecolor=BG2, edgecolor=GR, labelcolor=FG)
    ax.set_xlabel("Time (s)", color=FG, fontweight="600"); ax.set_ylabel("Importance", color=FG); ax.tick_params(colors=FG)
    ax.set_title("Temporal Occlusion — Which time regions matter most?", fontsize=11, fontweight="700", color=FG)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    for sp in ["bottom","left"]: ax.spines[sp].set_color(GR)
    plt.tight_layout(); return fig


# REPORT

def build_report(result, age, sex, threshold, pid=""):
    lines = [
        "      CHAGASVISION — CLINICAL ANALYSIS REPORT",
        "=" * 53,
        f"Date: {now_sl():%Y-%m-%d %H:%M}  Patient: {pid or 'N/A'}  Age: {age}  Sex: {sex}",
        f"\nRESULT: {result['prediction']}",
        f"Probability: {result['probability']*100:.1f}%  Confidence: {result['confidence']} ({result.get('confidence_score',0):.2f})",
        f"  -> {result.get('confidence_explanation','')}",
        f"Agreement: {result['model_consistency']*100:.0f}%  Threshold: {threshold}",
        f"XAI consistency: tau = {result.get('method_consistency',0):.2f}",
        f"Methods: {', '.join(result.get('method_types',[]))}",
        "\nTOP LEADS:"]
    for l, v in result["sorted_leads"][:6]:
        lines.append(f"  {l:>4}  {'#'*int(v*40)} {v*100:.1f}%")
    if result.get("detected_patterns"):
        lines.append("\nDETECTED PATTERNS:")
        for p in result["detected_patterns"]:
            line = f"  {p['name']} ({p['strength']*100:.1f}%) — {p['relevance']}"
            if p.get("temporal_region"): line += f" [{p['temporal_region']}]"
            lines.append(line)
    lines += ["\nDISCLAIMER: Clinical decision support only. Confirm with serology.",
              f"\n© ChagasVision {now_sl().year}"]
    return "\n".join(lines)



# PAGES


def page_home():
    st.markdown("""
    <div class="hero">
        <h1>Detect Chagas Disease<br>from <span class="acc">12-Lead ECG</span></h1>
        <p>AI-powered clinical screening using a hybrid Multi-Scale CNN + Transformer.
           Trained on ECG recordings from 3 datasets with Explainable AI for transparent decisions.</p>
    </div>""", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### About Chagas Disease")
        st.markdown("Chagas disease is caused by *Trypanosoma cruzi*, affecting **6.5 million** people. "
                     "Chronic infection causes cardiomyopathy: **RBBB** (40-50%), **LAFB** (30-40%), **AV block** (15-25%).")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### How ChagasVision Works")
        st.markdown("**1.** Upload 12-lead ECG (HDF5)\n\n**2.** 5-model ensemble analyses the waveform\n\n"
                     "**3.** 4 XAI methods explain the prediction\n\n**4.** Clinical report with recommendations")
        st.markdown('</div>', unsafe_allow_html=True)
    if not st.session_state["authenticated"]:
        st.markdown("---")
        _, c, _ = st.columns([1,2,1])
        with c:
            if st.button("Login to Access the System", type="primary", use_container_width=True):
                st.session_state["page"] = "login"; st.rerun()
    st.markdown("DISCLAIMER: ChagasVision is a clinical decision support tool for research purposes only.", unsafe_allow_html=True)
    st.markdown(f"© ChagasVision {now_sl().year}", unsafe_allow_html=True)

def page_login():
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("##### Sign in to your account")
        st.caption("Authorised personnel only")
        with st.form("login"):
            u = st.text_input("Username", placeholder="Enter your username")
            p = st.text_input("Password", type="password", placeholder="Enter your password")
            if st.form_submit_button("Sign In", use_container_width=True, type="primary"):
                if u and p:
                    r = verify_user(u, p)
                    if r:
                        if r[1] not in ("clinician", "admin"):
                            st.error("Unauthorised role.")
                        else:
                            log_login(u, r[1], "login_success")
                            landing = "scanner" if r[1] == "clinician" else "manage_users"
                            st.session_state.update({"authenticated":True,"username":u,"full_name":r[0],"role":r[1],"page":landing})
                            st.rerun()
                    else:
                        log_login(u, "unknown", "login_failed")
                        st.error("Invalid credentials.")
                else: st.warning("Please enter both fields.")
        st.markdown('<div class="disc">New accounts can only be created by a system administrator.</div>', unsafe_allow_html=True)


def page_scanner(models, results, default_threshold):
    preprocessor = ECGPreprocessor(); xai_engine = ComprehensiveXAI(models)

    st.markdown("#### ECG Analysis Console")
    st.caption(f"Clinician: **{st.session_state['full_name']}** • {now_sl():%Y-%m-%d}")

    with st.expander("How to use the scanner", expanded=False):
        st.markdown("""<div class="instr-box"><ol>
            <li><b>Prepare</b> — HDF5 file with 'tracings' dataset (12-lead ECG)</li>
            <li><b>Upload</b> — drag/drop or click below</li>
            <li><b>Enter info</b> — age, sex, optional Patient ID</li>
            <li><b>Analyse</b> — 5-model ensemble + 4 XAI methods</li>
            <li><b>Review</b> — probability, attention overlay, patterns</li>
            <li><b>Download</b> — clinical report for records</li>
        </ol></div>""", unsafe_allow_html=True)

    with st.expander("Threshold Settings", expanded=False):
        threshold = st.slider("Adjust Threshold", 0.30, 0.70, float(default_threshold), 0.01, label_visibility="collapsed")

    c1, c2 = st.columns([2, 1])
    with c1:
        uploaded = st.file_uploader("Upload 12-Lead ECG", type=["h5","hdf5"], label_visibility="collapsed")

    # Reset fields on file change
    current_file = uploaded.name if uploaded else None
    if "last_uploaded_file" not in st.session_state:
        st.session_state["last_uploaded_file"] = None
    if current_file != st.session_state["last_uploaded_file"]:
        st.session_state["last_uploaded_file"] = current_file
        st.session_state["scan_counter"] = st.session_state.get("scan_counter", 0) + 1
    sc = st.session_state.get("scan_counter", 0)

    with c2:
        pid = st.text_input("Patient ID", "", placeholder="Enter Patient ID", key=f"pid_{sc}")
        a1, a2 = st.columns(2)
        age = a1.number_input("Age", min_value=1, max_value=120, value=None, placeholder="Enter age", key=f"age_{sc}")
        sex_s = a2.selectbox("Sex", ["Select", "Female", "Male"], key=f"sex_{sc}")

    if uploaded and st.button("Analyse ECG", type="primary", use_container_width=True):
        if age is None:
            st.error("Please enter the patient's age.")
        elif sex_s == "Select":
            st.error("Please select the patient's sex.")
        else:
            sex_v = 1 if sex_s == "Male" else 0
            try:
                with h5py.File(uploaded, "r") as f:
                    raw = np.array(f["tracings"], dtype=np.float32)
                    if raw.ndim == 3: raw = raw[0]
                    if raw.shape[0] > raw.shape[1]: raw = raw.T
                processed = preprocessor.process(raw)
                with st.spinner("Running 5-model ensemble + 4 XAI methods..."):
                    result = xai_engine.explain(processed, age, sex_v, threshold)
                prob = result["probability"]

                # Result banner
                if prob >= threshold:
                    st.markdown(f'<div class="res-pos"><h4 style="color:#ef4444;margin:0">CHAGAS POSITIVE</h4><p style="color:#fca5a5">Prob: <b>{prob*100:.1f}%</b> • Confidence: <b>{result["confidence"]}</b> • Agreement: <b>{result["model_consistency"]*100:.0f}%</b></p></div>', unsafe_allow_html=True)
                elif prob >= threshold - 0.1:
                    st.markdown(f'<div class="res-bor"><h4 style="color:#eab308;margin:0">BORDERLINE</h4><p style="color:#fde68a">Prob: <b>{prob*100:.1f}%</b> • Confidence: <b>{result["confidence"]}</b> • Agreement: <b>{result["model_consistency"]*100:.0f}%</b></p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="res-neg"><h4 style="color:#22c55e;margin:0">CHAGAS NEGATIVE</h4><p style="color:#86efac">Prob: <b>{prob*100:.1f}%</b> • Confidence: <b>{result["confidence"]}</b> • Agreement: <b>{result["model_consistency"]*100:.0f}%</b></p></div>', unsafe_allow_html=True)

                # Metrics
                m1,m2,m3,m4 = st.columns(4)
                m1.metric("Probability", f"{prob*100:.1f}%")
                m2.metric("Agreement", f"{result['model_consistency']*100:.0f}%")
                m3.metric("XAI Consistency", f"{result['method_consistency']:.2f}")
                m4.metric("Confidence", result["confidence"])

                # 1. Probability Gauge
                st.markdown('<p class="shdr">Probability Gauge</p>', unsafe_allow_html=True)
                st.pyplot(plot_probability_gauge(prob, threshold)); plt.close()

                # 2. Ensemble Agreement — each model's prediction
                st.markdown('<p class="shdr">Ensemble Agreement</p>', unsafe_allow_html=True)
                st.pyplot(plot_model_agreement(result["predictions"], threshold)); plt.close()

                # 3. Lead Importance — aggregated from all 5 XAI methods
                st.markdown('<p class="shdr">Lead Importance </p>', unsafe_allow_html=True)
                st.pyplot(plot_lead_importance(result["lead_importance"])); plt.close()
                st.caption(f"{result.get('n_methods','?')} methods: {', '.join(result.get('method_types',[]))}")

                # 4. Grad-CAM ECG Overlay — WHERE the model looked
                st.markdown('<p class="shdr">Grad-CAM ECG Overlay</p>', unsafe_allow_html=True)
                st.pyplot(plot_ecg_gradcam(processed, result["lead_importance"], result.get("grad_cam"))); plt.close()

                
                # 5. Temporal Occlusion — which TIME regions matter
                if result.get("temporal_occlusion") is not None:
                    st.markdown('<p class="shdr">Temporal Occlusion</p>', unsafe_allow_html=True)
                    fig = plot_temporal_occlusion(result["temporal_occlusion"])
                    if fig: st.pyplot(fig); plt.close()


                # Disclaimer + save + download
                st.markdown("DISCLAIMER: ChagasVision is a clinical decision support tool for research purposes only.", unsafe_allow_html=True)
                st.markdown(f"© ChagasVision {now_sl().year}", unsafe_allow_html=True)
                top_str = ", ".join(f"{l}({v*100:.1f}%)" for l, v in result["sorted_leads"][:5])
                save_scan(user=st.session_state["username"], patient_id=pid or f"anon_{now_sl():%H%M%S}",
                          age=age, sex=sex_s, prob=round(prob,4), prediction=result["prediction"],
                          threshold=threshold, confidence=result["confidence"],
                          agreement=round(result["model_consistency"],3), top_leads=top_str)
                st.download_button("Download Report", build_report(result, age, sex_s, threshold, pid),
                                   f"chagas_{pid or 'report'}_{now_sl():%Y%m%d_%H%M%S}.txt", use_container_width=True)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.info("Ensure HDF5 contains 'tracings' dataset with 12-lead ECG data.")
    elif not uploaded:
        st.markdown('<div class="upbox"><div class="ic">📤</div><h4>Upload an ECG to begin</h4>'
                    '<p>Drag and drop HDF5 file, enter patient details, click Analyse</p></div>', unsafe_allow_html=True)


def page_history():
    st.markdown("#### My Scan History")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1: search_query = st.text_input("Search by Patient ID or Date", "", placeholder="Type to filter...")
    with col2: status_filter = st.selectbox("Filter by Status", ["All", "Positive", "Negative"])
    with col3: sort_order = st.selectbox("Sort by", ["Newest First", "Oldest First"])
    scans = get_scans(user=st.session_state["username"])
    if not scans: st.info("No scans yet."); return
    filtered = scans[:]
    if status_filter != "All":
        filtered = [s for s in filtered if status_filter.upper() in s.get("prediction","").upper()]
    if search_query:
        q = search_query.lower()
        filtered = [s for s in filtered if q in str(s.get('patient_id','')).lower() or q in str(s.get('scan_date','')).lower()]
    if sort_order == "Oldest First": filtered.reverse()
    pos = sum(1 for s in filtered if "POS" in s.get("prediction","").upper())
    c1,c2,c3 = st.columns(3)
    c1.metric("Total", len(filtered)); c2.metric("Positive", pos); c3.metric("Negative", len(filtered)-pos)
    st.markdown("---")
    for s in filtered:
        ip = "POS" in s.get("prediction","").upper()
        ps = f"{s['probability']*100:.1f}%" if s.get("probability") else "?"
        with st.expander(f"{'🔴' if ip else '🟢'} {s['scan_date'][:16]} • {s.get('patient_id','?')} • {ps}"):
            a,b,c,d = st.columns(4)
            a.markdown(f"**Age:** {s.get('age','')}"); b.markdown(f"**Sex:** {s.get('sex','')}")
            c.markdown(f"**Confidence:** {s.get('confidence','')}"); d.markdown(f"**Agreement:** {s.get('agreement','')}")
            st.markdown(f"**Top leads:** {s.get('top_leads','')}"); st.markdown(f"**Result:** {s.get('prediction','')}")


# ADMIN PAGES


def page_manage_users():
    st.markdown("#### User Management")
    search_users = st.text_input("Search users", "", placeholder="Type to filter...")
    users = get_all_users()
    if search_users:
        q = search_users.lower()
        users = [u for u in users if q in str(u.get('username','')).lower() or q in str(u.get('full_name','')).lower()]
    st.metric("Total Users", len(users))
    with st.expander("Add new user", expanded=False):
        with st.form("add_user", clear_on_submit=True):
            role_add = st.selectbox("Role", ["Clinician", "Administrator"], key="add_role")
            nu = st.text_input("Username", key="add_u"); nn = st.text_input("Full Name", key="add_n")
            did = st.text_input("Doctor/Employee ID", key="add_did")
            pw = st.text_input("Password", type="password", key="add_pw")
            pw2 = st.text_input("Confirm Password", type="password", key="add_pw2")
            if st.form_submit_button("Register User", use_container_width=True, type="primary"):
                role_val = "clinician" if role_add == "Clinician" else "admin"
                if not all([nu, nn, pw, pw2]): st.error("All fields required.")
                elif not did: st.error("ID required.")
                elif pw != pw2: st.error("Passwords do not match.")
                elif len(pw) < 6: st.error("Password must be 6+ characters.")
                elif register_user(nu, pw, nn, did, role_val):
                    log_login(st.session_state["username"], "admin", f"created_user:{nu}({role_val})")
                    st.success(f"User '{nu}' registered.")
                else: st.error("Username already taken.")
    st.markdown("---")
    for user in users:
        id_val = user.get("doctor_id", "")
        with st.expander(f"{user['username']} — {user['full_name']} ({user['role']})"):
            with st.form(f"edit_{user['id']}"):
                c1, c2 = st.columns(2)
                new_username = c1.text_input("Username", value=user["username"], key=f"uname_{user['id']}")
                new_name = c2.text_input("Full Name", value=user["full_name"], key=f"name_{user['id']}")
                c3, c4 = st.columns(2)
                new_did = c3.text_input("ID", value=id_val, key=f"did_{user['id']}")
                new_role = c4.selectbox("Role", ["clinician","admin"], index=0 if user["role"]=="clinician" else 1, key=f"role_{user['id']}")
                pc1, pc2 = st.columns(2)
                new_pw = pc1.text_input("New Password", type="password", key=f"pw_{user['id']}", placeholder="Leave empty to keep")
                new_pw2 = pc2.text_input("Confirm", type="password", key=f"pw2_{user['id']}")
                bc1, bc2 = st.columns(2)
                with bc1:
                    if st.form_submit_button("Save", use_container_width=True):
                        if new_pw and new_pw != new_pw2: st.error("Passwords don't match.")
                        elif new_pw and len(new_pw) < 6: st.error("Password too short.")
                        else:
                            pw_to_set = new_pw if new_pw else None
                            if update_user_full(user["id"], new_username, new_name, new_did, new_role, pw_to_set):
                                log_login(st.session_state["username"], "admin", f"edited_user:{user['username']}")
                                st.success("Updated."); st.rerun()
                            else: st.error("Update failed.")
                with bc2:
                    if st.form_submit_button("Delete", use_container_width=True):
                        if user["username"] == st.session_state["username"]: st.error("Cannot delete yourself.")
                        else:
                            delete_user(user["id"])
                            log_login(st.session_state["username"], "admin", f"deleted_user:{user['username']}")
                            st.success("Deleted."); st.rerun()

def page_login_log():
    st.markdown("#### Login Audit Log")
    col1, col2 = st.columns([2, 1])
    with col1: search_log = st.text_input("Search", "", placeholder="Filter...")
    with col2: log_filter = st.selectbox("Filter", ["All", "login_success", "login_failed", "logout", "created_user", "edited_user", "deleted_user"])
    logs = get_login_log()
    if not logs: st.info("No events."); return
    filtered = logs[:]
    if log_filter != "All": filtered = [l for l in filtered if log_filter in l.get("action","")]
    if search_log:
        q = search_log.lower()
        filtered = [l for l in filtered if q in str(l.get('username','')).lower() or q in str(l.get('action','')).lower()]
    st.metric("Events", len(filtered))
    st.markdown("---")
    for entry in filtered:
        action = entry.get("action","")
        icon = "✅" if "success" in action else "❌" if "failed" in action else "🚪" if "logout" in action else "➕" if "created" in action else "✏️" if "edited" in action else "🗑️" if "deleted" in action else "📌"
        st.markdown(f"{icon} **{entry.get('timestamp','')[:16]}** — `{entry.get('username','')}` ({entry.get('role','')}) — {action}")

def page_all_scans():
    st.markdown("#### Scan Activity Log - Admin View")
    st.caption("Patient data is confidential. Only scan metadata visible.")
    scans = get_scans(user=None, limit=200)
    if not scans: st.info("No scans."); return
    st.metric("Total Scans", len(scans))
    st.markdown("---")
    for s in scans:
        st.markdown(f"🩺 **{s.get('scan_date','')[:16]}** — Clinician: `{s.get('user','?')}`")


# MAIN
def main():
    init_db(); navbar()
    page = st.session_state["page"]; auth = st.session_state["authenticated"]; role = st.session_state.get("role","")
    if page == "home": page_home()
    elif page == "login":
        if auth: st.session_state["page"] = "scanner" if role == "clinician" else "manage_users"; st.rerun()
        else: page_login()
    elif page == "scanner":
        if not auth or role != "clinician": st.session_state["page"]="login"; st.rerun()
        else:
            models, res, thr, status = load_ensemble()
            if status != "ok": st.error("Models not found.")
            else: page_scanner(models, res, thr)
    elif page == "history":
        if not auth or role != "clinician": st.session_state["page"]="login"; st.rerun()
        else: page_history()
    elif page == "manage_users":
        if not auth or role != "admin": st.session_state["page"]="login"; st.rerun()
        else: page_manage_users()
    elif page == "login_log":
        if not auth or role != "admin": st.session_state["page"]="login"; st.rerun()
        else: page_login_log()
    elif page == "all_scans":
        if not auth or role != "admin": st.session_state["page"]="login"; st.rerun()
        else: page_all_scans()

if __name__ == "__main__":
    main()