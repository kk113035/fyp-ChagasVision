"""
ChagasVision Clinical Application
===================================
Dark-mode clinical interface with authentication, XAI analysis,
scan history, and Streamlit Cloud deployment support.

Run locally::
    streamlit run app.py

Deploy::
    Push to GitHub → connect to share.streamlit.io
"""

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
from datetime import datetime

from config import (
    MODELS_DIR, LEAD_NAMES, SAMPLING_RATE, CHAGAS_PATTERNS, ModelConfig,
)
from model import build_model
from preprocessing import ECGPreprocessor
from xai import ComprehensiveXAI

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="ChagasVision", page_icon="🫀", layout="wide")

for key, val in [("authenticated", False), ("page", "home"),
                 ("username", ""), ("full_name", ""), ("role", "")]:
    if key not in st.session_state:
        st.session_state[key] = val

# ═══════════════════════════════════════════════════════════════════════════
# DARK MODE CSS
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Force dark background globally ── */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
section[data-testid="stSidebar"] {
    background: #0a0f1a !important;
    color: #e2e8f0;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stSidebar"] > div { background: #0f1629 !important; }
h1, h2, h3, h4, h5 { color: #f1f5f9 !important; }
p, li, span { color: #cbd5e1; }
label { color: #94a3b8 !important; }

/* ── Inputs dark ── */
input, textarea, select, [data-baseweb="select"],
.stTextInput > div > div, .stNumberInput > div > div,
.stSelectbox > div > div {
    background: #1e293b !important;
    color: #e2e8f0 !important;
    border-color: #334155 !important;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
    border: none !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(37,99,235,0.3) !important;
}

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1a2744 40%, #0f172a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 2.8rem 2.2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%; right: -20%;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(59,130,246,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40%; left: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(239,68,68,0.06) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 {
    font-size: 2.4rem; font-weight: 800; margin: 0;
    letter-spacing: -0.5px; color: #f1f5f9 !important;
    position: relative; z-index: 1;
}
.hero .acc { color: #3b82f6; }
.hero p {
    color: #94a3b8; margin: 0.6rem 0 0 0;
    max-width: 620px; font-size: 1.05rem;
    position: relative; z-index: 1; line-height: 1.6;
}

/* ── Stat Grid ── */
.sgrid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.8rem; margin: 1.2rem 0; }
.scard {
    background: linear-gradient(135deg, #111827, #1e293b);
    border: 1px solid #1e3a5f;
    border-radius: 12px; padding: 1.2rem; text-align: center;
    transition: transform 0.15s, border-color 0.15s;
}
.scard:hover { transform: translateY(-3px); border-color: #3b82f6; }
.scard .v {
    font-size: 1.7rem; font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
}
.scard .l {
    font-size: 0.7rem; color: #64748b;
    text-transform: uppercase; letter-spacing: 1.2px; margin-top: 0.3rem;
}
.v-blue { color: #3b82f6; } .v-green { color: #10b981; }
.v-red { color: #ef4444; } .v-purple { color: #a78bfa; }

/* ── Cards ── */
.card {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.4rem;
    margin: 0.5rem 0;
}

/* ── Result Cards ── */
.res-pos {
    background: linear-gradient(135deg, #1c0a0a, #2d1111);
    border-left: 5px solid #dc2626;
    padding: 1.3rem; border-radius: 10px; margin: 1rem 0;
    box-shadow: 0 0 20px rgba(220,38,38,0.08);
}
.res-neg {
    background: linear-gradient(135deg, #051a0e, #0b2e1a);
    border-left: 5px solid #16a34a;
    padding: 1.3rem; border-radius: 10px; margin: 1rem 0;
    box-shadow: 0 0 20px rgba(22,163,74,0.08);
}
.res-bor {
    background: linear-gradient(135deg, #1a1500, #332b00);
    border-left: 5px solid #ca8a04;
    padding: 1.3rem; border-radius: 10px; margin: 1rem 0;
}

/* ── Section Headers ── */
.shdr {
    font-size: 1rem; font-weight: 700; color: #e2e8f0 !important;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 0.4rem; margin: 1.5rem 0 0.8rem 0;
}

/* ── Table ── */
.ctbl {
    width: 100%; border-collapse: collapse;
    border-radius: 10px; overflow: hidden;
    border: 1px solid #1e3a5f; margin: 0.8rem 0;
}
.ctbl th {
    background: #1e293b; color: #94a3b8;
    padding: 0.7rem 0.8rem; text-align: left;
    font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.8px;
}
.ctbl td {
    padding: 0.65rem 0.8rem; border-bottom: 1px solid #1e293b;
    font-size: 0.88rem; color: #cbd5e1;
}
.ctbl tr:nth-child(even) { background: #0f172a; }
.ctbl .hl { background: #172554 !important; color: #93c5fd; font-weight: 700; }

/* ── Login ── */
.login-hero {
    background: linear-gradient(135deg, #0f172a, #1a2744);
    border: 1px solid #1e3a5f;
    border-radius: 14px; padding: 2rem;
    text-align: center; margin-bottom: 1.5rem;
}
.login-hero h2 { color: #f1f5f9 !important; margin: 0; }
.login-hero p { color: #64748b; }

/* ── Upload ── */
.upbox {
    background: #111827; border: 2px dashed #1e3a5f;
    border-radius: 14px; padding: 2.5rem;
    text-align: center; margin: 0.5rem 0;
}
.upbox .ic { font-size: 3rem; margin-bottom: 0.5rem; }
.upbox h4 { color: #e2e8f0 !important; margin: 0.3rem 0; }
.upbox p { color: #64748b; font-size: 0.9rem; }

/* ── Instructions box ── */
.instr-box {
    background: #111827; border: 1px solid #1e3a5f;
    border-radius: 10px; padding: 1rem 1.2rem;
    font-size: 0.88rem; color: #94a3b8;
}
.instr-box ol { padding-left: 1.2rem; margin: 0.5rem 0; }
.instr-box li { margin: 0.4rem 0; }
.instr-box b { color: #e2e8f0; }

/* ── Disclaimer ── */
.disc {
    background: #0c1a33; border: 1px solid #1e40af;
    border-radius: 8px; padding: 0.9rem;
    font-size: 0.82rem; color: #93c5fd; margin-top: 1rem;
}

/* ── Tabs dark ── */
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: transparent; }
.stTabs [data-baseweb="tab"] {
    background: #1e293b !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
    padding: 0.5rem 1rem !important;
}
.stTabs [aria-selected="true"] {
    background: #2563eb !important;
    color: #ffffff !important;
}

/* ── Login Card ── */
.login-card {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 2rem;
    max-width: 480px;
    margin: 0 auto;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.login-card .icon-row {
    display: flex; justify-content: center; gap: 1.5rem; margin: 1rem 0 0.5rem 0;
}
.login-card .icon-item {
    text-align: center; font-size: 0.72rem; color: #64748b;
}
.login-card .icon-item .ic { font-size: 1.3rem; margin-bottom: 0.2rem; }
.login-divider {
    display: flex; align-items: center; gap: 0.8rem; margin: 1rem 0;
    color: #334155; font-size: 0.8rem;
}
.login-divider::before, .login-divider::after {
    content: ''; flex: 1; height: 1px; background: #1e3a5f;
}

/* ── Accessibility: focus outlines ── */
.stButton > button:focus-visible,
input:focus-visible, select:focus-visible, textarea:focus-visible {
    outline: 2px solid #3b82f6 !important;
    outline-offset: 2px !important;
}

/* ── Accessibility: skip link (hidden until focused) ── */
.skip-link {
    position: absolute; top: -100px; left: 0;
    background: #3b82f6; color: white;
    padding: 0.5rem 1rem; z-index: 9999;
    font-size: 0.9rem; font-weight: 600;
    border-radius: 0 0 8px 0;
    text-decoration: none;
}
.skip-link:focus { top: 0; }

/* ── Accessibility: screen reader only text ── */
.sr-only {
    position: absolute; width: 1px; height: 1px;
    padding: 0; margin: -1px; overflow: hidden;
    clip: rect(0,0,0,0); white-space: nowrap; border: 0;
}

/* ── Mobile Responsive ── */
@media (max-width: 768px) {
    .hero { padding: 1.5rem 1rem; }
    .hero h1 { font-size: 1.6rem; }
    .hero p { font-size: 0.9rem; }
    .sgrid { grid-template-columns: repeat(2, 1fr); gap: 0.5rem; }
    .scard .v { font-size: 1.3rem; }
    .scard .l { font-size: 0.6rem; }
    .ctbl th, .ctbl td { padding: 0.4rem 0.5rem; font-size: 0.75rem; }
    .res-pos, .res-neg, .res-bor { padding: 0.8rem; }
    .res-pos h4, .res-neg h4, .res-bor h4 { font-size: 0.95rem; }
    .upbox { padding: 1.5rem; }
    .upbox .ic { font-size: 2rem; }
    .login-hero { padding: 1.2rem; }
    .login-hero h2 { font-size: 1.4rem; }
    .login-card { padding: 1.2rem; }
    .disc { font-size: 0.75rem; padding: 0.6rem; }
    .shdr { font-size: 0.9rem; }
    .instr-box { font-size: 0.8rem; }
}

@media (max-width: 480px) {
    .hero h1 { font-size: 1.3rem; }
    .sgrid { grid-template-columns: 1fr 1fr; }
    .scard { padding: 0.8rem; }
    .scard .v { font-size: 1.1rem; }
    .ctbl { font-size: 0.7rem; }
}

/* ── Expander dark ── */
.streamlit-expanderHeader { background: #111827 !important; color: #e2e8f0 !important; }
.streamlit-expanderContent { background: #0f172a !important; }

/* ── Metrics dark ── */
[data-testid="stMetricValue"] { color: #e2e8f0 !important; }
[data-testid="stMetricLabel"] { color: #64748b !important; }

/* ── Hide defaults ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════════════════

DB_PATH = Path("chagasvision.db")

def init_db():
    conn = sqlite3.connect(str(DB_PATH)); c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password_hash TEXT, full_name TEXT, role TEXT DEFAULT 'clinician', created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
    c.execute("CREATE TABLE IF NOT EXISTS scans (id INTEGER PRIMARY KEY AUTOINCREMENT, user TEXT, patient_id TEXT, age INT, sex TEXT, probability REAL, prediction TEXT, threshold REAL, confidence TEXT, agreement REAL, top_leads TEXT, scan_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
    c.execute("SELECT COUNT(*) FROM users")
    if c.fetchone()[0] == 0:
        for u, p, n, r in [("admin","admin123","Administrator","admin"),
                           ("clinician","chagas2025","Clinical User","clinician")]:
            c.execute("INSERT INTO users (username,password_hash,full_name,role) VALUES (?,?,?,?)",
                      (u, hashlib.sha256(p.encode()).hexdigest(), n, r))
    conn.commit(); conn.close()

def verify_user(u, p):
    conn = sqlite3.connect(str(DB_PATH)); c = conn.cursor()
    c.execute("SELECT full_name, role FROM users WHERE username=? AND password_hash=?",
              (u, hashlib.sha256(p.encode()).hexdigest()))
    r = c.fetchone(); conn.close(); return r

def register_user(u, p, n):
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute("INSERT INTO users (username,password_hash,full_name) VALUES (?,?,?)",
                     (u, hashlib.sha256(p.encode()).hexdigest(), n))
        conn.commit(); conn.close(); return True
    except: conn.close(); return False

def save_scan(**kw):
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("INSERT INTO scans (user,patient_id,age,sex,probability,prediction,threshold,confidence,agreement,top_leads) VALUES (:user,:patient_id,:age,:sex,:prob,:prediction,:threshold,:confidence,:agreement,:top_leads)", kw)
    conn.commit(); conn.close()

def get_scans(user=None, limit=50):
    conn = sqlite3.connect(str(DB_PATH)); c = conn.cursor()
    if user: c.execute("SELECT * FROM scans WHERE user=? ORDER BY scan_date DESC LIMIT ?", (user, limit))
    else: c.execute("SELECT * FROM scans ORDER BY scan_date DESC LIMIT ?", (limit,))
    rows = c.fetchall(); cols = [d[0] for d in c.description]; conn.close()
    return [dict(zip(cols, r)) for r in rows]


# ═══════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
# NAVBAR
# ═══════════════════════════════════════════════════════════════════════════

def navbar():
    auth = st.session_state["authenticated"]
    if auth:
        cols = st.columns([3, 1, 1, 1, 1, 2])
        with cols[0]: st.markdown("### 🫀 ChagasVision")
        with cols[1]:
            if st.button("🏠 Home", use_container_width=True): st.session_state["page"]="home"; st.rerun()
        with cols[2]:
            if st.button("🔬 Scanner", use_container_width=True): st.session_state["page"]="scanner"; st.rerun()
        with cols[3]:
            if st.button("📋 History", use_container_width=True): st.session_state["page"]="history"; st.rerun()
        with cols[4]:
            if st.button("ℹ️ About", use_container_width=True): st.session_state["page"]="about"; st.rerun()
        with cols[5]:
            if st.button(f"🚪 {st.session_state['full_name']}", use_container_width=True):
                for k in ["authenticated","username","full_name","role"]:
                    st.session_state[k] = "" if k != "authenticated" else False
                st.session_state["page"]="home"; st.rerun()
    else:
        cols = st.columns([3, 1, 1, 1])
        with cols[0]: st.markdown("### 🫀 ChagasVision")
        with cols[1]:
            if st.button("🏠 Home", use_container_width=True): st.session_state["page"]="home"; st.rerun()
        with cols[2]:
            if st.button("🔐 Login", use_container_width=True): st.session_state["page"]="login"; st.rerun()
        with cols[3]:
            if st.button("ℹ️ About", use_container_width=True): st.session_state["page"]="about"; st.rerun()
    st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════
# PLOTS (dark theme)
# ═══════════════════════════════════════════════════════════════════════════

BG = "#0a0f1a"; BG2 = "#111827"; FG = "#e2e8f0"; GR = "#1e3a5f"; TX2 = "#64748b"

def plot_probability_gauge(prob, threshold):
    fig, ax = plt.subplots(figsize=(10, 1.5), facecolor=BG2)
    ax.set_facecolor(BG2)
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
    fig, ax = plt.subplots(figsize=(10, 3.5), facecolor=BG2)
    ax.set_facecolor(BG2)
    x = np.arange(len(preds))
    colors = ["#dc2626" if p >= threshold else "#10b981" for p in preds]
    bars = ax.bar(x, preds, color=colors, alpha=0.9, width=0.5, edgecolor=BG2, linewidth=2)
    ax.axhline(threshold, color=FG, ls="--", lw=1, alpha=0.3)
    ax.axhline(np.mean(preds), color="#3b82f6", lw=2, label=f"Mean: {np.mean(preds):.3f}")
    ax.set_xticks(x); ax.set_xticklabels([f"M{i+1}" for i in range(len(preds))], fontweight="600", color=FG)
    ax.set_ylim(0, 1); ax.set_ylabel("Probability", color=FG, fontweight="600")
    ax.tick_params(colors=FG)
    for b, p in zip(bars, preds):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.03, f"{p:.3f}",
                ha="center", fontsize=10, fontweight="bold", color=FG)
    ax.legend(fontsize=9, facecolor=BG2, edgecolor=GR, labelcolor=FG)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    for sp in ["bottom","left"]: ax.spines[sp].set_color(GR)
    plt.tight_layout(); return fig

def plot_lead_importance(lead_imp):
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG2)
    ax.set_facecolor(BG2)
    si = sorted(lead_imp.items(), key=lambda x: x[1], reverse=True)
    leads = [x[0] for x in si]; vals = [x[1]*100 for x in si]
    chl = {"V1","V2","V6","I","aVL","II"}
    colors = ["#dc2626" if l in chl else "#3b82f6" for l in leads]
    ax.barh(leads, vals, color=colors, alpha=0.85, height=0.6)
    ax.set_xlabel("Importance (%)", color=FG, fontweight="600"); ax.invert_yaxis()
    ax.tick_params(colors=FG)
    for i, v in enumerate(vals):
        ax.text(v+0.3, i, f"{v:.1f}%", va="center", fontsize=9, fontweight="600", color=FG)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#dc2626",label="Chagas-relevant"),Patch(color="#3b82f6",label="Other")],
              fontsize=9, loc="lower right", facecolor=BG2, edgecolor=GR, labelcolor=FG)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    for sp in ["bottom","left"]: ax.spines[sp].set_color(GR)
    plt.tight_layout(); return fig

def plot_ecg_gradcam(signal, lead_imp, grad_cam=None):
    fig = plt.figure(figsize=(16, 14), facecolor=BG)
    t = np.arange(signal.shape[1]) / SAMPLING_RATE
    cam_up = None
    if grad_cam is not None and len(grad_cam) > 0:
        cam_up = np.interp(np.linspace(0,1,signal.shape[1]), np.linspace(0,1,len(grad_cam)), grad_cam)
        cam_up = (cam_up-cam_up.min())/(cam_up.max()-cam_up.min()+1e-8)
    mx = max(lead_imp.values()) if lead_imp else 1
    mn = min(lead_imp.values()) if lead_imp else 0
    chl = {"V1","V2","V6","I","aVL","II"}
    for i, name in enumerate(LEAD_NAMES):
        ax = fig.add_subplot(6,2,i+1); ax.set_facecolor(BG2)
        ni = (lead_imp.get(name,0)-mn)/(mx-mn+1e-8)
        lc = plt.cm.Reds(ni*0.6+0.2) if ni > 0.5 else "#64748b"
        ax.plot(t, signal[i], color=lc, lw=0.9)
        if cam_up is not None:
            for j in range(len(t)-1):
                if cam_up[j] > 0.4:
                    ax.axvspan(t[j],t[j+1], alpha=cam_up[j]*0.3*ni, color="#dc2626", lw=0)
        lbl_c = "#dc2626" if name in chl else FG
        ax.set_ylabel(name, fontsize=10, fontweight="bold", color=lbl_c)
        ax.text(0.98,0.9, f"{lead_imp.get(name,0)*100:.1f}%", transform=ax.transAxes, fontsize=8,
                ha="right", va="top", color="#dc2626" if ni>0.5 else TX2, fontweight="bold" if ni>0.5 else "normal")
        ax.grid(True, alpha=0.08, color=GR); ax.set_xlim(0, t[-1])
        ax.tick_params(labelsize=7, colors=FG)
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)
        for sp in ["bottom","left"]: ax.spines[sp].set_color(GR)
        if i >= 10: ax.set_xlabel("Time (s)", fontsize=9, color=FG)
    fig.suptitle("12-Lead ECG — Grad-CAM Attention Overlay", fontsize=14, fontweight="800", color=FG, y=0.98)
    fig.text(0.99, 0.005, "Red = AI attention  •  Red labels = Chagas-relevant", fontsize=8, ha="right", color=TX2)
    plt.tight_layout(rect=[0,0.015,1,0.97]); return fig

def plot_gradcam_detail(cam, sig):
    if cam is None or len(cam) == 0: return None
    t = np.arange(len(sig)) / SAMPLING_RATE
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 4.5), sharex=True,
                                  gridspec_kw={"height_ratios": [1, 3]}, facecolor=BG2)
    # Heatmap: set extent to match the ECG time axis
    a1.imshow(cam.reshape(1, -1), aspect="auto", cmap="hot", interpolation="bilinear",
              extent=[0, t[-1], 0, 1])
    a1.set_yticks([]); a1.set_facecolor(BG2)
    a1.set_title("Grad-CAM Temporal Attention (conv2 layer)", fontsize=11, fontweight="700", color=FG)
    # Upsample Grad-CAM to signal length for overlay
    cn = np.interp(np.linspace(0, 1, len(sig)), np.linspace(0, 1, len(cam)), cam)
    cn = (cn - cn.min()) / (cn.max() - cn.min() + 1e-8)
    a2.set_facecolor(BG2)
    a2.plot(t, sig, color="#94a3b8", lw=1)
    a2.fill_between(t, sig.min(), sig, alpha=cn * 0.4, color="#dc2626")
    a2.set_xlabel("Time (s)", color=FG); a2.set_ylabel("Lead II", color=FG)
    a2.set_xlim(0, t[-1])
    a2.tick_params(colors=FG)
    for ax in [a1, a2]:
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        for sp in ["bottom", "left"]: ax.spines[sp].set_color(GR)
    plt.tight_layout(); return fig


def plot_temporal_occlusion(temp_occ, stride=32):
    """Visualize temporal occlusion importance as a heatmap."""
    if temp_occ is None or len(temp_occ) == 0: return None
    fig, ax = plt.subplots(figsize=(14, 2.5), facecolor=BG2)
    ax.set_facecolor(BG2)
    t = np.arange(len(temp_occ)) * stride / SAMPLING_RATE
    ax.fill_between(t, 0, temp_occ, color="#3b82f6", alpha=0.6)
    ax.plot(t, temp_occ, color="#60a5fa", lw=1.5)
    # Mark peaks automatically
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(temp_occ, height=np.percentile(temp_occ, 75), distance=3)
    if len(peaks) > 0:
        ax.scatter(t[peaks], temp_occ[peaks], color="#ef4444", s=40, zorder=5, label="Attention peaks")
        ax.legend(fontsize=8, facecolor=BG2, edgecolor=GR, labelcolor=FG)
    ax.set_xlabel("Time (s)", color=FG, fontweight="600")
    ax.set_ylabel("Importance", color=FG); ax.tick_params(colors=FG)
    ax.set_title("Temporal Occlusion — Which time regions matter most?", fontsize=11, fontweight="700", color=FG)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    for sp in ["bottom","left"]: ax.spines[sp].set_color(GR)
    plt.tight_layout(); return fig


def plot_per_method_comparison(per_method_imps):
    """Side-by-side lead rankings from each XAI method."""
    if not per_method_imps: return None
    # Group by method type
    groups = {}
    for key, imp in per_method_imps.items():
        base = key.split("_m")[0].replace("_"," ").title()
        if base not in groups: groups[base] = []
        groups[base].append(imp)
    # Average within each method type
    type_imps = {}
    for name, imps_list in groups.items():
        avg = {}
        for lead in LEAD_NAMES:
            avg[lead] = np.mean([d.get(lead, 0) for d in imps_list])
        type_imps[name] = avg

    n = len(type_imps)
    if n == 0: return None
    fig, axes = plt.subplots(1, n, figsize=(4*n, 5), facecolor=BG2, sharey=True)
    if n == 1: axes = [axes]
    chagas_set = {"V1","V2","V6","I","aVL","II"}

    for ax, (method, imp) in zip(axes, type_imps.items()):
        ax.set_facecolor(BG2)
        si = sorted(imp.items(), key=lambda x: x[1], reverse=True)
        leads = [x[0] for x in si]; vals = [x[1]*100 for x in si]
        colors = ["#dc2626" if l in chagas_set else "#3b82f6" for l in leads]
        ax.barh(leads, vals, color=colors, alpha=0.8, height=0.6)
        ax.invert_yaxis(); ax.set_title(method, fontsize=10, fontweight="700", color=FG)
        ax.tick_params(colors=FG)
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)
        for sp in ["bottom","left"]: ax.spines[sp].set_color(GR)
    axes[0].set_ylabel("Lead", color=FG)
    fig.suptitle("Per-Method Lead Ranking Comparison", fontsize=12, fontweight="700", color=FG, y=1.02)
    plt.tight_layout(); return fig


# ═══════════════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════════════

def build_report(result, age, sex, threshold, pid=""):
    interp = result["interpretation"]
    ea = result.get("ensemble_analysis", {})
    lines = [
        "╔══════════════════════════════════════════════════════╗",
        "║      CHAGASVISION — CLINICAL ANALYSIS REPORT        ║",
        "╚══════════════════════════════════════════════════════╝",
        f"Date: {datetime.now():%Y-%m-%d %H:%M}  Patient: {pid or 'N/A'}  Age: {age}  Sex: {sex}",
        f"\nRESULT: {result['prediction']}",
        f"Probability: {result['probability']*100:.1f}%  Confidence: {result['confidence']} ({result.get('confidence_score',0):.2f})",
        f"  → {result.get('confidence_explanation','')}",
        f"Agreement: {result['model_consistency']*100:.0f}%  Threshold: {threshold}",
        f"Ensemble vote: {ea.get('positive_votes','?')} ({ea.get('agreement_level','?')})",
        f"XAI consistency: τ = {result.get('method_consistency',0):.2f}",
        f"Methods used: {', '.join(result.get('method_types',[]))}",
        "\nTOP LEADS:"]
    for l, v in result["sorted_leads"][:6]:
        lines.append(f"  {l:>4}  {'█'*int(v*40)} {v*100:.1f}%")
    if ea.get("high_confidence_leads"):
        lines.append(f"\nAll models agree on: {', '.join(ea['high_confidence_leads'][:4])}")
    if ea.get("disagreement_leads"):
        lines.append(f"Models disagree on: {', '.join(ea['disagreement_leads'][:4])}")
    if result.get("detected_patterns"):
        lines.append("\nDETECTED PATTERNS:")
        for p in result["detected_patterns"]:
            line = f"  {p['name']} ({p['strength']*100:.1f}%) — {p['relevance']}"
            if p.get("temporal_region"): line += f" [localised to {p['temporal_region']}]"
            lines.append(line)
    if result.get("attention_peaks"):
        lines.append("\nATTENTION PEAKS:")
        for pk in result["attention_peaks"][:4]:
            lines.append(f"  {pk['region']} at {pk['position_pct']*100:.1f}% (strength {pk['strength']:.3f}) — {pk['clinical_significance']}")
    if interp.get("clinical_findings"):
        lines.append("\nCLINICAL FINDINGS:")
        for f in interp["clinical_findings"]: lines.append(f"  • {f}")
    if interp.get("technical_notes"):
        lines.append("\nTECHNICAL NOTES:")
        for n in interp["technical_notes"]: lines.append(f"  • {n}")
    lines += [f"\nASSESSMENT: {interp['summary']}", "\nRECOMMENDATIONS:"]
    for r in interp["recommendations"]: lines.append(f"  • {r}")
    lines += ["\nDISCLAIMER: Clinical decision support only. Confirm with serology.",
              f"\n© ChagasVision {datetime.now().year}"]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# PAGES
# ═══════════════════════════════════════════════════════════════════════════

def page_home():
    st.markdown("""
    <div class="hero">
        <h1>Detect Chagas Disease<br>from <span class="acc">12-Lead ECG</span></h1>
        <p>AI-powered clinical screening system using a hybrid Multi-Scale CNN + Transformer
           architecture. Trained on 143,114 ECG recordings from 3 international datasets with
           4 Explainable AI methods for transparent, trustworthy clinical decisions.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="sgrid">
        <div class="scard"><div class="v v-blue">0.837</div><div class="l">AUC-ROC Score</div></div>
        <div class="scard"><div class="v v-green">143,114</div><div class="l">Training ECGs</div></div>
        <div class="scard"><div class="v v-red">0.745</div><div class="l">Balanced Accuracy</div></div>
        <div class="scard"><div class="v v-purple">4</div><div class="l">XAI Methods</div></div>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🦠 About Chagas Disease")
        st.markdown("Chagas disease is caused by *Trypanosoma cruzi*, affecting **6.5 million** "
                     "people. Chronic infection causes cardiomyopathy with characteristic ECG "
                     "abnormalities: **RBBB** (40-50%), **LAFB** (30-40%), **AV block** (15-25%). "
                     "Serological testing is limited — ECG screening can prioritize patients.")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🔬 How ChagasVision Works")
        st.markdown("**1.** Upload 12-lead ECG recording (HDF5 format)\n\n"
                     "**2.** Signal preprocessed: bandpass → normalise → resample\n\n"
                     "**3.** 5-model ensemble analyses the waveform\n\n"
                     "**4.** 4 XAI methods explain the prediction\n\n"
                     "**5.** Clinical report generated with recommendations")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <table class="ctbl">
        <tr><th>Method</th><th>Dataset</th><th>AUC-ROC</th><th>XAI</th><th>Data</th></tr>
        <tr><td>Jidling et al. 2023</td><td>REDS-II</td><td>0.820</td><td>None</td><td>2M+</td></tr>
        <tr><td>Jidling et al. 2023</td><td>ELSA-Brasil</td><td>0.770</td><td>None</td><td>2M+</td></tr>
        <tr><td>Ribeiro et al. 2020</td><td>CODE (full)</td><td>0.870</td><td>None</td><td>2M+</td></tr>
        <tr class="hl"><td>ChagasVision</td><td>Combined</td><td>0.837</td><td>4 methods</td><td>143K</td></tr>
    </table>""", unsafe_allow_html=True)

    if not st.session_state["authenticated"]:
        st.markdown("---")
        _, c, _ = st.columns([1,2,1])
        with c:
            if st.button("🔐 Login to Access the Scanner →", type="primary", use_container_width=True):
                st.session_state["page"] = "login"; st.rerun()

    st.markdown('<div class="disc">⚕️ <b>Disclaimer:</b> ChagasVision is a clinical decision support '
                'tool for research purposes only. It does not replace professional medical diagnosis. '
                'Final confirmation requires serological testing.</div>', unsafe_allow_html=True)


def page_login():
    st.markdown('<div class="login-hero"><h2>🔐 Secure Clinical Access</h2>'
                '<p>Authorised healthcare personnel only — all sessions are encrypted and logged</p></div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="login-card">
        <div class="icon-row">
            <div class="icon-item"><div class="ic">🔒</div>Encrypted</div>
            <div class="icon-item"><div class="ic">🛡️</div>SHA-256 Hash</div>
            <div class="icon-item"><div class="ic">📋</div>Audit Logged</div>
            <div class="icon-item"><div class="ic">👤</div>Role-Based</div>
        </div>
    </div>""", unsafe_allow_html=True)

    _, col, _ = st.columns([1, 2, 1])
    with col:
        tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])
        with tab1:
            with st.form("login"):
                st.markdown("##### Sign in to your account")
                u = st.text_input("Username", placeholder="Enter your username")
                p = st.text_input("Password", type="password", placeholder="Enter your password")
                if st.form_submit_button("Sign In", use_container_width=True, type="primary"):
                    if u and p:
                        r = verify_user(u, p)
                        if r:
                            st.session_state.update({"authenticated":True,"username":u,"full_name":r[0],"role":r[1],"page":"scanner"})
                            st.rerun()
                        else: st.error("Invalid credentials. Please check your username and password.")
                    else: st.warning("Please enter both username and password.")
            st.markdown('<div class="login-divider">Demo Credentials</div>', unsafe_allow_html=True)
            dc1, dc2 = st.columns(2)
            with dc1:
                st.code("admin / admin123", language=None)
            with dc2:
                st.code("clinician / chagas2025", language=None)

        with tab2:
            with st.form("reg"):
                st.markdown("##### Create a new account")
                nu = st.text_input("Choose Username", key="ru", placeholder="Minimum 3 characters")
                nn = st.text_input("Full Name", placeholder="As it will appear on reports")
                np1 = st.text_input("Password", type="password", key="rp1", placeholder="Minimum 6 characters")
                np2 = st.text_input("Confirm Password", type="password", key="rp2", placeholder="Re-enter password")
                if st.form_submit_button("Create Account", use_container_width=True):
                    if not all([nu,nn,np1,np2]): st.error("All fields are required.")
                    elif len(nu) < 3: st.error("Username must be at least 3 characters.")
                    elif np1!=np2: st.error("Passwords do not match.")
                    elif len(np1)<6: st.error("Password must be at least 6 characters.")
                    elif register_user(nu,np1,nn): st.success("Account created! Switch to the Login tab to sign in.")
                    else: st.error("Username already taken. Please choose another.")


def page_scanner(models, results, default_threshold):
    preprocessor = ECGPreprocessor(); xai_engine = ComprehensiveXAI(models)

    with st.sidebar:
        st.markdown("### ⚙️ Analysis Settings")
        threshold = st.slider("Decision Threshold", 0.30, 0.70, float(default_threshold), 0.01)
        st.caption(f"Optimal: {default_threshold:.3f}")
        st.markdown("---")
        st.markdown("### 📊 Display Options")
        show = {k: st.checkbox(v, True) for k, v in [
            ("prob","Probability gauge"),("models","Ensemble agreement"),("xai","Lead importance"),
            ("gcam","Grad-CAM ECG overlay"),("gcam_d","Grad-CAM detail"),("patt","Clinical patterns"),
            ("recs","Recommendations"),("t_occ","Temporal occlusion"),("ens_dis","Ensemble disagreement"),
            ("per_method","Per-method breakdown"),("findings","Clinical findings & notes"),
            ("peaks","Attention peaks")]}

    # Title
    st.markdown("#### 🔬 ECG Analysis Console")
    st.caption(f"Logged in: **{st.session_state['full_name']}** • {datetime.now():%Y-%m-%d}")

    # Instructions dropdown
    with st.expander("📖 How to use the scanner", expanded=False):
        st.markdown("""
        <div class="instr-box">
        <ol>
            <li><b>Prepare your file</b> — ECG must be in HDF5 format (.h5 or .hdf5) with a dataset named <code>tracings</code> containing 12-lead ECG data.</li>
            <li><b>Upload the file</b> — drag and drop or click the upload area below.</li>
            <li><b>Enter patient info</b> — age (1-120) and biological sex. Optionally add a Patient ID for your records.</li>
            <li><b>Click "Analyse ECG"</b> — the system runs preprocessing, 5-model ensemble inference, and 4 XAI methods.</li>
            <li><b>Review results</b> — check the probability, model agreement, lead importance, Grad-CAM overlay, and clinical patterns.</li>
            <li><b>Download report</b> — save the clinical analysis report for your records.</li>
        </ol>
        <p><b>Supported formats:</b> HDF5 files from SaMi-Trop, CODE-15, or demo files generated by this system.</p>
        <p><b>Threshold:</b> Adjust in the sidebar. Lower = more sensitive (catches more Chagas). Higher = more specific (fewer false alarms). Default is optimised for balanced accuracy.</p>
        </div>""", unsafe_allow_html=True)

    # Upload
    c1, c2 = st.columns([2, 1])
    with c1:
        uploaded = st.file_uploader("Upload 12-Lead ECG", type=["h5","hdf5"], label_visibility="collapsed")
    with c2:
        pid = st.text_input("Patient ID", "", placeholder="Optional")
        a1, a2 = st.columns(2)
        age = a1.number_input("Age", 1, 120, 50); sex_s = a2.selectbox("Sex", ["Female","Male"])
        sex_v = 1 if sex_s == "Male" else 0

    if uploaded and st.button("🔍 Analyse ECG", type="primary", use_container_width=True):
        try:
            with h5py.File(uploaded, "r") as f:
                raw = np.array(f["tracings"], dtype=np.float32)
                if raw.ndim == 3: raw = raw[0]
                if raw.shape[0] > raw.shape[1]: raw = raw.T
            processed = preprocessor.process(raw)

            with st.spinner("Running 5-model ensemble + 4 XAI methods..."):
                result = xai_engine.explain(processed, age, sex_v, threshold)
            prob = result["probability"]

            # Result
            if prob >= threshold:
                st.markdown(f'<div class="res-pos"><h4 style="color:#ef4444;margin:0">⚠️ CHAGAS POSITIVE — Further Testing Recommended</h4>'
                            f'<p style="color:#fca5a5">Probability: <b>{prob*100:.1f}%</b> • Confidence: <b>{result["confidence"]}</b> • Agreement: <b>{result["model_consistency"]*100:.0f}%</b></p></div>', unsafe_allow_html=True)
            elif prob >= threshold - 0.1:
                st.markdown(f'<div class="res-bor"><h4 style="color:#eab308;margin:0">⚡ BORDERLINE — Close Monitoring Advised</h4>'
                            f'<p style="color:#fde68a">Probability: <b>{prob*100:.1f}%</b> • Confidence: <b>{result["confidence"]}</b> • Agreement: <b>{result["model_consistency"]*100:.0f}%</b></p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="res-neg"><h4 style="color:#22c55e;margin:0">✅ CHAGAS NEGATIVE — Low Risk</h4>'
                            f'<p style="color:#86efac">Probability: <b>{prob*100:.1f}%</b> • Confidence: <b>{result["confidence"]}</b> • Agreement: <b>{result["model_consistency"]*100:.0f}%</b></p></div>', unsafe_allow_html=True)

            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Probability", f"{prob*100:.1f}%"); m2.metric("Agreement", f"{result['model_consistency']*100:.0f}%")
            m3.metric("XAI Consistency", f"{result['method_consistency']:.2f}"); m4.metric("Confidence", result["confidence"])

            if show["prob"]:
                st.markdown('<p class="shdr">📊 Probability Gauge</p>', unsafe_allow_html=True)
                st.pyplot(plot_probability_gauge(prob, threshold)); plt.close()
            if show["models"]:
                st.markdown('<p class="shdr">🤝 Ensemble Agreement</p>', unsafe_allow_html=True)
                st.pyplot(plot_model_agreement(result["predictions"], threshold)); plt.close()
            if show["xai"]:
                st.markdown('<p class="shdr">🎯 Lead Importance (XAI — aggregated)</p>', unsafe_allow_html=True)
                st.pyplot(plot_lead_importance(result["lead_importance"])); plt.close()
                mt = result.get("method_types", [])
                st.caption(f"{result.get('n_methods', '?')} method types used: {', '.join(mt)}")
            if show["per_method"] and result.get("per_method_importances"):
                st.markdown('<p class="shdr">📊 Per-Method Lead Ranking Comparison</p>', unsafe_allow_html=True)
                fig = plot_per_method_comparison(result["per_method_importances"])
                if fig: st.pyplot(fig); plt.close()
            if show["gcam"]:
                st.markdown('<p class="shdr">🔥 ECG with Grad-CAM Attention Overlay</p>', unsafe_allow_html=True)
                st.pyplot(plot_ecg_gradcam(processed, result["lead_importance"], result.get("grad_cam"))); plt.close()
            if show["gcam_d"] and result.get("grad_cam") is not None:
                st.markdown('<p class="shdr">🔎 Grad-CAM Detail — Lead II</p>', unsafe_allow_html=True)
                fig = plot_gradcam_detail(result["grad_cam"], processed[1])
                if fig: st.pyplot(fig); plt.close()
            if show["t_occ"] and result.get("temporal_occlusion") is not None:
                st.markdown('<p class="shdr">⏱️ Temporal Occlusion — Which time regions matter?</p>', unsafe_allow_html=True)
                fig = plot_temporal_occlusion(result["temporal_occlusion"])
                if fig: st.pyplot(fig); plt.close()
                st.caption("Each bar shows how much the prediction changes when that time window is removed (Zeiler & Fergus, 2014)")
            if show["peaks"] and result.get("attention_peaks"):
                st.markdown('<p class="shdr">📍 Auto-Detected Attention Peaks</p>', unsafe_allow_html=True)
                for pk in result["attention_peaks"][:5]:
                    st.markdown(
                        f"- **{pk['region']}** at position {pk['position_pct']*100:.1f}% "
                        f"(strength {pk['strength']:.3f}) — {pk['clinical_significance']}")
            if show["ens_dis"] and result.get("ensemble_analysis"):
                st.markdown('<p class="shdr">🔀 Ensemble Disagreement Analysis</p>', unsafe_allow_html=True)
                ea = result["ensemble_analysis"]
                ea_c1, ea_c2 = st.columns(2)
                with ea_c1:
                    st.markdown(f"**Vote:** {ea.get('positive_votes','?')} models positive")
                    st.markdown(f"**Agreement:** {ea.get('agreement_level','?')} (unanimity {ea.get('unanimity',0)*100:.0f}%)")
                with ea_c2:
                    if ea.get("high_confidence_leads"):
                        st.markdown(f"✅ **All models agree on:** {', '.join(ea['high_confidence_leads'][:4])}")
                    if ea.get("disagreement_leads"):
                        st.markdown(f"⚠️ **Models disagree on:** {', '.join(ea['disagreement_leads'][:4])}")
                st.caption("Ensemble disagreement = epistemic uncertainty (Lakshminarayanan et al., 2017 NeurIPS)")
            if show["patt"] and result.get("detected_patterns"):
                st.markdown('<p class="shdr">🏥 Auto-Detected Clinical Patterns</p>', unsafe_allow_html=True)
                for p in result["detected_patterns"]:
                    title = f"🔬 {p['name']} — strength {p['strength']*100:.1f}%"
                    if p.get("temporal_region"):
                        title += f" — localised to {p['temporal_region']}"
                    with st.expander(title, expanded=True):
                        pc1, pc2 = st.columns(2)
                        with pc1:
                            st.markdown(f"**Relevant leads:** {', '.join(p['leads'])}")
                            st.markdown(f"**Leads above threshold:** {p.get('n_leads_above_threshold','?')}/{len(p['leads'])}")
                            st.markdown(f"**Avg lead importance:** {p.get('avg_lead_importance',0)*100:.2f}%")
                        with pc2:
                            st.markdown(f"**Clinical relevance:** {p['relevance']}")
                            if p.get("temporal_peaks"):
                                for tp in p["temporal_peaks"][:2]:
                                    st.markdown(f"Peak in **{tp['region']}** (strength {tp['strength']:.3f})")
            if show["findings"] and result.get("interpretation"):
                interp = result["interpretation"]
                if interp.get("clinical_findings"):
                    st.markdown('<p class="shdr">🩺 Clinical Findings (auto-generated from XAI)</p>', unsafe_allow_html=True)
                    for f in interp["clinical_findings"]:
                        st.markdown(f"- {f}")
                if interp.get("technical_notes"):
                    st.markdown('<p class="shdr">⚙️ Technical Notes</p>', unsafe_allow_html=True)
                    for n in interp["technical_notes"]:
                        st.markdown(f"- {n}")
            if show["recs"] and result.get("interpretation"):
                st.markdown('<p class="shdr">📋 Recommendations</p>', unsafe_allow_html=True)
                interp = result["interpretation"]
                if prob >= 0.5: st.error(f"**{interp['summary']}**")
                elif prob >= 0.35: st.warning(f"**{interp['summary']}**")
                else: st.success(f"**{interp['summary']}**")
                for r in interp["recommendations"]: st.markdown(f"- {r}")
                # Confidence explanation
                conf_exp = result.get("confidence_explanation", "")
                if conf_exp:
                    st.caption(f"Confidence: {result['confidence']} ({result.get('confidence_score',0):.2f}) — {conf_exp}")

            st.markdown('<div class="disc">⚕️ <b>Important:</b> Clinical decision support only. '
                        'Confirm with serological testing and qualified healthcare professionals.</div>',
                        unsafe_allow_html=True)

            top_str = ", ".join(f"{l}({v*100:.1f}%)" for l, v in result["sorted_leads"][:5])
            save_scan(user=st.session_state["username"], patient_id=pid or f"anon_{datetime.now():%H%M%S}",
                      age=age, sex=sex_s, prob=round(prob,4), prediction=result["prediction"],
                      threshold=threshold, confidence=result["confidence"],
                      agreement=round(result["model_consistency"],3), top_leads=top_str)

            st.download_button("📄 Download Clinical Report",
                               build_report(result, age, sex_s, threshold, pid),
                               f"chagas_{pid or 'report'}_{datetime.now():%Y%m%d_%H%M%S}.txt",
                               use_container_width=True)
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.info("Ensure HDF5 contains a 'tracings' dataset with 12-lead ECG data.")

    elif not uploaded:
        st.markdown("""
        <div class="upbox">
            <div class="ic">📤</div>
            <h4>Upload an ECG to begin analysis</h4>
            <p>Drag and drop an HDF5 file or click to browse. Then enter patient details and click Analyse.</p>
        </div>""", unsafe_allow_html=True)


def page_history():
    st.markdown("#### 📋 Scan Archive")
    scans = get_scans(user=None if st.session_state["role"]=="admin" else st.session_state["username"])
    if not scans:
        st.info("No scan records yet. Run an analysis from the Scanner page to create records.")
        return
    pos = sum(1 for s in scans if "POS" in s.get("prediction","").upper())
    c1,c2,c3 = st.columns(3)
    c1.metric("Total Scans", len(scans)); c2.metric("Positive", pos); c3.metric("Negative", len(scans)-pos)
    st.markdown("---")
    for s in scans:
        ip = "POS" in s.get("prediction","").upper()
        ps = f"{s['probability']*100:.1f}%" if s.get("probability") else "?"
        with st.expander(f"{'🔴' if ip else '🟢'} {s['scan_date'][:16]} • {s.get('patient_id','?')} • {ps}"):
            a,b,c,d = st.columns(4)
            a.markdown(f"**Age:** {s.get('age','')}"); b.markdown(f"**Sex:** {s.get('sex','')}")
            c.markdown(f"**Confidence:** {s.get('confidence','')}"); d.markdown(f"**Agreement:** {s.get('agreement','')}")
            st.markdown(f"**Top leads:** {s.get('top_leads','')}")
            st.markdown(f"**Result:** {s.get('prediction','')}")


def page_about():
    st.markdown("#### ℹ️ About ChagasVision")
    t1,t2,t3,t4 = st.tabs(["Architecture","Datasets","Performance","References"])
    with t1:
        st.markdown("##### Hybrid CNN + Transformer — 538,801 Parameters")
        st.code("ECG [B,12,2048]\n → Multi-Scale Conv1d (k=3,7,15) → SE Attention\n → Conv1d (k=7, 128ch) → SE Attention\n → Transformer Encoder (2 layers, 4 heads, d=128)\n → Metadata Fusion (age 4d + sex 4d = 8 dims)\n → MLP Classifier (136→128→64→1) → probability", language="text")
        st.markdown("##### Class Imbalance: 6 Research-Backed Techniques")
        for i,(n,d) in enumerate([
            ("Class-Balanced Focal Loss","Cui (2019) effective-number α + Lin (2017) focal γ=2.0"),
            ("Weighted Random Sampling","20× oversample → ~5 positives per batch"),
            ("Asymmetric Augmentation","80% positive / 20% negative (6 ECG transforms)"),
            ("Stratified K-Fold CV","Preserves 2.8% positive ratio per fold"),
            ("Threshold Optimisation","Post-training search [0.30, 0.70] on validation"),
            ("Gradient Clipping","max_norm=1.0 prevents explosion from rare samples")],1):
            st.markdown(f"**{i}.** {n} — {d}")
    with t2:
        st.markdown("""<table class="ctbl"><tr><th>Dataset</th><th>Country</th><th>Samples</th><th>Chagas+</th><th>Format</th><th>Labels</th></tr>
            <tr><td>SaMi-Trop</td><td>Brazil</td><td>1,631</td><td>100%</td><td>HDF5</td><td>Serology</td></tr>
            <tr><td>CODE-15%</td><td>Brazil</td><td>119,684</td><td>~2%</td><td>HDF5</td><td>Self-report</td></tr>
            <tr><td>PTB-XL</td><td>Germany</td><td>21,799</td><td>0%</td><td>WFDB</td><td>Geography</td></tr></table>""", unsafe_allow_html=True)
    with t3:
        st.markdown("""<table class="ctbl"><tr><th>Metric</th><th>Value</th><th>95% CI</th></tr>
            <tr><td>Balanced Accuracy</td><td>0.745</td><td>[0.726, 0.763]</td></tr>
            <tr><td>AUC-ROC</td><td>0.837</td><td>[0.819, 0.854]</td></tr>
            <tr><td>Sensitivity</td><td>0.684</td><td>[0.645, 0.721]</td></tr>
            <tr><td>Specificity</td><td>0.806</td><td>[0.800, 0.811]</td></tr>
            <tr><td>CinC TPR@5%</td><td>0.394</td><td>[0.355, 0.429]</td></tr></table>""", unsafe_allow_html=True)
    with t4:
        for i,r in enumerate(["Ribeiro et al. (2020) Nature Comms — DNN ECG diagnosis, AUC 0.87 for Chagas",
            "Jidling et al. (2023) PLoS NTD — Chagas ECG screening, AUC 0.82 REDS-II",
            "Reyna et al. (2025) CinC — PhysioNet Challenge 2025, TPR@5% metric",
            "Lin et al. (2017) ICCV — Focal Loss for Dense Object Detection",
            "Cui et al. (2019) CVPR — Class-Balanced Loss from Effective Samples",
            "Hu et al. (2018) CVPR — Squeeze-and-Excitation Networks",
            "Vaswani et al. (2017) NeurIPS — Attention Is All You Need",
            "Sundararajan et al. (2017) ICML — Integrated Gradients",
            "Selvaraju et al. (2017) ICCV — Grad-CAM Visual Explanations",
            "Rojas et al. (2018) PLoS NTD — Chagas ECG patterns, RBBB OR=4.60"],1):
            st.markdown(f"**[{i}]** {r}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    init_db()

    # Accessibility: skip-to-content link for keyboard/screen reader users
    st.markdown('<a href="#main-content" class="skip-link" tabindex="0">Skip to main content</a>',
                unsafe_allow_html=True)

    navbar()

    # Landmark for skip link
    st.markdown('<div id="main-content"></div>', unsafe_allow_html=True)

    page = st.session_state["page"]; auth = st.session_state["authenticated"]

    if page == "home": page_home()
    elif page == "login":
        if auth: st.session_state["page"]="scanner"; st.rerun()
        else: page_login()
    elif page == "scanner":
        if not auth: st.session_state["page"]="login"; st.rerun()
        else:
            models, res, thr, status = load_ensemble()
            if status!="ok": st.error("Trained models not found. Complete training first.")
            else: page_scanner(models, res, thr)
    elif page == "history":
        if not auth: st.session_state["page"]="login"; st.rerun()
        else: page_history()
    elif page == "about": page_about()

if __name__ == "__main__":
    main()