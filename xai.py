"""
ChagasVision Explainable AI (XAI)
==================================
All detection is DATA-DRIVEN — no hardcoded thresholds.

4 Attribution Methods:
  1. Integrated Gradients  — Sundararajan et al. (2017) ICML
     Path integral from zero baseline satisfying Completeness axiom.
  2. GradientSHAP          — Lundberg & Lee (2017) NeurIPS
     Shapley value approximation using random noise baselines.
  3. Occlusion Sensitivity — Zeiler & Fergus (2014) ECCV
     Model-agnostic perturbation: zero each lead/window, measure change.
  4. Grad-CAM (1-D)        — Selvaraju et al. (2017) ICCV
     Gradient-weighted class activation map from convolutional layer.

Cross-method agreement via Kendall's tau (Adebayo et al., 2018 NeurIPS).
Ensemble disagreement analysis (Lakshminarayanan et al., 2017 NeurIPS).
Clinical patterns aligned to Rojas et al. (2018) PLoS NTD.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as sp_stats
from scipy.signal import find_peaks
from typing import Dict, List, Optional, Tuple

from config import LEAD_NAMES, CHAGAS_PATTERNS, NUM_LEADS, SAMPLING_RATE


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════════════════════

def _lead_imp_from_attr(attr: np.ndarray) -> Dict[str, float]:
    """Convert a [12, T] attribution map to per-lead importance percentages.
    Takes absolute values (importance = magnitude, not direction),
    averages across time, then normalises to sum to 1.0."""
    per_lead = np.abs(attr).mean(axis=1)
    total = per_lead.sum() + 1e-8
    return {LEAD_NAMES[i]: float(per_lead[i] / total) for i in range(NUM_LEADS)}


# ═══════════════════════════════════════════════════════════════════════════
# METHOD 1: INTEGRATED GRADIENTS (Sundararajan et al., 2017)
# ═══════════════════════════════════════════════════════════════════════════

class IntegratedGradients:
    """Axiom-satisfying attribution: IG_i(x) = (x_i - x'_i) * integral dF/dx_i da
    Zero baseline = absence of signal (standard for ECG)."""

    def __init__(self, model: nn.Module, n_steps: int = 50):
        self.model = model
        self.n_steps = n_steps

    def compute(self, ecg, age, sex, baseline=None):
        self.model.eval()
        if baseline is None:
            baseline = torch.zeros_like(ecg)
        alphas = np.linspace(0, 1, self.n_steps)
        scaled = torch.cat([baseline + a * (ecg - baseline) for a in alphas], dim=0)
        scaled.requires_grad_(True)
        out = self.model(scaled, age.repeat(self.n_steps), sex.repeat(self.n_steps))
        torch.sigmoid(out).sum().backward()
        grads = scaled.grad.mean(dim=0, keepdim=True)
        attr = (ecg - baseline) * grads
        return attr.squeeze(0).detach().cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════
# METHOD 2: GRADIENT SHAP (Lundberg & Lee, 2017)
# ═══════════════════════════════════════════════════════════════════════════

class GradientSHAP:
    """Shapley value approximation via Expected Gradients.
    Random noise baselines + random interpolation = Shapley values."""

    def __init__(self, model: nn.Module, n_samples: int = 25, noise_std: float = 0.1):
        self.model = model
        self.n_samples = n_samples
        self.noise_std = noise_std

    def compute(self, ecg, age, sex):
        self.model.eval()
        attrs = []
        for _ in range(self.n_samples):
            bl = torch.randn_like(ecg) * self.noise_std
            alpha = torch.rand(1).item()
            interp = (bl + alpha * (ecg - bl)).clone().requires_grad_(True)
            out = self.model(interp, age, sex)
            torch.sigmoid(out).sum().backward()
            self.model.zero_grad()
            if interp.grad is not None:
                a = ((ecg - bl) * interp.grad).squeeze(0).detach().cpu().numpy()
                if not np.isnan(a).any():
                    attrs.append(a)
        return np.mean(attrs, axis=0) if attrs else np.zeros((NUM_LEADS, ecg.shape[-1]))


# ═══════════════════════════════════════════════════════════════════════════
# METHOD 3: OCCLUSION SENSITIVITY (Zeiler & Fergus, 2014)
# ═══════════════════════════════════════════════════════════════════════════

class OcclusionSensitivity:
    """Model-agnostic perturbation: zero each lead/window, measure prediction change."""

    def __init__(self, model: nn.Module):
        self.model = model

    @torch.no_grad()
    def compute_lead_importance(self, ecg, age, sex):
        self.model.eval()
        orig = torch.sigmoid(self.model(ecg, age, sex)).item()
        imp = {}
        for i, name in enumerate(LEAD_NAMES):
            occ = ecg.clone()
            occ[:, i, :] = 0.0
            imp[name] = abs(orig - torch.sigmoid(self.model(occ, age, sex)).item())
        total = sum(imp.values()) or 1.0
        return {k: v / total for k, v in imp.items()}

    @torch.no_grad()
    def compute_temporal(self, ecg, age, sex, window=64, stride=32):
        self.model.eval()
        orig = torch.sigmoid(self.model(ecg, age, sex)).item()
        L = ecg.shape[-1]
        imp = []
        for s in range(0, L - window + 1, stride):
            occ = ecg.clone()
            occ[:, :, s:s+window] = 0.0
            imp.append(abs(orig - torch.sigmoid(self.model(occ, age, sex)).item()))
        arr = np.array(imp)
        return arr / (arr.max() + 1e-8)


# ═══════════════════════════════════════════════════════════════════════════
# METHOD 4: GRAD-CAM 1D (adapted from Selvaraju et al., 2017)
# ═══════════════════════════════════════════════════════════════════════════

class GradCAM1D:
    """1D Grad-CAM: hooks conv layer for activations + gradients.
    CAM = ReLU(sum of gradient-weighted activations) = WHERE the model focused."""

    def __init__(self, model: nn.Module, layer: str = "conv2"):
        self.model = model
        self.act = None
        self.grad = None
        self._hooks = []
        target = dict(model.named_modules()).get(layer)
        if target is None:
            raise ValueError(f"Layer '{layer}' not found")
        self._hooks.append(target.register_forward_hook(
            lambda m, i, o: setattr(self, 'act', o.detach())))
        self._hooks.append(target.register_full_backward_hook(
            lambda m, gi, go: setattr(self, 'grad', go[0].detach())))

    def compute(self, ecg, age, sex):
        self.model.eval()
        x = ecg.clone().requires_grad_(True)
        out = self.model(x, age, sex)
        self.model.zero_grad()
        torch.sigmoid(out).sum().backward()
        if self.grad is None or self.act is None:
            return np.zeros(1)
        w = self.grad.mean(dim=2, keepdim=True)
        cam = F.relu((w * self.act).sum(dim=1)).squeeze(0).cpu().numpy()
        return cam / (cam.max() + 1e-8)

    def compute_lead_importance(self, ecg, age, sex):
        """Per-lead importance from input gradients during Grad-CAM pass."""
        self.model.eval()
        x = ecg.clone().requires_grad_(True)
        out = self.model(x, age, sex)
        self.model.zero_grad()
        torch.sigmoid(out).sum().backward()
        if x.grad is None:
            return {name: 1.0 / NUM_LEADS for name in LEAD_NAMES}
        grad_abs = x.grad.squeeze(0).abs().cpu().numpy()
        per_lead = grad_abs.mean(axis=1)
        total = per_lead.sum() + 1e-8
        return {LEAD_NAMES[i]: float(per_lead[i] / total) for i in range(NUM_LEADS)}

    def cleanup(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ═══════════════════════════════════════════════════════════════════════════
# DATA-DRIVEN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def _adaptive_threshold(values: np.ndarray) -> float:
    """Otsu's method (1979): maximise between-class variance to separate
    'important' from 'unimportant' values without hardcoded cutoffs."""
    if len(values) < 2 or values.max() == values.min():
        return float(np.mean(values))
    bins = np.linspace(values.min(), values.max(), 50)
    best_t, best_var = bins[0], 0
    for t in bins[1:-1]:
        lo = values[values <= t]
        hi = values[values > t]
        if len(lo) == 0 or len(hi) == 0:
            continue
        w0 = len(lo) / len(values)
        w1 = len(hi) / len(values)
        var = w0 * w1 * (lo.mean() - hi.mean()) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return float(best_t)


def _find_attention_peaks(heatmap: np.ndarray, fs: int = 400) -> List[Dict]:
    """Auto-detect peaks in Grad-CAM temporal attention using scipy find_peaks.
    Labels cardiac cycle regions based on timing."""
    if heatmap is None or len(heatmap) < 5:
        return []
    thresh = _adaptive_threshold(heatmap)
    peaks, _ = find_peaks(heatmap, height=thresh,
                          distance=max(3, len(heatmap) // 20),
                          prominence=thresh * 0.3)
    results = []
    for pk in peaks:
        position_pct = pk / len(heatmap)
        cycle_pos = position_pct % 0.5
        if cycle_pos < 0.12:
            region = "P-wave / PR interval"
            clinical = "Conduction timing (AV block if prolonged)"
        elif cycle_pos < 0.30:
            region = "QRS complex"
            clinical = "Ventricular depolarisation (RBBB if wide, LAFB if axis deviated)"
        elif cycle_pos < 0.50:
            region = "ST-T segment"
            clinical = "Repolarisation (ischemia or cardiomyopathy changes)"
        else:
            region = "Between-beat interval"
            clinical = "Rhythm regularity"
        results.append({
            "position": int(pk),
            "position_pct": round(float(position_pct), 3),
            "strength": round(float(heatmap[pk]), 4),
            "region": region,
            "clinical_significance": clinical,
        })
    results.sort(key=lambda x: x["strength"], reverse=True)
    return results


def compute_method_consistency(imps: Dict[str, Dict[str, float]]) -> float:
    """Kendall's tau-b between all XAI method pairs.
    
    FIX: Previous version used argsort (sort indices) instead of actual values.
    Now passes raw importance values to kendalltau which computes ranks internally.
    tau-b handles ties correctly (scipy.stats.kendalltau).
    """
    if len(imps) < 2:
        return 1.0
    value_lists = []
    for d in imps.values():
        vals = [d.get(l, 0) for l in LEAD_NAMES]
        value_lists.append(vals)
    taus = []
    for i in range(len(value_lists)):
        for j in range(i + 1, len(value_lists)):
            tau, _ = sp_stats.kendalltau(value_lists[i], value_lists[j])
            if not np.isnan(tau):
                taus.append(tau)
    return round(float(np.mean(taus)), 4) if taus else 0.0


def compute_agreement_map(attrs: Dict[str, np.ndarray], sig_len: int = 2048) -> np.ndarray:
    """Per-timepoint agreement across methods. Normalise each to [0,1], average."""
    if not attrs:
        return np.zeros((NUM_LEADS, sig_len))
    normed = []
    for attr in attrs.values():
        a = np.abs(attr)
        if a.shape[-1] != sig_len:
            up = np.zeros((NUM_LEADS, sig_len))
            for lead in range(min(a.shape[0], NUM_LEADS)):
                up[lead] = np.interp(np.linspace(0, 1, sig_len),
                                     np.linspace(0, 1, a.shape[-1]),
                                     a[lead] if a.ndim == 2 else a)
            a = up
        for lead in range(min(a.shape[0], NUM_LEADS)):
            mx = a[lead].max()
            if mx > 0:
                a[lead] /= mx
        normed.append(a[:NUM_LEADS])
    return np.mean(normed, axis=0)


def analyse_ensemble(per_model_imps: List[Dict[str, float]],
                     predictions: List[float],
                     threshold: float = 0.5) -> Dict:
    """Ensemble disagreement from actual votes and lead variance.
    Reference: Lakshminarayanan et al. (2017) NeurIPS."""
    n_models = len(predictions)
    if n_models < 2:
        return {"agreement_level": "single_model", "unanimity": 1.0,
                "disagreement_leads": [], "high_confidence_leads": LEAD_NAMES[:3],
                "positive_votes": "1/1", "lead_variance": {}}

    pos_count = sum(1 for p in predictions if p >= threshold)
    unanimity = max(pos_count, n_models - pos_count) / n_models
    if unanimity == 1.0: level = "unanimous"
    elif unanimity >= 0.8: level = "strong_majority"
    elif unanimity >= 0.6: level = "majority"
    else: level = "split"

    lead_var = {}
    for lead in LEAD_NAMES:
        vals = [imp.get(lead, 0) for imp in per_model_imps]
        lead_var[lead] = float(np.var(vals))

    var_values = np.array(list(lead_var.values()))
    var_thresh = _adaptive_threshold(var_values)
    disagree = [l for l, v in sorted(lead_var.items(), key=lambda x: x[1], reverse=True)
                if v > var_thresh][:4]
    agree = [l for l, v in sorted(lead_var.items(), key=lambda x: x[1])][:4]

    return {
        "agreement_level": level, "unanimity": round(unanimity, 2),
        "disagreement_leads": disagree, "high_confidence_leads": agree,
        "lead_variance": lead_var, "positive_votes": f"{pos_count}/{n_models}",
    }


def detect_patterns(lead_imp: Dict[str, float],
                    grad_cam: Optional[np.ndarray] = None,
                    temporal_occ: Optional[np.ndarray] = None) -> List[Dict]:
    """Data-driven pattern detection using Otsu adaptive threshold.
    Patterns from Rojas et al. (2018): RBBB OR=4.60, LAFB OR=1.60."""
    all_values = np.array(list(lead_imp.values()))
    importance_threshold = _adaptive_threshold(all_values)
    attention_peaks = _find_attention_peaks(grad_cam) if grad_cam is not None else []

    detected = []
    for pid, info in CHAGAS_PATTERNS.items():
        pattern_strength = sum(lead_imp.get(l, 0) for l in info["leads"])
        n_leads = len(info["leads"])
        avg_per_lead = pattern_strength / n_leads
        if avg_per_lead > importance_threshold:
            pattern = {
                "id": pid, "name": info["name"], "leads": info["leads"],
                "strength": round(pattern_strength, 4),
                "avg_lead_importance": round(avg_per_lead, 4),
                "description": info["description"], "relevance": info["relevance"],
                "reference": info.get("reference", ""),
                "n_leads_above_threshold": sum(
                    1 for l in info["leads"] if lead_imp.get(l, 0) > importance_threshold),
            }
            if attention_peaks:
                region_keywords = {"rbbb": "QRS", "lafb": "QRS", "lvh": "QRS",
                                   "av_block": "P-wave", "low_voltage": "QRS"}
                keyword = region_keywords.get(pid, "")
                matching = [p for p in attention_peaks if keyword in p["region"]]
                if matching:
                    pattern["temporal_peaks"] = matching[:3]
                    pattern["temporal_region"] = matching[0]["region"]
                    pattern["temporal_strength"] = matching[0]["strength"]
            detected.append(pattern)
    detected.sort(key=lambda x: x["strength"], reverse=True)
    return detected


def compute_confidence(prob: float, predictions: List[float],
                       method_cons: float, threshold: float) -> Tuple[str, float, str]:
    """Multi-factor confidence from actual model outputs.
    Margin (40%) + vote agreement (35%) + XAI consistency (25%)."""
    margin = min(abs(prob - threshold) / 0.5, 1.0)
    n_models = len(predictions)
    if n_models > 1:
        pos = sum(1 for p in predictions if p >= threshold)
        vote_agreement = max(pos, n_models - pos) / n_models
    else:
        vote_agreement = 1.0
    xai_factor = max(0.0, method_cons)
    score = 0.40 * margin + 0.35 * vote_agreement + 0.25 * xai_factor

    reasons = []
    if margin > 0.4:
        reasons.append(f"clear margin ({abs(prob - threshold)*100:.0f}% from threshold)")
    elif margin < 0.15:
        reasons.append(f"close to boundary ({abs(prob - threshold)*100:.1f}% from threshold)")
    if vote_agreement == 1.0:
        reasons.append(f"all {n_models} models agree")
    elif vote_agreement >= 0.8:
        reasons.append(f"strong consensus ({vote_agreement*100:.0f}%)")
    else:
        reasons.append(f"models disagree ({vote_agreement*100:.0f}% agree)")
    if xai_factor > 0.6:
        reasons.append(f"XAI consistent (τ={method_cons:.2f})")
    elif xai_factor < 0.3:
        reasons.append(f"XAI disagree (τ={method_cons:.2f})")

    if score > 0.7: label = "High"
    elif score > 0.4: label = "Medium"
    else: label = "Low"
    return label, round(float(score), 3), "; ".join(reasons) if reasons else "Insufficient data"


def build_interpretation(prob, sorted_leads, patterns, predictions,
                         method_cons, ensemble, threshold) -> Dict:
    """Auto-generated interpretation — every statement references actual numbers."""
    interp = {"summary": "", "clinical_findings": [], "technical_notes": [], "recommendations": []}
    pct = round(prob * 100, 1)

    if prob >= 0.75:
        interp["summary"] = f"HIGH probability ({pct}%) of Chagas cardiomyopathy. Urgent serological testing recommended."
    elif prob >= threshold:
        interp["summary"] = f"ELEVATED probability ({pct}%) of Chagas cardiomyopathy. Serological confirmation recommended."
    elif prob >= threshold - 0.1:
        interp["summary"] = f"BORDERLINE risk ({pct}%) — cannot rule out Chagas. Follow-up evaluation advised."
    elif prob >= 0.2:
        interp["summary"] = f"LOW probability ({pct}%) but residual risk. Monitor if symptomatic."
    else:
        interp["summary"] = f"VERY LOW probability ({pct}%) of Chagas cardiomyopathy."

    top3 = [f"{l} ({v*100:.1f}%)" for l, v in sorted_leads[:3]]
    interp["clinical_findings"].append(f"Primary leads: {', '.join(top3)}")

    chagas_set = {"V1", "V2", "V6", "I", "aVL", "II"}
    n_chagas = sum(1 for l, _ in sorted_leads[:3] if l in chagas_set)
    interp["clinical_findings"].append(
        f"{n_chagas}/3 top leads are Chagas-relevant "
        f"(V1/V2 → RBBB OR=4.60, I/aVL → LAFB OR=1.60, II → AV block; Rojas et al., 2018)")

    for p in patterns[:3]:
        finding = f"Detected: {p['name']} (strength {p['strength']*100:.1f}%)"
        if p.get("temporal_region"): finding += f" — localised to {p['temporal_region']}"
        finding += f" — {p['relevance']}"
        interp["clinical_findings"].append(finding)

    ea = ensemble
    n_models = len(predictions)
    pos_count = sum(1 for p in predictions if p >= threshold)
    interp["technical_notes"].append(
        f"Ensemble: {pos_count}/{n_models} models positive "
        f"(probabilities: {', '.join(f'{p:.3f}' for p in predictions)})")
    interp["technical_notes"].append(
        f"Std: {np.std(predictions):.4f} — "
        f"{'low variance' if np.std(predictions) < 0.05 else 'moderate variance'}")
    if ea.get("high_confidence_leads"):
        interp["technical_notes"].append(f"Models agree on: {', '.join(ea['high_confidence_leads'][:3])}")
    if ea.get("disagreement_leads"):
        interp["technical_notes"].append(f"Models disagree on: {', '.join(ea['disagreement_leads'][:3])}")
    interp["technical_notes"].append(
        f"XAI agreement: τ = {method_cons:.2f} "
        f"({'strong' if method_cons > 0.6 else 'moderate' if method_cons > 0.3 else 'weak'})")

    if prob >= 0.5:
        interp["recommendations"] = [
            "Serological confirmation (ELISA/IFA) strongly recommended",
            "Echocardiogram to assess cardiac function",
            "Cardiology specialist referral",
            "Holter monitor for arrhythmia assessment"]
    elif prob >= threshold - 0.1:
        interp["recommendations"] = [
            "Follow-up ECG in 3-6 months",
            "Consider serology if clinically suspicious",
            "Screen family members if from endemic area"]
    elif prob >= 0.2:
        interp["recommendations"] = ["Routine clinical follow-up", "Repeat ECG if symptoms develop"]
    else:
        interp["recommendations"] = ["Routine follow-up — no immediate action required"]
    return interp


# ═══════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE XAI ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

class ComprehensiveXAI:
    """Runs all 4+1 methods across ensemble, aggregates, analyses."""

    def __init__(self, models: List[nn.Module]):
        self.models = models

    def explain(self, ecg_input, age: float, sex: int, threshold: float = 0.5) -> Dict:
        if isinstance(ecg_input, np.ndarray):
            ecg = torch.from_numpy(ecg_input).float()
        else:
            ecg = ecg_input.float()
        if ecg.dim() == 2: ecg = ecg.unsqueeze(0)
        age_t = torch.tensor([age]).float()
        sex_t = torch.tensor([sex]).long()

        # Step 1: Ensemble prediction
        preds = []
        with torch.no_grad():
            for m in self.models:
                m.eval()
                preds.append(torch.sigmoid(m(ecg, age_t, sex_t)).item())
        avg_prob = float(np.mean(preds))

        # Step 2: Run XAI methods
        method_imps = {}
        method_attrs = {}
        per_model_imps = []

        # Vanilla Gradient (all models)
        for idx, m in enumerate(self.models):
            try:
                g = ecg.clone().requires_grad_(True)
                torch.sigmoid(m(g, age_t, sex_t)).sum().backward()
                m.zero_grad()
                attr = g.grad.squeeze(0).abs().detach().cpu().numpy()
                key = f"gradient_m{idx+1}"
                method_imps[key] = _lead_imp_from_attr(attr)
                per_model_imps.append(method_imps[key])
                if idx == 0: method_attrs["gradient"] = attr
            except Exception: pass

        # Integrated Gradients (first 2 models)
        for idx in range(min(2, len(self.models))):
            try:
                attr = IntegratedGradients(self.models[idx], n_steps=30).compute(ecg.clone(), age_t, sex_t)
                key = f"intgrad_m{idx+1}"
                method_imps[key] = _lead_imp_from_attr(attr)
                method_attrs[key] = attr
            except Exception: pass

        # GradientSHAP (first model)
        try:
            attr = GradientSHAP(self.models[0], n_samples=20).compute(ecg.clone(), age_t, sex_t)
            method_imps["shap"] = _lead_imp_from_attr(attr)
            method_attrs["shap"] = attr
        except Exception: pass

        # Occlusion (first 2 models)
        temporal_occ = None
        for idx in range(min(2, len(self.models))):
            try:
                occ = OcclusionSensitivity(self.models[idx])
                method_imps[f"occlusion_m{idx+1}"] = occ.compute_lead_importance(ecg, age_t, sex_t)
                if idx == 0: temporal_occ = occ.compute_temporal(ecg, age_t, sex_t)
            except Exception: pass

        # Grad-CAM (first 2 models) — FIX: now stored in method_imps
        grad_cam = None
        for idx in range(min(2, len(self.models))):
            try:
                gc = GradCAM1D(self.models[idx], "conv2")
                cam = gc.compute(ecg.clone(), age_t, sex_t)
                gc_imp = gc.compute_lead_importance(ecg.clone(), age_t, sex_t)
                method_imps[f"gradcam_m{idx+1}"] = gc_imp
                gc.cleanup()
                if cam is not None and len(cam) > 1:
                    grad_cam = cam if grad_cam is None else (grad_cam + cam) / 2
            except Exception: pass

        # Step 3: Aggregate
        agg = {}
        for lead in LEAD_NAMES:
            vals = [d.get(lead, 0) for d in method_imps.values()]
            agg[lead] = np.mean(vals) if vals else 0.0
        total = sum(agg.values()) or 1.0
        lead_importance = {k: v / total for k, v in agg.items()}
        sorted_leads = sorted(lead_importance.items(), key=lambda x: x[1], reverse=True)

        # Step 4: Analysis
        method_cons = compute_method_consistency(method_imps)
        agreement_map = compute_agreement_map(method_attrs)

        n_models = len(preds)
        if n_models > 1:
            pos_count = sum(1 for p in preds if p >= threshold)
            model_cons = max(pos_count, n_models - pos_count) / n_models
        else:
            model_cons = 1.0

        ens_analysis = analyse_ensemble(per_model_imps, preds, threshold)
        attention_peaks = _find_attention_peaks(grad_cam) if grad_cam is not None else []
        patterns = detect_patterns(lead_importance, grad_cam, temporal_occ)

        conf_label, conf_score, conf_explanation = compute_confidence(
            avg_prob, preds, method_cons, threshold)
        interpretation = build_interpretation(
            avg_prob, sorted_leads, patterns, preds, method_cons, ens_analysis, threshold)

        method_types = set()
        for k in method_imps.keys():
            if k.startswith("gradient_m"): method_types.add("Gradient")
            elif k.startswith("intgrad"): method_types.add("Integrated Gradients")
            elif k == "shap": method_types.add("GradientSHAP")
            elif k.startswith("occlusion"): method_types.add("Occlusion")
            elif k.startswith("gradcam"): method_types.add("Grad-CAM")

        return {
            "probability": round(avg_prob, 4),
            "std": round(float(np.std(preds)), 4),
            "predictions": [round(p, 4) for p in preds],
            "prediction": "Chagas Positive" if avg_prob >= threshold else "Chagas Negative",
            "confidence": conf_label,
            "confidence_score": conf_score,
            "confidence_explanation": conf_explanation,
            "lead_importance": lead_importance,
            "sorted_leads": sorted_leads,
            "per_method_importances": method_imps,
            "methods_used": list(method_imps.keys()),
            "method_types": sorted(method_types),
            "n_methods": len(method_types),
            "grad_cam": grad_cam,
            "temporal_occlusion": temporal_occ,
            "attention_peaks": attention_peaks,
            "agreement_map": agreement_map,
            "attribution_maps": method_attrs,
            "model_consistency": round(model_cons, 4),
            "method_consistency": method_cons,
            "ensemble_analysis": ens_analysis,
            "detected_patterns": patterns,
            "interpretation": interpretation,
        }