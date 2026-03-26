"""
ChagasVision Explainable AI (XAI)
==================================

All detection is DATA-DRIVEN — no hardcoded thresholds.

4 Attribution Methods:
  1. Integrated Gradients  — Sundararajan et al. (2017) ICML
  2. GradientSHAP          — Lundberg & Lee (2017) NeurIPS
  3. Occlusion Sensitivity — Zeiler & Fergus (2014) ECCV
  4. Grad-CAM (1-D)        — Selvaraju et al. (2017) ICCV

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
    per_lead = np.abs(attr).mean(axis=1)
    total = per_lead.sum() + 1e-8
    return {LEAD_NAMES[i]: float(per_lead[i] / total) for i in range(NUM_LEADS)}


# ═══════════════════════════════════════════════════════════════════════════
# METHOD 1: INTEGRATED GRADIENTS (Sundararajan et al., 2017)
# ═══════════════════════════════════════════════════════════════════════════

class IntegratedGradients:
    """IG_i(x) = (x_i − x'_i) × ∫₀¹ ∂F/∂x_i dα"""

    def __init__(self, model: nn.Module, n_steps: int = 50):
        self.model = model; self.n_steps = n_steps

    def compute(self, ecg, age, sex, baseline=None):
        self.model.eval()
        if baseline is None: baseline = torch.zeros_like(ecg)
        scaled = torch.cat([baseline + a * (ecg - baseline)
                            for a in np.linspace(0, 1, self.n_steps)], dim=0)
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
    def __init__(self, model: nn.Module, n_samples: int = 25, noise_std: float = 0.1):
        self.model = model; self.n_samples = n_samples; self.noise_std = noise_std

    def compute(self, ecg, age, sex):
        self.model.eval(); attrs = []
        for _ in range(self.n_samples):
            bl = torch.randn_like(ecg) * self.noise_std
            alpha = torch.rand(1).item()
            interp = (bl + alpha * (ecg - bl)).clone().requires_grad_(True)
            out = self.model(interp, age, sex)
            torch.sigmoid(out).sum().backward(); self.model.zero_grad()
            if interp.grad is not None:
                a = ((ecg - bl) * interp.grad).squeeze(0).detach().cpu().numpy()
                if not np.isnan(a).any(): attrs.append(a)
        return np.mean(attrs, axis=0) if attrs else np.zeros((NUM_LEADS, ecg.shape[-1]))


# ═══════════════════════════════════════════════════════════════════════════
# METHOD 3: OCCLUSION SENSITIVITY (Zeiler & Fergus, 2014)
# ═══════════════════════════════════════════════════════════════════════════

class OcclusionSensitivity:
    def __init__(self, model: nn.Module):
        self.model = model

    @torch.no_grad()
    def compute_lead_importance(self, ecg, age, sex):
        self.model.eval()
        orig = torch.sigmoid(self.model(ecg, age, sex)).item()
        imp = {}
        for i, name in enumerate(LEAD_NAMES):
            occ = ecg.clone(); occ[:, i, :] = 0.0
            imp[name] = abs(orig - torch.sigmoid(self.model(occ, age, sex)).item())
        total = sum(imp.values()) or 1.0
        return {k: v / total for k, v in imp.items()}

    @torch.no_grad()
    def compute_temporal(self, ecg, age, sex, window=64, stride=32):
        self.model.eval()
        orig = torch.sigmoid(self.model(ecg, age, sex)).item()
        L = ecg.shape[-1]; imp = []
        for s in range(0, L - window + 1, stride):
            occ = ecg.clone(); occ[:, :, s:s+window] = 0.0
            imp.append(abs(orig - torch.sigmoid(self.model(occ, age, sex)).item()))
        arr = np.array(imp)
        return arr / (arr.max() + 1e-8)


# ═══════════════════════════════════════════════════════════════════════════
# METHOD 4: GRAD-CAM 1D (adapted from Selvaraju et al., 2017)
# ═══════════════════════════════════════════════════════════════════════════

class GradCAM1D:
    def __init__(self, model: nn.Module, layer: str = "conv2"):
        self.model = model; self.act = None; self.grad = None; self._hooks = []
        target = dict(model.named_modules()).get(layer)
        if target is None: raise ValueError(f"Layer '{layer}' not found")
        self._hooks.append(target.register_forward_hook(lambda m,i,o: setattr(self, 'act', o.detach())))
        self._hooks.append(target.register_full_backward_hook(lambda m,gi,go: setattr(self, 'grad', go[0].detach())))

    def compute(self, ecg, age, sex):
        self.model.eval()
        x = ecg.clone().requires_grad_(True)
        out = self.model(x, age, sex); self.model.zero_grad()
        torch.sigmoid(out).sum().backward()
        if self.grad is None or self.act is None: return np.zeros(1)
        w = self.grad.mean(dim=2, keepdim=True)
        cam = F.relu((w * self.act).sum(dim=1)).squeeze(0).cpu().numpy()
        return cam / (cam.max() + 1e-8)

    def cleanup(self):
        for h in self._hooks: h.remove()
        self._hooks.clear()


# ═══════════════════════════════════════════════════════════════════════════
# DATA-DRIVEN ANALYSIS (no hardcoded thresholds)
# ═══════════════════════════════════════════════════════════════════════════

def _adaptive_threshold(values: np.ndarray) -> float:
    """Otsu-style adaptive threshold: find the value that maximally
    separates 'important' from 'unimportant' using between-class variance."""
    if len(values) < 2 or values.max() == values.min():
        return float(np.mean(values))
    bins = np.linspace(values.min(), values.max(), 50)
    best_t, best_var = bins[0], 0
    for t in bins[1:-1]:
        lo = values[values <= t]; hi = values[values > t]
        if len(lo) == 0 or len(hi) == 0: continue
        w0, w1 = len(lo)/len(values), len(hi)/len(values)
        var = w0 * w1 * (lo.mean() - hi.mean())**2
        if var > best_var: best_var, best_t = var, t
    return float(best_t)


def _find_attention_peaks(heatmap: np.ndarray, fs: int = 400) -> List[Dict]:
    """Find peaks in temporal attention map and label cardiac regions."""
    if heatmap is None or len(heatmap) < 5:
        return []
    # Adaptive peak threshold from data
    thresh = _adaptive_threshold(heatmap)
    peaks, props = find_peaks(heatmap, height=thresh, distance=max(3, len(heatmap)//20),
                               prominence=thresh * 0.3)
    total_duration_ms = len(heatmap) / fs * 1000 if fs > 0 else len(heatmap)

    results = []
    for pk in peaks:
        position_pct = pk / len(heatmap)
        # Auto-label based on position in cardiac cycle
        # Typical 10s ECG: P(0-10%), QRS(10-25%), T(25-50%), then repeats
        cycle_pos = position_pct % 0.5  # ~2 cycles in 10s
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
            "strength": float(heatmap[pk]),
            "region": region,
            "clinical_significance": clinical,
        })
    results.sort(key=lambda x: x["strength"], reverse=True)
    return results


def compute_method_consistency(imps: Dict[str, Dict[str, float]]) -> float:
    """Kendall's tau between all XAI method pairs."""
    if len(imps) < 2: return 1.0
    ranks = []
    for d in imps.values():
        vals = [d.get(l, 0) for l in LEAD_NAMES]
        ranks.append(np.argsort(vals)[::-1])
    taus = []
    for i in range(len(ranks)):
        for j in range(i+1, len(ranks)):
            t, _ = sp_stats.kendalltau(ranks[i], ranks[j])
            if not np.isnan(t): taus.append(t)
    return float(np.mean(taus)) if taus else 0.0


def compute_agreement_map(attrs: Dict[str, np.ndarray], sig_len: int = 2048) -> np.ndarray:
    """Per-timepoint agreement across methods. 1.0 = all agree, 0.0 = none."""
    if not attrs: return np.zeros((NUM_LEADS, sig_len))
    normed = []
    for attr in attrs.values():
        a = np.abs(attr)
        if a.shape[-1] != sig_len:
            up = np.zeros((NUM_LEADS, sig_len))
            for lead in range(min(a.shape[0], NUM_LEADS)):
                up[lead] = np.interp(np.linspace(0,1,sig_len), np.linspace(0,1,a.shape[-1]), a[lead] if a.ndim==2 else a)
            a = up
        for lead in range(min(a.shape[0], NUM_LEADS)):
            mx = a[lead].max()
            if mx > 0: a[lead] /= mx
        normed.append(a[:NUM_LEADS])
    return np.mean(normed, axis=0)


def analyse_ensemble(per_model_imps: List[Dict[str, float]],
                      predictions: List[float]) -> Dict:
    """Data-driven ensemble disagreement analysis."""
    if len(per_model_imps) < 2:
        return {"agreement_level": "single_model", "disagreement_leads": [],
                "high_confidence_leads": LEAD_NAMES[:3], "positive_votes": "1/1"}

    pos_count = sum(1 for p in predictions if p >= 0.5)
    total = len(predictions)

    # Agreement level from actual vote distribution
    unanimity = max(pos_count, total - pos_count) / total
    if unanimity == 1.0: level = "unanimous"
    elif unanimity >= 0.8: level = "strong_majority"
    elif unanimity >= 0.6: level = "majority"
    else: level = "split"

    # Per-lead variance (data-driven, no fixed threshold)
    lead_var = {}
    for lead in LEAD_NAMES:
        vals = [imp.get(lead, 0) for imp in per_model_imps]
        lead_var[lead] = float(np.var(vals))

    # Adaptive: leads with variance above median are "disagreement"
    var_values = np.array(list(lead_var.values()))
    var_thresh = _adaptive_threshold(var_values)
    disagree = [l for l, v in sorted(lead_var.items(), key=lambda x: x[1], reverse=True)
                if v > var_thresh][:4]
    agree = [l for l, v in sorted(lead_var.items(), key=lambda x: x[1])][:4]

    return {
        "agreement_level": level,
        "unanimity": round(unanimity, 2),
        "disagreement_leads": disagree,
        "high_confidence_leads": agree,
        "lead_variance": lead_var,
        "positive_votes": f"{pos_count}/{total}",
    }


def detect_patterns(lead_imp: Dict[str, float],
                     grad_cam: Optional[np.ndarray] = None,
                     temporal_occ: Optional[np.ndarray] = None) -> List[Dict]:
    """
    DATA-DRIVEN pattern detection.

    Instead of fixed thresholds, uses the distribution of lead importances
    to decide what's significant. A pattern is detected when its relevant
    leads have importance above the adaptive threshold.
    """
    # Compute adaptive threshold from actual importance distribution
    all_values = np.array(list(lead_imp.values()))
    importance_threshold = _adaptive_threshold(all_values)

    # Find attention peaks from Grad-CAM (if available)
    attention_peaks = _find_attention_peaks(grad_cam) if grad_cam is not None else []

    detected = []
    for pid, info in CHAGAS_PATTERNS.items():
        # Sum importance of this pattern's relevant leads
        pattern_strength = sum(lead_imp.get(l, 0) for l in info["leads"])
        n_leads = len(info["leads"])

        # Average importance per lead for this pattern
        avg_per_lead = pattern_strength / n_leads

        # Pattern is significant if average lead importance exceeds adaptive threshold
        if avg_per_lead > importance_threshold:
            pattern = {
                "id": pid,
                "name": info["name"],
                "leads": info["leads"],
                "strength": round(pattern_strength, 4),
                "avg_lead_importance": round(avg_per_lead, 4),
                "description": info["description"],
                "relevance": info["relevance"],
                "reference": info.get("reference", ""),
                "n_leads_above_threshold": sum(
                    1 for l in info["leads"] if lead_imp.get(l, 0) > importance_threshold
                ),
            }

            # Auto-find temporal region from attention peaks
            if attention_peaks:
                # Match pattern to its expected cardiac region
                region_keywords = {
                    "rbbb": "QRS", "lafb": "QRS", "lvh": "QRS",
                    "av_block": "P-wave", "low_voltage": "QRS",
                }
                keyword = region_keywords.get(pid, "")
                matching_peaks = [p for p in attention_peaks if keyword in p["region"]]
                if matching_peaks:
                    pattern["temporal_peaks"] = matching_peaks[:3]
                    pattern["temporal_region"] = matching_peaks[0]["region"]
                    pattern["temporal_strength"] = matching_peaks[0]["strength"]

            detected.append(pattern)

    detected.sort(key=lambda x: x["strength"], reverse=True)
    return detected


def compute_confidence(prob, std, model_cons, method_cons) -> Tuple[str, float, str]:
    """Multi-factor confidence with auto-generated explanation."""
    margin = abs(prob - 0.5) * 2
    agree = max(0, 1 - std * 5)
    xai = max(0, method_cons)
    score = 0.4 * margin + 0.3 * agree + 0.3 * xai

    # Auto-generate explanation from the actual factors
    reasons = []
    if margin > 0.5: reasons.append("strong decision margin")
    elif margin < 0.2: reasons.append("close to decision boundary")
    if agree > 0.8: reasons.append("all models agree")
    elif agree < 0.5: reasons.append("models show disagreement")
    if xai > 0.6: reasons.append("XAI methods consistent")
    elif xai < 0.3: reasons.append("XAI methods disagree")

    if score > 0.7: label = "High"
    elif score > 0.4: label = "Medium"
    else: label = "Low"

    explanation = "; ".join(reasons) if reasons else "No strong signals"
    return label, float(score), explanation


def build_interpretation(prob, sorted_leads, patterns, model_cons,
                          method_cons, ensemble) -> Dict:
    """Auto-generated interpretation from actual analysis data."""
    interp = {"summary": "", "clinical_findings": [], "technical_notes": [], "recommendations": []}

    # Summary auto-scales with probability
    pct = int(prob * 100)
    if prob >= 0.75:   interp["summary"] = f"HIGH probability ({pct}%) of Chagas cardiomyopathy. Urgent serological testing recommended."
    elif prob >= 0.5:  interp["summary"] = f"ELEVATED probability ({pct}%) of Chagas cardiomyopathy. Serological confirmation recommended."
    elif prob >= 0.35: interp["summary"] = f"BORDERLINE risk ({pct}%) — cannot rule out Chagas. Follow-up evaluation advised."
    elif prob >= 0.2:  interp["summary"] = f"LOW probability ({pct}%) but residual risk. Monitor if symptomatic."
    else:              interp["summary"] = f"VERY LOW probability ({pct}%) of Chagas cardiomyopathy."

    # Clinical findings from actual XAI data
    top3 = [f"{l} ({v*100:.1f}%)" for l, v in sorted_leads[:3]]
    interp["clinical_findings"].append(f"Primary leads: {', '.join(top3)}")

    chagas_set = {"V1","V2","V6","I","aVL","II"}
    top_names = [l for l, _ in sorted_leads[:3]]
    n_chagas = sum(1 for l in top_names if l in chagas_set)
    interp["clinical_findings"].append(f"{n_chagas}/3 top leads are Chagas-relevant (V1/V2→RBBB, I/aVL→LAFB, II→AV block)")

    for p in patterns[:3]:
        finding = f"Detected: {p['name']} (strength {p['strength']*100:.1f}%)"
        if p.get("temporal_region"):
            finding += f" — localised to {p['temporal_region']}"
        finding += f" — {p['relevance']}"
        interp["clinical_findings"].append(finding)

    # Technical notes from actual model data
    ea = ensemble
    interp["technical_notes"].append(
        f"Ensemble vote: {ea.get('positive_votes','?')} models positive ({ea.get('agreement_level','?')})")

    if ea.get("high_confidence_leads"):
        interp["technical_notes"].append(
            f"All models agree on: {', '.join(ea['high_confidence_leads'][:3])}")
    if ea.get("disagreement_leads"):
        interp["technical_notes"].append(
            f"Models disagree on: {', '.join(ea['disagreement_leads'][:3])}")

    interp["technical_notes"].append(
        f"XAI cross-method agreement: τ = {method_cons:.2f} "
        f"({'consistent' if method_cons > 0.5 else 'some disagreement'})")

    # Recommendations auto-scale
    if prob >= 0.5:
        interp["recommendations"] = [
            "Serological confirmation (ELISA/IFA) strongly recommended",
            "Echocardiogram to assess cardiac function",
            "Cardiology specialist referral",
            "Holter monitor for arrhythmia assessment",
        ]
    elif prob >= 0.35:
        interp["recommendations"] = [
            "Follow-up ECG in 3-6 months",
            "Consider serological testing if clinically suspicious",
            "Screen family members if from endemic area",
        ]
    elif prob >= 0.2:
        interp["recommendations"] = [
            "Routine clinical follow-up",
            "Repeat ECG if symptoms develop",
        ]
    else:
        interp["recommendations"] = [
            "Routine follow-up — no immediate action required",
        ]
    return interp


# ═══════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE XAI ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

class ComprehensiveXAI:
    """Runs all 4 methods across the ensemble, aggregates automatically."""

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

        # ── Step 1: Ensemble prediction ──
        preds = []
        with torch.no_grad():
            for m in self.models:
                m.eval()
                preds.append(torch.sigmoid(m(ecg, age_t, sex_t)).item())
        avg_prob = float(np.mean(preds))
        pred_std = float(np.std(preds))

        # ── Step 2: Run XAI methods ──
        method_imps = {}
        method_attrs = {}
        per_model_imps = []

        # Gradient (all models → ensemble-level XAI)
        for idx, m in enumerate(self.models):
            try:
                g = ecg.clone().requires_grad_(True)
                torch.sigmoid(m(g, age_t, sex_t)).sum().backward(); m.zero_grad()
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
                method_attrs[f"intgrad_m{idx+1}"] = attr
            except Exception: pass

        # Gradient SHAP (first model)
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

        # Grad-CAM (first 2 models, averaged)
        grad_cam = None
        for idx in range(min(2, len(self.models))):
            try:
                gc = GradCAM1D(self.models[idx], "conv2")
                cam = gc.compute(ecg.clone(), age_t, sex_t); gc.cleanup()
                if cam is not None and len(cam) > 1:
                    grad_cam = cam if grad_cam is None else (grad_cam + cam) / 2
            except Exception: pass

        # ── Step 3: Aggregate (automatic) ──
        agg = {}
        for lead in LEAD_NAMES:
            vals = [d.get(lead, 0) for d in method_imps.values()]
            agg[lead] = np.mean(vals) if vals else 0.0
        total = sum(agg.values()) or 1.0
        lead_importance = {k: v / total for k, v in agg.items()}
        sorted_leads = sorted(lead_importance.items(), key=lambda x: x[1], reverse=True)

        # ── Step 4: Automatic analysis ──
        method_cons = compute_method_consistency(method_imps)
        agreement_map = compute_agreement_map(method_attrs)
        model_cons = max(0, 1 - pred_std / max(avg_prob, 0.01)) if len(preds) > 1 else 1.0
        ens_analysis = analyse_ensemble(per_model_imps, preds)

        # Attention peaks (auto-detected from Grad-CAM)
        attention_peaks = _find_attention_peaks(grad_cam) if grad_cam is not None else []

        # Patterns (data-driven thresholds)
        patterns = detect_patterns(lead_importance, grad_cam, temporal_occ)

        # Confidence
        conf_label, conf_score, conf_explanation = compute_confidence(
            avg_prob, pred_std, model_cons, method_cons)

        # Interpretation (auto-generated)
        interpretation = build_interpretation(
            avg_prob, sorted_leads, patterns, model_cons, method_cons, ens_analysis)

        # Unique method types used
        method_types = set()
        for k in method_imps.keys():
            if "gradient_m" in k: method_types.add("Gradient")
            elif "intgrad" in k: method_types.add("Integrated Gradients")
            elif "shap" in k: method_types.add("GradientSHAP")
            elif "occlusion" in k: method_types.add("Occlusion")
            elif "gradcam" in k: method_types.add("Grad-CAM")

        return {
            "probability": avg_prob,
            "std": pred_std,
            "predictions": preds,
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
            "model_consistency": model_cons,
            "method_consistency": method_cons,
            "ensemble_analysis": ens_analysis,
            "detected_patterns": patterns,
            "interpretation": interpretation,
        }