#Name: Kaveesha Punchihewa
#ID: 20220094/w1959726
#Every code used in this file is either implemented by me or adapted from research articles and other sources, they are cited and referenced in a document. 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as sp_stats
from scipy.signal import find_peaks
from typing import Dict, List, Optional, Tuple

from config import LEAD_NAMES, CHAGAS_PATTERNS, NUM_LEADS, SAMPLING_RATE


def _lead_imp_from_attr(attr: np.ndarray) -> Dict[str, float]:
    per_lead = np.abs(attr).mean(axis=1)
    total = per_lead.sum() + 1e-8
    return {LEAD_NAMES[i]: float(per_lead[i] / total) for i in range(NUM_LEADS)}


class IntegratedGradients:
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


class GradientSHAP:
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


class OcclusionSensitivity:
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


class GradCAM1D:
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


def _adaptive_threshold(values: np.ndarray) -> float:
   
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


def detect_patterns(lead_imp: Dict[str, float],
                    grad_cam: Optional[np.ndarray] = None,
                    temporal_occ: Optional[np.ndarray] = None) -> List[Dict]:
    
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
        reasons.append(f"XAI consistent (tau={method_cons:.2f})")
    elif xai_factor < 0.3:
        reasons.append(f"XAI disagree (tau={method_cons:.2f})")

    if score > 0.7: label = "High"
    elif score > 0.4: label = "Medium"
    else: label = "Low"
    return label, round(float(score), 3), "; ".join(reasons) if reasons else "Insufficient data"


class ComprehensiveXAI:
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

        # Step 1: Ensemble prediction — each model votes independently
        preds = []
        with torch.no_grad():
            for m in self.models:
                m.eval()
                preds.append(torch.sigmoid(m(ecg, age_t, sex_t)).item())
        avg_prob = float(np.mean(preds))

        # Step 2: Run XAI methods
        method_imps = {}
        method_attrs = {}

        # Vanilla Gradient (all models — fast baseline)
        for idx, m in enumerate(self.models):
            try:
                g = ecg.clone().requires_grad_(True)
                torch.sigmoid(m(g, age_t, sex_t)).sum().backward()
                m.zero_grad()
                attr = g.grad.squeeze(0).abs().detach().cpu().numpy()
                method_imps[f"gradient_m{idx+1}"] = _lead_imp_from_attr(attr)
                if idx == 0: method_attrs["gradient"] = attr
            except Exception: pass

        # Integrated Gradients (first 2 models — axiomatic)
        for idx in range(min(2, len(self.models))):
            try:
                attr = IntegratedGradients(self.models[idx], n_steps=30).compute(ecg.clone(), age_t, sex_t)
                method_imps[f"intgrad_m{idx+1}"] = _lead_imp_from_attr(attr)
                method_attrs[f"intgrad_m{idx+1}"] = attr
            except Exception: pass

        # GradientSHAP (first model — Shapley values)
        try:
            attr = GradientSHAP(self.models[0], n_samples=20).compute(ecg.clone(), age_t, sex_t)
            method_imps["shap"] = _lead_imp_from_attr(attr)
            method_attrs["shap"] = attr
        except Exception: pass

        # Occlusion Sensitivity (first 2 models — perturbation-based)
        temporal_occ = None
        for idx in range(min(2, len(self.models))):
            try:
                occ = OcclusionSensitivity(self.models[idx])
                method_imps[f"occlusion_m{idx+1}"] = occ.compute_lead_importance(ecg, age_t, sex_t)
                if idx == 0: temporal_occ = occ.compute_temporal(ecg, age_t, sex_t)
            except Exception: pass

        # Grad-CAM (first 2 models — temporal attention heatmap)
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

        # Step 3: Aggregate all methods into one lead importance ranking
        agg = {}
        for lead in LEAD_NAMES:
            vals = [d.get(lead, 0) for d in method_imps.values()]
            agg[lead] = np.mean(vals) if vals else 0.0
        total = sum(agg.values()) or 1.0
        lead_importance = {k: v / total for k, v in agg.items()}
        sorted_leads = sorted(lead_importance.items(), key=lambda x: x[1], reverse=True)

        # Step 4: Cross-method agreement
        method_cons = compute_method_consistency(method_imps)

        # Model consistency = vote agreement
        n_models = len(preds)
        if n_models > 1:
            pos_count = sum(1 for p in preds if p >= threshold)
            model_cons = max(pos_count, n_models - pos_count) / n_models
        else:
            model_cons = 1.0

        # Step 5: Pattern detection & attention peaks
        attention_peaks = _find_attention_peaks(grad_cam) if grad_cam is not None else []
        patterns = detect_patterns(lead_importance, grad_cam, temporal_occ)

        # Step 6: Confidence
        conf_label, conf_score, conf_explanation = compute_confidence(
            avg_prob, preds, method_cons, threshold)

        # Count method types actually used
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
            "method_types": sorted(method_types),
            "n_methods": len(method_types),
            "grad_cam": grad_cam,
            "temporal_occlusion": temporal_occ,
            "attention_peaks": attention_peaks,
            "model_consistency": round(model_cons, 4),
            "method_consistency": method_cons,
            "detected_patterns": patterns,
        }