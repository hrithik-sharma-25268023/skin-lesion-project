"""
Skin lesion project, binary gate, then the relevant subtype model.

    stage 1  binary:- benign vs malignant
    stage 2  benign:- branch  -> Swin-T {NV / BKL / DF / VASC}
             malignant:- branch  -> EfficientNet-B3  {MEL / BCC / AK / SCC}
"""

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import boto3
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from skin_lesion_project import file_system

BUCKET = "s3://skin-lesion-data-bucket"
RESULTS_PREFIX = "s3://skin-lesion-data-bucket/diagnosis"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


BINARY_MODELS = {
    "Swin-T": {
        "path": f"{BUCKET}/saved_models/swin_t_binary_finetuned/swin_t_binary_finetuned_checkpoint.pth",
        "arch": "swin", "img_size": 224,
    },
    "EfficientNet-B3 v2": {
        "path": f"{BUCKET}/saved_models/efficientnet_b3_binary_finetuned_v2/efficientnet_b3_binary_finetuned_checkpoint_v2.pth",
        "arch": "efficientnet", "img_size": 300,
    },
}

BRANCH_MODELS = {
    "benign": {
        "path": f"{BUCKET}/saved_models/swin_t_benign_subtype/swin_t_benign_subtype_checkpoint.pth",
        "arch": "swin", "img_size": 224,
        "classes": ["NV", "BKL", "DF", "VASC"],
        "model_name": "Swin-T",
    },
    "malignant": {
        "path": f"{BUCKET}/saved_models/pretrained_malignant_subtype/malignant_subtype_checkpoint_v2_.pth",
        "arch": "efficientnet", "img_size": 300,
        "classes": ["MEL", "BCC", "AK", "SCC"],
        "model_name": "EfficientNet-B3",
    },
}

FULL_NAME = {
    "NV": "Melanocytic nevus", "BKL": "Benign keratosis", "DF": "Dermatofibroma",
    "VASC": "Vascular lesion", "MEL": "Melanoma", "BCC": "Basal cell carcinoma",
    "AK": "Actinic keratosis", "SCC": "Squamous cell carcinoma",
}

st.set_page_config(page_title="Skin Lesion Project", layout="wide")


@st.cache_resource(show_spinner="Loading model…")
def load(path, arch, img_size, fallback_classes=None):
    ckpt = file_system.read(path, map_location=DEVICE, weights_only=False)
    sd = {k.replace("module.", "", 1): v for k, v in ckpt["model_state_dict"].items()}

    prefix = "head" if arch == "swin" else "classifier"
    head_key = [k for k in sd if k.startswith(prefix) and k.endswith("weight") and sd[k].dim() == 2][-1]
    n_out = sd[head_key].shape[0]

    model = models.swin_t(weights=None) if arch == "swin" else models.efficientnet_b3(weights=None)
    in_f = model.head.in_features if arch == "swin" else model.classifier[1].in_features
    setattr(model, prefix,
            nn.Sequential(nn.Dropout(0.0), nn.Linear(in_f, n_out))
            if head_key.count(".") == 2 else nn.Linear(in_f, n_out))
    model.load_state_dict(sd)
    model.to(DEVICE).eval()

    size = ckpt.get("image_size") or img_size
    log_prior = ckpt.get("log_prior")
    return {
        "model": model, "n_out": n_out, "arch": arch, "size": size,
        "classes": list(ckpt.get("class_names") or fallback_classes or []),
        "threshold": float(ckpt["chosen_threshold"]) if "chosen_threshold" in ckpt else None,
        "tau": float(ckpt.get("logit_adjust_tau", 0.0)),
        "log_prior": np.asarray(log_prior, dtype=np.float64) if log_prior is not None else None,
        "tf": transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }


def load_branch(name):
    c = BRANCH_MODELS[name]
    return load(c["path"], c["arch"], c["img_size"], c["classes"])


def save_json(s3_path, payload):
    u = urlparse(s3_path)
    boto3.client("s3").put_object(
        Bucket=u.netloc,
        Key=u.path.lstrip("/"),
        Body=json.dumps(payload, indent=2).encode(),
        ContentType="application/json",
    )


@torch.inference_mode()
def logits(m, img, tta=True):
    x = m["tf"](img).unsqueeze(0).to(DEVICE)
    views = [x, torch.flip(x, [3]), torch.flip(x, [2])] if tta else [x]
    acc = sum(F.log_softmax(m["model"](v).float(), dim=1) for v in views) / len(views)
    return acc.cpu().numpy()[0]


def p_malignant(m, img, tta=True):
    lg = logits(m, img, tta)
    if m["n_out"] == 1:
        return float(1 / (1 + np.exp(-lg[0])))
    e = np.exp(lg - lg.max())
    return float((e / e.sum())[1])


def subtype_probs(m, img, tta=True):
    lg = logits(m, img, tta)
    if m["tau"] and m["log_prior"] is not None:
        lg = lg - m["tau"] * m["log_prior"]
    e = np.exp(lg - lg.max())
    p = e / e.sum()
    return sorted(zip(m["classes"], p.tolist()), key=lambda t: -t[1])


st.title("Skin lesion Project")
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("Specimen")

    img = None
    f = st.file_uploader("Choose a lesion image", type=["jpg", "jpeg", "png", "webp"])
    if f:
        img = Image.open(io.BytesIO(f.getvalue())).convert("RGB")
        st.image(img, use_container_width=True)
        st.caption(f"{f.name} · {100}×{100} · {len(f.getvalue())/1024:.0f} KB")

    if img is None:
        st.info("Add a lesion image to run the cascade.")

    st.divider()
    st.subheader("Settings")
    binary_name = st.selectbox("Stage 1 model", list(BINARY_MODELS.keys()))
    use_tta = st.checkbox("Test-time augmentation", value=True)

    if st.button("Run Diagnosis", type="primary", disabled=img is None,
                 use_container_width=True):
        st.session_state.run = True

bcfg = BINARY_MODELS[binary_name]

sig = (f.file_id if f else None, binary_name, use_tta)
if st.session_state.get("sig") != sig:
    st.session_state.sig = sig
    st.session_state.run = False

with col2:
    st.subheader("Pathway")

    if not st.session_state.get("run"):
        with st.container(border=True):
            st.markdown("**Stage 1 · Binary gate**")
            st.caption(f"{binary_name} · {bcfg['img_size']}px")
        with st.container(border=True):
            st.markdown("**Stage 2 · Subtype**")
            st.caption("Runs on whichever branch stage 1 selects.")
        st.stop()

    gate = load(bcfg["path"], bcfg["arch"], bcfg["img_size"], ["benign", "malignant"])
    thr = gate["threshold"] if gate["threshold"] is not None else 0.5

    p = p_malignant(gate, img, use_tta)
    branch = "malignant" if p >= thr else "benign"
    sub = load_branch(branch)
    pairs = subtype_probs(sub, img, use_tta)
    top, top_p = pairs[0]
    p_branch = p if branch == "malignant" else 1 - p
    joint = p_branch * top_p

    with st.container(border=True):
        st.markdown("**Stage 1 · Binary gate**")
        st.caption(f"{binary_name} · {gate['size']}px")

        a, b, c = st.columns(3)
        a.metric("P(malignant)", f"{p:.3f}", f"{p - thr:+.3f} vs threshold",
                 delta_color="inverse" if branch == "benign" else "normal")
        b.metric("Threshold", f"{thr:.3f}")
        c.metric("Routed to", branch.capitalize())
        st.progress(p)

    f1, f2 = st.columns(2)
    f1.success("Benign branch — selected" if branch == "benign" else "Benign branch — not taken")
    f2.error("Malignant branch — selected" if branch == "malignant" else "Malignant branch — not taken")

    with st.container(border=True):
        st.markdown(f"**Stage 2 · {branch.capitalize()} subtype**")
        st.caption(f"{BRANCH_MODELS[branch]['model_name']} · {sub['size']}px")

        st.dataframe(
            pd.DataFrame([{"Code": c, "Subtype": FULL_NAME[c], "Probability": v}
                          for c, v in pairs]),
            hide_index=True, use_container_width=True,
            column_config={"Probability": st.column_config.ProgressColumn(
                "Probability", format="%.3f", min_value=0.0, max_value=1.0)},
        )

    with st.container(border=True):
        st.markdown("**Result**")
        r1, r2 = st.columns([2, 1])
        r1.metric(top, FULL_NAME[top])
        r2.metric("Combined", f"{joint:.4f}")

        record = {
            "image": f.name,
            "stage1_model": binary_name,
            "stage1": {
                "diagnosis": branch,
                "threshold": round(thr, 4),
                "probabilities": {
                    "benign": round(1 - p, 4),
                    "malignant": round(p, 4),
                },
            },
            "stage2": {
                "model": BRANCH_MODELS[branch]["model_name"],
                "code": top,
                "diagnosis": FULL_NAME[top],
                "probabilities": {c: round(v, 4) for c, v in pairs},
            },
            "combined": round(joint, 4),
            "tta": use_tta,
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        out_path = f"{RESULTS_PREFIX}/{Path(f.name).stem}.json"

        if st.button("Save results", use_container_width=True):
            try:
                save_json(out_path, record)
                st.success(f"Saved {Path(f.name).stem}.json")
            except Exception as e:
                st.error(f"Save failed: {e}")
        st.caption(out_path)
