"""V190b: MiMo encoder probe with richer pooling.

Tries to improve over V190 (mean-pool, 0.83 AUC) by:
  1. Multi-statistic pooling: mean + max + std → 3*1280 = 3840-d
  2. MLP probe (2 hidden layers) instead of plain LR

Predict-then-run:
  - Mean+Max+Std + LR: predict 0.83-0.86 (small lift from 0.83 plain mean)
  - MLP probe on mean: predict 0.84-0.87 (non-linearity helps)
  - Both combined: predict 0.85-0.88
"""
import argparse
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


# flash_attn shim
def _flash_attn_varlen_func_sdpa(q, k, v, cu_q, cu_k, max_q, max_k,
                                  causal=False, window_size=(-1, -1), **kw):
    bsz = cu_q.shape[0] - 1
    H, D = q.shape[1], q.shape[2]
    seq_lens = (cu_q[1:] - cu_q[:-1]).tolist()
    if all(s == seq_lens[0] for s in seq_lens):
        T = seq_lens[0]
        q_b = q.view(bsz, T, H, D).transpose(1, 2).contiguous()
        k_b = k.view(bsz, T, H, D).transpose(1, 2).contiguous()
        v_b = v.view(bsz, T, H, D).transpose(1, 2).contiguous()
        out = F.scaled_dot_product_attention(q_b, k_b, v_b, is_causal=causal)
        return out.transpose(1, 2).reshape(bsz * T, H, D)
    out = torch.zeros_like(q)
    for i in range(bsz):
        s = int(cu_q[i].item())
        e = int(cu_q[i + 1].item())
        if e == s:
            continue
        q_i = q[s:e].transpose(0, 1).unsqueeze(0)
        k_i = k[s:e].transpose(0, 1).unsqueeze(0)
        v_i = v[s:e].transpose(0, 1).unsqueeze(0)
        o = F.scaled_dot_product_attention(q_i, k_i, v_i, is_causal=causal)
        out[s:e] = o.squeeze(0).transpose(0, 1)
    return out

_fake_flash_attn = types.ModuleType("flash_attn")
_fake_flash_attn.flash_attn_varlen_func = _flash_attn_varlen_func_sdpa
sys.modules["flash_attn"] = _fake_flash_attn

sys.path.insert(0, "/tmp/MiMo-Audio/src")
from mimo_audio_tokenizer import MiMoAudioTokenizer, MiMoAudioTokenizerConfig  # noqa: E402


def load_mimo_tokenizer(model_id: str):
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    repo_path = snapshot_download(model_id)
    cfg = MiMoAudioTokenizerConfig.from_pretrained(repo_path)
    model = MiMoAudioTokenizer(cfg)
    state = load_file(f"{repo_path}/model.safetensors")
    model.load_state_dict(state, strict=False)
    return model


SR_NATIVE = 32000
SR_MIMO = 24000
CHUNK_SECONDS = 5


def load_chunk_wav(filename, end_sec, audio_dir):
    fp = audio_dir / filename
    audio, sr = sf.read(str(fp), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    start = (end_sec - CHUNK_SECONDS) * sr
    end = end_sec * sr
    if start < 0:
        target = audio[0:max(0, end)]
        if len(target) < CHUNK_SECONDS * sr:
            target = np.pad(target, (CHUNK_SECONDS * sr - len(target), 0))
    elif end > len(audio):
        target = audio[start:]
        if len(target) < CHUNK_SECONDS * sr:
            target = np.pad(target, (0, CHUNK_SECONDS * sr - len(target)))
    else:
        target = audio[start:end]
    if sr != SR_MIMO:
        target = librosa.resample(target, orig_sr=sr, target_sr=SR_MIMO)
    return target.astype(np.float32)


def wav_to_mel(wav, mel_transform):
    spec = mel_transform(wav[None, :])
    log_mel = torch.log(torch.clip(spec, min=1e-7)).squeeze(0)
    return log_mel.transpose(0, 1)


@torch.no_grad()
def extract_pooled(model, mel_transform, wavs_batch, device):
    """Returns (B, 3*d_model) — mean, max, std concatenated."""
    mels = []
    lens = []
    for w in wavs_batch:
        wav_t = torch.from_numpy(w).to(device)
        mel = wav_to_mel(wav_t, mel_transform)
        mels.append(mel)
        lens.append(mel.shape[0])
    packed = torch.cat(mels, dim=0)
    input_lens = torch.tensor(lens, dtype=torch.long, device=device)
    h, hp, ol, _ = model.encode(packed, input_lens=input_lens, use_quantizer=False)
    feats = []
    for i, l in enumerate(ol.tolist()):
        if l == 0:
            d = h.shape[-1]
            feats.append(torch.zeros(3 * d, device=h.device))
            continue
        valid = h[i, :l].float()  # (T, D)
        mu = valid.mean(dim=0)
        mx = valid.max(dim=0).values
        sd = valid.std(dim=0)
        feats.append(torch.cat([mu, mx, sd]))
    return torch.stack(feats).cpu().numpy()


class MLPProbe(nn.Module):
    def __init__(self, in_dim, hidden=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp_probe(X_tr, y_tr, X_te, epochs=80, lr=1e-3, batch=64, device="cuda"):
    """Train binary MLP probe with class-balanced weights; return predicted probs on test."""
    pos_weight = float((len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1))
    model = MLPProbe(X_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    X_tr_t = torch.from_numpy(X_tr).float().to(device)
    y_tr_t = torch.from_numpy(y_tr).float().to(device)
    X_te_t = torch.from_numpy(X_te).float().to(device)
    n = len(X_tr_t)
    for ep in range(epochs):
        idx = torch.randperm(n)
        for s in range(0, n, batch):
            sl = idx[s:s + batch]
            out = model(X_tr_t[sl])
            l = loss_fn(out, y_tr_t[sl])
            opt.zero_grad(); l.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(X_te_t)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soundscape_dir", default="data/train_soundscapes")
    ap.add_argument("--soundscape_csv", default="data/train_soundscapes_labels.csv")
    ap.add_argument("--taxonomy", default="data/taxonomy.csv")
    ap.add_argument("--model_id", default="XiaomiMiMo/MiMo-Audio-Tokenizer")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--out_features", default="cache/mimo_v190b_features.npz")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mlp", action="store_true", help="Also train MLP probe")
    args = ap.parse_args()

    tax = pd.read_csv(args.taxonomy)
    label_cols = tax["primary_label"].astype(str).tolist()
    l2i = {c: i for i, c in enumerate(label_cols)}

    sc = pd.read_csv(args.soundscape_csv)
    sc["end_sec"] = pd.to_timedelta(sc["end"]).dt.total_seconds().astype(int)

    def parse_lbl(x):
        if pd.isna(x) or x == "nan":
            return set()
        return set(t.strip() for t in str(x).split(";") if t.strip())

    sc["label_set"] = sc["primary_label"].apply(parse_lbl)
    grouped = sc.groupby(["filename", "end_sec"])["label_set"].apply(
        lambda s: set().union(*s)).reset_index()
    audio_dir = Path(args.soundscape_dir)
    existing = set(p.name for p in audio_dir.glob("*.ogg"))
    grouped = grouped[grouped["filename"].isin(existing)].reset_index(drop=True)
    print(f"Chunks: {len(grouped)}")

    out_path = Path(args.out_features)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"Loading cached features from {out_path}")
        z = np.load(out_path)
        X = z["X"]; Y = z["Y"]; groups = z["groups"]
    else:
        print(f"Loading {args.model_id} on {args.device}...")
        t0 = time.time()
        model = load_mimo_tokenizer(args.model_id)
        model.eval().to(args.device)
        print(f"  loaded in {time.time() - t0:.1f}s")

        from torchaudio.transforms import MelSpectrogram
        mel_transform = MelSpectrogram(
            sample_rate=model.config.sampling_rate,
            n_fft=model.config.nfft, hop_length=model.config.hop_length,
            win_length=model.config.window_size,
            f_min=model.config.fmin, f_max=model.config.fmax,
            n_mels=model.config.n_mels, power=1.0, center=True,
        ).to(args.device)

        print("\nExtracting MiMo (mean+max+std pooled)...")
        t0 = time.time()
        feats_list = []
        labels_list = []
        groups_list = []
        records = grouped.to_dict("records")
        for batch_start in range(0, len(records), args.batch_size):
            batch = records[batch_start:batch_start + args.batch_size]
            wavs = []
            keep_idx = []
            for j, r in enumerate(batch):
                try:
                    w = load_chunk_wav(r["filename"], int(r["end_sec"]), audio_dir)
                except Exception:
                    continue
                wavs.append(w); keep_idx.append(j)
            if not wavs:
                continue
            try:
                feats = extract_pooled(model, mel_transform, wavs, args.device)
            except Exception as e:
                print(f"  batch {batch_start} extraction error: {e}")
                continue
            for k, j in enumerate(keep_idx):
                r = batch[j]
                feats_list.append(feats[k])
                y = np.zeros(len(label_cols), dtype=np.float32)
                for lbl in r["label_set"]:
                    if lbl in l2i:
                        y[l2i[lbl]] = 1.0
                labels_list.append(y)
                groups_list.append(r["filename"])
            if (batch_start // args.batch_size) % 20 == 0:
                done = batch_start + len(batch)
                print(f"  {done}/{len(records)} chunks — {time.time() - t0:.0f}s")

        X = np.stack(feats_list)
        Y = np.stack(labels_list)
        groups = np.array(groups_list)
        np.savez(out_path, X=X, Y=Y, groups=groups)
        print(f"  saved features: X={X.shape}  time {time.time() - t0:.0f}s")
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # Phase: 5-fold GroupKFold per-class LR on richer features
    print("\nPhase 3: 5-fold GroupKFold LR on mean+max+std features...")
    gkf = GroupKFold(n_splits=5)
    auc_folds_lr = []
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, Y, groups)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        Y_tr, Y_te = Y[tr_idx], Y[te_idx]
        valid = (Y_tr.sum(axis=0) >= 3) & (Y_te.sum(axis=0) >= 1)
        if valid.sum() == 0:
            continue
        mu = X_tr.mean(axis=0, keepdims=True)
        sd = X_tr.std(axis=0, keepdims=True) + 1e-6
        X_tr_n = (X_tr - mu) / sd
        X_te_n = (X_te - mu) / sd
        scores = np.zeros((len(Y_te), len(Y_te[0])), dtype=np.float32)
        for ci in np.where(valid)[0]:
            try:
                lr = LogisticRegression(C=1.0, max_iter=300, solver="liblinear",
                                         class_weight="balanced")
                lr.fit(X_tr_n, Y_tr[:, ci])
                scores[:, ci] = lr.predict_proba(X_te_n)[:, 1]
            except Exception:
                pass
        try:
            auc = roc_auc_score(Y_te[:, valid], scores[:, valid], average="macro")
        except ValueError:
            auc = 0.0
        n_classes = int(valid.sum())
        print(f"  fold {fold}: AUC={auc:.4f} on {n_classes} classes")
        auc_folds_lr.append(auc)
    if auc_folds_lr:
        print(f"\n=== V190b LR (mean+max+std, 3840-d) AUC: "
              f"{np.mean(auc_folds_lr):.4f} ± {np.std(auc_folds_lr):.4f} ===")

    if args.mlp and torch.cuda.is_available():
        print("\nPhase 4: 5-fold GroupKFold MLP probe (binary per class)...")
        auc_folds_mlp = []
        for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, Y, groups)):
            X_tr, X_te = X[tr_idx], X[te_idx]
            Y_tr, Y_te = Y[tr_idx], Y[te_idx]
            valid = (Y_tr.sum(axis=0) >= 3) & (Y_te.sum(axis=0) >= 1)
            if valid.sum() == 0:
                continue
            mu = X_tr.mean(axis=0, keepdims=True)
            sd = X_tr.std(axis=0, keepdims=True) + 1e-6
            X_tr_n = ((X_tr - mu) / sd).astype(np.float32)
            X_te_n = ((X_te - mu) / sd).astype(np.float32)
            scores = np.zeros((len(Y_te), len(Y_te[0])), dtype=np.float32)
            for ci in np.where(valid)[0]:
                scores[:, ci] = train_mlp_probe(X_tr_n, Y_tr[:, ci], X_te_n,
                                                 device="cuda")
            try:
                auc = roc_auc_score(Y_te[:, valid], scores[:, valid], average="macro")
            except ValueError:
                auc = 0.0
            n_classes = int(valid.sum())
            print(f"  fold {fold}: MLP AUC={auc:.4f} on {n_classes} classes")
            auc_folds_mlp.append(auc)
        if auc_folds_mlp:
            print(f"\n=== V190b MLP (mean+max+std, 3840-d) AUC: "
                  f"{np.mean(auc_folds_mlp):.4f} ± {np.std(auc_folds_mlp):.4f} ===")


if __name__ == "__main__":
    main()
