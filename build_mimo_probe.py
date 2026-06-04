"""V190 MVT: MiMo-Audio-Tokenizer pre-RVQ encoder embeddings as per-species probes.

Hypothesis: MiMo's frozen encoder (trained at LM-loss-driven 1.2B Transformer
on speech+general audio at 24kHz with 25Hz output) produces continuous
representations whose linear probe on labeled soundscape carries species
signal — testing whether the speech-tokenizer encoder transfers to bird audio.

This is option 1 from the MiMo research findings: a low-cost MVT that decides
whether option 3 (3-5 day classification-gradient VQ-VAE) is justified.

PREDICTION (Predict-Then-Run):
  - Held-out macro AUC: 0.65-0.80 (medium confidence)
  - Direction: weaker than bird-specialty probes (Bird-MAE ~0.85), but should
    show signal because 1280-dim encoder features have enough capacity
  - Confidence: medium for direction, low for magnitude

  Caveat (informed by V188 disconfirm): even a high AUC here does NOT
  guarantee LB transfer. This MVT ONLY filters out the case where MiMo
  features carry no signal at all. If AUC ≥ 0.80, option 3 (gradient-trained
  VQ-VAE) becomes plausible. Below 0.65, abandon MiMo direction entirely.

DISCONFIRM (kill the direction):
  - AUC < 0.65 → speech-domain encoder doesn't transfer to bird audio
                 → no point investing in MiMo-style architecture

CONFIRM (worth option 3 investment):
  - AUC > 0.80 → encoder-level transfer exists; gradient-trained VQ on bird
                 audio could plausibly produce useful tokens

AMBIGUOUS (0.65-0.80):
  - Marginal signal; MiMo as-is won't help, but a domain-adapted version
    might. Decide based on cost-of-pursuit vs other directions.

Cost: ~1-2h CPU/GPU (model load + 1500 chunk forward passes).
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
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


# --- flash_attn shim (Pascal GPU lacks SM 7.5 needed for real flash-attn) ---
def _flash_attn_varlen_func_sdpa(q, k, v, cu_q, cu_k, max_q, max_k,
                                  causal=False, window_size=(-1, -1), **kwargs):
    """Drop-in replacement for flash_attn_varlen_func using torch SDPA.

    q, k, v: (total_tokens, num_heads, head_dim)
    cu_q, cu_k: int32 (B+1,) cumulative seq lens
    Returns: (total_tokens, num_heads, head_dim)
    """
    bsz = cu_q.shape[0] - 1
    H = q.shape[1]
    D = q.shape[2]
    seq_lens = (cu_q[1:] - cu_q[:-1]).tolist()
    if all(s == seq_lens[0] for s in seq_lens):
        T = seq_lens[0]
        # (B*T, H, D) → (B, T, H, D) → (B, H, T, D)
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

# Add MiMo source to path so we can import the tokenizer module without pip install
sys.path.insert(0, "/tmp/MiMo-Audio/src")
from mimo_audio_tokenizer import MiMoAudioTokenizer, MiMoAudioTokenizerConfig  # noqa: E402


def load_mimo_tokenizer(model_id: str) -> MiMoAudioTokenizer:
    """Workaround for transformers<4.49 not handling missing safetensors
    metadata: instantiate from config, then load state_dict manually.
    """
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    repo_path = snapshot_download(model_id)
    cfg = MiMoAudioTokenizerConfig.from_pretrained(repo_path)
    model = MiMoAudioTokenizer(cfg)
    state = load_file(f"{repo_path}/model.safetensors")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  load_state_dict missing keys: {len(missing)} (first 3: {missing[:3]})")
    if unexpected:
        print(f"  load_state_dict unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")
    return model

SR_NATIVE = 32000
SR_MIMO = 24000
CHUNK_SECONDS = 5
N_FFT = 960
HOP_LENGTH = 240
WIN_LENGTH = 960
N_MELS = 128
FMIN = 0
FMAX = None  # config has fmax=null → use Nyquist (12kHz)


def load_chunk_wav(filename, end_sec, audio_dir):
    """Load 5s audio chunk, mono, resampled to MiMo's 24kHz."""
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


def wav_to_mel(wav: torch.Tensor, mel_transform) -> torch.Tensor:
    """wav (T,) → log-mel (T_mel, n_mels) matching MiMo wav2mel."""
    spec = mel_transform(wav[None, :])  # (1, n_mels, T_mel)
    log_mel = torch.log(torch.clip(spec, min=1e-7)).squeeze(0)  # (n_mels, T_mel)
    return log_mel.transpose(0, 1)  # (T_mel, n_mels)


@torch.no_grad()
def extract_features(model, mel_transform, wavs_batch, device):
    """wavs_batch: list of np.float32 arrays of shape (T_samples,) at 24kHz.

    Returns (B, d_model) mean-pooled encoder features (use_quantizer=False).
    """
    mels = []
    lens = []
    for w in wavs_batch:
        wav_t = torch.from_numpy(w).to(device)
        mel = wav_to_mel(wav_t, mel_transform)  # (T_mel, n_mels)
        mels.append(mel)
        lens.append(mel.shape[0])
    # Pack into a single (sum_T, n_mels) tensor; encoder unpacks via input_lens
    packed = torch.cat(mels, dim=0)
    input_lens = torch.tensor(lens, dtype=torch.long, device=device)
    hidden_states, hidden_states_packed, output_length, codes = model.encode(
        packed, input_lens=input_lens, use_quantizer=False
    )
    # hidden_states: (B, T_out, d_model) with attention_mask zeros for padding
    # Mean-pool over valid time steps
    feats = []
    for i, ol in enumerate(output_length.tolist()):
        if ol == 0:
            feats.append(torch.zeros(hidden_states.shape[-1], device=hidden_states.device))
            continue
        feats.append(hidden_states[i, :ol].float().mean(dim=0))
    return torch.stack(feats).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soundscape_dir", default="data/train_soundscapes")
    ap.add_argument("--soundscape_csv", default="data/train_soundscapes_labels.csv")
    ap.add_argument("--taxonomy", default="data/taxonomy.csv")
    ap.add_argument("--model_id", default="XiaomiMiMo/MiMo-Audio-Tokenizer")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out_features", default="cache/mimo_v190_features.npz")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # Phase 0: alignment
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
        lambda s: set().union(*s)
    ).reset_index()

    audio_dir = Path(args.soundscape_dir)
    existing = set(p.name for p in audio_dir.glob("*.ogg"))
    grouped = grouped[grouped["filename"].isin(existing)].reset_index(drop=True)
    print(f"Chunks: {len(grouped)}")

    # Phase 1: load model
    print(f"Loading {args.model_id} on {args.device}...")
    t0 = time.time()
    model = load_mimo_tokenizer(args.model_id)
    model.eval().to(args.device)
    # Note: 1080 Ti is Pascal SM 6.1 — no native bf16/fp16 tensor cores.
    # Stay in fp32 for stability; SDPA on Pascal uses math kernel.
    print(f"  loaded in {time.time() - t0:.1f}s; d_model={model.config.d_model}")

    from torchaudio.transforms import MelSpectrogram
    mel_transform = MelSpectrogram(
        sample_rate=model.config.sampling_rate,
        n_fft=model.config.nfft,
        hop_length=model.config.hop_length,
        win_length=model.config.window_size,
        f_min=model.config.fmin,
        f_max=model.config.fmax,
        n_mels=model.config.n_mels,
        power=1.0,
        center=True,
    ).to(args.device)

    # Phase 2: extract features
    out_path = Path(args.out_features)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"Loading cached features from {out_path}")
        z = np.load(out_path)
        X = z["X"]
        Y = z["Y"]
        groups = z["groups"]
    else:
        print("\nPhase 2: Extract MiMo encoder features (use_quantizer=False)...")
        t0 = time.time()
        feats_list = []
        labels_list = []
        groups_list = []
        skipped = 0

        records = grouped.to_dict("records")
        for batch_start in range(0, len(records), args.batch_size):
            batch = records[batch_start:batch_start + args.batch_size]
            wavs = []
            keep_idx = []
            for j, r in enumerate(batch):
                try:
                    w = load_chunk_wav(r["filename"], int(r["end_sec"]), audio_dir)
                except Exception as e:
                    skipped += 1
                    continue
                wavs.append(w)
                keep_idx.append(j)
            if not wavs:
                continue
            try:
                feats = extract_features(model, mel_transform, wavs, args.device)
            except Exception as e:
                print(f"  batch {batch_start} extraction error: {type(e).__name__} {e}")
                skipped += len(wavs)
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
                print(f"  {done}/{len(records)} chunks — {time.time() - t0:.0f}s "
                      f"(skipped: {skipped})")

        X = np.stack(feats_list)
        Y = np.stack(labels_list)
        groups = np.array(groups_list)
        np.savez(out_path, X=X, Y=Y, groups=groups)
        print(f"  saved features: X={X.shape}, Y={Y.shape}, time {time.time() - t0:.0f}s")

    # Free model from VRAM/RAM before LR
    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()

    # Phase 3: 5-fold GroupKFold per-species LR
    print("\nPhase 3: 5-fold GroupKFold LR over MiMo features...")
    gkf = GroupKFold(n_splits=5)
    auc_folds = []
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, Y, groups)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        Y_tr, Y_te = Y[tr_idx], Y[te_idx]
        valid = (Y_tr.sum(axis=0) >= 3) & (Y_te.sum(axis=0) >= 1)
        if valid.sum() == 0:
            continue
        # Standardize features for LR stability
        mu = X_tr.mean(axis=0, keepdims=True)
        sd = X_tr.std(axis=0, keepdims=True) + 1e-6
        X_tr_n = (X_tr - mu) / sd
        X_te_n = (X_te - mu) / sd
        scores = np.zeros((len(Y_te), len(Y_te[0])), dtype=np.float32)
        for ci in np.where(valid)[0]:
            try:
                lr = LogisticRegression(
                    C=1.0, max_iter=300, solver="liblinear",
                    class_weight="balanced",
                )
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
        auc_folds.append(auc)

    if auc_folds:
        mean_auc = float(np.mean(auc_folds))
        std_auc = float(np.std(auc_folds))
        print("\n=== V190 MIMO MVT RESULT ===")
        print(f"Mean macro AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        print("Decision rules:")
        if mean_auc < 0.65:
            print(f"  ABANDON: AUC {mean_auc:.4f} < 0.65 → MiMo encoder fails to "
                  "transfer to bird audio. Skip option 3.")
        elif mean_auc < 0.80:
            print(f"  AMBIGUOUS: AUC {mean_auc:.4f} ∈ [0.65, 0.80) → marginal "
                  "transfer. Domain-adapted MiMo might work, but other directions "
                  "likely higher EV.")
        else:
            print(f"  CONFIRMED: AUC {mean_auc:.4f} ≥ 0.80 → MiMo encoder transfers. "
                  "Option 3 (classification-gradient VQ-VAE) is justified.")


if __name__ == "__main__":
    main()
