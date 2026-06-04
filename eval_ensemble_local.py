"""Local ensemble gate (1080 Ti) — spend-free. Does the 4-arch diverse SED ensemble
GAIN over the best single on the held-out 13 soundscape files? Plus leak-biased teacher ref."""
import os, glob, numpy as np, pandas as pd, soundfile as sf, librosa, torch, torch.nn as nn, timm
import onnxruntime as ort
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

DATA = "data"; CKPTS = "sed_ens_ckpts"
SR = 32000; WS = SR * 5
N_MELS, N_FFT, HOP, FMIN, FMAX, N_TIME = 256, 2048, 512, 20, 16000, 313
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TAGS = ["effv2s", "convnext", "effb3", "seresnext"]


def make_mel(wav):
    s = librosa.feature.melspectrogram(y=wav, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0)
    s = librosa.power_to_db(s, top_db=80); s = (s - s.mean()) / (s.std() + 1e-6)
    if s.shape[-1] < N_TIME: s = np.pad(s, ((0, 0), (0, N_TIME - s.shape[-1])))
    else: s = s[:, :N_TIME]
    return s.astype(np.float32)


def parse_t(t):
    p = str(t).split(":"); return int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2])


class SEDModel(nn.Module):
    def __init__(self, bn, n=234):
        super().__init__()
        self.bk = timm.create_model(bn, pretrained=False, num_classes=0, global_pool="", in_chans=1)
        d = self.bk.num_features
        self.gem_p = nn.Parameter(torch.tensor(3.0)); self.drop = nn.Dropout(0.3)
        self.frame_head = nn.Linear(d, n); self.att_head = nn.Linear(d, n)

    def forward(self, x):
        f = self.bk(x); p = self.gem_p.clamp(min=1.0)
        ff = (f.clamp(min=1e-6).pow(p)).mean(dim=2).pow(1.0 / p); ft = ff.transpose(1, 2)
        fr = self.frame_head(ft); at = torch.softmax(self.att_head(ft), dim=1)
        return (at * fr).sum(dim=1), fr


def main():
    tax = pd.read_csv(f"{DATA}/taxonomy.csv")
    PRIMARY = sorted(tax["primary_label"].astype(str).tolist()); l2i = {l: i for i, l in enumerate(PRIMARY)}; NC = len(PRIMARY)
    cks = {t: torch.load(f"{CKPTS}/fold0_best_{t}.pt", map_location="cpu") for t in TAGS if os.path.exists(f"{CKPTS}/fold0_best_{t}.pt")}
    val_files = set(next(iter(cks.values()))["val_files"])
    print(f"held-out files: {len(val_files)} | models: {[(t, round(cks[t]['heldout_auc'],4)) for t in cks]}", flush=True)

    lab = pd.read_csv(f"{DATA}/train_soundscapes_labels.csv"); lab["primary_label"] = lab["primary_label"].astype(str)
    rows = []
    for _, r in lab.iterrows():
        fn = str(r["filename"])
        if fn not in val_files: continue
        yv = np.zeros(NC, np.float32)
        for s in str(r["primary_label"]).split(";"):
            s = s.strip()
            if s in l2i: yv[l2i[s]] = 1.0
        rows.append((fn, parse_t(r["start"]), parse_t(r["end"]), yv))
    YY = np.stack([r[3] for r in rows])
    wav_cache = {}

    def seg_mel(fn, st, en):
        if fn not in wav_cache:
            w, sr = sf.read(f"{DATA}/train_soundscapes/{fn}", dtype="float32", always_2d=False)
            if w.ndim > 1: w = w.mean(1)
            if sr != SR: w = librosa.resample(w, orig_sr=sr, target_sr=SR)
            wav_cache[fn] = w
        w = wav_cache[fn]; sg = w[int(st * SR):int(en * SR)]
        sg = np.pad(sg, (0, WS - len(sg))) if len(sg) < WS else sg[:WS]
        return make_mel(sg)

    preds = {}
    for t, ck in cks.items():
        m = SEDModel(ck["backbone"], NC).to(DEVICE); m.load_state_dict(ck["state"]); m.eval()
        P = np.zeros((len(rows), NC), np.float32)
        with torch.no_grad():
            for i, (fn, st, en, yv) in enumerate(rows):
                x = torch.from_numpy(seg_mel(fn, st, en))[None, None].to(DEVICE)
                clip, _ = m(x.float()); P[i] = torch.sigmoid(clip).float().cpu().numpy()
        preds[t] = P; del m
        if DEVICE == "cuda": torch.cuda.empty_cache()

    v = (YY > 0.5).sum(0) > 0; yy = (YY[:, v] > 0.5).astype(int)
    def macro(P): return roc_auc_score(yy, P[:, v], average="macro")
    print(f"\n=== held-out ({int(v.sum())} classes w/ positives) ===")
    singles = {t: macro(preds[t]) for t in preds}
    for t in singles: print(f"  single {t}: {singles[t]:.4f}")
    best_single = max(singles.values())
    rk = sum(np.apply_along_axis(rankdata, 0, preds[t]) for t in preds) / len(preds)
    ens = macro(rk)
    print(f"=== ENSEMBLE ({len(preds)} diverse archs): {ens:.4f} ===")
    print(f"=== best single: {best_single:.4f}  ->  ensemble gain: {ens-best_single:+.4f} ===")

    # public SED teacher (leak-biased ref)
    def sig(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
    sess = [ort.InferenceSession(f"public_sed/sed_fold{f}.onnx", providers=["CPUExecutionProvider"]) for f in range(5)]
    inp = sess[0].get_inputs()[0].name
    T = np.zeros((len(rows), NC), np.float32)
    for i, (fn, st, en, yv) in enumerate(rows):
        x = seg_mel(fn, st, en)[None, None]; ps = np.zeros(NC, np.float32)
        for s_ in sess:
            o = s_.run(None, {inp: x}); clip = o[0][0]
            fm = o[1][0].max(0) if len(o) > 1 and o[1].ndim == 3 else clip
            ps += 0.5 * sig(clip) + 0.5 * sig(fm)
        T[i] = ps / len(sess)
    taiuc = macro(T)
    print(f"=== public SED teacher (LEAK-biased): {taiuc:.4f} ===")
    rt = np.apply_along_axis(rankdata, 0, T)
    for w in [0.2, 0.3, 0.4, 0.5]:
        print(f"  teacher+ourEnsemble blend w_ours={w}: {macro((1-w)*rt + w*rk):.4f} ({macro((1-w)*rt+w*rk)-taiuc:+.4f})")
    print(f"\n=== VERDICT: {'diverse-ensemble GAINS over single' if ens > best_single + 0.005 else 'no ensemble gain over single'} ===")


if __name__ == "__main__":
    main()
