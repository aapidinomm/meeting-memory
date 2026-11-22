"""Baseline speaker diarization: energy VAD, MFCC window embeddings, agglomerative clustering.

diarize(samples, rate) -> list of (start_s, end_s, speaker_label)
Only numpy, scipy and scikit-learn.
"""
import numpy as np
from scipy.fft import dct
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

HOP = 0.01
WIN = 0.025


def _mel_filters(rate, n_fft, n_mels=40):
    def hz2mel(h):
        return 2595 * np.log10(1 + h / 700.0)

    def mel2hz(m):
        return 700 * (10 ** (m / 2595.0) - 1)

    pts = mel2hz(np.linspace(hz2mel(60), hz2mel(rate / 2 - 200), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / rate).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        for k in range(l, c):
            fb[i - 1, k] = (k - l) / max(1, c - l)
        for k in range(c, r):
            fb[i - 1, k] = (r - k) / max(1, r - c)
    return fb


def _mfcc(x, rate):
    hop, win = int(HOP * rate), int(WIN * rate)
    n_fft = 512
    n = 1 + max(0, (len(x) - win) // hop)
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx] * np.hamming(win)
    spec = np.abs(np.fft.rfft(frames, n_fft)) ** 2
    logmel = np.log(spec @ _mel_filters(rate, n_fft).T + 1e-10)
    energy = np.log(np.sum(frames ** 2, axis=1) + 1e-10)
    return dct(logmel, type=2, axis=1, norm="ortho")[:, 1:20], energy


def diarize(samples, rate):
    feats, energy = _mfcc(samples, rate)
    speech = energy > np.percentile(energy, 30) + 0.5 * (np.percentile(energy, 95) - np.percentile(energy, 30)) * 0.3
    win, step = 150, 75                      # 1.5 s windows, 0.75 s step (in 10 ms frames)
    starts, embs = [], []
    for s in range(0, len(feats) - win + 1, step):
        m = speech[s:s + win]
        if m.mean() < 0.5:
            continue
        f = feats[s:s + win][m]
        embs.append(np.concatenate([f.mean(0), f.std(0)]))
        starts.append(s)
    if len(embs) < 3:
        return [(0.0, len(samples) / rate, "spk0")] if speech.any() else []
    E = StandardScaler().fit_transform(np.array(embs))
    best, best_k = None, 2
    for k in range(2, min(7, len(E))):
        lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(E)
        sc = silhouette_score(E, lab)
        if best is None or sc > best:
            best, best_k, best_lab = sc, k, lab
    votes = np.zeros((len(feats), best_k))
    for s, l in zip(starts, best_lab):
        votes[s:s + win, l] += 1
    frame_lab = np.where(votes.sum(1) > 0, votes.argmax(1), -1)
    frame_lab[~speech] = -1
    out, cur, beg = [], None, 0
    for i, l in enumerate(list(frame_lab) + [-1]):
        if l != cur:
            if cur is not None and cur >= 0 and i - beg >= 20:
                out.append((beg * HOP, i * HOP + WIN, "spk%d" % cur))
            cur, beg = l, i
    return out
