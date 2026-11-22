"""Diarization: adaptive smoothed-energy VAD with morphological cleanup, MFCC window
clustering (Ward + silhouette) for speaker labels, frame-level output segmentation."""
import numpy as np
from scipy.fft import dct
from scipy.ndimage import binary_closing, binary_opening, uniform_filter1d
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


def _vad(energy):
    e = uniform_filter1d(energy, 5)
    lo, hi = np.percentile(e, 10), np.percentile(e, 95)
    sp = e > lo + 0.3 * (hi - lo)
    sp = binary_closing(sp, np.ones(15, bool))
    sp = binary_opening(sp, np.ones(10, bool))
    return sp


def diarize(samples, rate):
    feats, energy = _mfcc(samples, rate)
    speech = _vad(energy)
    win, step = 150, 75
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
    best, best_k, best_lab = None, 2, None
    for k in range(2, min(7, len(E))):
        lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(E)
        sc = silhouette_score(E, lab)
        if best is None or sc > best:
            best, best_k, best_lab = sc, k, lab
    K = best_k
    votes = np.zeros((len(feats), K))
    for s, l in zip(starts, best_lab):
        votes[s:s + win, l] += 1
    init = np.where(votes.sum(1) > 0, votes.argmax(1), -1)

    idx = np.where(speech)[0]
    if len(idx) < 50:
        return []
    lab = init[idx]
    path = np.where(lab < 0, 0, lab)
    out = []
    n = len(idx)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and path[j + 1] == path[i] and idx[j + 1] - idx[j] <= 30:
            j += 1
        if idx[j] - idx[i] >= 15:
            out.append((idx[i] * HOP, idx[j] * HOP + WIN, "spk%d" % path[i]))
        i = j + 1
    return out
