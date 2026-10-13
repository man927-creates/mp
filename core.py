"""Core pipeline for the ML-based NIDS.

* Synthetic traffic is generated in-process, so no dataset download is needed.
* Models: XGBoost (detection + attack type), LSTM, Autoencoder, weighted ensemble.
* Explanations: XGBoost SHAP (pred_contribs) when XGBoost is available,
  otherwise a sensitivity analysis that gives the same kind of per-feature view.
* Optional dependencies (xgboost, tensorflow, scapy) never crash the app; the
  backend that is actually used is reported to the dashboard.
"""
import os
import threading
import time
import datetime as dt
from collections import defaultdict, deque

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import xgboost as xgb
    HAS_XGB, XGB_ERR = True, ""
except Exception as exc:  # missing package or broken DLL: fall back, do not crash
    xgb, HAS_XGB, XGB_ERR = None, False, f"{type(exc).__name__}: {exc}"

try:
    import tensorflow as tf
    from tensorflow import keras # type: ignore
    HAS_TF, TF_ERR = True, ""
except Exception as exc:
    tf, keras, HAS_TF, TF_ERR = None, None, False, f"{type(exc).__name__}: {exc}"

try:
    from scapy.all import IP, TCP, sniff
    HAS_SCAPY, SCAPY_ERR = True, ""
except Exception as exc:
    IP = TCP = sniff = None
    HAS_SCAPY, SCAPY_ERR = False, f"{type(exc).__name__}: {exc}"


T = 5                      # windows per sequence
WIN_SEC = 2.0              # seconds per traffic window (live capture)
FEATURES = ["Packet count", "Byte count", "Packet rate", "Avg packet size",
            "SYN ratio", "Unique ports", "Unique destinations"]
F = len(FEATURES)
CLASSES = ["Normal", "DDoS", "PortScan", "BruteForce"]
W_TREE, W_LSTM, W_AE = 0.40, 0.35, 0.25   # ensemble weights
SUSP_T, MAL_T = 35, 65                    # risk thresholds (0-100)
HOSTS = [f"192.168.1.{i}" for i in range(10, 16)]


def to_vec(pkts, byts, syn_ratio, ports, dsts):
    pkts = max(float(pkts), 0.0)
    return [np.log1p(pkts), np.log1p(byts), pkts / WIN_SEC, byts / max(pkts, 1.0),
            float(syn_ratio), np.log1p(ports), np.log1p(dsts)]


# ---------------------------------------------------------------- synthetic data
def gen_window(kind, rng, hard=False):
    # Some windows are deliberately ambiguous (bursty normal traffic, stealthy attacks)
    # so the metrics are realistic instead of trivially perfect.
    if kind == "Normal":
        if hard:                         # bursty but benign (e.g. file download)
            p = int(rng.integers(300, 700))
            a = np.clip(rng.normal(900, 150), 300, 1500)
            syn, ports, dsts = rng.beta(2, 12), rng.integers(1, 4), rng.integers(1, 3)
        else:
            p = max(5, int(rng.poisson(60)))
            a = np.clip(rng.normal(550, 120), 60, 1500)
            syn, ports, dsts = rng.beta(1, 20), rng.integers(1, 6), rng.integers(1, 7)
    elif kind == "DDoS":
        if hard:                         # low-rate DoS that looks like heavy normal traffic
            p = int(rng.integers(250, 500))
            a = np.clip(rng.normal(200, 60), 60, 600)
            syn, ports, dsts = rng.beta(10, 4), rng.integers(1, 3), rng.integers(1, 3)
            return to_vec(p, p * a, syn, ports, dsts)
        p = int(rng.integers(800, 2500))
        a = np.clip(rng.normal(120, 30), 40, 300)
        syn, ports, dsts = rng.beta(40, 2), rng.integers(1, 3), rng.integers(1, 3)
    elif kind == "PortScan":
        slow = hard                      # slow scan: few ports per window
        p = int(rng.integers(20, 60)) if slow else int(rng.integers(60, 160))
        a = np.clip(rng.normal(60, 10), 40, 120)
        syn, ports, dsts = rng.beta(30, 2), (rng.integers(5, 20) if slow else rng.integers(30, 250)), 1
    else:  # BruteForce (repeated SSH logins to one port)
        p = int(rng.integers(40, 120))
        a = np.clip(rng.normal(180, 40), 80, 400)
        syn, ports, dsts = (rng.beta(1, 8) if hard else rng.beta(3, 6)), 1, 1
    return to_vec(p, p * a, syn, ports, dsts)


def make_dataset(n, rng):
    y = rng.choice(len(CLASSES), size=n, p=[0.55, 0.15, 0.15, 0.15])
    hard = rng.random(n) < 0.20          # ~20% of sequences are ambiguous
    X = np.array([[gen_window(CLASSES[c], rng, hard[i]) for _ in range(T)]
                  for i, c in enumerate(y)])
    return X, y


# ---------------------------------------------------------------- model helpers
def fit_tree(X, y, binary):
    if HAS_XGB:
        if binary:
            m = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                  eval_metric="logloss", random_state=42)
        else:
            m = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                  objective="multi:softprob", num_class=len(CLASSES),
                                  eval_metric="mlogloss", random_state=42)
    else:
        m = HistGradientBoostingClassifier(max_iter=200, max_depth=4, random_state=42)
    return m.fit(X, y)


def fit_lstm(X3, y):
    """LSTM when TensorFlow works, otherwise a sequence MLP with the same input shape."""
    if HAS_TF:
        try:
            tf.random.set_seed(42)
            m = keras.Sequential([
                keras.layers.Input(shape=(T, F)),
                keras.layers.LSTM(32),
                keras.layers.Dense(16, activation="relu"),
                keras.layers.Dense(1, activation="sigmoid"),
            ])
            m.compile(optimizer="adam", loss="binary_crossentropy")
            m.fit(X3, y, epochs=8, batch_size=64, verbose=0)
            return "TensorFlow LSTM", (lambda Z: m.predict(Z, verbose=0).ravel())
        except Exception as exc:
            print(f"[warn] TensorFlow LSTM failed, using MLP fallback: {exc}")
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    clf.fit(X3.reshape(len(X3), -1), y)
    return ("MLP sequence model · TensorFlow unavailable",
            (lambda Z: clf.predict_proba(Z.reshape(len(Z), -1))[:, 1]))


def metric_row(name, y, p, thr):
    yp = (p >= thr).astype(int)
    acc = accuracy_score(y, yp)
    pr, rc, f1, _ = precision_recall_fscore_support(y, yp, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    spec = float(tn) / float(tn + fp) if (tn + fp) else 0.0
    return {"model": name, "Accuracy": round(acc * 100, 2), "Precision": round(pr * 100, 2),
            "Recall": round(rc * 100, 2), "F1-Score": round(f1 * 100, 2),
            "Specificity": round(spec * 100, 2),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def train_bundle(progress):
    """Train every model on synthetic traffic. `progress(stage, percent)` reports status."""
    t0 = time.time()
    rng = np.random.default_rng(42)

    progress("Generating synthetic traffic", 5)
    X, y = make_dataset(4000, rng)

    progress("Feature extraction & scaling", 15)
    flat = X.reshape(len(X), -1)
    scaler = StandardScaler().fit(flat)
    Xs = scaler.transform(flat)
    yb = (y > 0).astype(int)
    Xtr, Xte, ytr, yte, btr, bte = train_test_split(
        Xs, y, yb, test_size=0.2, random_state=42, stratify=y)

    progress("Training XGBoost (detection + attack type)", 30)
    tree_bin = fit_tree(Xtr, btr, binary=True)
    tree_multi = fit_tree(Xtr, ytr, binary=False)

    progress("Training LSTM", 55)
    lstm_name, lstm_pred = fit_lstm(Xtr.reshape(-1, T, F), btr)

    progress("Training Autoencoder on normal traffic", 75)
    normal = Xtr[ytr == 0]
    ae = MLPRegressor(hidden_layer_sizes=(24, 8, 24), max_iter=300, random_state=42)
    ae.fit(normal, normal)
    err_normal = ((ae.predict(normal) - normal) ** 2).mean(axis=1)
    ae_thr = float(np.percentile(err_normal, 95))

    progress("Evaluating models & ensemble", 90)
    p_tree = tree_bin.predict_proba(Xte)[:, 1]
    p_lstm = lstm_pred(Xte.reshape(-1, T, F))
    err = ((ae.predict(Xte) - Xte) ** 2).mean(axis=1)
    p_ae = np.clip(err / (2 * ae_thr), 0, 1)
    p_ens = W_TREE * p_tree + W_LSTM * p_lstm + W_AE * p_ae

    metrics = [
        metric_row("XGBoost" if HAS_XGB else "Tree model · fallback", bte, p_tree, 0.5),
        metric_row("LSTM", bte, p_lstm, 0.5),
        metric_row("Autoencoder", bte, p_ae, 0.5),
        metric_row("Ensemble", bte, p_ens, SUSP_T / 100),
    ]
    mask = yte > 0
    type_acc = accuracy_score(yte[mask], np.asarray(tree_multi.predict(Xte[mask])).astype(int))

    shap_method = ("XGBoost SHAP (pred_contribs)" if HAS_XGB
                   else "Sensitivity analysis · SHAP fallback")
    summary = {"samples": len(X), "train": len(Xtr), "test": len(Xte),
               "seconds": round(time.time() - t0, 1), "lstm": lstm_name,
               "attack_type_accuracy": round(type_acc * 100, 2),
               "explain": shap_method}

    progress("Ready", 100)
    return {"scaler": scaler, "bin": tree_bin, "multi": tree_multi,
            "lstm": lstm_pred, "lstm_name": lstm_name, "ae": ae, "ae_thr": ae_thr,
            "shap": shap_method, "metrics": metrics, "summary": summary,
            "bg": Xtr[rng.choice(len(Xtr), size=30, replace=False)]}


# ---------------------------------------------------------------- inference
def predict_batch(B, seqs):
    """seqs: raw array (n, T, F). Returns per-sequence risk, verdict and scores."""
    n = len(seqs)
    Xs = B["scaler"].transform(seqs.reshape(n, -1))
    p_tree = B["bin"].predict_proba(Xs)[:, 1]
    p_lstm = np.asarray(B["lstm"](Xs.reshape(n, T, F)), dtype=float)
    err = ((B["ae"].predict(Xs) - Xs) ** 2).mean(axis=1)
    p_ae = np.clip(err / (2 * B["ae_thr"]), 0, 1)
    p_ens = W_TREE * p_tree + W_LSTM * p_lstm + W_AE * p_ae
    risk = np.round(p_ens * 100, 1)
    verdict = np.where(risk >= MAL_T, "Malicious", np.where(risk >= SUSP_T, "Suspicious", "Normal"))
    attack = B["multi"].predict_proba(Xs)[:, 1:].argmax(axis=1) + 1
    return {"risk": risk, "verdict": verdict, "attack": attack,
            "p_tree": p_tree, "p_lstm": p_lstm, "p_ae": p_ae}


def explain(B, seq):
    """Per-feature contribution toward 'attack' for one sequence (raw, shape (T, F))."""
    x = B["scaler"].transform(seq.reshape(1, -1))
    if B["shap"].startswith("XGBoost"):
        contrib = B["bin"].get_booster().predict(xgb.DMatrix(x), pred_contribs=True)[0, :-1]
        vals = contrib.reshape(T, F).sum(axis=0)
    else:
        def logit(p):   # log-odds keep differences visible when probabilities saturate at 0/1
            p = np.clip(p, 1e-6, 1 - 1e-6)
            return np.log(p / (1 - p))
        bg = B["bg"]                       # background windows from the training data
        base = float(logit(B["bin"].predict_proba(x)[:, 1])[0])
        vals = np.zeros(F)
        for j in range(F):
            idx = [t * F + j for t in range(T)]
            x2 = np.repeat(x, len(bg), axis=0)
            x2[:, idx] = bg[:, idx]        # swap this feature for the value seen in background traffic
            vals[j] = base - float(np.mean(logit(B["bin"].predict_proba(x2)[:, 1])))
    order = np.argsort(-np.abs(vals))[:5]
    return {"method": B["shap"],
            "items": [{"feature": FEATURES[j], "value": round(float(vals[j]), 4)} for j in order]}


def backend_info(B=None):
    return {
        "classifier": "XGBoost" if HAS_XGB else "HistGradientBoosting · XGBoost unavailable",
        "sequence": B["lstm_name"] if B else ("TensorFlow LSTM" if HAS_TF else "MLP sequence model"),
        "explain": ("XGBoost SHAP" if HAS_XGB else "Sensitivity analysis · SHAP fallback"),
        "capture": "scapy live capture" if HAS_SCAPY else "unavailable · simulator only",
        "notes": [n for n in [
            "" if HAS_XGB else f"XGBoost not loaded: {XGB_ERR}",
            "" if HAS_TF else f"TensorFlow not loaded: {TF_ERR}",
            "" if HAS_SCAPY else f"scapy not loaded: {SCAPY_ERR}"] if n],
    }


# ---------------------------------------------------------------- traffic sources
def _sample_kind(rng):
    return str(rng.choice(CLASSES, p=[0.90, 0.035, 0.035, 0.03]))


class Simulator:
    """Emits one window per host per tick, with occasional attack episodes."""

    def __init__(self, seed=None, hosts=None):
        self.hosts = list(hosts) if hosts else list(HOSTS)
        self.rng = np.random.default_rng(seed)
        self.episode = {h: ("Normal", 0) for h in self.hosts}
        self.hard = {h: False for h in self.hosts}

    def step(self):
        out = []
        for h in self.hosts:
            kind, left = self.episode[h]
            if left <= 0:
                kind = _sample_kind(self.rng)
                left = int(self.rng.integers(4, 9)) if kind != "Normal" else 1
                self.hard[h] = bool(self.rng.random() < 0.2)
            self.episode[h] = (kind, left - 1)
            out.append((h, kind, gen_window(kind, self.rng, self.hard[h])))
        return out


class LiveCapture:
    """Sniffs packets with scapy and aggregates them per source IP every WIN_SEC seconds."""

    def __init__(self, iface=None, ip=None):
        self.iface = iface
        self.bpf = f"src host {ip}" if ip else None
        self.lock = threading.Lock()
        self.flows = self._new()
        self.stop_evt = threading.Event()
        self.thread = None
        self.error = ""

    @staticmethod
    def _new():
        return defaultdict(lambda: {"pk": 0, "by": 0, "syn": 0, "tcp": 0,
                                    "ports": set(), "dsts": set()})

    def _on_packet(self, pkt):
        if not pkt.haslayer(IP):
            return
        src, dst, size = pkt[IP].src, pkt[IP].dst, len(pkt)
        with self.lock:
            f = self.flows[src]
            f["pk"] += 1
            f["by"] += size
            f["dsts"].add(dst)
            if pkt.haslayer(TCP):
                flags = int(pkt[TCP].flags)
                f["tcp"] += 1
                if flags & 0x02 and not flags & 0x10:   # SYN without ACK
                    f["syn"] += 1
                f["ports"].add(int(pkt[TCP].dport))

    def _run(self):
        try:
            while not self.stop_evt.is_set():
                sniff(iface=self.iface, prn=self._on_packet, store=False, timeout=1, filter=self.bpf)
        except Exception as exc:
            self.error = (f"Live capture failed: {type(exc).__name__}: {exc}. Run as "
                          "Administrator and install Npcap on Windows or libpcap on Linux.")

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_evt.set()

    def drain(self):
        with self.lock:
            flows, self.flows = self.flows, self._new()
        out = []
        for ip, f in flows.items():
            if f["pk"] == 0:
                continue
            syn_ratio = f["syn"] / max(f["tcp"], 1)
            out.append((ip, to_vec(f["pk"], f["by"], syn_ratio, len(f["ports"]), len(f["dsts"]))))
        return out


# ---------------------------------------------------------------- monitor
MODEL_NAMES = ["XGBoost", "LSTM", "Autoencoder", "Ensemble"]


def _empty_cm():
    return {m: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for m in MODEL_NAMES}


def _update_cm(cm, truth, res, i, risk):
    """Add one labelled window to the running confusion counts of every model."""
    y = int(truth != "Normal")
    preds = {"XGBoost": res["p_tree"][i] >= 0.5, "LSTM": res["p_lstm"][i] >= 0.5,
             "Autoencoder": res["p_ae"][i] >= 0.5, "Ensemble": risk >= SUSP_T}
    for name, flag in preds.items():
        p = int(bool(flag))
        c = cm[name]
        if y and p:
            c["tp"] += 1
        elif not y and p:
            c["fp"] += 1
        elif y and not p:
            c["fn"] += 1
        else:
            c["tn"] += 1


def _cm_row(name, c):
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    n = tp + fp + fn + tn
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    acc = (tp + tn) / n if n else 0.0
    sp = tn / (tn + fp) if tn + fp else 0.0
    return {"model": name, "Accuracy": round(acc * 100, 2), "Precision": round(pr * 100, 2),
            "Recall": round(rc * 100, 2), "F1-Score": round(f1 * 100, 2),
            "Specificity": round(sp * 100, 2), "confusion": dict(c), "n": n}


class Monitor:
    def __init__(self, bundle):
        self.B = bundle
        self.lock = threading.Lock()
        self.stop_evt = None
        self.thread = None
        self.source = None
        self.target = None
        self.started = None
        self.error = ""
        self.buf = defaultdict(lambda: deque(maxlen=T))
        self.hosts = {}
        self.history = deque(maxlen=200)
        self.risk = deque(maxlen=60)
        self.counts = {"Normal": 0, "Suspicious": 0, "Malicious": 0}
        self.windows = 0
        self.explanation = None
        self.live_cm = _empty_cm()

    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self, source, ip=None):
        if self.running:
            raise RuntimeError("Monitoring is already running.")
        with self.lock:
            self.error = ""
            self.buf.clear()
            self.hosts.clear()
            self.history.clear()
            self.risk.clear()
            self.counts = {k: 0 for k in self.counts}
            self.windows = 0
            self.explanation = None
            self.live_cm = _empty_cm()
            self.source = source
            self.target = ip
            self.started = time.time()
        self.stop_evt = threading.Event()
        self.thread = threading.Thread(target=self._loop, args=(source, self.stop_evt, ip), daemon=True)
        self.thread.start()

    def stop(self):
        if self.stop_evt is not None:
            self.stop_evt.set()

    def _loop(self, source, stop, ip=None):
        sim = Simulator(hosts=[ip] if ip else None) if source == "simulator" else None
        cap = None
        if sim is None:
            cap = LiveCapture(ip=ip)
            cap.start()
        tick = 1.0 if sim else WIN_SEC
        try:
            while not stop.is_set():
                if sim:
                    items = [(h, kind, vec) for h, kind, vec in sim.step()]
                else:
                    if cap.error:
                        raise RuntimeError(cap.error)
                    items = [(h, None, vec) for h, vec in cap.drain()]
                self._process(items)
                stop.wait(tick)
        except Exception as exc:
            print(f"[monitor] stopped with error: {exc}")
            with self.lock:
                self.error = f"{type(exc).__name__}: {exc}"
        finally:
            if cap is not None:
                cap.stop()

    def _process(self, items):
        now = dt.datetime.now().strftime("%H:%M:%S")
        ready = []
        with self.lock:
            for host, truth, vec in items:
                self.buf[host].append(vec)
                h = self.hosts.setdefault(host, {"host": host, "windows": 0, "verdict": "Normal",
                                                 "risk": 0.0})
                h["windows"] += 1
                if len(self.buf[host]) == T:
                    ready.append((host, truth, np.array(self.buf[host])))
        if not ready:
            return

        seqs = np.stack([r[2] for r in ready])
        res = predict_batch(self.B, seqs)

        top = None
        with self.lock:
            tick_max = 0.0
            for i, (host, truth, _) in enumerate(ready):
                verdict = str(res["verdict"][i])
                risk = float(res["risk"][i])
                if truth is not None:
                    _update_cm(self.live_cm, truth, res, i, risk)
                attack = CLASSES[int(res["attack"][i])] if verdict != "Normal" else "None"
                self.windows += 1
                self.counts[verdict] += 1
                tick_max = max(tick_max, risk)
                self.hosts[host].update(verdict=verdict, risk=risk)
                if verdict != "Normal":
                    self.history.appendleft({
                        "time": now, "host": host, "verdict": verdict, "attack": attack,
                        "risk": round(risk, 1), "truth": truth or "—",
                        "scores": {"XGBoost": round(float(res["p_tree"][i]) * 100, 1),
                                   "LSTM": round(float(res["p_lstm"][i]) * 100, 1),
                                   "Autoencoder": round(float(res["p_ae"][i]) * 100, 1)},
                    })
                    if top is None or risk > top[0]:
                        top = (risk, i)
            self.risk.append(tick_max)

        if top is not None:
            risk, i = top
            expl = explain(self.B, seqs[i])
            with self.lock:
                self.explanation = {"host": ready[i][0], "risk": round(risk, 1), **expl}

    def snapshot(self):
        with self.lock:
            risks = list(self.risk)
            alerts = self.counts["Suspicious"] + self.counts["Malicious"]
            w = self.windows
            return {
                "running": self.running,
                "source": self.source,
                "target": self.target,
                "error": self.error,
                "started": self.started,
                "stats": {"windows": w, **self.counts, "hosts": len(self.hosts),
                          "avg_risk": round(sum(risks) / len(risks), 1) if risks else 0.0,
                          "alert_rate": round(100 * alerts / w, 1) if w else 0.0},
                "risk": risks,
                "history": [dict(e) for e in list(self.history)[:30]],
                "hosts": [dict(h) for h in sorted(self.hosts.values(), key=lambda h: h["host"])],
                "explanation": self.explanation,
                "labelled": sum(self.live_cm["Ensemble"].values()),
                "live_metrics": [_cm_row(k, v) for k, v in self.live_cm.items()]
                if sum(self.live_cm["Ensemble"].values()) else [],
            }