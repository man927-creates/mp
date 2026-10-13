"""ML-based NIDS dashboard. Run: python app.py  ->  http://127.0.0.1:5000

No login and no dataset download: models train automatically on synthetic
traffic when the server starts.
"""
import csv
import ipaddress
import datetime as dt
import html
import io
import threading
import traceback

from flask import Flask, Response, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

import core

app = Flask(__name__)

TRAIN = {"ready": False, "stage": "Starting", "progress": 0, "error": ""}
BUNDLE = None
MON = None
_lock = threading.Lock()


def _train_in_background():
    global BUNDLE, MON

    def progress(stage, pct):
        TRAIN.update(stage=stage, progress=pct)

    try:
        bundle = core.train_bundle(progress)
        with _lock:
            BUNDLE = bundle
            MON = core.Monitor(bundle)
            TRAIN.update(ready=True, stage="Ready", progress=100, error="")
    except Exception as exc:
        traceback.print_exc()
        TRAIN.update(stage="Training failed", error=f"{type(exc).__name__}: {exc}")


threading.Thread(target=_train_in_background, daemon=True).start()


def _error(message, code=400):
    return jsonify(ok=False, error=message), code


def _fmt_time(ts):
    return dt.datetime.fromtimestamp(ts).strftime("%d %b %Y, %H:%M:%S") if ts else "—"


def _history_copy():
    """All alerts of the current session (oldest first)."""
    if MON is None:
        return []
    with MON.lock:
        return [dict(e) for e in reversed(MON.history)]


@app.get("/")
def index():
    return render_template("index.html")

@app.post("/api/dataset/upload")
def upload_dataset():
    file = request.files.get("dataset")

    if not file or not file.filename:
        return _error("Please select a CSV dataset.")

    if not file.filename.lower().endswith(".csv"):
        return _error("Only CSV files are allowed.")

    try:
        content = file.stream.read().decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(content))

        if not reader.fieldnames:
            return _error("The CSV file is empty or has no header.")

        rows = list(reader)

        if not rows:
            return _error("The CSV contains no data rows.")

        return jsonify(
            ok=True,
            filename=html.escape(file.filename),
            row_count=len(rows),
            columns=reader.fieldnames,
            preview=rows[:5]
        )

    except (UnicodeDecodeError, csv.Error) as exc:
        return _error(f"Could not read CSV: {exc}")


@app.get("/api/status")
def status():
    mon = MON.snapshot() if MON is not None else None
    return jsonify(
        training=dict(TRAIN),
        backends=core.backend_info(BUNDLE),
        summary=BUNDLE["summary"] if BUNDLE else None,
        metrics=BUNDLE["metrics"] if BUNDLE else None,
        monitor=mon,
    )


@app.post("/api/monitor/start")
def monitor_start():
    if MON is None:
        return _error("Models are still training. Please wait a moment.", 409)
    body = request.get_json(silent=True) or {}
    source = body.get("source", "simulator")
    if source not in ("simulator", "live"):
        return _error("Unknown source. Use 'simulator' or 'live'.")
    if source == "live" and not core.HAS_SCAPY:
        return _error("Live capture is unavailable because scapy could not load: "
                      f"{core.SCAPY_ERR}. Use the simulator, or install scapy and Npcap.")
    ip = str(body.get("ip") or "").strip()
    if ip:
        try:
            ip = str(ipaddress.ip_address(ip))
        except ValueError:
            return _error("Invalid source IP. Enter an address such as 192.168.1.10, or leave it empty.")
    try:
        MON.start(source, ip or None)
    except RuntimeError as exc:
        return _error(str(exc), 409)
    return jsonify(ok=True, source=source, target=ip or None)


@app.post("/api/monitor/stop")
def monitor_stop():
    if MON is None:
        return _error("Models are still training.", 409)
    MON.stop()
    return jsonify(ok=True)


@app.get("/api/report.csv")
def report_csv():
    if MON is None:
        return _error("Models are still training. No alerts to export yet.", 409)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["time", "host", "verdict", "attack", "risk",
                     "xgboost_pct", "lstm_pct", "autoencoder_pct", "ground_truth"])
    for e in _history_copy():
        s = e["scores"]
        writer.writerow([e["time"], e["host"], e["verdict"], e["attack"], e["risk"],
                         s["XGBoost"], s["LSTM"], s["Autoencoder"], e["truth"]])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=nids_alerts.csv"})


CSS = """
body{font-family:"Segoe UI",Arial,sans-serif;color:#1d2433;margin:0;padding:32px;background:#f5f7fb}
h1{margin:0 0 4px;font-size:24px}
h2{font-size:16px;margin:28px 0 10px;border-bottom:2px solid #dbe2f0;padding-bottom:6px}
.meta{color:#5c677d;font-size:13px}
.kpis{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px;margin-top:18px}
.kpi{background:#fff;border:1px solid #dbe2f0;border-radius:10px;padding:12px}
.kpi small{color:#5c677d;font-size:11px;text-transform:uppercase}
.kpi b{display:block;font-size:22px;margin-top:4px}
table{width:100%;border-collapse:collapse;background:#fff;font-size:13px}
th,td{padding:8px;border-bottom:1px solid #e6ebf5;text-align:left;vertical-align:middle}
th{background:#eef2fa;font-size:11px;text-transform:uppercase;color:#5c677d}
.mono{font-family:Consolas,monospace;font-size:12px}
.b{padding:2px 8px;border-radius:99px;font-size:11px;font-weight:700}
.Normal{background:#dff7ee;color:#0f7a52}.Suspicious{background:#fff3d6;color:#8a5a00}.Malicious{background:#ffe1e6;color:#a1243a}
.note{background:#fff;border-left:4px solid #6aa3ff;padding:12px 14px;margin-top:14px;font-size:13px;line-height:1.6}
.toolbar{margin-bottom:14px}
.toolbar button{background:#3a6ff7;color:#fff;border:0;border-radius:8px;padding:9px 16px;font-weight:700;cursor:pointer}
.muted{color:#5c677d}
.up{color:#0f7a52;font-weight:700}.down{color:#a1243a;font-weight:700}
@media print{.toolbar{display:none}body{background:#fff;padding:12px}}
"""


def build_live_tables(snap, metrics, metric_cols):
    """Live confusion/metrics table and a live-vs-training comparison table."""
    e = html.escape
    live_rows = snap["live_metrics"] if snap else []
    if not live_rows:
        return ("<p class='muted'>No labelled live windows in this session. Run the Simulator mode "
                "to measure live accuracy. Live capture has no ground-truth labels.</p>", "")

    head = "<tr><th>Model</th>" + "".join("<th>{}</th>".format(c) for c in metric_cols) + \
           "<th>TP / FP / FN / TN</th></tr>"
    body = "".join(
        "<tr><td><b>{}</b></td>{}<td class='mono'>{} / {} / {} / {}</td></tr>".format(
            e(r["model"]),
            "".join("<td>{:.2f}%</td>".format(r[c]) for c in metric_cols),
            r["confusion"]["tp"], r["confusion"]["fp"], r["confusion"]["fn"], r["confusion"]["tn"])
        for r in live_rows)
    live_table = "<table><thead>" + head + "</thead><tbody>" + body + "</tbody></table>"

    def cell(live, train):
        d = live - train
        cls = "up" if d >= 0 else "down"
        return ("<td>{:.2f}%</td><td>{:.2f}%</td><td class='{}'>{:+.2f}</td>"
                .format(live, train, cls, d))

    cmp_head = ("<tr><th rowspan='2'>Model</th><th colspan='3'>Accuracy</th>"
                "<th colspan='3'>Recall</th><th colspan='3'>F1-Score</th></tr>"
                "<tr><th>Live</th><th>Training</th><th>Δ</th>"
                "<th>Live</th><th>Training</th><th>Δ</th>"
                "<th>Live</th><th>Training</th><th>Δ</th></tr>")
    cmp_body = "".join(
        "<tr><td><b>{}</b></td>{}{}{}</tr>".format(
            e(lr["model"]),
            cell(lr["Accuracy"], tr["Accuracy"]),
            cell(lr["Recall"], tr["Recall"]),
            cell(lr["F1-Score"], tr["F1-Score"]))
        for lr, tr in zip(live_rows, metrics))
    note = ("<div class='note'>Live numbers come from {n} labelled windows measured while the "
            "simulator ran (one window per host per tick). Training numbers come from the fixed "
            "held-out test split. Δ = live minus training: green means live is equal or higher, "
            "red means live is lower.</div>").format(n=snap["labelled"])
    compare = ("<table><thead>" + cmp_head + "</thead><tbody>" + cmp_body + "</tbody></table>" + note)
    return live_table, compare


def build_report_html():
    snap = MON.snapshot() if MON is not None else None
    hist = _history_copy()
    summary = BUNDLE["summary"]
    metrics = BUNDLE["metrics"]
    backends = core.backend_info(BUNDLE)
    st = snap["stats"] if snap else {"windows": 0, "Normal": 0, "Suspicious": 0,
                                     "Malicious": 0, "hosts": 0, "avg_risk": 0.0, "alert_rate": 0.0}
    e = html.escape

    if snap is None or not snap["source"]:
        session_state, source_label, started = "No monitoring session yet", "—", "—"
    else:
        session_state = "Running" if snap["running"] else "Stopped"
        source_label = ("Simulator (synthetic traffic)" if snap["source"] == "simulator"
                        else "Live capture (real packets)")
        started = _fmt_time(snap["started"])

    metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "Specificity"]
    metric_rows = "".join(
        "<tr><td><b>{}</b></td>{}<td class='mono'>{} / {} / {} / {}</td></tr>".format(
            e(r["model"]),
            "".join("<td>{}%</td>".format(r[c]) for c in metric_cols),
            r["confusion"]["tp"], r["confusion"]["fp"], r["confusion"]["fn"], r["confusion"]["tn"])
        for r in metrics)

    hist_rows = "".join(
        "<tr><td class='mono'>{}</td><td>{}</td><td><span class='b {}'>{}</span></td>"
        "<td>{}</td><td>{}%</td><td>{}</td></tr>".format(
            e(h["time"]), e(h["host"]), e(h["verdict"]), e(h["verdict"]),
            e(h["attack"]), h["risk"], e(h["truth"]))
        for h in reversed(hist[-300:])) or \
        "<tr><td colspan='6' class='muted'>No alerts recorded in this session.</td></tr>"

    host_rows = "".join(
        "<tr><td>{}</td><td><span class='b {}'>{}</span></td><td>{}</td><td>{}%</td></tr>".format(
            e(h["host"]), e(h["verdict"]), e(h["verdict"]), h["windows"], round(h["risk"]))
        for h in (snap["hosts"] if snap else [])) or \
        "<tr><td colspan='4' class='muted'>No hosts observed yet.</td></tr>"

    expl = snap["explanation"] if snap else None
    if expl:
        expl_rows = "".join(
            "<tr><td>{}</td><td class='mono'>{}</td></tr>".format(
                e(i["feature"]),
                ("+" if i["value"] >= 0 else "") + "{:.4f}".format(i["value"]))
            for i in expl["items"])
        expl_block = ("<p class='muted'>Host <b>{}</b> · risk {}% · method: {}. Positive values push "
                      "the prediction toward attack.</p><table><thead><tr><th>Feature</th>"
                      "<th>Contribution</th></tr></thead><tbody>{}</tbody></table>").format(
            e(expl["host"]), expl["risk"], e(expl["method"]), expl_rows)
    else:
        expl_block = "<p class='muted'>No alert has been explained yet in this session.</p>"

    kpis = "".join(
        "<div class='kpi'><small>{}</small><b>{}</b></div>".format(label, value)
        for label, value in [
            ("Windows analysed", st["windows"]), ("Normal", st["Normal"]),
            ("Suspicious", st["Suspicious"]), ("Malicious", st["Malicious"]),
            ("Active hosts", st["hosts"])])

    live_table, compare = build_live_tables(snap, metrics, metric_cols)
    best = max(metrics, key=lambda r: r["F1-Score"])
    now = dt.datetime.now().strftime("%d %b %Y, %H:%M:%S")

    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>ML-NIDS Report</title><style>{css}</style></head><body>"
        "<div class='toolbar'><button onclick='window.print()'>Print / Save as PDF</button></div>"
        "<h1>ML-Based Network Intrusion Detection — Report</h1>"
        "<div class='meta'>Generated {now} · Session: {state} · Source: {source}</div>"
        "<div class='kpis'>{kpis}</div>"

        "<h2>Session details</h2>"
        "<table><tbody>"
        "<tr><th style='width:220px'>Monitoring started</th><td>{started}</td></tr>"
        "<tr><th>Average risk</th><td>{avg_risk}/100</td></tr>"
        "<tr><th>Session state</th><td>{state}</td></tr>"
        "</tbody></table>"

        "<h2>Detection pipeline</h2>"
        "<table><tbody>"
        "<tr><th style='width:220px'>Classifier</th><td>{clf}</td></tr>"
        "<tr><th>Sequence model (LSTM)</th><td>{seq}</td></tr>"
        "<tr><th>Autoencoder</th><td>Reconstruction-error anomaly detector (trained on normal traffic)</td></tr>"
        "<tr><th>Ensemble</th><td>Risk = 0.40 × XGBoost + 0.35 × LSTM + 0.25 × Autoencoder (0–100); "
        "Normal &lt; 35 ≤ Suspicious &lt; 65 ≤ Malicious</td></tr>"
        "<tr><th>Explanation</th><td>{expl_method}</td></tr>"
        "<tr><th>Capture</th><td>{capture}</td></tr>"
        "</tbody></table>"

        "<h2>Model performance (held-out test split)</h2>"
        "<table><thead><tr><th>Model</th>{mh}<th>TP / FP / FN / TN</th></tr></thead>"
        "<tbody>{metric_rows}</tbody></table>"
        "<div class='note'>Training data: {samples} synthetic sequences ({train} train / {test} test), "
        "trained in {seconds}s. Attack-type accuracy: {type_acc}%. Best F1: {best_name} "
        "({best_f1}%). Results come from synthetic traffic, so they show the pipeline works; "
        "they are not a guarantee on real networks.</div>"

        "<h2>Live metrics (simulator ground truth)</h2>{live_table}"
        "<h2>Live vs training comparison</h2>{compare}"

        "<h2>Hosts</h2>"
        "<table><thead><tr><th>Host</th><th>Latest verdict</th><th>Windows</th><th>Risk</th></tr></thead>"
        "<tbody>{host_rows}</tbody></table>"

        "<h2>Latest explanation</h2>{expl_block}"

        "<h2>Detection history (alerts)</h2>"
        "<table><thead><tr><th>Time</th><th>Host</th><th>Verdict</th><th>Attack</th><th>Risk</th>"
        "<th>Ground truth</th></tr></thead><tbody>{hist_rows}</tbody></table>"
        "<div class='note'>Ground truth is shown only in Simulator mode. Live capture has no "
        "labels, so it shows “—”.</div>"

        "</body></html>"
    ).format(
        css=CSS, now=now, state=session_state, source=e(source_label), kpis=kpis,
        started=e(started), avg_risk=st["avg_risk"],
        clf=e(backends["classifier"]), seq=e(backends["sequence"]),
        expl_method=e(backends["explain"]), capture=e(backends["capture"]),
        mh="".join("<th>{}</th>".format(c) for c in metric_cols),
        metric_rows=metric_rows, samples=summary["samples"], train=summary["train"],
        test=summary["test"], seconds=summary["seconds"],
        type_acc=summary["attack_type_accuracy"],
        best_name=e(best["model"]), best_f1=best["F1-Score"],
        host_rows=host_rows, expl_block=expl_block, hist_rows=hist_rows,
        live_table=live_table, compare=compare,
    )


@app.get("/report")
def report_page():
    if BUNDLE is None:
        return _error("Models are still training. Try again after training finishes.", 409)
    return Response(build_report_html(), mimetype="text/html")


@app.errorhandler(Exception)
def handle_exception(exc):
    if isinstance(exc, HTTPException):
        if request.path.startswith("/api/"):
            return jsonify(ok=False, error=exc.description), exc.code
        return exc
    traceback.print_exc()
    if request.path.startswith("/api/"):
        return jsonify(ok=False, error=f"{type(exc).__name__}: {exc}"), 500
    return "Internal server error", 500


if __name__ == "__main__":
    print("ML-NIDS dashboard: http://127.0.0.1:5000  (training starts automatically)")
    app.run(host="127.0.0.1", port=5000, threaded=True, debug=False)