# ML-Based Network Intrusion Detection System

Final-year project: real-time network traffic classification using
XGBoost, LSTM, Autoencoder, Ensemble Fusion and SHAP-style explanations.

## Run

```bash
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000

* No login and no dataset download. Models train automatically on synthetic traffic
  when the server starts (about 10-40 seconds).
* **Simulator** mode works everywhere and shows attack episodes with ground truth.
* **Live capture** needs `scapy`, Npcap (Windows) or libpcap (Linux), and
  Administrator / root rights.

## Backends

The dashboard shows which backend is really used:

| Component   | Preferred           | Fallback (automatic)                       |
|-------------|---------------------|--------------------------------------------|
| Classifier  | XGBoost             | sklearn HistGradientBoosting               |
| Sequence    | TensorFlow LSTM     | MLP over the same 5-window sequence        |
| Explanation | XGBoost SHAP        | Feature sensitivity analysis               |
| Capture     | scapy live sniffing | Simulator only                             |

If TensorFlow is installed but fails to load (common DLL issues on Windows),
the app logs the error and keeps running with the fallback.

## Layout

* `core.py`: synthetic data, models, ensemble, explanations, traffic sources, monitor
* `app.py`: Flask routes, background training, JSON error handling
* `templates/index.html`: dashboard

## Ensemble

Risk = 0.40 × XGBoost + 0.35 × LSTM + 0.25 × Autoencoder (0-100).
Normal < 35 ≤ Suspicious < 65 ≤ Malicious.
