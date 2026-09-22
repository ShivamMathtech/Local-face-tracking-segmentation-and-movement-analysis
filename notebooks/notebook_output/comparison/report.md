# Tracker experiment

Same source replayed for each algorithm. First largest visible face is explicitly auto-selected for this benchmark. No ground-truth identities are available: continuity measures observed-frame availability, not identity accuracy. The demo uses scripted detections and is not a detector accuracy benchmark.

| Algorithm | FPS | Latency ms | Continuity % | Lost |
|---|---:|---:|---:|---:|
| detector-only | 144.07 | 6.94 | 71.42857142857143 | 60 |
| detector+kalman | 150.47 | 6.65 | 71.42857142857143 | 60 |
| detector+flow | 45.08 | 22.18 | 71.42857142857143 | 60 |
| detector+kalman+flow | 28.38 | 35.24 | 71.42857142857143 | 60 |