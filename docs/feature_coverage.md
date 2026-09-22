# Feature coverage and limitations

| Brief area | Delivered implementation | Conditions |
|---|---|---|
| Desktop dashboard | PySide6 split dashboard, plots, target/display/session tabs | Graphical OS required |
| Camera/image/video/RTSP | OpenCV input with source timestamps | Camera/codec/network support varies |
| Multi-face detection | Haar, MediaPipe Tasks adapter, YuNet adapter | Optional backends need model files |
| Face locking | Click/list selection, explicit lock, conservative association, lost state | Not guaranteed biometric identity |
| Kalman / optical flow | Working six-state filter and LK assistance | CPU implementation |
| Dense landmarks | MediaPipe Tasks ROI landmarking, feature lines/mesh/contour | Requires extra and .task model |
| Lightweight landmarks | Actual sparse corners or detector keypoints | Not dense/anatomical fallback |
| Coordinates / motion | Origins, normalization, velocities, vector acceleration, direction, distance | Pixel units; no metric depth |
| Trajectory / maps | Segmented trails, dwell/density/movement maps | Bounded history; unlimited trail optional |
| Segmentation | Dense polygon/hull/refinement or labelled ellipse fallback | Geometric rather than semantic masks |
| Blur / edge / modes | All requested basic filters and scope controls | Simulated night/thermal clearly labelled |
| Head pose | solvePnP and axes, optional intrinsics/distortion | Dense model; generic head geometry |
| Calibration | Chessboard CLI, JSON loading, scaled intrinsics | Six successful views minimum |
| Recording/export | MP4, CSV, JSONL, JSON, PNG/JPEG, graph and ZIP export | MPEG-4 encoder must be available |
| Session controls | Explicit recording, finalization, ZIP, confirmed delete | Local files only |
| Analytics | Real measured continuity, time, counts, speed/area statistics | No fabricated identity accuracy |
| Experiments | Four tracker variants replay same video with selected detector | Ground truth accuracy not implemented |
| Notebooks/tests | Ten notebooks; core and Qt integration checks | See validation report |
| Optional ML backends | Generic ONNX/TorchScript interface + explicit Nx6 adapter | Arbitrary YOLO models need decoding logic |
| Thermal hardware extension | Frame-source boundary documented | No FLIR driver/radiometry implemented |
| GPU | Optional ONNX provider capability | No mandatory or guaranteed GPU path |

This table records actual behavior rather than treating every optional extension as already supported hardware.
