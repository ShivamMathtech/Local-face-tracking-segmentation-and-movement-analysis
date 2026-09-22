# Release validation

The source was validated in Linux with Python 3.12.14 and Qt offscreen rendering. Exact dependency versions are in `validation-environment.json`; the main requirements use compatible version ranges. `requirements-tested.txt` records direct runtime versions for optional reproducibility.

## Completed checks
- **48 automated tests passed**: geometry, coordinate origins, all direction classes, irregular-timestamp velocity/acceleration, gap handling, bounded trajectory, Kalman covariance, conservative lock expiry and ambiguity, short-gap reacquisition, optical-flow translation, masks, display filters, sharpness, synthetic solvePnP, settings, Haar blank input, recording/export, video timestamps, still-image measurement export and the Qt dashboard worker/lock/save/stop lifecycle.
- All **10 notebooks** executed successfully, including their assertions and rendered plots. Stored outputs are actual executions; timing values reflect this runtime and are not hardware promises.
- A 210-frame synthetic CLI run created a decodable MP4, streaming tracking CSV, landmarks JSONL and session metadata, plus analysis graphs, mask, heatmap and trajectory exports. Its deliberate occlusion caused a persistent lost lock; reappearance did not silently switch the expired target.
- OpenCV Haar and optional YuNet each detected one face in OpenCV's public `lena.jpg` sample. That image was used only for local validation and is not redistributed. This is a basic integration check, not an accuracy evaluation.
- The Haar still-image CLI auto-lock path completed and exported metrics. The demo UI was visually inspected using the included dashboard screenshot.
- All optional model download commands retrieved nonempty upstream files. YuNet loaded and returned no detections for a blank image.
- Python compilation and undefined-name/unused-import static checks passed.

## Not verified in this environment
- No physical webcam, RTSP endpoint, Windows/macOS desktop, real multi-person crossing sequence, GPU or thermal sensor was available. These require deployment-machine testing.
- MediaPipe 0.10.35 imported and the detector/landmarker files downloaded, but its native inference runtime could not load because this container lacks `libGLESv2.so.2`. System package installation was unavailable here. Dense-landmark/MediaPipe detector inference therefore remains **unverified**. Install the platform's EGL/GLES runtime (typically `libegl1` and `libgles2` on Debian/Ubuntu) to use it. The default Haar pipeline does not require MediaPipe.
- Camera calibration and real head-pose accuracy were not measured against physical ground truth. Pose geometry was checked using known synthetic projections.

## Interpretation
Synthetic detections test application mechanics, not face-recognition accuracy. Test counts do not establish production robustness. The association score and continuity percentage are computed from observed data; neither is a verified identity metric. Models are optional downloads and are not bundled.
