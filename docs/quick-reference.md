# Quick reference

| Action | Desktop | CLI |
|---|---|---|
| Launch | run_windows.bat or run_mac_linux.sh | `python -m face_motion_lab` |
| No-camera demo | Demo button | `--demo` |
| Webcam | Start camera | `--camera 0` |
| Video | Open video | `--video input.mp4` |
| Image | Open image | `--image image.jpg` |
| Lock | Select face, Lock selected / L | `--headless --auto-lock` |
| Unlock | U | Desktop action |
| Pause | Space | Desktop action |
| Record | Record / End recording | `--headless --record` |
| Export | Export analysis | `--headless --output folder` |
| Compare trackers | CLI | `--video input.mp4 --compare` |
| Calibration | CLI then Device settings | `--calibrate folder` |
| Tests | Terminal | `python -m pytest -q` |

Dense model installation and platform notes are in README.md.
