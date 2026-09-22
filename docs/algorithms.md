# Algorithms

## Detection and association
Haar uses OpenCV's bundled frontal cascade with scale factor 1.12, five neighbours and 32px minimum size. Its output has no calibrated confidence; `None` is shown as `n/a`. YuNet and MediaPipe provide model scores. A 16×16 hue/saturation histogram describes each detection's crop; this is a short-term appearance cue, not an identity embedding.

Association cost is `0.40(1 − IoU) + 0.30 min(normalized center distance, 1) + 0.30 histogram distance`. If compatible anatomical keypoint configurations exist, 10% of the cost uses scale-normalized point-shape distance. A Hungarian assignment yields a one-to-one matching. Position, area-ratio and histogram gates reject implausible candidates; locked targets use tighter gates and explicit ambiguity margins. The displayed match score is `1 − cost`, a heuristic, not probability.

## Kalman state
State `s = [cx, cy, vx, vy, width, height]`. Transition matrix is identity with `A[0,2]=A[1,3]=dt`. Observation selects center and dimensions. Prediction: `s⁻=As`, `P⁻=APAᵀ+Q`. Correction: `K=P⁻Hᵀ(HP⁻Hᵀ+R)⁻¹`, `s=s⁻+K(z−Hs⁻)`. Covariance uses the Joseph update. Variable dt is clipped for numerical stability; camera timestamps still drive measured motion.

Optical flow uses Shi–Tomasi features, pyramidal Lucas–Kanade tracking, a 1.5px forward/backward residual test and median displacement. At least five valid points and a consistency bound are required. Flow aids association prediction; it does not turn missing face detections into confirmed observations.

## Pose and blur
Six dense landmarks map to a generic 3D head model for iterative solvePnP. Intrinsics and distortion come from calibration or an approximate focal length equal to image width. Euler angles describe the chosen generic model axes and are not a medical/biometric measurement. Reprojected X/Y/Z axes are shown on the face.

Sharpness is variance of the ROI's grayscale Laplacian. Blur labels use fixed illustrative thresholds of 50 and 150 and must be calibrated per camera. The bounded `blur_score = 1/(1+sharpness/100)` is a visualization heuristic, not a probability.

## Primary API references
- [MediaPipe Face Landmarker Python](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python)
- [MediaPipe Face Detector Python](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector/python)
- [OpenCV YuNet API](https://docs.opencv.org/4.x/df/d20/classcv_1_1FaceDetectorYN.html)
- [OpenCV optical flow](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html)

The implementation was checked against the Tasks API rather than assuming the legacy `mp.solutions` module exists.
