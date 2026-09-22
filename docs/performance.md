# Performance and operational limits

No frame-rate promise is made. Displayed processing FPS is the inverse of mean pipeline processing time over at most 300 frames; it is not sensor FPS or display delivery FPS. Latency measures one processing call. Process CPU is normalized by logical CPU count; memory is process RSS. Live acquisition uses monotonic timestamps; file playback uses decoder timestamps with FPS fallback for unsupported codecs.

Set processing width to 640 for lower latency, reduce dense landmark use, disable geometric mesh, turn off optical flow and shorten graph/history windows as needed. Haar, YuNet, built-in MediaPipe and NumPy paths use CPU. The optional ONNX adapter selects available providers. No unsupported GPU speedup is claimed.

The GUI consumes only the latest result. Capture and inference share a background worker, so a slow detector can still increase camera-backend buffering; hardware-dependent low-latency RTSP tuning remains necessary. Network timeouts depend on FFmpeg support. Stream reconnection is manual and explicit.

CSV and landmark JSONL recording flush after each frame. MP4 uses constant-rate resampling; exact timestamps live in CSV. Real-time graphs render a maximum of 2,000 retained samples, even when the configured time window spans more. Export retains up to history_limit rows; a recording streams all processed rows. Unlimited trails can grow without bound. Do not use unlimited mode for unattended multi-hour sessions.

The built-in demo tests plumbing, selection, occlusion/loss and exports with scripted detections. It cannot establish detection quality, identity accuracy, camera driver reliability or device throughput. Evaluate those with representative consenting subjects and ground truth before research conclusions.
