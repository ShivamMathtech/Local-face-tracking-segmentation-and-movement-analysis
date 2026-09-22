# Architecture and ownership

`input.FrameSource` emits a BGR8 image, source-relative timestamp and frame number. A future thermal sensor adapter must preserve radiometric values separately rather than relabel RGB intensity as temperature. Current built-in sources emit RGB/BGR images only.

`Pipeline.process` resizes to a configurable maximum processing width, detects faces, extracts optional landmarks and performs association. `FaceTracker` owns temporary IDs and the selected lock. `MovementAnalyzer` receives only observed locked-target positions. Masks and overlays use copies of the frame.

`Worker` owns every pipeline mutation, the capture object and the recorder in its QThread. UI commands are queued. UI polling takes the newest result from a locked single-result mailbox every 33 ms; old preview packets are overwritten rather than queued indefinitely. Graphs refresh at roughly 6 Hz. QImage makes an owned copy of array pixels before painting. The pipeline does not depend on Qt.

At EOF the worker remains available for image selection and export until Stop. Source replacement first requests a stop and starts the next source after the previous thread finishes. Shutdown never forcibly terminates a running capture thread. Network capture requests FFmpeg open/read timeouts; some device backends may take longer to release. Recording is finalized before resources are released.

Errors during one frame are reported; five consecutive processing failures suspend processing. Original frame arrays never receive grayscale, night/thermal or mask display transforms. Source resolution changes reset tracking; recording stops if its resolution contract changes.

## Extension points
- Detector protocol: `name`, `detect(frame) -> list[FaceDetection]`, `close()`.
- MediaPipe detector and landmarker use the Tasks IMAGE API, called synchronously inside the worker. Track association is owned by this application; MediaPipe's internal IDs are not treated as identities.
- Optional `ModelDetector` loads an ONNX session or TorchScript module with caller-supplied preprocessing/decoding.
- `FrameSource` can be replaced by a device-specific adapter that yields `Frame` objects with source timestamps.
- `export_analysis` and `SessionRecorder` are shared by CLI and desktop.

The directory layout groups substantive modules by responsibility. It intentionally does not create one-line proxy files for every suggested filename in the brief.
