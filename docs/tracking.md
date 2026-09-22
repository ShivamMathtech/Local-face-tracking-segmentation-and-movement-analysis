# Lock policy

A track is DETECTED on creation and TRACKING when matched subsequently. Selecting a visible track creates a LOCKED target; Unlock produces UNLOCKED. Missing detections produce TEMPORARILY_LOST, with the last motion model predicting a box. Predictions are visually identified and never added to motion statistics.

Within `max_lost_frames`, a sufficiently unambiguous compatible detection can reacquire the same ID. The reacquisition counter increments. An ambiguous pair of candidates, a contested detection or a failed gate suspends matching. After the threshold, the locked track becomes REACQUIRING and is removed from automatic assignment. New detections receive new IDs. The user must click/select and Lock to establish a new target. No age-based fallback picks the nearest face.

This policy reduces swaps but cannot mathematically ensure identity. Similar-looking people, severe overlaps and drastic appearance changes can still cause a wrong short-term association. For safety-sensitive identity verification, this implementation is insufficient. There is no named identity database or biometric recognition subsystem.

Switching targets starts new target motion statistics. Explicit Reset clears tracking IDs and retained analysis rows. Source changes create a fresh pipeline. Association score and observed-frame continuity must not be marketed as recognition accuracy.
