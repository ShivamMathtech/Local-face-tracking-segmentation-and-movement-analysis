# Movement and coordinate conventions

For box `(x,y,w,h)`, center is `(x+w/2,y+h/2)`, area is `wh`, and aspect ratio is `w/h`. Top-left coordinates normalize by processing width/height into `[0,1]` when inside frame. Center mode subtracts half width/height and keeps positive Y downward. Cartesian mode subtracts half width and flips Y upward; normalized centered values typically lie in `[-0.5,0.5]`. Predictions may extend outside the frame; they are not clamped into apparently observed measurements.

Observed centers are exponentially smoothed: `c_t = alpha*r_t + (1-alpha)*c_prev`. Displacement is `dc = c_t-c_prev`, distance is Euclidean norm. With actual `dt`, velocity is `alpha*(dc/dt)+(1-alpha)*v_prev`. Acceleration is `(v-v_prev)/dt`; the displayed scalar is the magnitude of this 2D acceleration vector, not the signed derivative of speed.

Direction bins divide `atan2(vy,vx)` into eight sectors. A speed below `direction_threshold` is stationary. Relative CLOSER/FARTHER is a 3.5% area-change threshold between observations; pose, expression and detector noise can also change area. It is not calibrated distance.

Every lost interval breaks the derivative chain and trajectory line. The first renewed observation has zero velocity/acceleration, avoiding a fictitious jump across missing time. Non-increasing timestamps are ignored. A paused live stream must also break the derivative chain; video playback timestamps preserve source time.

Dwell heatmaps weight consecutive observations by time (intervals capped at 0.25s to avoid excessive sparse-frame dwell), omit missing segments, blur the spatial histogram and normalize it for display. Movement intensity weights those intervals by speed. The map is a relative visualization; its colors are not calibrated physical units.

CSV keeps top-left processing-pixel coordinates regardless of the display origin. All-source rows retain their face IDs. Motion graphs/statistics reset on a new lock. Bounded export history and unlimited trajectory have separate memory semantics. No physical speed is reported without scale calibration.
