# Segmentation

Dense MediaPipe landmarks supply the ordered facial oval for a polygon; the hull mode computes a convex hull of dense points. The refined mode performs a 5×5 elliptical opening/closing, fills external contours and applies a 7×7 Gaussian edge feather. Sparse corners are never treated as a reliable face outline; without a dense model, a bounding-box ellipse is explicitly identified as approximate.

The optional skin heuristic thresholds YCrCb within the face mask. It is a teaching tool with strong lighting/skin-tone limitations, not semantic segmentation. Every mask is an 8-bit array in processing-frame coordinates. PNG export preserves its soft boundary values.

Original, Face Mask overlay, black-background Masked Face, light-background Background Removed, desaturated-background Face Highlighted and white Face Silhouette are available. Mask fill, boundary and opacity are separate controls. An unobserved target has an empty mask. Display transformations do not alter detector input or the source file.
