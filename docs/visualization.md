# Visualization

RGB, grayscale, Canny, gradient-magnitude Sobel, Laplacian, Gaussian blur, median blur, horizontal motion-blur simulation, high-contrast CLAHE, binary mask and heat/density/movement maps are display-only transforms.

Night Vision applies grayscale, CLAHE, Gaussian denoising, gain/contrast/brightness, optional synthetic noise and a green tint. Thermal Style normalizes luminance and applies INFERNO/JET/HOT/TURBO. Both modes carry permanent **SIMULATED** labels on screen and in frame/video exports. Neither provides temperature, infrared sensitivity or additional scene information.

Edges can cover the full frame, the locked ROI or the segmented region. Sparse feature points always render as points. Dense landmarks support anatomical feature polylines, face contour, or a geometric Delaunay mesh (not the official MediaPipe tessellation topology).

Overlays include individually toggled boxes, center markers, coordinate grid, trajectory, landmarks, mask, movement vector and pose axes. The arrow spans the previous-to-current smoothed observed displacement. Lost trajectories are separated rather than connected across gaps.
