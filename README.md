
# Fake People Computer Vision Project

This project contains synthetic portraits plus ready-to-run Python scripts for class examples in computer vision.
The people are fictional and generated for educational use.

## Included topics

1. Compare OpenCV, Pillow, scikit-image and imageio
2. Inspect image tensors: shape, dtype, min, max, memory
3. BGR/RGB, channels, grayscale, magenta and normalization
4. Resize, crop, rotation and reflection
5. Reflection with respect to an arbitrary line defined by two points
6. Filtering, thresholding, Sobel and Canny
7. HOG features and visualization
8. Face bounding box and landmarks overlays
9. Illustrative pitch, yaw and roll (PYR) overlay
10. Preparing image tensors for CNNs
11. Optional MediaPipe Face Mesh demo

## Project tree

- `images/raw/`: original synthetic portraits
- `metadata/`: approximate bbox, landmarks and pose values
- `scripts/`: all code files
- `outputs/`: generated output images when you run the scripts

## How to run

From PyCharm or a terminal, place the folder in your project root and run:

```bash
python scripts/12_run_all.py
```

Or run scripts individually, for example:

```bash
python scripts/03_color_channels_gray_normalization.py
python scripts/07_hog_demo.py
```

## Suggested packages

```bash
pip install numpy opencv-contrib-python pillow matplotlib scikit-image imageio
```

Optional:

```bash
pip install mediapipe tensorflow
```

## Notes

- The overlays in `metadata/` are pedagogical approximations.
- The optional MediaPipe script may or may not detect the synthetic faces, depending on your environment.
- All scripts use relative paths, so the project is easy to copy into PyCharm.
