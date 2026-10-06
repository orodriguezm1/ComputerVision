
"""Run all main demos in sequence."""
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
scripts = [
    '01_compare_readers.py',
    '02_inspect_tensors.py',
    '03_color_channels_gray_normalization.py',
    '04_geometric_transforms.py',
    '05_reflection_about_arbitrary_line.py',
    '06_filters_threshold_edges.py',
    '07_hog_demo.py',
    '08_bbox_and_landmarks_from_metadata.py',
    '09_pose_pyr_demo.py',
    '10_prepare_tensor_for_cnn.py',
]

for script in scripts:
    print('\n' + '='*90)
    print('Running', script)
    subprocess.run([sys.executable, str(HERE / script)], check=True)
