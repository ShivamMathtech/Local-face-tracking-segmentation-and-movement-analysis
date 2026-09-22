import sys
from face_motion_lab.main import main

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python examples/video_tracking.py input.mp4")
    raise SystemExit(main(["--video", sys.argv[1]]))
