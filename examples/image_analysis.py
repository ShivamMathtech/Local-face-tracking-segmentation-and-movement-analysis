import sys
from face_motion_lab.main import main

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python examples/image_analysis.py image.jpg")
    raise SystemExit(main(["--image", sys.argv[1]]))
