"""Optional ONNX/Torch detector adapter. Model-specific preprocessing and decoding are explicit."""

from pathlib import Path
import numpy as np


class ModelDetector:
    def __init__(self, backend: str, path: str, preprocess, decode):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        self.name = f"Custom {backend} detector"
        self.backend = backend
        self.preprocess = preprocess
        self.decode = decode
        if backend == "onnx":
            import onnxruntime as ort

            self.model = ort.InferenceSession(path, providers=ort.get_available_providers())
        elif backend == "torch":
            import torch

            self.torch = torch
            self.model = torch.jit.load(path, map_location="cpu").eval()
        else:
            raise ValueError("backend must be onnx or torch")

    def detect(self, frame):
        tensor = self.preprocess(frame)
        if self.backend == "onnx":
            output = self.model.run(None, {self.model.get_inputs()[0].name: np.asarray(tensor)})
        else:
            with self.torch.inference_mode():
                output = self.model(self.torch.as_tensor(tensor))
        return self.decode(output, frame.shape)

    def close(self):
        self.model = None
