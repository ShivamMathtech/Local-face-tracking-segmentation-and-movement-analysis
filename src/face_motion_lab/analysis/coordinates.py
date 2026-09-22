def coordinates(x: float, y: float, width: int, height: int, origin: str = "top-left") -> dict:
    if width <= 0 or height <= 0:
        raise ValueError("Frame dimensions must be positive")
    if origin == "top-left":
        px, py = x, y
    elif origin == "center":
        px, py = x - width / 2, y - height / 2
    elif origin == "cartesian":
        px, py = x - width / 2, height / 2 - y
    else:
        raise ValueError("Invalid coordinate origin")
    return {"x": px, "y": py, "x_norm": px / width, "y_norm": py / height, "origin": origin}
