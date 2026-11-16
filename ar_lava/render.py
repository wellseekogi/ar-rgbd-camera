import cv2
import numpy as np


def render_with_depth(frame, depth, calibration, game):
    """Compare real and virtual optical Z at the same source frame and pixel."""
    output = render(frame, None, calibration, game)
    h, w = depth.shape
    pixels = np.indices((h, w), dtype=np.float32)[::-1].transpose(1, 2, 0).reshape(-1, 1, 2)
    rays_xy = cv2.undistortPoints(pixels, calibration.matrix, calibration.distortion)
    rays = np.c_[rays_xy.reshape(-1, 2), np.ones(h * w)]
    normal = cv2.Rodrigues(calibration.rvec)[0][:, 2]
    numerator = float(normal @ calibration.tvec.reshape(3))
    denominator = rays @ normal
    floor_z = np.full(h * w, np.nan)
    np.divide(numerator, denominator, out=floor_z, where=np.abs(denominator) > 1e-8)
    floor_z = floor_z.reshape(h, w)
    occluder = np.isfinite(depth) & (floor_z > 0) & (depth < floor_z - 0.05)
    output[occluder] = (frame[occluder] * 0.45 + output[occluder] * 0.55).astype(np.uint8)
    return output


def render(frame, mask, calibration, game, background=None, opacity=0.45):
    if calibration is None or not calibration.ready:
        return frame
    cfg = game["config"]
    base = background.copy() if background is not None else frame.copy()
    overlay = base.copy()
    dangerous = set(game["cells"])
    for row in range(cfg["rows"]):
        for col in range(cfg["cols"]):
            x = cfg["origin_x"] + col * cfg["cell_m"]
            y = cfg["origin_y"] + row * cfg["cell_m"]
            side = cfg["cell_m"]
            world = [[x, y, 0], [x + side, y, 0], [x + side, y + side, 0], [x, y + side, 0]]
            camera_points = (calibration.projection @ np.c_[world, np.ones(4)].T).T
            if np.any(camera_points[:, 2] <= 0.05):
                continue
            polygon = calibration.project(world)
            if not np.isfinite(polygon).all() or np.max(np.abs(polygon)) > 100_000:
                continue
            polygon = polygon.astype(np.int32)
            cell = row * cfg["cols"] + col
            color = (115, 115, 48)
            if cell in dangerous:
                color = (40, 50, 235) if game["phase"] == "lava" else (40, 165, 245)
            cv2.fillConvexPoly(overlay, polygon, color)
            cv2.polylines(overlay, [polygon], True, (215, 220, 220), 1, cv2.LINE_AA)
    output = cv2.addWeighted(base, 0.25, overlay, 0.75, 0)
    if mask is not None:
        alpha = np.clip(cv2.resize(mask, (frame.shape[1], frame.shape[0])), 0, 1)
        alpha = (alpha * opacity)[..., None]
        output = (frame * alpha + output * (1 - alpha)).astype(np.uint8)
    return output
