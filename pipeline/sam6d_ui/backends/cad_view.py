"""Render the tracked CAD and the same centered axes as the camera overlay."""
import cv2
import numpy as np


def draw_reference_axes(image, pose, K, length):
    points = np.array([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]])
    camera_points = points @ pose[:3, :3].T + pose[:3, 3]
    if camera_points[0, 2] <= .001:
        return image
    projected = camera_points @ K.T
    projected = projected[:, :2] / projected[:, 2:]
    if not np.isfinite(projected).all() or np.abs(projected).max() > 1e6:
        return image
    pixels = np.round(projected).astype(int)
    origin = tuple(pixels[0])
    for index, (label, color) in enumerate((('X', (255, 65, 65)), ('Y', (70, 255, 100)), ('Z', (80, 150, 255))), 1):
        if camera_points[index, 2] <= .001:
            continue
        endpoint = tuple(pixels[index])
        cv2.arrowedLine(image, origin, endpoint, color, 2, cv2.LINE_AA, tipLength=.15)
        cv2.putText(image, label, (endpoint[0]+4, endpoint[1]-4), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
    return image


def render_cad_view(mesh, pose, K, shape, mesh_tensors, glctx):
    import torch
    from Utils import nvdiffrast_render
    height, width = shape[:2]
    scale = min(1., 640 / width)
    height, width = round(height * scale), round(width * scale)
    intrinsics = np.asarray(K, dtype=float).copy()
    intrinsics[:2] *= scale
    with torch.inference_mode():
        color, _, _ = nvdiffrast_render(
            K=intrinsics, H=height, W=width,
            ob_in_cams=torch.as_tensor(pose, device=mesh_tensors['pos'].device, dtype=torch.float32)[None],
            glctx=glctx, mesh_tensors=mesh_tensors, use_light=True, extra={})
        image = (color[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    image = np.ascontiguousarray(image)
    draw_reference_axes(image, pose, intrinsics, min(float(mesh.extents.max()) / 2, .1))
    ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError('Failed to encode the CAD view.')
    return encoded.tobytes()
