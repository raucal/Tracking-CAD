"""RGB-D acquisition and rectification; importing the SDK does not open the camera."""
import time

import cv2
import numpy as np


def devices():
    import pyrealsense2 as rs
    return [{'serial': item.get_info(rs.camera_info.serial_number),
             'name': item.get_info(rs.camera_info.name)}
            for item in rs.context().query_devices()]


class RealSenseSource:
    def __init__(self, options):
        import pyrealsense2 as rs
        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        available = devices()
        serial = options.get('serial')
        if not serial:
            if len(available) != 1:
                raise ValueError('Select a connected RealSense camera.' if available else
                                 'No Intel RealSense camera was detected on the server.')
            serial = available[0]['serial']
        config.enable_device(serial)
        width, height, fps = options['width'], options['height'], options['fps']
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.profile = self.pipeline.start(config)
        try:
            # A live consumer must discard old frames when slower than acquisition.
            for sensor in self.profile.get_device().query_sensors():
                if sensor.supports(rs.option.frames_queue_size):
                    try:sensor.set_option(rs.option.frames_queue_size,1)
                    except RuntimeError:pass  # Some firmware exposes a read-only option.
            self.align = rs.align(rs.stream.color)
            intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
            coefficients = np.asarray(intr.coeffs)
            self.maps = None
            if np.any(np.abs(coefficients) > 1e-10):
                if intr.model not in (rs.distortion.brown_conrady, rs.distortion.modified_brown_conrady):
                    raise ValueError('This distortion model requires compatible rectification calibration.')
                self.maps = cv2.initUndistortRectifyMap(K, coefficients, None, K, (width, height), cv2.CV_32FC1)
            self.camera = {'cam_K': K.reshape(-1).tolist(),
                           'depth_scale': self.profile.get_device().first_depth_sensor().get_depth_scale() * 1000,
                           'depth_scale_unit': 'millimeters_per_raw_unit', 'dist_coeffs': [0.] * 5,
                           'source_dist_coeffs': coefficients.tolist(), 'rectified': True,
                           'aligned_to': 'color', 'optical_frame': 'realsense_color_optical',
                           'serial': serial, 'width': width, 'height': height, 'fps': fps}
        except BaseException:
            self.pipeline.stop()
            raise

    def read(self):
        frames = self.pipeline.wait_for_frames(timeout_ms=2000)
        # Align only the newest synchronized frameset, not every queued pair.
        for _ in range(64):
            newer=self.pipeline.poll_for_frames()
            if not newer:break
            frames=newer
        frames = self.align.process(frames)
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color or not depth:
            raise RuntimeError('RealSense did not deliver an aligned RGB-D pair.')
        rgb = np.asanyarray(color.get_data()).copy()
        raw = np.asanyarray(depth.get_data()).copy()
        if self.maps is not None:
            rgb = cv2.remap(rgb, *self.maps, cv2.INTER_LINEAR)
            raw = cv2.remap(raw, *self.maps, cv2.INTER_NEAREST)
        return {'rgb': cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), 'depth': raw,
                'camera': dict(self.camera), 'number': color.get_frame_number(),
                'device_timestamp_ms': color.get_timestamp(), 'timestamp': time.time()}

    def close(self):
        self.pipeline.stop()
