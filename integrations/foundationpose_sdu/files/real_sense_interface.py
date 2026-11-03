from dataclasses import dataclass
import cv2
import numpy as np
import pyrealsense2 as rs


def find_realsense_devices():
    ctx = rs.context()
    devices = ctx.query_devices()
    serial_numbers = []
    
    for device in devices:
        serial_number = device.get_info(rs.camera_info.serial_number)
        serial_numbers.append(serial_number)
        print(f"Found device: {device.get_info(rs.camera_info.name)}")
        print(f"Serial number: {serial_number}")
    
    return serial_numbers


@dataclass
class CaptureResult:
    ir_left: np.ndarray
    ir_right: np.ndarray


class RealsenseCamera:
    def __init__(self, device_number=0):
        self.pipeline = rs.pipeline()
        config = rs.config()
        h, w = 480, 640
        self.w_offset = int(0.17 * w)

        ctx = rs.context()
        devices = ctx.query_devices()
        serial_numbers = devices[device_number].get_info(rs.camera_info.serial_number)
        print("serial_number: ", serial_numbers)
        config.enable_device(serial_numbers)

        for sensor in devices[device_number].query_sensors():
            print(f"  {sensor.get_info(rs.camera_info.name)}")
  
        # Enable all streams
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        pipeline_profile = self.pipeline.start(config)
        # Create alignment object to align depth to color
        self.align = rs.align(rs.stream.color)

        depth_profile = pipeline_profile.get_stream(rs.stream.depth)
        color_profile = pipeline_profile.get_stream(rs.stream.color)
        depth_intr = depth_profile.as_video_stream_profile().get_intrinsics()
        color_intr = color_profile.as_video_stream_profile().get_intrinsics()
        self.depth_intr = depth_intr
        self.color_intr = color_intr
        self.intr = color_intr

        # self.cam_intrinsics = np.array([
        # [intr.ppx, 0, intr.fx], 
        # [0, intr.ppy, intr.fy], 
        # [0, 0, 1]])

        self.cam_intrinsics = np.array([
        [color_intr.fx, 0, color_intr.ppx], 
        [0, color_intr.fy, color_intr.ppy], 
        [0, 0, 1]])
        
        # Set high accuracy for depth sensor
        depth_sensor = pipeline_profile.get_device().first_depth_sensor()
        depth_sensor.set_option(
            rs.option.visual_preset, 5
        )
        self.depth_scale = depth_sensor.get_depth_scale()

        presets = depth_sensor.get_option_range(rs.option.visual_preset)
        print(f"Available presets: {presets}")  # Will show min/max values

        if depth_sensor.supports(rs.option.laser_power):
            laser_range = depth_sensor.get_option_range(rs.option.laser_power)
            depth_sensor.set_option(rs.option.laser_power, laser_range.max)
            print(f"Laser power set to max: {laser_range.max}")

         # Get depth scale (important!)
        # depth_sensor = pipeline_profile.get_device().first_depth_sensor()
        # depth_scale = depth_sensor.get_depth_scale()
        # print(f"Depth Scale: {depth_scale}")  # Typically 0.001 for millimeters to meters
        self.pipeline.wait_for_frames()


    def get_aligned_frames(self):
        """
        Get aligned color and depth frames
        """
        # Wait for a coherent pair of frames
        frames = self.pipeline.wait_for_frames()
        
        # Align depth frame to color frame
        aligned_frames = self.align.process(frames)
        
        # Get aligned frames
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            return None, None
        
        # Convert to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        return color_image, depth_image

    def capture(self):
        frames = self.pipeline.wait_for_frames(timeout_ms=200) # originally 200 ms
        img = []
        for idx in 1, 2:
            ir = frames.get_infrared_frame(idx)
            ir = np.asanyarray(ir.data)
            if False:
                if idx == 1:
                    ir = ir[:, self.w_offset :]
                else:
                    ir = ir[:, : -self.w_offset]
            img.append(ir)

        return CaptureResult(ir_left=img[0], ir_right=img[1])


    def get_frame(self):
        frames = self.pipeline.wait_for_frames(timeout_ms=200) # originally 200 ms
        # Align depth frame to color frame
        # Create alignment object to align depth to color
        self.align = rs.align(rs.stream.color)

        # Extract different frames
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        # ir1_frame = frames.get_infrared_frame(1)  # Left IR camera
        # ir2_frame = frames.get_infrared_frame(2)  # Right IR camera
        
        # frame = np.asanyarray(ir1_frame.data)
        frame = np.asanyarray(color_frame.data)
        depth_frame = np.asanyarray(depth_frame.data)
        
        return frame, depth_frame

    def get_depth_at_pixel(self, depth_image, x, y):
        """Get depth value in meters at specific pixel"""
        if depth_image is None:
            return None
        
        # Get depth value in millimeters (original unit)
        depth_value = depth_image[y, x]
        
        if depth_value == 0:
            return 0  # No depth data at this pixel
        
        # Convert to meters (RealSense typically stores depth in millimeters)
        depth_in_meters = depth_value / 1000.0
        
        return depth_in_meters
    
    def visualize_depth(self, depth_image):
        """Create a colorized visualization of depth data"""
        if depth_image is None:
            return None
        
        # Apply colormap for better visualization
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), 
            cv2.COLORMAP_JET
        )
        
        return depth_colormap



    @staticmethod
    def ir_to_bgr(capture_result: CaptureResult, align=0):
        h, w = capture_result.ir_left.shape
        w = w - align
        bgr = np.zeros(shape=(h, w, 3), dtype=np.uint8)
        bgr[..., 2] = capture_result.ir_left[:, align:]
        bgr[..., 0] = capture_result.ir_right[:, :w]
        return bgr


def _main():
    import cv2

    cam = RealsenseCamera()
    # while True:
    #     img = cam.ir_to_bgr(cam.capture(), align=128)
    #     # scale = 0.5
    #     # img = cv2.resize(
    #     #     img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
    #     # )
    #     cv2.imshow("", img)
    #     if cv2.waitKey(1) == ord("q"):
    #         break


if __name__ == "__main__":
    _main()
