"""
@file multi_sensor_processing.py
@brief A pipeline for fetching, processing, and rendering frames from multiple Orbbec sensors.

This script utilizes Open3D and OpenCV and pyorbbecsdk library to fetch frames from multiple Orbbec sensors, 
apply transformations, register color and depth frames, and process point cloud data in real-time.

@note This script is adapted and extended from Orbbec's official examples.
"""

import os
import sys
import cv2
import time
import yaml
import numpy as np
import open3d as o3d
import argparse

# Load SDK path
sdk_path = os.path.join(os.path.expanduser("~"), "pyorbbecsdk/install/lib")
sys.path.append(sdk_path)

from pyorbbecsdk import *  # Import SDK modules
from utils import frame_to_bgr_image, convert_to_o3d_point_cloud

def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
def main(cfg: dict):
    """Main function to run the frame processing pipeline."""
    save_dir = os.path.join(os.getcwd(), cfg['saving']['save_directory'])
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    tfile = os.path.join(cfg['calibration']['path'], cfg['calibration']['transformation_file'])
    with open(tfile, 'r') as file:
        values = [float(val) for val in file.read().strip().split(',')]
    if len(values) != 16:
        raise ValueError("The file does not contain 16 values required for a 4x4 matrix.")
    T = np.array(values).reshape((4, 4))
    print("Transformation matrix loaded.")

    # Initialize pyorbbec context, filters, and pipelines.
    ctx = Context()
    device_list = ctx.query_devices()
    num_devices = device_list.get_count()

    pipelines = []
    for i in range(num_devices):
        device = device_list.get_device_by_index(i)
        pipeline = Pipeline(device)
        pipeline.enable_frame_sync()
        config = Config()

        # Enable color stream.
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = profile_list.get_default_video_stream_profile()
        config.enable_stream(color_profile)

        # Enable depth stream.
        profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = profile_list.get_video_stream_profile(640, 0, OBFormat.Y16, 30)
        config.enable_stream(depth_profile)
        pipeline.start(config)
        pipelines.append(pipeline)

    camera_param = pipeline.get_camera_param()
    point_cloud_filter = PointCloudFilter()
    point_cloud_filter.set_camera_param(camera_param)

    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

    vis = o3d.visualization.Visualizer()
    vis.create_window("Online Point Cloud", cfg['visualization']['window_width'], cfg['visualization']['window_height'])
    render_option = vis.get_render_option()
    render_option.point_size = cfg['visualization']['point_size']
    pcds = {}  
    count = 0
    while True:
        for i, pipeline in enumerate(pipelines):
            loop_start = time.time()
            frames = pipeline.wait_for_frames(50)  # Adjust wait time as needed.
            if frames is None:
                continue

            aligned_frames = align_filter.process(frames)

            if aligned_frames is None:
                print(f"Alignment failed for sensor {i}")
                continue
            aligned_frames = aligned_frames.as_frame_set()
            color_frame_aligned = aligned_frames.get_color_frame()
            depth_frame_aligned = aligned_frames.get_depth_frame()
            if color_frame_aligned is None or depth_frame_aligned is None:
                continue

            color_image = frame_to_bgr_image(color_frame_aligned)
            if color_image is None:
                print(f"Color conversion failed for sensor {i}")
                continue
            has_color_sensor = True

            try:
                depth_data = np.frombuffer(
                    depth_frame_aligned.get_data(), dtype=np.uint16
                ).reshape((depth_frame_aligned.get_height(), depth_frame_aligned.get_width()))
            except ValueError:
                print(f"Depth reshape failed for sensor {i}")
                continue
            depth_data = depth_data.astype(np.float32) * depth_frame_aligned.get_depth_scale()
            depth_data = np.where((depth_data > cfg['processing']['min_depth']) & (depth_data < cfg['processing']['max_depth']), depth_data, 0)
            depth_data = depth_data.astype(np.uint16)

            if cfg['saving']['enable_saving']:
                if np.mod(count, cfg['saving']['save_frequency']) == 0:
                    depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
                    depth_image = cv2.applyColorMap(depth_image.astype(np.uint8), cv2.COLORMAP_JET)
                    combined = cv2.addWeighted(color_image, 0.5, depth_image, 0.5, 0)
                    cv2.imwrite(os.path.join(cfg['saving']['save_directory'], f"color_{i}_{count}.png"), color_image)
                    cv2.imwrite(os.path.join(cfg['saving']['save_directory'], f"depth_{i}_{count}.png"), depth_data)
                    cv2.imwrite(os.path.join(cfg['saving']['save_directory'], f"depth_{i}_vis_{count}.png"), depth_image)
                    cv2.imwrite(os.path.join(cfg['saving']['save_directory'], f"combined_{i}_{count}.png"), combined)

            scale = depth_frame_aligned.get_depth_scale()
            point_format = OBFormat.RGB_POINT if has_color_sensor else OBFormat.POINT
            point_cloud_filter.set_create_point_format(point_format)
            point_cloud_filter.set_position_data_scaled(scale)
            point_cloud_frame = point_cloud_filter.process(aligned_frames)
            pts = point_cloud_filter.calculate(point_cloud_frame)
            if pts is None or len(pts) == 0:
                continue

            pts = np.array(pts)
            points_array = pts[:, :3]
            colors_array = pts[:, 3:6].astype(np.uint8) if has_color_sensor else None

            pcd = convert_to_o3d_point_cloud(points_array, colors_array)
            if cfg['saving']['enable_saving']:
                if np.mod(count, cfg['saving']['save_frequency']) == 0:
                    file_name = os.path.join(cfg['saving']['save_directory'], f"point_cloud_{i}_{count}.ply")
                    o3d.io.write_point_cloud(file_name, pcd)
                    print(f"Point cloud saved to: {file_name}")

            if i == cfg['processing']['target_sensor']:
                pcd.transform(T)

            if i in pcds:
                pcds[i].points = pcd.points
                pcds[i].colors = pcd.colors
                vis.update_geometry(pcds[i])
            else:
                pcds[i] = pcd
                vis.add_geometry(pcds[i])
            
        loop_end = time.time()
        print(f"Frame processing: {1/(loop_end - loop_start)} FPS\n")

        vis.poll_events()
        vis.update_renderer()

        count += 1
        if cv2.waitKey(1) in [cfg['visualization']['ESC_key'], ord('q')]:
            break

    for pipeline in pipelines:
        pipeline.stop()
    vis.destroy_window()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frame processing pipeline for multiple Orbbec Sensors.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()
    
    config = load_config(args.config)
    main(config)