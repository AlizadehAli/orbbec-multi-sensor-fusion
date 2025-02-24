"""
@file multi_sensor_processing_multithread.py
@brief A multi-threaded pipeline for fetching, processing, and rendering frames from multiple Orbbec sensors.

This script utilizes Open3D and OpenCV and pyorbbecsdk library to fetch frames from multiple Orbbec sensors, 
apply transformations, register color and depth frames, and process point cloud data in real-time. The processing 
pipeline is designed for multi-threading, allowing separate nodes for frame fetching and processing.

@note This script is adapted and extended from Orbbec's official examples.
"""

import os
import sys
import cv2
import time
import yaml
import numpy as np
import open3d as o3d
import threading
import argparse
from queue import Queue
from typing import List

# Load SDK path
sdk_path = os.path.join(os.path.expanduser("~"), "pyorbbecsdk/install/lib")
sys.path.append(sdk_path)

from pyorbbecsdk import *  # Import SDK modules
from utils import frame_to_bgr_image, convert_to_o3d_point_cloud

def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def fetch_frames(pipeline: Pipeline, index: int, aligned_frames_queue: List[Queue], align_filter: AlignFilter, stop_event: threading.Event, config: dict):
    """Fetch frames from a given pipeline and add them to the queue."""
    while not stop_event.is_set():
        frames = pipeline.wait_for_frames(50)
        if frames is None:
            print(f"[Warning] No frames received from sensor {index}")
            continue
        aligned_frames = align_filter.process(frames).as_frame_set()
        if aligned_frames_queue[index].qsize() >= config['processing']['max_queue_size']:
            aligned_frames_queue[index].get()
        aligned_frames_queue[index].put(aligned_frames)

def start_streams(pipelines: List[Pipeline], aligned_frames_queue: List[Queue], align_filter: AlignFilter, stop_event: threading.Event, config: dict):
    """Start fetching frames from multiple pipelines in separate threads."""
    threads = []
    for index, pipeline in enumerate(pipelines):
        print(f"Starting device {index}")
        thread = threading.Thread(target=fetch_frames, args=(pipeline, index, aligned_frames_queue, align_filter, stop_event, config))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    return threads

def stop_streams(pipelines: List[Pipeline], stop_event: threading.Event):
    """Stop all running pipelines."""
    stop_event.set()
    for pipeline in pipelines:
        pipeline.stop()

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
    
    ctx = Context()
    device_list = ctx.query_devices()
    num_devices = device_list.get_count()
    if num_devices == 0:
        print("No device connected")
        return
    if num_devices > cfg['processing']['max_devices']:
        print("Too many devices connected")
        return
    
    pipelines = []
    aligned_frames_queue: List[Queue] = [Queue() for _ in range(num_devices)]
    has_color_sensor: List[bool] = [False for _ in range(num_devices)]
    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
    stop_event = threading.Event()
    
    for i in range(num_devices):
        device = device_list.get_device_by_index(i)
        pipeline = Pipeline(device)
        pipeline.enable_frame_sync()
        config_obj = Config()
        
        # Enable color stream
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = profile_list.get_default_video_stream_profile()
        config_obj.enable_stream(color_profile)
        has_color_sensor[i] = True
        
        # Enable depth stream
        profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = profile_list.get_video_stream_profile(640, 0, OBFormat.Y16, 30)
        config_obj.enable_stream(depth_profile)
        pipeline.start(config_obj)
        pipelines.append(pipeline)
    
    camera_param = pipeline.get_camera_param()
    point_cloud_filter = PointCloudFilter()
    point_cloud_filter.set_camera_param(camera_param)

    threads = start_streams(pipelines, aligned_frames_queue, align_filter, stop_event, config)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window("Online Point Cloud", cfg['visualization']['window_width'], cfg['visualization']['window_height'])
    render_option = vis.get_render_option()
    render_option.point_size = cfg['visualization']['point_size']
    pcds = {}
    count = 0
    try:
        while not stop_event.is_set():
            loop_start = time.time()
            for i in range(num_devices):
                if aligned_frames_queue[i].empty():
                    continue
                
                aligned_frames = aligned_frames_queue[i].get()
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if color_frame is None or depth_frame is None:
                    print(f"[Warning] Missing frames for sensor {i}")
                    continue
                
                # color_image = frame_to_bgr_image(color_frame) if color_frame else None
                depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                depth_data = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))
                depth_data = depth_data.astype(np.float32) * depth_frame.get_depth_scale()
                depth_data = np.clip(depth_data, cfg['processing']['min_depth'], cfg['processing']['max_depth'])
                depth_data = depth_data.astype(np.uint16)

                if cfg['saving']['enable_saving']:
                    if np.mod(count, cfg['saving']['save_frequency']) == 0:
                        color_image = frame_to_bgr_image(color_frame) if color_frame else None
                        depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
                        depth_image = cv2.applyColorMap(depth_image.astype(np.uint8), cv2.COLORMAP_JET)
                        combined = cv2.addWeighted(color_image, 0.5, depth_image, 0.5, 0)
                        cv2.imwrite(os.path.join(cfg['saving']['save_directory'], f"color_{i}_{count}.png"), color_image)
                        cv2.imwrite(os.path.join(cfg['saving']['save_directory'], f"depth_{i}_{count}.png"), depth_data)
                        cv2.imwrite(os.path.join(cfg['saving']['save_directory'], f"depth_{i}_vis_{count}.png"), depth_image)
                        cv2.imwrite(os.path.join(cfg['saving']['save_directory'], f"combined_{i}_{count}.png"), combined)

                scale = depth_frame.get_depth_scale()
                point_format = OBFormat.RGB_POINT if has_color_sensor[i] else OBFormat.POINT
                point_cloud_filter.set_create_point_format(point_format)
                point_cloud_filter.set_position_data_scaled(scale)
                point_cloud_frame = point_cloud_filter.process(aligned_frames)
                pts = point_cloud_filter.calculate(point_cloud_frame)
                
                if pts is None or len(pts) == 0:
                    continue
                
                pts = np.array(pts)
                points_array = pts[:, :3]
                colors_array = pts[:, 3:6] if has_color_sensor[i] else None
                
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
            print(f"Frame processing: {1/(loop_end - loop_start)} FPS")
            count += 1
            vis.poll_events()
            vis.update_renderer()
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_streams(pipelines, stop_event)
        vis.destroy_window()
        cv2.destroyAllWindows()
        for thread in threads:
            thread.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-thread frame processing pipeline for multiple Orbbec sensors.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()
    
    config = load_config(args.config)
    main(config)