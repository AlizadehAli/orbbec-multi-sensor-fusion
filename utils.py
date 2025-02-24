"""
@file utils.py
@brief Utility functions for frame format conversion and point cloud processing using Open3D and OpenCV.

This module provides utility functions to convert various image formats (YUYV, UYVY, I420, NV12, NV21, MJPG, RGB, BGR)
into BGR or RGB formats, as well as functions to handle Open3D point clouds.
"""

from typing import Union, Any, Optional
import cv2
import numpy as np
import open3d as o3d
from pyorbbecsdk import FormatConvertFilter, VideoFrame
from pyorbbecsdk import OBFormat, OBConvertFormat

def yuyv_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Convert a YUYV image frame to a BGR image.

    @param frame: Input YUYV image as a NumPy array.
    @param width: Image width.
    @param height: Image height.
    @return: Converted BGR image.
    """
    yuyv = frame.reshape((height, width, 2))
    return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)

def uyvy_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Convert a UYVY image frame to a BGR image.
    """
    uyvy = frame.reshape((height, width, 2))
    return cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)

def i420_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Convert an I420 image frame to a BGR image.
    """
    y = frame[0:height, :]
    u = frame[height:height + height // 4].reshape(height // 2, width // 2)
    v = frame[height + height // 4:].reshape(height // 2, width // 2)
    yuv_image = cv2.merge([y, u, v])
    return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_I420)

def nv21_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Convert an NV21 image frame to a BGR image.
    """
    y = frame[0:height, :]
    uv = frame[height:height + height // 2].reshape(height // 2, width)
    yuv_image = cv2.merge([y, uv])
    return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV21)

def nv12_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Convert an NV12 image frame to a BGR image.
    """
    y = frame[0:height, :]
    uv = frame[height:height + height // 2].reshape(height // 2, width)
    yuv_image = cv2.merge([y, uv])
    return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV12)

def determine_convert_format(frame: VideoFrame):
    """
    Determine the appropriate format conversion for the given video frame.
    """
    format_map = {
        OBFormat.I420: OBConvertFormat.I420_TO_RGB888,
        OBFormat.MJPG: OBConvertFormat.MJPG_TO_RGB888,
        OBFormat.YUYV: OBConvertFormat.YUYV_TO_RGB888,
        OBFormat.NV21: OBConvertFormat.NV21_TO_RGB888,
        OBFormat.NV12: OBConvertFormat.NV12_TO_RGB888,
        OBFormat.UYVY: OBConvertFormat.UYVY_TO_RGB888
    }
    return format_map.get(frame.get_format(), None)

def frame_to_rgb_frame(frame: VideoFrame) -> Union[Optional[VideoFrame], Any]:
    """
    Convert a video frame to an RGB frame if necessary.
    """
    if frame.get_format() == OBFormat.RGB:
        return frame
    convert_format = determine_convert_format(frame)
    if convert_format is None:
        print("Unsupported format")
        return None
    convert_filter = FormatConvertFilter()
    convert_filter.set_format_convert_format(convert_format)
    return convert_filter.process(frame)

def frame_to_bgr_image(frame: VideoFrame) -> Union[Optional[np.array], Any]:
    """
    Convert a video frame to a BGR image.
    """
    width, height = frame.get_width(), frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())
    
    format_converters = {
        OBFormat.RGB: lambda d: cv2.cvtColor(np.resize(d, (height, width, 3)), cv2.COLOR_RGB2BGR),
        OBFormat.BGR: lambda d: np.resize(d, (height, width, 3)),
        OBFormat.YUYV: lambda d: cv2.cvtColor(np.resize(d, (height, width, 2)), cv2.COLOR_YUV2BGR_YUYV),
        OBFormat.MJPG: lambda d: cv2.imdecode(d, cv2.IMREAD_COLOR),
        OBFormat.I420: lambda d: i420_to_bgr(d, width, height),
        OBFormat.NV12: lambda d: nv12_to_bgr(d, width, height),
        OBFormat.NV21: lambda d: nv21_to_bgr(d, width, height),
        OBFormat.UYVY: lambda d: cv2.cvtColor(np.resize(d, (height, width, 2)), cv2.COLOR_YUV2BGR_UYVY)
    }
    
    return format_converters.get(color_format, lambda _: None)(data)

def convert_to_o3d_point_cloud(points: np.ndarray, colors: np.ndarray = None) -> o3d.geometry.PointCloud:
    """
    Optimized conversion of NumPy arrays to Open3D PointCloud.
    Uses Open3D Tensor API for efficiency and then converts to legacy format for visualization.
    """
    points = np.asarray(points, dtype=np.float32)
    
    if colors is not None:
        colors = np.asarray(colors, dtype=np.float32) / 255.0  # Normalize to [0, 1]

    pcd_t = o3d.t.geometry.PointCloud()
    pcd_t.point.positions = o3d.core.Tensor(points)

    if colors is not None:
        pcd_t.point.colors = o3d.core.Tensor(colors)

    pcd = pcd_t.to_legacy()
    return pcd