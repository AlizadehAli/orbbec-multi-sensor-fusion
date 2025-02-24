# Multi-Sensor Processing Pipeline
## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Available Scripts](#available-scripts)
6. [File Structure](#file-structure)
7. [Author](#author)

## Overview

The **Multi-Sensor Processing Pipeline** is a multi-threaded system designed to fetch, process, and render frames from multiple Orbbec sensors in real-time. It uses **Open3D**, **OpenCV** and **pyorbbecsdk** library to fetch frames from multiple Orbbec sensors, apply transformations, register color and depth frames, and process point cloud data in real-time. This project extends examples from **Orbbec 3D Technology, Inc.** and is optimized for **sensor fusion** and **point cloud processing**.

## Features

- **Multi-sensor support**: Handles multiple Orbbec sensors simultaneously.
- **Frame synchronization**: Ensures accurate depth and color alignment.
- **Real-time processing**: Utilizes multi-threading for efficiency.
- **Point cloud visualization**: Uses Open3D to render processed point clouds.
- **Configurable pipeline**: Uses a YAML configuration file for easy customization.
- **Data saving**: Optionally saves color, depth, and point cloud files for calibration and analysis.

## Installation

### Dependencies

Ensure you have the following dependencies installed:

```bash
pip install open3d opencv-python numpy pyyaml
```

### Orbbec SDK

The **pyorbbecsdk** library must be installed and accessible. Follow the official Orbbec installation guide to set up the SDK.

## Usage

### Running the Script

To execute the script, use:

```bash
python multi_sensor_processing.py --config config.yml
```
```bash
python multi_sensor_processing_multithread.py --config config.yml
```

### Configuration

Modify the `config.yml` file to set parameters for:

- **Calibration transformation matrix**
- **Sensor processing options**
- **Saving frequency and directories**
- **Visualization settings**

### Example YAML Configuration

```yaml
calibration:
  path: "./calib_data"
  transformation_file: "transformation_mtx.txt"

processing:
  target_sensor: 1
  max_devices: 2
  max_queue_size: 5
  min_depth: 20  # in mm
  max_depth: 10000  # in mm

saving:
  enable_saving: false
  save_directory: "./output"
  save_frequency: 10

visualization:
  window_name: "Online Point Cloud"
  window_width: 640
  window_height: 480
  point_size: 0.2
  ESC_key: 27

```

## Available Scripts

This repository contains two scripts for processing multiple Orbbec sensors:

- **multi\_sensor\_processing.py**: Standard processing pipeline for handling multiple sensors.
- **multi\_sensor\_processing\_multithread.py**: A multithreaded version for improved performance and efficiency.

## File Structure

```plaintext
project-root/
├── multi_sensor_processing.py  # Main script for sensor processing
├── multi_sensor_processing_multithread.py  # Main script for multi-thread sensor processing
├── config.yml                  # Configuration file
├── utils.py                     # Utility functions for image and point cloud processing
├── output/                      # Directory for saved output files
│   ├── color_0_xxx.png
│   ├── depth_0_xxx.png
│   ├── point_cloud_0_xxx.ply
│   └── ...
├── calib_data/                  # Directory for calibration files
│   ├── transformation_mtx.txt
│   └── ...
└── README.md                   # Documentation
```

## Author

**Ali Alizadeh** - Based on examples from Orbbec 3D Technology, Inc.

