import sys
from omegaconf import OmegaConf
from colorama import init, Fore, Style
import numpy as np
from scipy.interpolate import CubicSpline

import os
from pathlib import Path

def setup_path_variables():
    
    PROJECT_ROOT = str(Path(__file__).resolve().parent)
    SRC_ROOT = str(Path(__file__).resolve().parent)
    os.environ['PROJECT_ROOT'] = PROJECT_ROOT
    os.environ['SRC_ROOT'] = SRC_ROOT

    return PROJECT_ROOT, SRC_ROOT

def load_config(main_config_path, task_config_path):
    """
    Load a main Hydra configuration and its referenced files from config_path using config_name
    without the YAML extension.
    """

    
    default_cfg = OmegaConf.create({
        "hydra": {
            "job": {
                "num": 0,  
                "override_dirname": "${name}"
            }
        }
    })

    
    cfg = OmegaConf.load(main_config_path)

    
    cfg = OmegaConf.merge(default_cfg, cfg)
    task_cfg = OmegaConf.load(task_config_path)
    cfg["task"] = task_cfg

    
    OmegaConf.resolve(cfg)

    return cfg

def linear_interpolation(points, step=0.01, target_dim=7):
    
    assert points.shape[1] == target_dim, "Input points must be 7-dimensional"
    num_points = points.shape[0]

    
    t = np.linspace(0, 1, num_points)
    
    
    t_interp = np.arange(0, 1, step)
    
    
    interpolated_points = np.zeros((len(t_interp), target_dim))
    for i in range(target_dim):
        interpolated_points[:, i] = np.interp(t_interp, t, points[:, i])

    return interpolated_points

def cubic_spline_interpolation(points, step=0.01, target_dim=7):
    
    assert points.shape[1] == target_dim, "Input points must be 7-dimensional"
    num_points = points.shape[0]

    
    t = np.linspace(0, 1, num_points)
    
    
    cs_list = [CubicSpline(t, points[:, i]) for i in range(target_dim)]
    
    
    t_interp = np.arange(0, 1, step)
    interpolated_points = np.array([cs(t_interp) for cs in cs_list]).T

    return interpolated_points

def get_user_input():
    prompt = f"""
    {Fore.CYAN}╭─ Command ──────────────╮
    │ {Fore.GREEN}▶ [Enter]{Fore.WHITE} Start        │
    │ {Fore.RED}✖ [q]{Fore.WHITE}     Quit         │
    ╰────────────────────────╯{Style.RESET_ALL}
    >>> """
    return input(prompt)

def get_start_command():
    
    sys.stdin.flush()
    while True:
        user_input = get_user_input()
        if user_input == 'q':
            return False
        elif user_input == '':
            return True
        else:
            print("Invalid input. Please press <Enter> or <q>.")


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def update_array(existing_array, new_array):
    """
    Append a new array along the leading dimension and move the preceding last entry into the
    second position.
    """
    
    
    updated_array = np.empty_like(existing_array)


    for i in range(0, existing_array.shape[0]):
        if i < existing_array.shape[0]-1:
            updated_array[i, ...] = existing_array[i+1, ...]
        else:
            
            updated_array[i, ...] = new_array

    return updated_array


def scale(x, lower, upper)-> np.ndarray:
    """
    Scale a list of angles from [-1, 1] to a specified range [lower, upper].

    Parameters:
    x (list or np.ndarray): List of angles in the range [-1, 1].
    lower (list or np.ndarray): Lower bounds for each angle.
    upper (list or np.ndarray): Upper bounds for each angle.

    Returns:
    np.ndarray: Scaled angles.
    """
    x = np.array(x, dtype=np.float32)
    lower = np.array(lower, dtype=np.float32)
    upper = np.array(upper, dtype=np.float32)

    if len(x.shape) > 1:
        lower = np.expand_dims(lower, axis=0)  # Shape (1, 7)
        upper = np.expand_dims(upper, axis=0)  # Shape (1, 7)

    return 0.5 * (x + 1.0) * (upper - lower) + lower


def unscale(x, lower, upper)-> np.ndarray:
    """
    Convert a list of scaled values to [-1, 1].

    Parameters:
    x (list of float): List of scaled values.
    lower (list of float): List of lower bounds for scaling.
    upper (list of float): List of upper bounds for scaling.
    
    Returns:
    list of float: List of unscaled values.
    """
    x = np.array(x, dtype=np.float32)
    lower = np.array(lower, dtype=np.float32)
    upper = np.array(upper, dtype=np.float32)
    
    
    if len(x.shape) > 1:
        lower = np.expand_dims(lower, axis=0)  # Shape (1, 7)
        upper = np.expand_dims(upper, axis=0)  # Shape (1, 7)
    
    
    return 2.0 * (x - lower) / (upper - lower) - 1.0

from typing import Dict, Callable, List
import torch

def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result

import threading
import open3d as o3d
import time

class MultiCameraVisualizer(threading.Thread): 
    def __init__(self, camera1, camera2, update_rate=10):
        super().__init__(daemon=True)  
        self.camera1 = camera1
        self.camera2 = camera2
        self.update_rate = update_rate  # Hz
        self.running = True
        self.vis = None
        
    def setup(self):
        
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        
        
        self.pcd1 = self.camera1.capture_point_cloud()
        self.pcd2 = self.camera2.capture_point_cloud()
        
        
        self.pcd1.paint_uniform_color([1, 0, 0])    
        self.pcd2.paint_uniform_color([0, 0, 1])    
        
        
        self.vis.add_geometry(self.pcd1)
        self.vis.add_geometry(self.pcd2)
        
        
        render_option = self.vis.get_render_option()
        render_option.point_size = 1.0
        render_option.background_color = np.array([0, 0, 0])
        
    def stop(self):
        self.running = False
        if self.vis is not None:
            self.vis.destroy_window()
            
    def run(self):
        """Run the worker thread."""
        try:
            self.setup()
            
            while self.running:
                
                try:
                    new_pcd1 = self.camera1.capture_point_cloud()
                    new_pcd2 = self.camera2.capture_point_cloud()
                    
                    
                    self.pcd1.points = new_pcd1.points
                    self.pcd1.colors = new_pcd1.colors
                    self.pcd2.points = new_pcd2.points
                    self.pcd2.colors = new_pcd2.colors
                    
                    
                    self.vis.update_geometry(self.pcd1)
                    self.vis.update_geometry(self.pcd2)
                    self.vis.poll_events()
                    self.vis.update_renderer()
                    
                    
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"Error updating point cloud: {e}")
                    
        except Exception as e:
            print(f"Visualizer thread error: {e}")
        finally:
            if self.vis is not None:
                self.vis.destroy_window()