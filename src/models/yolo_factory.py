"""
YOLO Factory module for loading different YOLO model versions using a unified interface.
"""

import os
from enum import Enum
from typing import Union, Dict, Any, Optional, List, Tuple

import torch
import numpy as np


class YOLOVersion(Enum):
    """Enum for supported YOLO versions."""
    YOLOV3 = "yolov3"
    YOLOV4 = "yolov4"
    YOLOV5 = "yolov5"
    YOLOV6 = "yolov6"
    YOLOV7 = "yolov7"
    YOLOV8 = "yolov8"
    YOLO_NAS = "yolo_nas"


class YOLOFactory:
    """Factory class for creating YOLO model instances."""
    
    @staticmethod
    def create(version: Union[str, YOLOVersion], 
               model_size: str = "m",
               pretrained: bool = True,
               **kwargs) -> Any:
        """
        Create a YOLO model of the specified version.
        
        Args:
            version: YOLO version to use (e.g., "yolov8", YOLOVersion.YOLOV8)
            model_size: Model size/variant ("n", "s", "m", "l", "x")
            pretrained: Whether to load pretrained weights
            **kwargs: Additional arguments for specific model versions
            
        Returns:
            YOLO model instance
        
        Raises:
            ValueError: If the specified version is not supported
            ImportError: If the required package for the model is not installed
        """
        if isinstance(version, str):
            try:
                version = YOLOVersion(version.lower())
            except ValueError:
                raise ValueError(f"Unknown YOLO version: {version}. "
                                f"Supported versions: {[v.value for v in YOLOVersion]}")
        
        if version == YOLOVersion.YOLOV8:
            try:
                from ultralytics import YOLO
                model_path = f"yolov8{model_size}.pt" if pretrained else None
                return YOLO(model_path) if model_path else YOLO(kwargs.get("cfg", f"yolov8{model_size}.yaml"))
            except ImportError:
                raise ImportError("Please install ultralytics package: pip install ultralytics")
        
        elif version == YOLOVersion.YOLOV5:
            try:
                # Using PyTorch Hub to load YOLOv5
                model_name = f"yolov5{model_size}"
                if pretrained:
                    model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
                else:
                    model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=False)
                return model
            except Exception as e:
                raise ImportError(f"Error loading YOLOv5: {str(e)}. "
                                 f"Try: pip install -U torch torchvision")
        
        elif version == YOLOVersion.YOLOV3 or version == YOLOVersion.YOLOV4:
            # For these versions we could use different implementations
            # For example, Darknet, PyTorch-YOLOv3/v4, etc.
            raise NotImplementedError(f"{version.value} implementation coming soon")
        
        elif version == YOLOVersion.YOLOV6:
            raise NotImplementedError(f"{version.value} implementation coming soon")
        
        elif version == YOLOVersion.YOLOV7:
            raise NotImplementedError(f"{version.value} implementation coming soon")
        
        elif version == YOLOVersion.YOLO_NAS:
            raise NotImplementedError(f"{version.value} implementation coming soon")
        
        else:
            raise ValueError(f"Unsupported YOLO version: {version}")

    @staticmethod
    def get_available_versions() -> List[str]:
        """Get list of available YOLO versions that can be created."""
        # In a real implementation, we would check which versions are actually
        # installable given the current environment
        return [v.value for v in YOLOVersion]


if __name__ == "__main__":
    # Simple test to verify the factory works
    print(f"Available YOLO versions: {YOLOFactory.get_available_versions()}")
    
    try:
        model = YOLOFactory.create("yolov8", model_size="n", pretrained=True)
        print(f"Successfully created YOLOv8 model: {model}")
    except Exception as e:
        print(f"Error creating YOLOv8 model: {str(e)}") 