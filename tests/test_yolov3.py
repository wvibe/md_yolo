"""
Tests for YOLOv3 model implementation using pytest
"""

import torch
from src.models.yolo_factory import YOLOFactory, YOLOVersion


def test_model_creation():
    """Test if the model can be created successfully"""
    model = YOLOFactory.create(YOLOVersion.YOLOV3, model_size="m", pretrained=False)
    assert model is not None


def test_forward_pass():
    """Test forward pass with a dummy input"""
    model = YOLOFactory.create(YOLOVersion.YOLOV3, model_size="m", pretrained=False)
    # Create a dummy input tensor with batch size of 1, 3 channels, and 416x416 image size
    x = torch.randn(1, 3, 416, 416)
    outputs = model(x)

    # Check if we get the expected number of outputs (3 detection scales)
    assert len(outputs) == 3

    # Check detection heads output shapes
    # Large objects: 1/32 scale
    assert outputs[0].shape[2:4] == torch.Size([13, 13])
    # Medium objects: 1/16 scale
    assert outputs[1].shape[2:4] == torch.Size([26, 26])
    # Small objects: 1/8 scale
    assert outputs[2].shape[2:4] == torch.Size([52, 52])

    # Check output channels: (num_classes + 5) * 3 anchors
    expected_channels = (80 + 5) * 3  # 80 COCO classes + 5 box params, with 3 anchors
    for output in outputs:
        assert output.shape[1] == expected_channels


def test_predict_method():
    """Test the prediction method"""
    model = YOLOFactory.create(YOLOVersion.YOLOV3, model_size="m", pretrained=False)
    # Random input image
    x = torch.randn(1, 3, 416, 416)
    # Set model to eval mode for inference
    model.eval()

    with torch.no_grad():
        detections = model.predict(x, conf_threshold=0.1)

    # We may or may not have detections with random input, but the function should return
    # a list containing a list (possibly empty) for each image in the batch
    assert len(detections) == 1  # 1 batch
    assert isinstance(detections[0], list)
