"""
YOLOv3 model implementation based on the paper:
"YOLOv3: An Incremental Improvement" by Joseph Redmon and Ali Farhadi (2018)
https://arxiv.org/abs/1804.02767

This implementation includes the full YOLOv3 architecture with Darknet-53 backbone.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Standard convolution block with BatchNorm and LeakyReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2  # Same spatial dimensions

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.leaky_relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """
    Residual block used in Darknet-53: conv1x1 followed by conv3x3 with skip connection
    """

    def __init__(self, channels: int):
        super().__init__()
        reduced_channels = channels // 2

        self.conv1 = ConvBlock(channels, reduced_channels, kernel_size=1)
        self.conv2 = ConvBlock(reduced_channels, channels, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        return out + residual


class Darknet53(nn.Module):
    """
    Darknet-53 backbone architecture for YOLOv3
    """

    def __init__(self):
        super().__init__()

        # Initial conv layer
        self.conv1 = ConvBlock(3, 32, kernel_size=3)

        # Downsample + Residual blocks
        self.conv2 = ConvBlock(32, 64, kernel_size=3, stride=2)
        self.res_block1 = self._make_layer(64, 1)

        self.conv3 = ConvBlock(64, 128, kernel_size=3, stride=2)
        self.res_block2 = self._make_layer(128, 2)

        self.conv4 = ConvBlock(128, 256, kernel_size=3, stride=2)
        self.res_block3 = self._make_layer(256, 8)

        self.conv5 = ConvBlock(256, 512, kernel_size=3, stride=2)
        self.res_block4 = self._make_layer(512, 8)

        self.conv6 = ConvBlock(512, 1024, kernel_size=3, stride=2)
        self.res_block5 = self._make_layer(1024, 4)

    def _make_layer(self, channels: int, num_blocks: int) -> nn.Sequential:
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(channels))
        return nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res_block1(x)

        x = self.conv3(x)
        x = self.res_block2(x)

        x = self.conv4(x)
        x = self.res_block3(x)
        route1 = x  # First route: layer 36 (after 8 residual blocks at 256 channels)

        x = self.conv5(x)
        x = self.res_block4(x)
        route2 = x  # Second route: layer 61 (after 8 residual blocks at 512 channels)

        x = self.conv6(x)
        x = self.res_block5(x)
        route3 = x  # Last feature map

        return route1, route2, route3


class YOLODetectionBlock(nn.Module):
    """
    YOLO Detection Block that processes feature maps at different scales
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, out_channels // 2, kernel_size=1)
        self.conv2 = ConvBlock(out_channels // 2, out_channels, kernel_size=3)
        self.conv3 = ConvBlock(out_channels, out_channels // 2, kernel_size=1)
        self.conv4 = ConvBlock(out_channels // 2, out_channels, kernel_size=3)
        self.conv5 = ConvBlock(out_channels, out_channels // 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        feature_map = self.conv5(x)

        return feature_map


class YOLOv3(nn.Module):
    """
    YOLOv3 model with Darknet-53 backbone
    """

    def __init__(
        self, num_classes: int = 80, anchors: Optional[List[Tuple[int, int]]] = None
    ):
        super().__init__()

        # Default COCO anchors (3 anchor boxes per scale, 3 scales)
        if anchors is None:
            self.anchors = [
                # Large objects (stride 32)
                [(116, 90), (156, 198), (373, 326)],
                # Medium objects (stride 16)
                [(30, 61), (62, 45), (59, 119)],
                # Small objects (stride 8)
                [(10, 13), (16, 30), (33, 23)],
            ]
        else:
            # Group anchors into 3 scales, 3 anchors per scale
            assert len(anchors) == 9, "Expected 9 anchor boxes (3 per scale)"
            self.anchors = [anchors[i : i + 3] for i in range(0, 9, 3)]

        self.num_classes = num_classes

        # Output channels: (5 + num_classes) * 3 for each box (5 = 4 box coordinates + 1 objectness)
        self.out_channels = (5 + num_classes) * 3

        # Darknet-53 backbone
        self.darknet = Darknet53()

        # Detection layers
        # Large object detection (stride 32)
        self.detect1 = YOLODetectionBlock(1024, 1024)
        self.conv_out1 = nn.Conv2d(512, self.out_channels, kernel_size=1)

        # Upsampling and medium object detection (stride 16)
        self.conv_up1 = ConvBlock(512, 256, kernel_size=1)
        self.detect2 = YOLODetectionBlock(768, 512)  # 512 + 256 = 768 (after concat)
        self.conv_out2 = nn.Conv2d(256, self.out_channels, kernel_size=1)

        # Upsampling and small object detection (stride 8)
        self.conv_up2 = ConvBlock(256, 128, kernel_size=1)
        self.detect3 = YOLODetectionBlock(384, 256)  # 256 + 128 = 384 (after concat)
        self.conv_out3 = nn.Conv2d(128, self.out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Backbone forward pass
        route_36, route_61, x = self.darknet(x)

        # Large object detection (first yolo layer)
        x = self.detect1(x)
        detect1 = self.conv_out1(x)

        # Medium object detection (second yolo layer)
        x = self.conv_up1(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat((x, route_61), dim=1)
        x = self.detect2(x)
        detect2 = self.conv_out2(x)

        # Small object detection (third yolo layer)
        x = self.conv_up2(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat((x, route_36), dim=1)
        x = self.detect3(x)
        detect3 = self.conv_out3(x)

        return [detect1, detect2, detect3]

    def _reshape_output(
        self, outputs: List[torch.Tensor], input_shape: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reshape the outputs to be more interpretable.

        Args:
            outputs: Raw outputs from the network [detect1, detect2, detect3]
            input_shape: Original input shape (height, width)

        Returns:
            Tuple of tensors: (boxes, objectness, class_probs, raw_outputs)
        """
        batch_size = outputs[0].size(0)

        boxes_list = []
        objectness_list = []
        class_probs_list = []

        for i, output in enumerate(outputs):
            anchors = torch.tensor(self.anchors[i], device=output.device)

            # Get grid size based on output size
            grid_size = output.shape[2:4]
            stride = torch.tensor(
                [input_shape[1] / grid_size[1], input_shape[0] / grid_size[0]],
                device=output.device,
            )

            # Reshape output to [batch, num_anchors, grid_h, grid_w, 5+num_classes]
            output = (
                output.reshape(
                    batch_size, 3, 5 + self.num_classes, grid_size[0], grid_size[1]
                )
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )

            # Get outputs
            x = torch.sigmoid(output[..., 0])  # Center x
            y = torch.sigmoid(output[..., 1])  # Center y
            w = output[..., 2]  # Width
            h = output[..., 3]  # Height
            objectness = torch.sigmoid(output[..., 4])  # Objectness confidence
            class_probs = torch.sigmoid(output[..., 5:])  # Class predictions

            # Generate grid
            grid_x = torch.arange(
                grid_size[1], dtype=torch.float, device=output.device
            ).repeat(grid_size[0], 1)
            grid_y = (
                torch.arange(grid_size[0], dtype=torch.float, device=output.device)
                .repeat(grid_size[1], 1)
                .t()
            )

            # Shape: [1, 1, grid_h, grid_w]
            grid_x = grid_x.unsqueeze(0).unsqueeze(0)
            grid_y = grid_y.unsqueeze(0).unsqueeze(0)

            # Add offset to center predictions
            x = (x + grid_x) * stride[0]
            y = (y + grid_y) * stride[1]

            # Scale anchors to the prediction scale
            anchors = anchors / stride.unsqueeze(0)

            # Transform width and height
            w = torch.exp(w) * anchors[:, 0:1].view(1, 3, 1, 1)
            h = torch.exp(h) * anchors[:, 1:2].view(1, 3, 1, 1)

            # Scale back to original dimensions
            w = w * stride[0]
            h = h * stride[1]

            # Convert to x1, y1, x2, y2 format
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2

            # Stack boxes: [batch, num_anchors, grid_h, grid_w, 4]
            boxes = torch.stack([x1, y1, x2, y2], dim=-1)

            boxes_list.append(boxes.reshape(batch_size, -1, 4))
            objectness_list.append(objectness.reshape(batch_size, -1, 1))
            class_probs_list.append(
                class_probs.reshape(batch_size, -1, self.num_classes)
            )

        # Concatenate predictions from the three scales
        boxes = torch.cat(boxes_list, dim=1)
        objectness = torch.cat(objectness_list, dim=1)
        class_probs = torch.cat(class_probs_list, dim=1)

        return boxes, objectness, class_probs, outputs

    def predict(
        self, x: torch.Tensor, conf_threshold: float = 0.5, nms_threshold: float = 0.4
    ) -> List[List[Dict[str, torch.Tensor]]]:
        """
        Perform object detection on the input image.

        Args:
            x: Input tensor of shape [batch_size, 3, height, width]
            conf_threshold: Confidence threshold for filtering detections
            nms_threshold: IoU threshold for non-maximum suppression

        Returns:
            List of detections for each image in the batch, where each detection is a
            dictionary containing 'box', 'score', and 'class' keys
        """
        input_shape = x.shape[2:4]  # (height, width)
        outputs = self.forward(x)

        # Reshape outputs to standard format
        boxes, objectness, class_probs, _ = self._reshape_output(outputs, input_shape)

        # Convert to CPU for post-processing
        boxes = boxes.detach().cpu()
        objectness = objectness.detach().cpu()
        class_probs = class_probs.detach().cpu()

        batch_detections = []

        # Process each image in batch
        for b in range(boxes.shape[0]):
            img_boxes = boxes[b]
            img_objectness = objectness[b].squeeze()
            img_class_probs = class_probs[b]

            # Calculate class scores (objectness * class probability)
            class_scores = img_objectness.unsqueeze(1) * img_class_probs

            # Get max scores and classes
            max_scores, max_classes = torch.max(class_scores, dim=1)

            # Filter by confidence threshold
            keep = max_scores > conf_threshold
            if not keep.any():
                # No detections above threshold
                batch_detections.append([])
                continue

            filtered_boxes = img_boxes[keep]
            filtered_scores = max_scores[keep]
            filtered_classes = max_classes[keep]

            # Apply NMS class-wise
            unique_classes = filtered_classes.unique()
            detection_list = []

            for cls in unique_classes:
                cls_mask = filtered_classes == cls

                # NMS operations for current class
                cls_boxes = filtered_boxes[cls_mask]
                cls_scores = filtered_scores[cls_mask]

                # Sort by confidence
                sorted_idx = torch.argsort(cls_scores, descending=True)
                cls_boxes = cls_boxes[sorted_idx]
                cls_scores = cls_scores[sorted_idx]

                keep_idx = []
                while cls_boxes.size(0) > 0:
                    # Add the box with highest confidence
                    keep_idx.append(sorted_idx[0].item())

                    # If only one box remains, stop
                    if cls_boxes.size(0) == 1:
                        break

                    # Calculate IoU with other boxes
                    IoU = self._box_iou(cls_boxes[0].unsqueeze(0), cls_boxes[1:])

                    # Filter boxes with IoU > threshold
                    mask = IoU.squeeze(0) <= nms_threshold
                    cls_boxes = cls_boxes[1:][mask]
                    cls_scores = cls_scores[1:][mask]
                    sorted_idx = sorted_idx[1:][mask]

                # Add detections for this class
                for idx in keep_idx:
                    detection_list.append(
                        {
                            "box": filtered_boxes[idx],
                            "score": filtered_scores[idx],
                            "class": cls.item(),
                        }
                    )

            batch_detections.append(detection_list)

        return batch_detections

    @staticmethod
    def _box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """
        Calculate IoU between boxes.

        Args:
            box1: First box tensor of shape (..., 4)
            box2: Second box tensor of shape (..., 4)

        Returns:
            IoU between box1 and box2
        """
        # Calculate intersection area
        x1 = torch.max(box1[..., 0].unsqueeze(-1), box2[..., 0])
        y1 = torch.max(box1[..., 1].unsqueeze(-1), box2[..., 1])
        x2 = torch.min(box1[..., 2].unsqueeze(-1), box2[..., 2])
        y2 = torch.min(box1[..., 3].unsqueeze(-1), box2[..., 3])

        # Intersection area
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

        # Union area
        box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
        box2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
        union = box1_area.unsqueeze(-1) + box2_area - intersection

        # IoU
        return intersection / (union + 1e-6)

    def load_darknet_weights(self, weights_path: str) -> None:
        """
        Load pre-trained weights from Darknet format.

        Args:
            weights_path: Path to the .weights file
        """
        # This would be a more complex method to parse the binary darknet weights format
        # For simplicity, we'll just acknowledge that this method would be needed
        # for loading official pre-trained weights
        raise NotImplementedError("Loading Darknet weights not implemented yet")
