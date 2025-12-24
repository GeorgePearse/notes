# YOLO (You Only Look Once)

The most popular real-time object detection family.

## Evolution Timeline

```
2016        2017        2018        2020        2021        2022        2023
  │           │           │           │           │           │           │
  ▼           ▼           ▼           ▼           ▼           ▼           ▼
┌─────┐   ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐
│v1   │──►│v2   │────►│v3   │────►│v4   │────►│v5   │────►│v6/v7│────►│v8   │
│     │   │9000 │     │     │     │     │     │     │     │     │     │     │
└─────┘   └─────┘     └─────┘     └─────┘     └─────┘     └─────┘     └─────┘
Darknet   Darknet     Darknet     Alexey      Ultra-      Meituan/    Ultra-
Joseph    Joseph      Joseph      Bochkovskiy lytics      Megvii      lytics
Redmon    Redmon      Redmon

63.4 mAP  78.6 mAP    57.9 AP     65.7 AP     55.4 AP     52.8 AP     53.9 AP
(VOC)     (VOC)       (COCO)      (COCO)      (COCO)      (COCO)      (COCO)
```

## YOLOv1 (2016)

**Key insight**: Frame detection as a single regression problem.

### Architecture

```
Input: 448 × 448 × 3
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│                     24 Conv Layers                           │
│                     (inspired by GoogLeNet)                  │
│                                                              │
│   Conv 7×7×64 → MaxPool → Conv 3×3×192 → MaxPool →          │
│   Conv 1×1×128 → Conv 3×3×256 → ... → Conv 3×3×1024         │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│                      2 FC Layers                             │
│                                                              │
│   4096 → 4096 (dropout 0.5) → 7×7×30                        │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
Output: 7 × 7 × 30 (S×S×(B×5+C))
```

### Grid-Based Prediction

```
         Input Image                     Output Grid (7×7)
    ┌─────────────────────┐            ┌─────────────────────┐
    │                     │            │ ┌───┬───┬───┬───┐   │
    │                     │            │ │   │   │   │   │   │
    │       ┌─────┐       │            │ ├───┼───┼───┼───┤   │
    │       │ obj │       │   ────►    │ │   │ * │   │   │   │
    │       └─────┘       │            │ ├───┼───┼───┼───┤   │
    │                     │            │ │   │   │   │   │   │
    │                     │            │ └───┴───┴───┴───┘   │
    └─────────────────────┘            └─────────────────────┘

    * = grid cell responsible for detecting the object
        (cell containing object center)
```

### Output Tensor

```python
# Output: S × S × (B × 5 + C)
# S = 7 (grid size)
# B = 2 (boxes per cell)
# C = 20 (PASCAL VOC classes)

# Per cell: 2 boxes × (x, y, w, h, conf) + 20 class probs = 30 values

# For each bounding box:
# x, y: offset from cell corner (0-1)
# w, h: relative to image size (0-1)
# conf: P(Object) × IoU(pred, truth)

output_shape = (7, 7, 30)
#                    │
#        ┌───────────┴───────────┐
#        │                       │
#    Box 1 (5) + Box 2 (5)    Classes (20)
#    x,y,w,h,c   x,y,w,h,c    P(class|obj)
```

### Loss Function

```python
def yolov1_loss(predictions, targets, lambda_coord=5, lambda_noobj=0.5):
    """
    Multi-part loss function:
    1. Localization loss (coord)
    2. Confidence loss (obj)
    3. Confidence loss (noobj)
    4. Classification loss
    """
    # Only responsible cell predicts (cell containing object center)

    # 1. Coordinate loss (only for responsible predictor)
    loss_xy = lambda_coord * sum((x - x_hat)² + (y - y_hat)²)
    loss_wh = lambda_coord * sum((√w - √w_hat)² + (√h - √h_hat)²)

    # 2. Confidence loss for cells with objects
    loss_conf_obj = sum((C - C_hat)²)

    # 3. Confidence loss for cells without objects (down-weighted)
    loss_conf_noobj = lambda_noobj * sum((C - C_hat)²)

    # 4. Classification loss (only for cells with objects)
    loss_cls = sum((p_i - p_i_hat)²)

    return loss_xy + loss_wh + loss_conf_obj + loss_conf_noobj + loss_cls
```

### Limitations

- Only 2 boxes per cell → struggles with small clustered objects
- No feature pyramid → poor on small objects
- Coarse grid (7×7) limits localization accuracy

## YOLOv2 / YOLO9000 (2017)

**Key improvements**: Batch norm, anchor boxes, multi-scale training.

### Architecture Changes

```
YOLOv1                              YOLOv2 (Darknet-19)
──────                              ──────────────────
GoogLeNet-inspired                  VGG-inspired
No batch norm                       Batch norm on all convs
FC layers at end                    Fully convolutional
448×448 input                       Multi-scale (320-608)
7×7 grid                            13×13 grid
```

### Key Innovations

1. **Batch Normalization**: Added to all conv layers
2. **High Resolution Classifier**: Pretrain on 448×448 (not 224×224)
3. **Anchor Boxes**: Predefined shapes instead of arbitrary boxes
4. **Dimension Clusters**: K-means on training boxes to find anchor shapes
5. **Direct Location Prediction**: Constrain predictions to cell
6. **Fine-Grained Features**: Passthrough layer from earlier features
7. **Multi-Scale Training**: Random resize every 10 batches

### Anchor Box Prediction

```python
# Constrained prediction (prevents instability)
# bx = σ(tx) + cx    (center x relative to cell)
# by = σ(ty) + cy    (center y relative to cell)
# bw = pw × e^tw     (width relative to anchor)
# bh = ph × e^th     (height relative to anchor)

def decode_yolov2_box(tx, ty, tw, th, anchor, cell_x, cell_y, stride):
    # Sigmoid constrains center to within cell
    bx = (torch.sigmoid(tx) + cell_x) * stride
    by = (torch.sigmoid(ty) + cell_y) * stride

    # Exponential for width/height
    bw = anchor[0] * torch.exp(tw)
    bh = anchor[1] * torch.exp(th)

    return bx, by, bw, bh
```

## YOLOv3 (2018)

**Key improvements**: Multi-scale predictions, better backbone.

### Architecture

```
                    Input (416 × 416)
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    Darknet-53 Backbone                        │
│                                                               │
│   Conv/Residual blocks at multiple scales                    │
│   Output: features at 1/8, 1/16, 1/32                        │
└──────────────────────────────────────────────────────────────┘
           │                    │                    │
           │                    │                    │
           ▼                    ▼                    ▼
    ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
    │ Scale 1     │      │ Scale 2     │      │ Scale 3     │
    │ 52×52       │      │ 26×26       │      │ 13×13       │
    │ (small obj) │      │ (med obj)   │      │ (large obj) │
    │             │      │             │      │             │
    │ 3 anchors   │      │ 3 anchors   │      │ 3 anchors   │
    └─────────────┘      └─────────────┘      └─────────────┘
           │                    │                    │
           └────────────────────┴────────────────────┘
                                │
                                ▼
                    9 anchor boxes total
                    (3 per scale)
```

### Darknet-53 Backbone

```python
class DarknetBlock(nn.Module):
    """Residual block used in Darknet-53."""
    def __init__(self, in_channels):
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = ConvBNLeaky(in_channels, mid_channels, kernel_size=1)
        self.conv2 = ConvBNLeaky(mid_channels, in_channels, kernel_size=3)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))

# Darknet-53: 53 convolutional layers
# Much deeper than Darknet-19, uses residual connections
```

### Multi-Scale Prediction

```python
def yolov3_head(features_small, features_med, features_large, num_classes=80):
    """
    Predict at 3 scales for different object sizes.
    """
    num_anchors = 3
    output_channels = num_anchors * (5 + num_classes)  # 3 × 85 = 255

    # Large objects (13×13)
    out_large = conv_1x1(features_large, output_channels)

    # Medium objects (26×26)
    upsampled = upsample(features_large)
    concat_med = torch.cat([upsampled, features_med], dim=1)
    out_med = conv_1x1(concat_med, output_channels)

    # Small objects (52×52)
    upsampled = upsample(concat_med)
    concat_small = torch.cat([upsampled, features_small], dim=1)
    out_small = conv_1x1(concat_small, output_channels)

    return out_small, out_med, out_large

# Anchors (COCO):
# Scale 1 (52×52): (10,13), (16,30), (33,23)      - small objects
# Scale 2 (26×26): (30,61), (62,45), (59,119)     - medium objects
# Scale 3 (13×13): (116,90), (156,198), (373,326) - large objects
```

### Loss Changes

```python
# YOLOv3 uses:
# - Binary cross-entropy for class predictions (multi-label)
# - Binary cross-entropy for objectness
# - MSE for box coordinates

def yolov3_loss(pred, target):
    # Objectness loss
    obj_loss = F.binary_cross_entropy(pred_obj, target_obj)

    # Class loss (independent sigmoids, not softmax)
    cls_loss = F.binary_cross_entropy(pred_cls, target_cls)

    # Box loss
    box_loss = F.mse_loss(pred_box, target_box)

    return obj_loss + cls_loss + box_loss
```

## YOLOv4 (2020)

**Key improvements**: Bag of freebies, bag of specials, CSPDarknet.

### Architecture

```
                        Input
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    CSPDarknet53 Backbone                      │
│                    (Cross Stage Partial)                      │
└──────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                         SPP                                   │
│                (Spatial Pyramid Pooling)                      │
│                                                               │
│     MaxPool(5×5) + MaxPool(9×9) + MaxPool(13×13) + Identity  │
└──────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                        PANet                                  │
│              (Path Aggregation Network)                       │
│                                                               │
│   Top-down FPN + Bottom-up path aggregation                  │
└──────────────────────────────────────────────────────────────┘
                           │
                           ▼
                    YOLO Head (3 scales)
```

### Bag of Freebies (Training Tricks)

Data augmentation and training strategies that don't increase inference cost:

| Technique | Description |
|-----------|-------------|
| **Mosaic** | Combine 4 images into one |
| **CutMix** | Cut and paste image regions |
| **DropBlock** | Structured dropout for conv layers |
| **Label Smoothing** | Soft labels instead of hard 0/1 |
| **CIoU Loss** | Better box regression loss |
| **Self-Adversarial Training** | Network attacks itself |

### Bag of Specials (Architecture Tricks)

Module enhancements:

| Technique | Description |
|-----------|-------------|
| **CSP** | Cross Stage Partial connections |
| **SPP** | Spatial Pyramid Pooling |
| **PANet** | Path Aggregation Network |
| **SAM** | Spatial Attention Module |
| **Mish** | Mish activation (smooth ReLU) |

### Mosaic Augmentation

```python
def mosaic_augmentation(images, labels):
    """Combine 4 images into one with their labels."""
    h, w = target_size
    center_x = random.randint(w // 4, 3 * w // 4)
    center_y = random.randint(h // 4, 3 * h // 4)

    mosaic_image = np.zeros((h, w, 3))
    mosaic_labels = []

    # Place 4 images in quadrants
    for i, (img, lbl) in enumerate(zip(images[:4], labels[:4])):
        if i == 0:  # top-left
            x1, y1, x2, y2 = 0, 0, center_x, center_y
        elif i == 1:  # top-right
            x1, y1, x2, y2 = center_x, 0, w, center_y
        # ... etc

        # Resize and place
        mosaic_image[y1:y2, x1:x2] = resize(img, (y2-y1, x2-x1))
        mosaic_labels.extend(transform_labels(lbl, x1, y1, x2, y2))

    return mosaic_image, mosaic_labels
```

## YOLOv5 (2020)

**Note**: Not an official paper, developed by Ultralytics.

### Key Features

- PyTorch native (previous versions were Darknet)
- Extensive hyperparameter optimization
- Easy to use API
- Multiple model sizes (n, s, m, l, x)

### Model Sizes

```
Model    Params    FLOPs     mAP      Speed
─────    ──────    ─────     ───      ─────
v5n      1.9M      4.5G      28.0     45 fps
v5s      7.2M      16.5G     37.4     42 fps
v5m      21.2M     49.0G     45.4     35 fps
v5l      46.5M     109.1G    49.0     28 fps
v5x      86.7M     205.7G    50.7     20 fps
```

### Ultralytics Usage

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov5s.pt')

# Train
model.train(data='coco128.yaml', epochs=100, imgsz=640)

# Inference
results = model('image.jpg')
results[0].show()

# Export
model.export(format='onnx')
```

## YOLOv8 (2023)

**Key changes**: Anchor-free, decoupled head.

### Architecture

```
                        Input
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    CSPDarknet Backbone                        │
│                    (modified from v5)                         │
│                                                               │
│   C2f blocks (CSP with 2 convs, inspired by ELAN)            │
└──────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                     PANet + SPPF                              │
└──────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                   Decoupled Head                              │
│                                                               │
│   ┌─────────────┐         ┌─────────────┐                    │
│   │ Classification│         │  Regression │                    │
│   │    Branch    │         │   Branch    │                    │
│   │              │         │             │                    │
│   │  Conv → Conv │         │ Conv → Conv │                    │
│   │      ↓       │         │     ↓       │                    │
│   │   Classes    │         │  Box + DFL  │                    │
│   └─────────────┘         └─────────────┘                    │
└──────────────────────────────────────────────────────────────┘
```

### Anchor-Free Detection

```python
# YOLOv8 predicts:
# - Object center (like FCOS)
# - Distance to 4 box edges
# - No predefined anchor boxes!

def yolov8_decode(pred, stride):
    """Decode anchor-free predictions."""
    # pred shape: [B, 4 + num_classes, H, W]

    # Box predictions (distribution over distances)
    box_pred = pred[:, :4]  # distances to left, top, right, bottom

    # Using DFL (Distribution Focal Loss) for more precise regression
    # Instead of single value, predict distribution over discrete bins

    # Class predictions
    cls_pred = pred[:, 4:]

    return box_pred, cls_pred
```

### DFL (Distribution Focal Loss)

Instead of regressing a single value, predict a distribution:

```python
def dfl_loss(pred_dist, target, reg_max=16):
    """
    Distribution Focal Loss for box regression.

    Instead of predicting box coordinates directly,
    predict distribution over discrete values.
    """
    # pred_dist: [N, reg_max] - probabilities for each bin
    # target: [N] - ground truth distance

    # Target falls between two bins
    target_left = target.floor().long()
    target_right = target_left + 1

    # Weight for each bin
    weight_left = target_right.float() - target
    weight_right = target - target_left.float()

    # Cross-entropy with soft labels
    loss = (
        F.cross_entropy(pred_dist, target_left, reduction='none') * weight_left +
        F.cross_entropy(pred_dist, target_right, reduction='none') * weight_right
    )

    return loss.mean()
```

### Task-Specific Heads

```python
from ultralytics import YOLO

# Detection
model = YOLO('yolov8m.pt')

# Segmentation
model = YOLO('yolov8m-seg.pt')

# Pose estimation
model = YOLO('yolov8m-pose.pt')

# Classification
model = YOLO('yolov8m-cls.pt')

# Oriented bounding boxes
model = YOLO('yolov8m-obb.pt')
```

### Complete Training Example

```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8m.pt')  # or yolov8m.yaml for training from scratch

# Train
results = model.train(
    data='coco128.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    workers=8,
    optimizer='AdamW',
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.0,
)

# Validate
metrics = model.val()
print(f"mAP50-95: {metrics.box.map}")

# Predict
results = model.predict('image.jpg', conf=0.25, iou=0.7)

# Export
model.export(format='onnx', dynamic=True, simplify=True)
```

## YOLO Comparison Table

| Version | Year | Key Innovation | mAP (COCO) | FPS (V100) |
|---------|------|----------------|------------|------------|
| v1 | 2016 | Single-pass detection | - | 45 |
| v2 | 2017 | Anchor boxes, batch norm | - | 67 |
| v3 | 2018 | Multi-scale, FPN | 33.0 | 35 |
| v4 | 2020 | CSP, PANet, Mosaic | 43.5 | 62 |
| v5m | 2020 | PyTorch, auto-anchor | 45.4 | 35 |
| v6m | 2022 | RepVGG, hardware-aware | 49.0 | - |
| v7 | 2022 | E-ELAN, model scaling | 51.4 | - |
| v8m | 2023 | Anchor-free, DFL | 50.2 | 40 |

## When to Use Which YOLO

| Use Case | Recommendation |
|----------|----------------|
| **Edge/Mobile** | YOLOv8n, YOLOv5n |
| **Balanced** | YOLOv8m, YOLOv5m |
| **Maximum Accuracy** | YOLOv8x, YOLO-NAS-L |
| **Real-time Video** | YOLOv8s, YOLOv5s |
| **Custom Training** | YOLOv8 (best tooling) |
| **Research/Comparison** | Any (well-documented) |
