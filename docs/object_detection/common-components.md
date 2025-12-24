# Common Components in Object Detection

Building blocks shared across most modern detectors.

## Backbones

The feature extractor that converts images to feature maps.

### ResNet (2015)

The workhorse backbone. Skip connections enable very deep networks.

```
                    ┌───────────┐
           x ──────►│   Conv    │──────► x + F(x)
                    │   BN      │
                    │   ReLU    │
                    │   Conv    │
                    │   BN      │
                    └─────┬─────┘
                          │
           x ─────────────┴──────────────►
              (identity or 1x1 conv)
```

```python
import torchvision.models as models

# Common choices
resnet50 = models.resnet50(weights='IMAGENET1K_V2')
resnet101 = models.resnet101(weights='IMAGENET1K_V2')

# Extract features at different scales
class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights='IMAGENET1K_V2')
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  # /4,  256 channels
        self.layer2 = resnet.layer2  # /8,  512 channels
        self.layer3 = resnet.layer3  # /16, 1024 channels
        self.layer4 = resnet.layer4  # /32, 2048 channels

    def forward(self, x):
        c1 = self.layer0(x)
        c2 = self.layer1(c1)  # 1/4 resolution
        c3 = self.layer2(c2)  # 1/8 resolution
        c4 = self.layer3(c3)  # 1/16 resolution
        c5 = self.layer4(c4)  # 1/32 resolution
        return {'c2': c2, 'c3': c3, 'c4': c4, 'c5': c5}
```

### CSPDarknet (YOLO v4+)

Cross-Stage Partial connections reduce computation while maintaining accuracy.

```
              ┌────────────────────────────┐
              │                            │
    Input ────┼──► Dense Block ──► Concat ─┼──► Output
              │         │            ▲     │
              │         └────────────┘     │
              │    (partial feature reuse) │
              └────────────────────────────┘
```

### Vision Transformer (ViT) / Swin

Transformer-based backbones for detection.

```python
# Swin Transformer backbone (used in DINO, etc.)
from torchvision.models import swin_t, swin_b

backbone = swin_b(weights='IMAGENET1K_V1')
```

### Backbone Comparison

| Backbone | Params | FLOPs | Speed | Accuracy | Use Case |
|----------|--------|-------|-------|----------|----------|
| ResNet-50 | 25M | 4.1G | Fast | Good | General purpose |
| ResNet-101 | 44M | 7.8G | Medium | Better | When accuracy matters |
| CSPDarknet | 27M | 5.3G | Fast | Good | YOLO models |
| Swin-T | 28M | 4.5G | Medium | Better | Transformer detectors |
| Swin-B | 88M | 15.4G | Slow | Best | Maximum accuracy |

## Neck: Feature Pyramid Network (FPN)

Combines multi-scale features for detecting objects of different sizes.

### The Problem

```
Small objects  ──► Need high-resolution features (early layers)
                   But early layers have weak semantics

Large objects ──► Need strong semantic features (deep layers)
                   But deep layers have low resolution
```

### FPN Solution

```
Backbone                    FPN (Top-Down + Lateral)

C5 (1/32) ────────────────► P5 ◄─────────────────────
    │                         │
    ▼                         │ upsample 2x
C4 (1/16) ──── 1x1 conv ────► + ──► P4
    │                         │
    ▼                         │ upsample 2x
C3 (1/8)  ──── 1x1 conv ────► + ──► P3
    │                         │
    ▼                         │ upsample 2x
C2 (1/4)  ──── 1x1 conv ────► + ──► P2

Result: All P levels have same channel dimension (usually 256)
        but different spatial resolutions
```

### FPN Implementation

```python
class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        # Lateral connections (1x1 conv to reduce channels)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1)
            for in_ch in in_channels_list
        ])
        # Output convolutions (3x3 to smooth after addition)
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in in_channels_list
        ])

    def forward(self, features):
        # features = [c2, c3, c4, c5] from backbone
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down pathway
        for i in range(len(laterals) - 2, -1, -1):
            upsampled = F.interpolate(laterals[i + 1], scale_factor=2, mode='nearest')
            laterals[i] = laterals[i] + upsampled

        # Output convolutions
        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
        return outputs  # [p2, p3, p4, p5]
```

### FPN Variants

| Variant | Description | Used In |
|---------|-------------|---------|
| **FPN** | Top-down only | Faster R-CNN |
| **PANet** | Top-down + Bottom-up | YOLOv4, v5 |
| **BiFPN** | Weighted bidirectional | EfficientDet |
| **NAS-FPN** | Neural architecture search | Auto-designed |

### PANet (Path Aggregation Network)

```
Top-Down (FPN):           Bottom-Up (PANet addition):

P5 ──────────────────────► N5
 │                          ▲
 ▼                          │
P4 ──────────────────────► N4
 │                          ▲
 ▼                          │
P3 ──────────────────────► N3
```

```python
class PANet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fpn = FPN(...)  # Top-down

        # Bottom-up path
        self.downsample_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, stride=2, padding=1)
            for _ in range(3)
        ])

    def forward(self, features):
        # Top-down (FPN)
        fpn_features = self.fpn(features)  # [p2, p3, p4, p5]

        # Bottom-up
        n2 = fpn_features[0]
        n3 = fpn_features[1] + self.downsample_convs[0](n2)
        n4 = fpn_features[2] + self.downsample_convs[1](n3)
        n5 = fpn_features[3] + self.downsample_convs[2](n4)

        return [n2, n3, n4, n5]
```

## Anchors

Predefined bounding box shapes that the network refines.

### Anchor Generation

```python
def generate_anchors(
    feature_map_size,
    stride,
    scales=[32, 64, 128],
    ratios=[0.5, 1.0, 2.0]
):
    """Generate anchor boxes for a single feature map level."""
    anchors = []
    for y in range(feature_map_size):
        for x in range(feature_map_size):
            cx = (x + 0.5) * stride
            cy = (y + 0.5) * stride

            for scale in scales:
                for ratio in ratios:
                    w = scale * np.sqrt(ratio)
                    h = scale / np.sqrt(ratio)
                    anchors.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])

    return np.array(anchors)

# Example: 3 scales × 3 ratios = 9 anchors per location
# Feature map 80×80 → 80 × 80 × 9 = 57,600 anchors
```

### Anchor Matching

```python
def match_anchors_to_gt(anchors, gt_boxes, pos_iou_thresh=0.7, neg_iou_thresh=0.3):
    """Assign ground truth boxes to anchors."""
    ious = compute_iou(anchors, gt_boxes)  # [num_anchors, num_gt]
    max_iou_per_anchor = ious.max(dim=1)
    best_gt_per_anchor = ious.argmax(dim=1)

    # Positive: IoU > 0.7
    positive_mask = max_iou_per_anchor > pos_iou_thresh

    # Negative: IoU < 0.3
    negative_mask = max_iou_per_anchor < neg_iou_thresh

    # Ignore: 0.3 <= IoU <= 0.7
    ignore_mask = ~positive_mask & ~negative_mask

    return positive_mask, negative_mask, best_gt_per_anchor
```

### Anchor-Free Alternative

Instead of predefined anchors, predict:
- Center point location
- Distance to box edges (FCOS)
- Or: Corner points (CornerNet)

## Non-Maximum Suppression (NMS)

Removes duplicate detections for the same object.

### Standard NMS

```python
def nms(boxes, scores, iou_threshold=0.5):
    """Standard NMS implementation."""
    # Sort by score (descending)
    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0:
        # Keep highest scoring box
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        # Compute IoU with remaining boxes
        ious = compute_iou(boxes[i:i+1], boxes[order[1:]])[0]

        # Remove boxes with IoU > threshold
        mask = ious <= iou_threshold
        order = order[1:][mask]

    return keep
```

### Soft-NMS

Instead of removing overlapping boxes, reduce their scores:

```python
def soft_nms(boxes, scores, iou_threshold=0.5, sigma=0.5, score_threshold=0.001):
    """Soft-NMS: decay scores instead of removing."""
    for i in range(len(boxes)):
        max_idx = i + scores[i:].argmax()
        # Swap to front
        boxes[i], boxes[max_idx] = boxes[max_idx], boxes[i]
        scores[i], scores[max_idx] = scores[max_idx], scores[i]

        ious = compute_iou(boxes[i:i+1], boxes[i+1:])[0]

        # Gaussian decay
        decay = np.exp(-(ious ** 2) / sigma)
        scores[i+1:] *= decay

    keep = scores > score_threshold
    return keep
```

### NMS Variants Comparison

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| **Standard NMS** | Hard threshold removal | Fast, simple | Misses occluded objects |
| **Soft-NMS** | Score decay | Better for occlusion | Slower |
| **DIoU-NMS** | Uses DIoU instead of IoU | Better for overlapping | Slightly slower |
| **NMS-Free** | End-to-end (DETR) | No hyperparameter | Requires transformer |

### Batched NMS (torchvision)

```python
from torchvision.ops import batched_nms

# Per-class NMS
keep = batched_nms(
    boxes,      # [N, 4]
    scores,     # [N]
    classes,    # [N] - class indices
    iou_threshold=0.5
)
```

## Loss Functions

### Classification Loss

**Cross-Entropy Loss:**
```python
cls_loss = F.cross_entropy(predictions, targets)
```

**Focal Loss** (for class imbalance):
```python
def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    Focal Loss: FL(p_t) = -α_t (1 - p_t)^γ log(p_t)

    Down-weights easy examples, focuses on hard ones.
    """
    ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    p_t = torch.exp(-ce_loss)
    focal_weight = alpha * (1 - p_t) ** gamma
    return (focal_weight * ce_loss).mean()
```

### Box Regression Loss

**Smooth L1 Loss:**
```python
def smooth_l1_loss(pred, target, beta=1.0):
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return loss.mean()
```

**IoU-based Losses** (directly optimize IoU):

```python
def compute_iou(box1, box2):
    """Compute IoU between two boxes."""
    inter_x1 = torch.max(box1[..., 0], box2[..., 0])
    inter_y1 = torch.max(box1[..., 1], box2[..., 1])
    inter_x2 = torch.min(box1[..., 2], box2[..., 2])
    inter_y2 = torch.min(box1[..., 3], box2[..., 3])

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])

    union = area1 + area2 - inter_area
    iou = inter_area / union
    return iou

def giou_loss(pred, target):
    """GIoU Loss: IoU + penalty for enclosing box."""
    iou = compute_iou(pred, target)

    # Enclosing box
    enc_x1 = torch.min(pred[..., 0], target[..., 0])
    enc_y1 = torch.min(pred[..., 1], target[..., 1])
    enc_x2 = torch.max(pred[..., 2], target[..., 2])
    enc_y2 = torch.max(pred[..., 3], target[..., 3])
    enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)

    inter_area = ...  # from iou calculation
    union = ...
    giou = iou - (enc_area - union) / enc_area

    return 1 - giou

def diou_loss(pred, target):
    """DIoU Loss: IoU + center distance penalty."""
    iou = compute_iou(pred, target)

    # Center points
    pred_cx = (pred[..., 0] + pred[..., 2]) / 2
    pred_cy = (pred[..., 1] + pred[..., 3]) / 2
    target_cx = (target[..., 0] + target[..., 2]) / 2
    target_cy = (target[..., 1] + target[..., 3]) / 2

    center_dist = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2

    # Diagonal of enclosing box
    enc_x1 = torch.min(pred[..., 0], target[..., 0])
    enc_y1 = torch.min(pred[..., 1], target[..., 1])
    enc_x2 = torch.max(pred[..., 2], target[..., 2])
    enc_y2 = torch.max(pred[..., 3], target[..., 3])
    enc_diag = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2

    diou = iou - center_dist / enc_diag
    return 1 - diou

def ciou_loss(pred, target):
    """CIoU Loss: DIoU + aspect ratio consistency."""
    diou = ...  # from diou calculation

    # Aspect ratio penalty
    pred_w = pred[..., 2] - pred[..., 0]
    pred_h = pred[..., 3] - pred[..., 1]
    target_w = target[..., 2] - target[..., 0]
    target_h = target[..., 3] - target[..., 1]

    v = (4 / np.pi ** 2) * (torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h)) ** 2
    alpha = v / (1 - iou + v)

    ciou = diou - alpha * v
    return 1 - ciou
```

### Loss Comparison

| Loss | Formula | Advantage |
|------|---------|-----------|
| **L1** | \|Δ\| | Simple |
| **Smooth L1** | Quadratic near 0 | Less sensitive to outliers |
| **IoU** | 1 - IoU | Scale invariant |
| **GIoU** | 1 - GIoU | Works for non-overlapping boxes |
| **DIoU** | 1 - DIoU | Faster convergence |
| **CIoU** | 1 - CIoU | Best for box regression |

## RoI Operations

### RoI Pooling (Fast R-CNN)

```python
from torchvision.ops import roi_pool

# Extract fixed-size features from variable-size regions
pooled = roi_pool(
    features,     # [B, C, H, W]
    boxes,        # [N, 5] - (batch_idx, x1, y1, x2, y2)
    output_size=(7, 7),
    spatial_scale=1/16  # feature map stride
)  # Output: [N, C, 7, 7]
```

### RoI Align (Mask R-CNN)

Fixes misalignment from quantization:

```python
from torchvision.ops import roi_align

# Bilinear interpolation instead of quantization
aligned = roi_align(
    features,
    boxes,
    output_size=(7, 7),
    spatial_scale=1/16,
    aligned=True  # Important: use correct alignment
)
```

```
RoI Pool:              RoI Align:
┌─┬─┬─┬─┐              ┌───────────┐
│ │ │ │ │              │ ● ● ● ●   │  ● = sample points
├─┼─┼─┼─┤              │           │
│ │ │ │ │  Quantized   │ ● ● ● ●   │  Bilinear
├─┼─┼─┼─┤  boundaries  │           │  interpolation
│ │ │ │ │              │ ● ● ● ●   │
└─┴─┴─┴─┘              └───────────┘

Misaligned              Precise
```

## Data Augmentation

### Common Augmentations

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.RandomResizedCrop(height=640, width=640, scale=(0.8, 1.0)),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
```

### Mosaic Augmentation (YOLO v4+)

Combines 4 images into one:

```
┌─────────┬─────────┐
│  Img 1  │  Img 2  │
│         │         │
├─────────┼─────────┤
│  Img 3  │  Img 4  │
│         │         │
└─────────┴─────────┘
```

### MixUp / CutMix

```python
def mixup(images, labels, alpha=0.2):
    """Blend two images and their labels."""
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size)

    mixed_images = lam * images + (1 - lam) * images[index]
    return mixed_images, labels, labels[index], lam
```
