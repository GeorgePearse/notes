# R-CNN Family

The two-stage detector lineage: from R-CNN to Mask R-CNN.

## Evolution Overview

```
R-CNN (2014)          Fast R-CNN (2015)       Faster R-CNN (2015)      Mask R-CNN (2017)
─────────────         ────────────────        ──────────────────       ─────────────────
47.3 mAP (VOC)        66.9 mAP (VOC)          73.2 mAP (VOC)           39.8 mAP (COCO)
~50s per image        ~2s per image           ~0.2s per image          ~0.2s + masks

Selective Search      Selective Search        Region Proposal          + Instance
+ CNN + SVM           + Shared CNN            Network (RPN)            Segmentation
                      + Multi-task                                     + RoI Align
```

## R-CNN (2014)

**Regions with CNN features** - The paper that started deep learning for detection.

### Architecture

```
        ┌─────────────────────────────────────────────────────────────┐
        │                        Input Image                           │
        └─────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
        ┌─────────────────────────────────────────────────────────────┐
        │               Selective Search (~2000 proposals)             │
        │                   (External algorithm)                       │
        └─────────────────────────────────────────────────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
               ┌─────────┐     ┌─────────┐     ┌─────────┐
               │ Region 1│     │ Region 2│ ... │Region N │
               │  (crop) │     │  (crop) │     │ (crop)  │
               └────┬────┘     └────┬────┘     └────┬────┘
                    │               │               │
                    ▼               ▼               ▼
               ┌─────────┐     ┌─────────┐     ┌─────────┐
               │  Warp   │     │  Warp   │     │  Warp   │
               │ 227×227 │     │ 227×227 │     │ 227×227 │
               └────┬────┘     └────┬────┘     └────┬────┘
                    │               │               │
                    ▼               ▼               ▼
               ┌─────────────────────────────────────────┐
               │           CNN (AlexNet/VGG)              │
               │        (run separately for each!)        │
               └─────────────────────────────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
               ┌─────────┐     ┌─────────┐     ┌─────────┐
               │   SVM   │     │   SVM   │ ... │   SVM   │
               │ (per    │     │ (per    │     │ (per    │
               │  class) │     │  class) │     │  class) │
               └─────────┘     └─────────┘     └─────────┘
```

### Problems with R-CNN

1. **Slow**: CNN runs ~2000 times per image
2. **Multi-stage**: Separate CNN, SVM, and bbox regressor training
3. **Storage**: Features must be stored on disk

## Fast R-CNN (2015)

**Key insight**: Run CNN once, extract features for all regions.

### Architecture

```
        ┌─────────────────────────────────────────────────────────────┐
        │                        Input Image                           │
        └─────────────────────────────────────────────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                  │
                    ▼                                  ▼
        ┌───────────────────────┐          ┌─────────────────────────┐
        │   CNN Backbone        │          │   Selective Search      │
        │   (run ONCE)          │          │   (~2000 proposals)     │
        │                       │          └───────────┬─────────────┘
        │   ┌────────────┐      │                      │
        │   │Feature Map │      │                      │
        │   └────────────┘      │                      │
        └───────────┬───────────┘                      │
                    │                                  │
                    └──────────────┬───────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────┐
                    │         RoI Pooling              │
                    │   (extract fixed-size features   │
                    │    for each proposal)            │
                    └───────────────┬─────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────┐
                    │          FC Layers               │
                    └───────────────┬─────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
            ┌─────────────┐                ┌─────────────┐
            │   Softmax   │                │  Bbox Reg   │
            │ (K+1 class) │                │  (4K values)│
            └─────────────┘                └─────────────┘
```

### RoI Pooling

```python
def roi_pool(feature_map, roi, output_size=(7, 7)):
    """
    Extract fixed-size feature from variable-size RoI.

    Problem: RoI can be any size, but FC layers need fixed input.
    Solution: Divide RoI into output_size grid, max-pool each cell.
    """
    x1, y1, x2, y2 = roi
    roi_width = x2 - x1
    roi_height = y2 - y1

    # Divide into 7×7 grid
    bin_height = roi_height / output_size[0]
    bin_width = roi_width / output_size[1]

    pooled = []
    for i in range(output_size[0]):
        for j in range(output_size[1]):
            # Quantize to integer coordinates (causes misalignment)
            start_h = int(y1 + i * bin_height)
            end_h = int(y1 + (i + 1) * bin_height)
            start_w = int(x1 + j * bin_width)
            end_w = int(x1 + (j + 1) * bin_width)

            pooled.append(feature_map[start_h:end_h, start_w:end_w].max())

    return pooled.reshape(output_size)
```

### Multi-task Loss

```python
def fast_rcnn_loss(cls_logits, bbox_pred, cls_targets, bbox_targets):
    """
    L = L_cls + λ L_loc

    L_cls: Cross-entropy for classification
    L_loc: Smooth L1 for box regression (only for positive samples)
    """
    # Classification loss (all samples)
    cls_loss = F.cross_entropy(cls_logits, cls_targets)

    # Box regression loss (positive samples only)
    positive_mask = cls_targets > 0  # Not background
    bbox_loss = F.smooth_l1_loss(
        bbox_pred[positive_mask],
        bbox_targets[positive_mask]
    )

    return cls_loss + bbox_loss
```

### Improvements over R-CNN

- **9× faster training** (no feature caching)
- **213× faster inference** (single CNN pass)
- **End-to-end training** (multi-task loss)

### Remaining Problem

Still uses **Selective Search** - slow (~2 seconds) and not GPU-accelerated.

## Faster R-CNN (2015)

**Key insight**: Replace Selective Search with a neural network (RPN).

### Architecture

```
        ┌─────────────────────────────────────────────────────────────┐
        │                        Input Image                           │
        └─────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
        ┌─────────────────────────────────────────────────────────────┐
        │                    CNN Backbone (VGG/ResNet)                 │
        │                                                              │
        │   Input: 800×600×3                                          │
        │   Output: 50×38×512 (or 256 for C4)                         │
        └───────────────────────────────┬─────────────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                                       │
                    ▼                                       ▼
        ┌───────────────────────┐               ┌───────────────────────┐
        │  Region Proposal      │               │   (shared features)    │
        │  Network (RPN)        │               │                        │
        │                       │               │                        │
        │  3×3 conv → 512       │               │                        │
        │       │               │               │                        │
        │   ┌───┴───┐           │               │                        │
        │   ▼       ▼           │               │                        │
        │  cls    bbox          │               │                        │
        │  2k     4k            │               │                        │
        │  (obj/  (box          │               │                        │
        │  not)   deltas)       │               │                        │
        └───────────┬───────────┘               │                        │
                    │                           │                        │
                    │ ~300 proposals            │                        │
                    │ (after NMS)               │                        │
                    └───────────────────────────┤                        │
                                                │                        │
                                                ▼                        │
                                    ┌───────────────────────┐           │
                                    │      RoI Pooling      │◄──────────┘
                                    │    (7×7 per RoI)      │
                                    └───────────┬───────────┘
                                                │
                                                ▼
                                    ┌───────────────────────┐
                                    │       FC Layers        │
                                    │    (4096 → 4096)       │
                                    └───────────┬───────────┘
                                                │
                                    ┌───────────┴───────────┐
                                    ▼                       ▼
                            ┌─────────────┐         ┌─────────────┐
                            │   Softmax   │         │  Bbox Reg   │
                            │  (K+1 cls)  │         │ (4K deltas) │
                            └─────────────┘         └─────────────┘
```

### Region Proposal Network (RPN)

```python
class RPN(nn.Module):
    def __init__(self, in_channels, num_anchors=9):
        super().__init__()
        # 3×3 conv for feature extraction
        self.conv = nn.Conv2d(in_channels, 512, 3, padding=1)

        # Classification: objectness score (2 per anchor: obj/not-obj)
        self.cls_layer = nn.Conv2d(512, num_anchors * 2, 1)

        # Regression: box deltas (4 per anchor: dx, dy, dw, dh)
        self.reg_layer = nn.Conv2d(512, num_anchors * 4, 1)

    def forward(self, feature_map):
        x = F.relu(self.conv(feature_map))
        cls_logits = self.cls_layer(x)   # [B, 2k, H, W]
        bbox_deltas = self.reg_layer(x)  # [B, 4k, H, W]
        return cls_logits, bbox_deltas
```

### Anchor Generation

```python
def generate_anchors(base_size=16, ratios=[0.5, 1, 2], scales=[8, 16, 32]):
    """
    Generate k=9 anchors at each location.

    Scales: [128, 256, 512] pixels
    Ratios: [1:2, 1:1, 2:1]
    """
    anchors = []
    for scale in scales:
        for ratio in ratios:
            w = base_size * scale * np.sqrt(ratio)
            h = base_size * scale / np.sqrt(ratio)
            anchors.append([-w/2, -h/2, w/2, h/2])
    return np.array(anchors)

# Result: 9 anchor templates
# Applied at every location in feature map
# 50×38 feature map → 50×38×9 = 17,100 anchors
```

### Box Parameterization

```python
def encode_boxes(anchors, gt_boxes):
    """
    Encode ground truth boxes relative to anchors.

    t_x = (x_gt - x_a) / w_a
    t_y = (y_gt - y_a) / h_a
    t_w = log(w_gt / w_a)
    t_h = log(h_gt / h_a)
    """
    # Anchor centers and dimensions
    anchor_cx = (anchors[:, 0] + anchors[:, 2]) / 2
    anchor_cy = (anchors[:, 1] + anchors[:, 3]) / 2
    anchor_w = anchors[:, 2] - anchors[:, 0]
    anchor_h = anchors[:, 3] - anchors[:, 1]

    # GT centers and dimensions
    gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]

    # Encode
    tx = (gt_cx - anchor_cx) / anchor_w
    ty = (gt_cy - anchor_cy) / anchor_h
    tw = torch.log(gt_w / anchor_w)
    th = torch.log(gt_h / anchor_h)

    return torch.stack([tx, ty, tw, th], dim=1)

def decode_boxes(anchors, deltas):
    """Decode predicted deltas to boxes."""
    # Reverse of encode
    pred_cx = deltas[:, 0] * anchor_w + anchor_cx
    pred_cy = deltas[:, 1] * anchor_h + anchor_cy
    pred_w = torch.exp(deltas[:, 2]) * anchor_w
    pred_h = torch.exp(deltas[:, 3]) * anchor_h

    # Convert to x1, y1, x2, y2
    x1 = pred_cx - pred_w / 2
    y1 = pred_cy - pred_h / 2
    x2 = pred_cx + pred_w / 2
    y2 = pred_cy + pred_h / 2

    return torch.stack([x1, y1, x2, y2], dim=1)
```

### Training Strategy

**4-Step Alternating Training** (original paper):
1. Train RPN
2. Train Fast R-CNN using RPN proposals
3. Fix Fast R-CNN, fine-tune RPN
4. Fix RPN, fine-tune Fast R-CNN

**End-to-End Training** (modern):
```python
def faster_rcnn_loss(rpn_cls, rpn_box, roi_cls, roi_box, targets):
    # RPN losses
    rpn_cls_loss = F.binary_cross_entropy_with_logits(rpn_cls, rpn_targets)
    rpn_box_loss = F.smooth_l1_loss(rpn_box[positive], rpn_box_targets)

    # ROI head losses
    roi_cls_loss = F.cross_entropy(roi_cls, roi_targets)
    roi_box_loss = F.smooth_l1_loss(roi_box[positive], roi_box_targets)

    return rpn_cls_loss + rpn_box_loss + roi_cls_loss + roi_box_loss
```

### torchvision Implementation

```python
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

# Load pretrained model
model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
model.eval()

# Inference
image = torch.randn(3, 800, 600)
predictions = model([image])

# predictions[0] contains:
# - 'boxes': [N, 4] tensor of boxes
# - 'labels': [N] tensor of class labels
# - 'scores': [N] tensor of confidence scores

# Custom number of classes
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
num_classes = 10  # Your dataset classes + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
```

## Mask R-CNN (2017)

**Key insight**: Add a parallel mask prediction branch for instance segmentation.

### Architecture

```
                         Faster R-CNN
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │              RoI Features                │
        │               (7×7×256)                  │
        └─────────────────┬───────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
    ┌───────────┐   ┌───────────┐   ┌───────────┐
    │  FC × 2   │   │  FC × 2   │   │  Conv × 4 │
    │  (1024)   │   │  (1024)   │   │  + Deconv │
    └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
          │               │               │
          ▼               ▼               ▼
    ┌───────────┐   ┌───────────┐   ┌───────────┐
    │    Cls    │   │   Bbox    │   │   Mask    │
    │  (K+1)    │   │   (4K)    │   │  (K×28×28)│
    └───────────┘   └───────────┘   └───────────┘

    Softmax         Box deltas       Per-class
    class scores    regression       binary masks
```

### RoI Align (Critical Improvement)

The mask branch requires pixel-level alignment. RoI Pooling's quantization hurts.

```python
# RoI Pooling problem: quantization misaligns features
roi_x1 = int(roi[0] / stride)  # Rounding loses precision!

# RoI Align solution: bilinear interpolation at exact positions
def roi_align(feature_map, roi, output_size, sampling_ratio=2):
    """
    Sample at regular points using bilinear interpolation.
    No quantization = precise alignment.
    """
    roi_width = roi[2] - roi[0]
    roi_height = roi[3] - roi[1]

    bin_width = roi_width / output_size[1]
    bin_height = roi_height / output_size[0]

    output = []
    for i in range(output_size[0]):
        for j in range(output_size[1]):
            # Sample points within each bin
            for si in range(sampling_ratio):
                for sj in range(sampling_ratio):
                    # Exact coordinates (no rounding!)
                    y = roi[1] + (i + (si + 0.5) / sampling_ratio) * bin_height
                    x = roi[0] + (j + (sj + 0.5) / sampling_ratio) * bin_width

                    # Bilinear interpolation
                    value = bilinear_interpolate(feature_map, x, y)
                    output.append(value)

    return average_pool(output, output_size)
```

### Mask Head

```python
class MaskHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # 4 conv layers
        self.conv1 = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)

        # Upsample to 28×28
        self.deconv = nn.ConvTranspose2d(256, 256, 2, stride=2)

        # Per-class mask prediction
        self.mask_pred = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        # x: [N, 256, 14, 14] from RoI Align
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.deconv(x))  # [N, 256, 28, 28]
        return self.mask_pred(x)    # [N, K, 28, 28]
```

### Mask Loss

```python
def mask_loss(mask_logits, mask_targets, class_labels):
    """
    Binary cross-entropy for predicted class mask only.
    Not softmax across classes - each class mask is independent.
    """
    # mask_logits: [N, K, 28, 28]
    # Only compute loss for ground truth class
    num_rois = mask_logits.size(0)
    indices = torch.arange(num_rois)
    mask_logits_for_class = mask_logits[indices, class_labels]  # [N, 28, 28]

    return F.binary_cross_entropy_with_logits(
        mask_logits_for_class,
        mask_targets
    )
```

### torchvision Implementation

```python
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2

model = maskrcnn_resnet50_fpn_v2(weights='DEFAULT')
model.eval()

predictions = model([image])

# predictions[0] contains:
# - 'boxes': [N, 4]
# - 'labels': [N]
# - 'scores': [N]
# - 'masks': [N, 1, H, W] binary masks (same size as input)
```

## Cascade R-CNN (2018)

**Key insight**: Multiple detection heads with increasing IoU thresholds.

### The Problem

- Training with IoU=0.5: Many low-quality positives → noisy predictions
- Training with IoU=0.7: Too few positives → overfitting

### Architecture

```
        ┌─────────────────────────────────────────────────────────────┐
        │                         RPN                                  │
        └─────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
        ┌─────────────────────────────────────────────────────────────┐
        │                    Stage 1 (IoU=0.5)                         │
        │                                                              │
        │   RoI Align → Head 1 → cls + bbox                           │
        │                          │                                   │
        └──────────────────────────┼───────────────────────────────────┘
                                   │ refined boxes
                                   ▼
        ┌─────────────────────────────────────────────────────────────┐
        │                    Stage 2 (IoU=0.6)                         │
        │                                                              │
        │   RoI Align → Head 2 → cls + bbox                           │
        │                          │                                   │
        └──────────────────────────┼───────────────────────────────────┘
                                   │ refined boxes
                                   ▼
        ┌─────────────────────────────────────────────────────────────┐
        │                    Stage 3 (IoU=0.7)                         │
        │                                                              │
        │   RoI Align → Head 3 → cls + bbox → Final output            │
        │                                                              │
        └─────────────────────────────────────────────────────────────┘
```

### Key Ideas

1. **Progressive refinement**: Each stage refines proposals from previous
2. **Increasing IoU threshold**: 0.5 → 0.6 → 0.7
3. **Separate heads**: Each stage has own classifier/regressor

```python
class CascadeRCNN(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.rpn = RPN(...)

        # Three cascade stages
        self.roi_heads = nn.ModuleList([
            RoIHead(iou_thresh=0.5),
            RoIHead(iou_thresh=0.6),
            RoIHead(iou_thresh=0.7)
        ])

    def forward(self, images, targets=None):
        features = self.backbone(images)
        proposals = self.rpn(features)

        # Cascade through stages
        for stage, roi_head in enumerate(self.roi_heads):
            cls_logits, bbox_deltas = roi_head(features, proposals)

            if stage < len(self.roi_heads) - 1:
                # Refine proposals for next stage
                proposals = apply_deltas(proposals, bbox_deltas)

        return cls_logits, bbox_deltas
```

## Performance Comparison

### COCO val2017

| Model | Backbone | AP | AP50 | AP75 | Params |
|-------|----------|-----|------|------|--------|
| Faster R-CNN | R-50-FPN | 37.4 | 58.1 | 40.4 | 41M |
| Faster R-CNN | R-101-FPN | 39.4 | 60.1 | 43.1 | 60M |
| Mask R-CNN | R-50-FPN | 38.2 | 58.8 | 41.4 | 44M |
| Mask R-CNN | R-101-FPN | 40.0 | 60.9 | 43.7 | 63M |
| Cascade R-CNN | R-50-FPN | 40.3 | 58.6 | 44.0 | 69M |
| Cascade R-CNN | R-101-FPN | 42.1 | 60.4 | 45.9 | 88M |

### Speed (V100 GPU)

| Model | Backbone | FPS |
|-------|----------|-----|
| Faster R-CNN | R-50-FPN | 26 |
| Faster R-CNN | R-101-FPN | 20 |
| Mask R-CNN | R-50-FPN | 22 |
| Cascade R-CNN | R-50-FPN | 15 |

## Summary

| Model | Year | Key Innovation | Use Case |
|-------|------|----------------|----------|
| R-CNN | 2014 | CNN for detection | Historical |
| Fast R-CNN | 2015 | RoI pooling, shared features | Historical |
| Faster R-CNN | 2015 | RPN (end-to-end) | General detection |
| Mask R-CNN | 2017 | RoI Align, mask branch | Instance segmentation |
| Cascade R-CNN | 2018 | Multi-stage refinement | High-quality detection |
