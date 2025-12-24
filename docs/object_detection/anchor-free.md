# Anchor-Free Detectors

Eliminating predefined anchor boxes for simpler, more flexible detection.

## Why Anchor-Free?

### Problems with Anchors

```
1. Hyperparameter Sensitivity
   - Aspect ratios: [0.5, 1.0, 2.0] or [0.33, 0.5, 1.0, 2.0, 3.0]?
   - Scales: [32, 64, 128] or [16, 32, 64, 128, 256]?
   - Need domain-specific tuning

2. Scale Mismatch
   - Anchor sizes designed for COCO may not work for your dataset
   - Small/large objects may not match any anchor well

3. Computation Overhead
   - ~100k anchors per image
   - IoU calculation for matching
   - Complex positive/negative sampling

4. Imbalanced Sampling
   - Most anchors are negative (no object)
   - Requires hard negative mining or focal loss
```

### Anchor-Free Paradigm

```
Anchor-Based:                    Anchor-Free:
─────────────                    ────────────
Predict offset from anchor   →   Directly predict box location
Match anchors to GT          →   Assign points to GT
Multiple anchors per loc     →   One prediction per location
IoU-based assignment         →   Center-based assignment
```

## CornerNet (2018)

**Key insight**: Detect objects as pairs of corners (top-left, bottom-right).

### Architecture

```
                        Input Image
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│                    Hourglass Network                            │
│              (Stacked hourglass for multi-scale)                │
└────────────────────────────────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │ Top-Left │   │ Bottom-  │   │Embeddings│
        │ Heatmap  │   │ Right    │   │          │
        │    +     │   │ Heatmap  │   │  (for    │
        │ Offsets  │   │    +     │   │ pairing) │
        └──────────┘   │ Offsets  │   └──────────┘
                       └──────────┘
```

### Corner Detection

```python
def corner_net_head(features, num_classes):
    """
    Predict corner heatmaps and embeddings.
    """
    # Top-left corner
    tl_heatmap = Conv(features, num_classes)   # [H, W, C]
    tl_embedding = Conv(features, 1)           # [H, W, 1]
    tl_offset = Conv(features, 2)              # [H, W, 2]

    # Bottom-right corner
    br_heatmap = Conv(features, num_classes)
    br_embedding = Conv(features, 1)
    br_offset = Conv(features, 2)

    return tl_heatmap, tl_embedding, tl_offset, br_heatmap, br_embedding, br_offset
```

### Corner Pooling

Special pooling to find corners where edges meet:

```
Top-Left Corner Pooling:
                    max→
    ┌───────────────────┐
    │ ● ────────────────│  Take max from right
    │ │                 │
    │ │                 │  Take max from below
    │ ▼                 │
    │ max               │
    └───────────────────┘

Output at corner = max(all values to right) + max(all values below)
```

```python
def corner_pool_tl(x):
    """Top-left corner pooling."""
    # Max from right to left
    batch, ch, h, w = x.shape
    horizontal = torch.zeros_like(x)
    for i in range(w - 1, -1, -1):
        horizontal[:, :, :, i] = torch.max(
            x[:, :, :, i:], dim=3
        )[0]

    # Max from bottom to top
    vertical = torch.zeros_like(x)
    for i in range(h - 1, -1, -1):
        vertical[:, :, i, :] = torch.max(
            x[:, :, i:, :], dim=2
        )[0]

    return horizontal + vertical
```

### Corner Pairing

```python
def pair_corners(tl_heatmap, br_heatmap, tl_embed, br_embed, tl_offset, br_offset):
    """
    Match top-left and bottom-right corners using embeddings.
    """
    # Find corner locations from heatmaps
    tl_locations = find_peaks(tl_heatmap)
    br_locations = find_peaks(br_heatmap)

    # Get embeddings at corner locations
    tl_embeddings = tl_embed[tl_locations]
    br_embeddings = br_embed[br_locations]

    # Pair corners with similar embeddings
    # (same object should have similar embeddings)
    distances = torch.cdist(tl_embeddings, br_embeddings)
    pairs = hungarian_matching(distances)

    # Create boxes from paired corners
    boxes = []
    for tl_idx, br_idx in pairs:
        x1, y1 = tl_locations[tl_idx] + tl_offset[tl_locations[tl_idx]]
        x2, y2 = br_locations[br_idx] + br_offset[br_locations[br_idx]]
        boxes.append([x1, y1, x2, y2])

    return boxes
```

## CenterNet (2019)

**Key insight**: Detect objects as center points, regress size.

### Architecture

```
                        Input Image
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│                    Backbone (ResNet/DLA)                        │
│                                                                 │
│           Output stride: 4 (1/4 resolution)                    │
└────────────────────────────────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │  Center  │   │   Size   │   │  Offset  │
        │ Heatmap  │   │ (w, h)   │   │ (Δx, Δy) │
        │ [H,W,C]  │   │ [H,W,2]  │   │ [H,W,2]  │
        └──────────┘   └──────────┘   └──────────┘

One prediction per center point location!
```

### Detection as Keypoint Estimation

```python
class CenterNet(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone

        # Three output heads
        self.heatmap = nn.Conv2d(64, num_classes, 1)  # Center heatmap
        self.wh = nn.Conv2d(64, 2, 1)                  # Width, height
        self.offset = nn.Conv2d(64, 2, 1)              # Sub-pixel offset

    def forward(self, x):
        features = self.backbone(x)
        return {
            'hm': torch.sigmoid(self.heatmap(features)),
            'wh': self.wh(features),
            'offset': self.offset(features)
        }
```

### Training Target Generation

```python
def generate_centernet_targets(gt_boxes, gt_classes, output_size, num_classes):
    """
    Generate training targets for CenterNet.
    """
    H, W = output_size
    heatmap = torch.zeros(num_classes, H, W)
    wh = torch.zeros(2, H, W)
    offset = torch.zeros(2, H, W)
    mask = torch.zeros(H, W)

    for box, cls in zip(gt_boxes, gt_classes):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        # Downsample to feature map resolution
        cx_int, cy_int = int(cx), int(cy)
        cx_offset = cx - cx_int
        cy_offset = cy - cy_int

        # Gaussian kernel for heatmap (soft label)
        radius = gaussian_radius(h, w)
        draw_gaussian(heatmap[cls], (cx_int, cy_int), radius)

        # Size and offset at center point
        wh[0, cy_int, cx_int] = w
        wh[1, cy_int, cx_int] = h
        offset[0, cy_int, cx_int] = cx_offset
        offset[1, cy_int, cx_int] = cy_offset
        mask[cy_int, cx_int] = 1

    return heatmap, wh, offset, mask
```

### Gaussian Kernel

```python
def gaussian_radius(height, width, min_overlap=0.7):
    """
    Compute radius such that IoU with GT > min_overlap.
    """
    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    # ... similar for other cases ...

    return int(min(r1, r2, r3))

def draw_gaussian(heatmap, center, radius, k=1):
    """Draw Gaussian kernel on heatmap."""
    diameter = 2 * radius + 1
    gaussian = gaussian_2d(diameter, sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
```

### Loss Function

```python
def centernet_loss(pred, target):
    """
    CenterNet loss = heatmap loss + size loss + offset loss
    """
    # Focal loss for heatmap
    hm_loss = focal_loss(pred['hm'], target['hm'])

    # L1 loss for size (only at center points)
    wh_loss = F.l1_loss(
        pred['wh'] * target['mask'],
        target['wh'] * target['mask']
    )

    # L1 loss for offset (only at center points)
    offset_loss = F.l1_loss(
        pred['offset'] * target['mask'],
        target['offset'] * target['mask']
    )

    return hm_loss + 0.1 * wh_loss + offset_loss
```

### Inference

```python
def centernet_decode(heatmap, wh, offset, K=100):
    """
    Decode CenterNet predictions to boxes.
    """
    batch, num_classes, H, W = heatmap.shape

    # Find top-K peaks in heatmap
    heatmap = nms_pool(heatmap)  # 3x3 max pool NMS
    scores, indices = heatmap.view(batch, -1).topk(K)

    # Convert flat indices to (class, y, x)
    classes = indices // (H * W)
    indices = indices % (H * W)
    ys = indices // W
    xs = indices % W

    # Get size and offset at peak locations
    wh = wh.view(batch, 2, -1).gather(2, indices.unsqueeze(1).expand(-1, 2, -1))
    offset = offset.view(batch, 2, -1).gather(2, indices.unsqueeze(1).expand(-1, 2, -1))

    # Compute boxes
    xs = xs.float() + offset[:, 0, :]
    ys = ys.float() + offset[:, 1, :]
    x1 = xs - wh[:, 0, :] / 2
    y1 = ys - wh[:, 1, :] / 2
    x2 = xs + wh[:, 0, :] / 2
    y2 = ys + wh[:, 1, :] / 2

    boxes = torch.stack([x1, y1, x2, y2], dim=2)

    return boxes, scores, classes
```

### CenterNet Advantages

- **Simple**: No anchors, no NMS (optional), no IoU calculations
- **Fast**: Single forward pass, simple post-processing
- **Flexible**: Same architecture for detection, pose, 3D, tracking

## FCOS (2019)

**Key insight**: Per-pixel prediction with centerness for quality estimation.

### Architecture

```
                        Input Image
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│                    ResNet + FPN                                 │
│                                                                 │
│   P3 (1/8) ─── P4 (1/16) ─── P5 (1/32) ─── P6 ─── P7          │
└────────────────────────────────────────────────────────────────┘
                             │
                             │ (shared head)
                             ▼
┌────────────────────────────────────────────────────────────────┐
│                      Per-pixel predictions                      │
│                                                                 │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│   │Classification│  │ Box (l,t,r,b)│  │  Centerness  │        │
│   │   [H,W,C]    │  │   [H,W,4]    │  │   [H,W,1]    │        │
│   └──────────────┘  └──────────────┘  └──────────────┘        │
└────────────────────────────────────────────────────────────────┘
```

### Per-Pixel Box Prediction

Instead of predicting (x, y, w, h), predict distances to edges:

```
             l (left)
        ←─────────────┐
                      │
    t   ┌─────────────┤ ────→ r (right)
   (top)│             │
        │      ●      │  ● = prediction point
        │             │
        └─────────────┘
              ↓
            b (bottom)

Box = (x - l, y - t, x + r, y + b)
```

```python
class FCOSHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # Shared conv layers
        self.cls_tower = nn.Sequential(*[
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU()
        ] * 4)

        self.bbox_tower = nn.Sequential(*[
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU()
        ] * 4)

        # Output layers
        self.cls_logits = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, 4, 3, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, 3, padding=1)

        # Learnable scale per FPN level
        self.scales = nn.ModuleList([nn.Parameter(torch.ones(1)) for _ in range(5)])

    def forward(self, features, level):
        cls_tower = self.cls_tower(features)
        bbox_tower = self.bbox_tower(features)

        cls_logits = self.cls_logits(cls_tower)
        centerness = self.centerness(cls_tower)

        # Box prediction with per-level scale
        bbox_pred = self.scales[level] * self.bbox_pred(bbox_tower)
        bbox_pred = F.relu(bbox_pred)  # Distances must be positive

        return cls_logits, bbox_pred, centerness
```

### Point Assignment

```python
def fcos_target_assignment(gt_boxes, points, strides, size_ranges):
    """
    Assign ground truth boxes to feature map points.

    Rules:
    1. Point must be inside GT box
    2. GT box size must match FPN level range
    3. If multiple GT boxes, assign smallest
    """
    targets = []

    for level, (stride, (min_size, max_size)) in enumerate(zip(strides, size_ranges)):
        # Points at this level
        level_points = points[level]

        for point in level_points:
            px, py = point

            for gt_box in gt_boxes:
                x1, y1, x2, y2 = gt_box

                # Check if point inside box
                if x1 <= px <= x2 and y1 <= py <= y2:
                    # Compute l, t, r, b
                    l = px - x1
                    t = py - y1
                    r = x2 - px
                    b = y2 - py

                    # Check size constraint
                    max_dist = max(l, t, r, b)
                    if min_size <= max_dist <= max_size:
                        targets.append({
                            'point': (px, py),
                            'box': (l, t, r, b),
                            'level': level
                        })

    return targets
```

### Centerness

Quality metric to down-weight predictions far from object center:

```python
def compute_centerness(l, t, r, b):
    """
    Centerness = sqrt(min(l,r)/max(l,r) × min(t,b)/max(t,b))

    At center: l≈r, t≈b → centerness = 1
    At edge: l>>r or t>>b → centerness → 0
    """
    lr = torch.min(l, r) / torch.max(l, r)
    tb = torch.min(t, b) / torch.max(t, b)
    centerness = torch.sqrt(lr * tb)
    return centerness
```

```
Centerness visualization:

    ┌─────────────────────────────────────┐
    │  0.1   0.2   0.3   0.4   0.3   0.2  │
    │  0.2   0.4   0.6   0.7   0.6   0.4  │
    │  0.3   0.6   0.9   1.0   0.9   0.6  │  ← Center has highest
    │  0.2   0.4   0.6   0.7   0.6   0.4  │
    │  0.1   0.2   0.3   0.4   0.3   0.2  │
    └─────────────────────────────────────┘

During inference: final_score = cls_score × centerness
→ Suppresses low-quality predictions at object edges
```

### FCOS Loss

```python
def fcos_loss(cls_pred, box_pred, centerness_pred, targets):
    """
    FCOS loss = focal loss + IoU loss + BCE loss
    """
    # Classification: Focal loss
    cls_loss = focal_loss(cls_pred, targets['cls'])

    # Box regression: IoU loss (only for positive samples)
    pos_mask = targets['cls'] > 0
    box_loss = giou_loss(box_pred[pos_mask], targets['box'][pos_mask])

    # Centerness: BCE loss (only for positive samples)
    centerness_target = compute_centerness(
        targets['box'][:, 0], targets['box'][:, 1],
        targets['box'][:, 2], targets['box'][:, 3]
    )
    centerness_loss = F.binary_cross_entropy_with_logits(
        centerness_pred[pos_mask],
        centerness_target[pos_mask]
    )

    return cls_loss + box_loss + centerness_loss
```

## Comparison: Anchor-Based vs Anchor-Free

| Aspect | Anchor-Based | Anchor-Free |
|--------|--------------|-------------|
| **Hyperparameters** | Scales, ratios, IoU thresholds | Minimal |
| **# Predictions** | K anchors × locations | 1 per location |
| **Assignment** | IoU-based | Point-based |
| **Positive sampling** | IoU > threshold | Inside object |
| **Flexibility** | Limited by anchor shapes | Any box shape |
| **Computation** | Anchor generation + IoU | Direct |

## Performance Comparison

### COCO val2017

| Model | Backbone | AP | AP50 | AP75 |
|-------|----------|-----|------|------|
| RetinaNet | R-50-FPN | 37.4 | 56.7 | 39.6 |
| FCOS | R-50-FPN | 38.7 | 57.4 | 41.8 |
| CenterNet | R-50 | 37.4 | 55.1 | 40.8 |
| FCOS | R-101-FPN | 41.5 | 60.7 | 45.0 |
| CenterNet | Hourglass-104 | 42.1 | 61.1 | 45.9 |

## Code Examples

### CenterNet with torchvision

```python
# CenterNet-like detection using keypoint estimation
import torchvision
from torchvision.models.detection import keypointrcnn_resnet50_fpn

# For custom CenterNet, use centernet-better-pytorch or mmdetection
```

### FCOS with detectron2

```python
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/fcos_R_50_FPN_1x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/fcos_R_50_FPN_1x.yaml")

predictor = DefaultPredictor(cfg)
outputs = predictor(image)
```

### mmdetection

```python
from mmdet.apis import init_detector, inference_detector

# FCOS
config = 'configs/fcos/fcos_r50_caffe_fpn_1x_coco.py'
checkpoint = 'fcos_r50_caffe_fpn_1x_coco.pth'
model = init_detector(config, checkpoint, device='cuda:0')
result = inference_detector(model, 'image.jpg')

# CenterNet
config = 'configs/centernet/centernet_resnet18_dcnv2_140e_coco.py'
```

## When to Use Anchor-Free

| Scenario | Recommendation |
|----------|----------------|
| **Dense small objects** | FCOS (multi-scale FPN) |
| **Real-time + simplicity** | CenterNet |
| **Research baseline** | FCOS (well-documented) |
| **Multi-task (pose, 3D)** | CenterNet framework |
| **Maximum accuracy** | Still competitive with anchor-based |
