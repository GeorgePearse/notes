# SSD & RetinaNet

One-stage anchor-based detectors that rivaled two-stage accuracy.

## SSD (Single Shot MultiBox Detector, 2016)

**Key insight**: Predict at multiple feature map scales.

### Architecture

```
Input: 300 × 300
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│                    VGG-16 (truncated)                        │
│                                                              │
│   Conv1 → Conv2 → Conv3 → Conv4 → Conv5 (dilated)          │
│                              │                               │
│                        38×38×512                             │
└──────────────────────────────┬──────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│    Conv6        │   │    Conv7        │   │    Conv8        │
│   19×19×1024    │   │   10×10×512     │   │    5×5×256      │
└────────┬────────┘   └────────┬────────┘   └────────┬────────┘
         │                     │                     │
         │                     │                     │
         ▼                     ▼                     ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│    Conv9        │   │    Conv10       │   │    Conv11       │
│    3×3×256      │   │    1×1×256      │   │                 │
└────────┬────────┘   └────────┬────────┘   │    Not used     │
         │                     │            │                 │
         ▼                     ▼            └─────────────────┘

Detection from 6 feature maps:
38×38, 19×19, 10×10, 5×5, 3×3, 1×1
```

### Multi-Scale Feature Maps

```
Feature Map     Size        Default Boxes    Total Boxes
───────────     ────        ─────────────    ───────────
Conv4_3         38 × 38     4 per location   5,776
Conv7           19 × 19     6 per location   2,166
Conv8_2         10 × 10     6 per location   600
Conv9_2         5 × 5       6 per location   150
Conv10_2        3 × 3       4 per location   36
Conv11_2        1 × 1       4 per location   4
                                             ─────
                            Total:           8,732 boxes
```

### Default Box Generation

```python
def generate_ssd_default_boxes(feature_map_sizes, image_size=300):
    """
    Generate default boxes (anchors) for SSD.

    Scales increase with feature map depth:
    - Early layers (high res) → small objects
    - Later layers (low res) → large objects
    """
    # Scale range
    s_min, s_max = 0.2, 0.9
    num_maps = len(feature_map_sizes)

    default_boxes = []
    for k, fm_size in enumerate(feature_map_sizes):
        # Scale for this feature map
        s_k = s_min + (s_max - s_min) * k / (num_maps - 1)
        s_k1 = s_min + (s_max - s_min) * (k + 1) / (num_maps - 1)

        # Aspect ratios: 1, 2, 1/2, 3, 1/3
        aspect_ratios = [1, 2, 0.5, 3, 1/3] if fm_size <= 10 else [1, 2, 0.5]

        for i in range(fm_size):
            for j in range(fm_size):
                cx = (j + 0.5) / fm_size
                cy = (i + 0.5) / fm_size

                for ar in aspect_ratios:
                    w = s_k * np.sqrt(ar)
                    h = s_k / np.sqrt(ar)
                    default_boxes.append([cx, cy, w, h])

                # Extra box with scale sqrt(s_k * s_k1)
                w = h = np.sqrt(s_k * s_k1)
                default_boxes.append([cx, cy, w, h])

    return np.array(default_boxes)
```

### Detection Head

```python
class SSDHead(nn.Module):
    def __init__(self, in_channels_list, num_classes, num_anchors_list):
        super().__init__()
        self.cls_heads = nn.ModuleList()
        self.reg_heads = nn.ModuleList()

        for in_ch, num_anchors in zip(in_channels_list, num_anchors_list):
            # Classification: num_anchors × num_classes
            self.cls_heads.append(
                nn.Conv2d(in_ch, num_anchors * num_classes, 3, padding=1)
            )
            # Regression: num_anchors × 4
            self.reg_heads.append(
                nn.Conv2d(in_ch, num_anchors * 4, 3, padding=1)
            )

    def forward(self, features):
        cls_preds = []
        reg_preds = []

        for feat, cls_head, reg_head in zip(features, self.cls_heads, self.reg_heads):
            cls_preds.append(cls_head(feat).permute(0, 2, 3, 1).flatten(1, 2))
            reg_preds.append(reg_head(feat).permute(0, 2, 3, 1).flatten(1, 2))

        return torch.cat(cls_preds, dim=1), torch.cat(reg_preds, dim=1)
```

### Hard Negative Mining

```python
def hard_negative_mining(cls_loss, pos_mask, neg_ratio=3):
    """
    Select hard negatives (highest loss) to balance with positives.

    Problem: ~8700 boxes, only ~10 are positive
    Solution: Select top-k negative boxes by loss
    """
    num_pos = pos_mask.sum()
    num_neg = min(neg_ratio * num_pos, (~pos_mask).sum())

    # Sort negative losses, take top-k
    neg_loss = cls_loss.clone()
    neg_loss[pos_mask] = 0
    _, neg_indices = neg_loss.sort(descending=True)
    neg_mask = torch.zeros_like(pos_mask)
    neg_mask[neg_indices[:num_neg]] = True

    return pos_mask | neg_mask
```

### SSD Loss

```python
def ssd_loss(cls_pred, reg_pred, cls_target, reg_target, pos_mask):
    """
    L = (L_conf + α × L_loc) / N

    L_conf: Softmax cross-entropy
    L_loc: Smooth L1 for positive boxes only
    """
    # Classification loss (with hard negative mining)
    cls_loss = F.cross_entropy(cls_pred, cls_target, reduction='none')
    selected = hard_negative_mining(cls_loss, pos_mask)
    cls_loss = cls_loss[selected].mean()

    # Localization loss (positive boxes only)
    reg_loss = F.smooth_l1_loss(
        reg_pred[pos_mask],
        reg_target[pos_mask]
    )

    return cls_loss + reg_loss
```

## RetinaNet (2017)

**Key insight**: Class imbalance is the main problem, solved by Focal Loss.

### The Class Imbalance Problem

```
One-Stage Detector (e.g., SSD):
─────────────────────────────
Total anchors:  ~100,000
Positive:       ~10 (0.01%)
Negative:       ~99,990 (99.99%)

Problem: Easy negatives dominate the loss
         Detector learns to predict "background" for everything

Two-Stage Detector:
─────────────────
RPN filters to ~2000 proposals
Then ~1:3 pos:neg sampling
Much more balanced!
```

### Architecture

```
                        Input Image
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│                    ResNet Backbone                              │
│                                                                 │
│   C1 ─► C2 ─► C3 ─► C4 ─► C5                                   │
│              │      │      │                                    │
└──────────────┼──────┼──────┼────────────────────────────────────┘
               │      │      │
               ▼      ▼      ▼
┌────────────────────────────────────────────────────────────────┐
│                         FPN                                     │
│                                                                 │
│   P3 (1/8) ◄── P4 (1/16) ◄── P5 (1/32)                        │
│        │           │             │                              │
│        │           │             ├──► P6 (1/64)                │
│        │           │             │       │                      │
│        │           │             │       └──► P7 (1/128)       │
└────────┼───────────┼─────────────┼───────────┼──────────────────┘
         │           │             │           │
         ▼           ▼             ▼           ▼
┌────────────────────────────────────────────────────────────────┐
│                Classification Subnet                            │
│                  (shared across scales)                         │
│                                                                 │
│   4 × Conv(256) → Conv(K×A) → Sigmoid                          │
│   K = num_classes, A = num_anchors                             │
└────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│                  Box Regression Subnet                          │
│                  (shared across scales)                         │
│                                                                 │
│   4 × Conv(256) → Conv(4×A)                                    │
└────────────────────────────────────────────────────────────────┘
```

### Focal Loss

The key contribution: down-weight easy examples.

```python
def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    Focal Loss: FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)

    Standard CE:   -log(p_t)
    Focal Loss:    -(1 - p_t)^γ × log(p_t)

    When p_t is high (easy example):
        (1 - p_t)^γ is small → loss is down-weighted

    When p_t is low (hard example):
        (1 - p_t)^γ ≈ 1 → loss unchanged
    """
    # pred: raw logits [N, num_classes]
    # target: class indices [N]

    p = torch.sigmoid(pred)
    ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

    # p_t: probability of correct class
    p_t = p * target + (1 - p) * (1 - target)

    # Focal weight
    focal_weight = (1 - p_t) ** gamma

    # Alpha weighting (balance pos/neg)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)

    loss = alpha_t * focal_weight * ce_loss

    return loss.mean()
```

### Focal Loss Visualization

```
Loss
 │
 │  ______ γ=0 (standard CE)
 │ /
 │/   ____ γ=0.5
 │   /
 │  /  ___ γ=1
 │ /  /
 │/  /  __ γ=2
 │  /  /
 │ /  /  _ γ=5
 │/  /  /
 │──────────────────────► p_t (probability of correct class)
 0               0.5              1.0

As γ increases:
- Easy examples (high p_t) contribute less
- Hard examples (low p_t) dominate training
```

### Focal Loss Implementation

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: [N, C] logits
        targets: [N, C] one-hot or soft labels
        """
        p = torch.sigmoid(inputs)

        # Binary cross entropy
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )

        # p_t
        p_t = p * targets + (1 - p) * (1 - targets)

        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weight
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Final loss
        loss = alpha_t * focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
```

### RetinaNet Subnets

```python
class RetinaNetHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()

        # Classification subnet
        cls_subnet = []
        for _ in range(4):
            cls_subnet.append(nn.Conv2d(in_channels, in_channels, 3, padding=1))
            cls_subnet.append(nn.ReLU())
        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.cls_score = nn.Conv2d(in_channels, num_anchors * num_classes, 3, padding=1)

        # Box regression subnet
        reg_subnet = []
        for _ in range(4):
            reg_subnet.append(nn.Conv2d(in_channels, in_channels, 3, padding=1))
            reg_subnet.append(nn.ReLU())
        self.reg_subnet = nn.Sequential(*reg_subnet)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, 3, padding=1)

        # Initialize cls_score with prior probability
        # (addresses class imbalance at initialization)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_score.bias, bias_value)

    def forward(self, features):
        cls_preds = []
        reg_preds = []

        for feature in features:
            cls_preds.append(self.cls_score(self.cls_subnet(feature)))
            reg_preds.append(self.bbox_pred(self.reg_subnet(feature)))

        return cls_preds, reg_preds
```

### Initialization Trick

```python
# Problem: At start of training, model predicts ~0.5 for all classes
# With 100,000 anchors → huge loss from easy negatives

# Solution: Initialize classification bias so initial predictions are low
# P(foreground) ≈ 0.01 at initialization

prior_prob = 0.01
bias = -math.log((1 - prior_prob) / prior_prob)  # ≈ -4.6
nn.init.constant_(cls_layer.bias, bias)

# Now initial sigmoid outputs ≈ 0.01
# Focal loss on easy negatives is tiny
```

## EfficientDet (2020)

**Key improvements**: BiFPN, compound scaling.

### BiFPN (Bidirectional Feature Pyramid Network)

```
                    P7_in ─────────────────► P7_out
                      │                         ▲
                      ▼                         │
                    P6_in ──► P6_td ──► + ──► P6_out
                      │         │       ▲       ▲
                      │         ▼       │       │
                    P5_in ──► P5_td ──► + ──► P5_out
                      │         │       ▲       ▲
                      │         ▼       │       │
                    P4_in ──► P4_td ──► + ──► P4_out
                      │         │       ▲       ▲
                      │         ▼       │       │
                    P3_in ──────────► + ──────► P3_out

Legend:
_in  = input from backbone
_td  = top-down (like regular FPN)
_out = output (top-down + bottom-up + input)
```

### Weighted Feature Fusion

```python
class WeightedFeatureFusion(nn.Module):
    """
    BiFPN uses learned weights for feature fusion.
    """
    def __init__(self, num_inputs, epsilon=1e-4):
        super().__init__()
        # Learnable weights (one per input)
        self.weights = nn.Parameter(torch.ones(num_inputs))
        self.epsilon = epsilon

    def forward(self, inputs):
        # Fast normalized fusion
        # w_i' = w_i / (sum(w_j) + ε)
        weights = F.relu(self.weights)
        weights = weights / (weights.sum() + self.epsilon)

        # Weighted sum
        output = sum(w * x for w, x in zip(weights, inputs))
        return output
```

### Compound Scaling

```python
# Scale all dimensions together
# width, depth, resolution

def get_efficientdet_config(phi):
    """
    phi: scaling coefficient (0-7)
    """
    configs = {
        # (width_mult, depth_mult, resolution)
        0: (1.0, 1.0, 512),
        1: (1.0, 1.1, 640),
        2: (1.1, 1.2, 768),
        3: (1.2, 1.4, 896),
        4: (1.4, 1.8, 1024),
        5: (1.6, 2.2, 1280),
        6: (1.8, 2.6, 1280),
        7: (2.0, 3.1, 1536),
    }
    return configs[phi]

# EfficientDet-D0 to D7
# Accuracy increases with phi
# Speed decreases with phi
```

### EfficientDet Performance

| Model | AP (COCO) | Params | FLOPs |
|-------|-----------|--------|-------|
| D0 | 34.6 | 3.9M | 2.5B |
| D1 | 40.5 | 6.6M | 6.1B |
| D2 | 43.0 | 8.1M | 11B |
| D3 | 47.5 | 12M | 25B |
| D4 | 49.7 | 21M | 55B |
| D5 | 51.5 | 34M | 130B |
| D6 | 52.6 | 52M | 226B |
| D7 | 53.7 | 77M | 410B |

## Comparison Summary

| Model | Year | Key Innovation | AP (COCO) | Speed |
|-------|------|----------------|-----------|-------|
| SSD300 | 2016 | Multi-scale detection | 25.1 | 46 FPS |
| SSD512 | 2016 | Higher resolution | 28.8 | 19 FPS |
| RetinaNet-R50 | 2017 | Focal Loss | 37.4 | 14 FPS |
| RetinaNet-R101 | 2017 | Deeper backbone | 39.1 | 11 FPS |
| EfficientDet-D0 | 2020 | BiFPN + scaling | 34.6 | 62 FPS |
| EfficientDet-D4 | 2020 | Larger scale | 49.7 | 13 FPS |

## When to Use What

| Scenario | Recommendation |
|----------|----------------|
| **Real-time, resource constrained** | SSD MobileNet |
| **Balanced accuracy/speed** | RetinaNet R-50 |
| **Maximum accuracy (one-stage)** | EfficientDet-D7 |
| **Transfer learning baseline** | RetinaNet (well-documented) |
| **Mobile deployment** | EfficientDet-D0 |
