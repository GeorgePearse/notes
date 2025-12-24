# DETR Family

Transformer-based end-to-end object detection.

## DETR (2020)

**Detection Transformer** - The first fully end-to-end detector with no NMS, no anchors.

### Key Innovations

1. **Set prediction**: Direct prediction of object set (no duplicates)
2. **Bipartite matching**: Hungarian algorithm for GT assignment
3. **Transformers**: Global reasoning through self-attention
4. **No post-processing**: No NMS, no anchor tuning

### Architecture

```
                        Input Image (800 × 1066)
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         CNN Backbone                                  │
│                        (ResNet-50)                                    │
│                                                                       │
│   Output: Feature map [B, C, H, W] = [B, 2048, 25, 34]               │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        1×1 Conv                                       │
│                  Reduce to 256 channels                               │
│                                                                       │
│   Output: [B, 256, 25, 34]                                           │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  Positional Encoding                                  │
│              (Fixed sinusoidal, 2D)                                   │
│                                                                       │
│   Add spatial information since transformers are permutation-invariant│
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    Transformer Encoder                                │
│                      (6 layers)                                       │
│                                                                       │
│   Self-attention over all spatial positions                          │
│   Global reasoning about the image                                   │
│                                                                       │
│   Input:  850 tokens (25×34 positions)                               │
│   Output: 850 tokens with global context                             │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    Transformer Decoder                                │
│                      (6 layers)                                       │
│                                                                       │
│   Object Queries: N=100 learnable embeddings                         │
│   Cross-attention: queries attend to encoder output                  │
│   Self-attention: queries attend to each other                       │
│                                                                       │
│   Output: 100 detection embeddings                                   │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    Prediction FFN                                     │
│            (Per-query, shared weights)                                │
│                                                                       │
│   Class prediction: Linear(256, num_classes + 1)  (+ "no object")   │
│   Box prediction: MLP(256, 256, 4)  (normalized xywh)               │
│                                                                       │
│   Output: 100 (class, box) predictions                               │
└──────────────────────────────────────────────────────────────────────┘
```

### Object Queries

```python
# Learnable query embeddings
# Each query "learns" to detect a different type of object/location

class DETR(nn.Module):
    def __init__(self, num_queries=100, hidden_dim=256):
        super().__init__()
        # Object queries - learned positional embeddings
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # These queries specialize during training:
        # - Some queries specialize in large objects
        # - Some in specific locations
        # - Some in specific classes
```

### Bipartite Matching (Hungarian Algorithm)

```python
from scipy.optimize import linear_sum_assignment

def hungarian_matching(pred_boxes, pred_classes, gt_boxes, gt_classes):
    """
    Find optimal 1-to-1 assignment between predictions and ground truth.

    Cost = λ_cls × class_cost + λ_L1 × L1_cost + λ_giou × GIoU_cost
    """
    num_preds = len(pred_boxes)
    num_gt = len(gt_boxes)

    # Cost matrix [num_preds, num_gt]
    cost_matrix = torch.zeros(num_preds, num_gt)

    for i in range(num_preds):
        for j in range(num_gt):
            # Classification cost
            cls_cost = -pred_classes[i, gt_classes[j]]  # Negative prob

            # L1 cost
            l1_cost = F.l1_loss(pred_boxes[i], gt_boxes[j], reduction='none').sum()

            # GIoU cost
            giou_cost = -generalized_box_iou(pred_boxes[i], gt_boxes[j])

            cost_matrix[i, j] = cls_cost + 5 * l1_cost + 2 * giou_cost

    # Hungarian algorithm finds minimum cost assignment
    row_indices, col_indices = linear_sum_assignment(cost_matrix.cpu().numpy())

    return row_indices, col_indices

# Example:
# Predictions: [pred_0, pred_1, ..., pred_99]
# GT boxes: [gt_0, gt_1, gt_2]
# Matching might be: pred_23→gt_0, pred_45→gt_1, pred_12→gt_2
# Other predictions matched to "no object"
```

### DETR Loss

```python
def detr_loss(pred_classes, pred_boxes, gt_classes, gt_boxes, indices):
    """
    Compute loss only for matched predictions.

    Args:
        indices: Output of Hungarian matching [(pred_idx, gt_idx), ...]
    """
    # Get matched predictions and targets
    pred_idx = indices[0]
    gt_idx = indices[1]

    # Classification loss (all predictions)
    # Matched predictions should predict GT class
    # Unmatched predictions should predict "no object"
    target_classes = torch.full(pred_classes.shape[:1], num_classes)  # "no object"
    target_classes[pred_idx] = gt_classes[gt_idx]
    cls_loss = F.cross_entropy(pred_classes, target_classes)

    # Box loss (only matched predictions)
    matched_pred_boxes = pred_boxes[pred_idx]
    matched_gt_boxes = gt_boxes[gt_idx]

    l1_loss = F.l1_loss(matched_pred_boxes, matched_gt_boxes)
    giou_loss = 1 - generalized_box_iou(matched_pred_boxes, matched_gt_boxes).diag()

    return cls_loss + 5 * l1_loss + 2 * giou_loss.mean()
```

### Positional Encoding

```python
def positional_encoding_2d(d_model, height, width):
    """
    2D sinusoidal positional encoding.
    """
    pe = torch.zeros(d_model, height, width)

    # Separate encodings for x and y
    d_model_half = d_model // 2
    div_term = torch.exp(torch.arange(0., d_model_half, 2) *
                         -(math.log(10000.0) / d_model_half))

    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)

    pe[0:d_model_half:2, :, :] = torch.sin(pos_w * div_term).T.unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model_half:2, :, :] = torch.cos(pos_w * div_term).T.unsqueeze(1).repeat(1, height, 1)
    pe[d_model_half::2, :, :] = torch.sin(pos_h * div_term).T.unsqueeze(2).repeat(1, 1, width)
    pe[d_model_half+1::2, :, :] = torch.cos(pos_h * div_term).T.unsqueeze(2).repeat(1, 1, width)

    return pe
```

### DETR Limitations

1. **Slow convergence**: 500 epochs vs ~90 for Faster R-CNN
2. **Poor small object detection**: Global attention loses local detail
3. **High computational cost**: O(n²) attention over all pixels

## Deformable DETR (2021)

**Key insight**: Sparse attention to a small set of keys around reference points.

### The Problem with DETR

```
DETR Attention:
Every query attends to ALL spatial locations
850 × 850 = 722,500 attention weights per layer!

For multi-scale features (needed for small objects):
Even more locations → Quadratic explosion
```

### Deformable Attention

```python
class DeformableAttention(nn.Module):
    """
    Instead of attending to all locations,
    attend to K learned offset positions per query.
    """
    def __init__(self, d_model, n_heads=8, n_levels=4, n_points=4):
        super().__init__()
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points

        # Learn offsets from reference point
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        # Learn attention weights
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        # Value projection
        self.value_proj = nn.Linear(d_model, d_model)
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes):
        """
        query: [B, num_queries, d_model]
        reference_points: [B, num_queries, n_levels, 2]  (normalized x, y)
        input_flatten: [B, sum(H_i × W_i), d_model]  (multi-scale features)
        """
        B, Len_q, _ = query.shape

        # Predict sampling offsets (small offsets from reference points)
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(B, Len_q, self.n_heads,
                                                  self.n_levels, self.n_points, 2)

        # Predict attention weights
        attention_weights = self.attention_weights(query)
        attention_weights = attention_weights.view(B, Len_q, self.n_heads,
                                                    self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, dim=-1)

        # Sample features at offset locations
        sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets
        # Use bilinear interpolation to sample at non-integer locations
        sampled_values = bilinear_sample(input_flatten, sampling_locations)

        # Weighted sum
        output = (attention_weights.unsqueeze(-1) * sampled_values).sum(-2).sum(-2)
        output = self.output_proj(output)

        return output
```

### Multi-Scale Features

```python
class DeformableDETR(nn.Module):
    def __init__(self, backbone, num_classes, num_queries=300):
        super().__init__()
        self.backbone = backbone

        # Multi-scale feature maps (like FPN)
        # C3 (1/8), C4 (1/16), C5 (1/32), C6 (1/64)
        self.input_proj = nn.ModuleList([
            nn.Conv2d(256, 256, 1),
            nn.Conv2d(512, 256, 1),
            nn.Conv2d(1024, 256, 1),
            nn.Conv2d(2048, 256, 1),
        ])

        # Encoder with deformable attention
        self.encoder = DeformableTransformerEncoder(...)

        # Decoder with deformable cross-attention
        self.decoder = DeformableTransformerDecoder(...)
```

### Two-Stage Variant

```python
# Optional: Generate initial reference points from encoder
class TwoStageDeformableDETR(nn.Module):
    def forward(self, x):
        # Backbone + encoder
        memory = self.encoder(self.backbone(x))

        # Stage 1: Generate proposals from encoder output
        proposals = self.proposal_generator(memory)  # [B, num_proposals, 4]
        top_k_proposals = select_top_k(proposals)

        # Stage 2: Refine proposals in decoder
        # Use proposals as reference points for object queries
        refined = self.decoder(memory, top_k_proposals)

        return refined
```

### Improvements Over DETR

| Aspect | DETR | Deformable DETR |
|--------|------|-----------------|
| Convergence | 500 epochs | 50 epochs |
| Multi-scale | No | Yes (FPN-like) |
| Small objects | Poor | Good |
| Memory | O(HW × HW) | O(HW × K) |

## DINO (2022)

**DETR with Improved deNoising anchOr boxes**

### Key Innovations

1. **Contrastive denoising**: Add noise to GT, train to denoise
2. **Mixed query selection**: Combine learned + content queries
3. **Look forward twice**: Use next-layer prediction for current refinement

### Contrastive Denoising

```python
def contrastive_denoising(gt_boxes, gt_classes, num_dn_groups=5, noise_scale=0.4):
    """
    Create denoising queries by adding noise to ground truth.

    Positive queries: GT + small noise → should reconstruct GT
    Negative queries: GT + large noise → should predict "no object"
    """
    num_gt = len(gt_boxes)

    # Positive queries (small noise)
    positive_boxes = gt_boxes.repeat(num_dn_groups, 1)
    positive_boxes += torch.randn_like(positive_boxes) * noise_scale * 0.5
    positive_labels = gt_classes.repeat(num_dn_groups)

    # Negative queries (large noise)
    negative_boxes = gt_boxes.repeat(num_dn_groups, 1)
    negative_boxes += torch.randn_like(negative_boxes) * noise_scale * 2.0
    negative_labels = torch.full((num_gt * num_dn_groups,), num_classes)  # "no object"

    # Concatenate
    dn_boxes = torch.cat([positive_boxes, negative_boxes])
    dn_labels = torch.cat([positive_labels, negative_labels])

    return dn_boxes, dn_labels
```

### Architecture

```
                    Input Image
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│                    Backbone + FPN                               │
└────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│                Deformable Encoder                               │
│                (6 layers)                                       │
└────────────────────────────────────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
┌──────────────────┐         ┌──────────────────┐
│  Mixed Queries   │         │  Denoising       │
│                  │         │  Queries         │
│ Content + Pos    │         │ GT + noise       │
└──────────────────┘         └──────────────────┘
          │                             │
          └──────────────┬──────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│                Deformable Decoder                               │
│                (6 layers)                                       │
│                                                                 │
│  Self-attention mask prevents DN queries seeing matching queries│
└────────────────────────────────────────────────────────────────┘
                         │
                         ▼
                   Predictions
```

### DINO Performance

| Model | Backbone | Epochs | AP |
|-------|----------|--------|-----|
| DETR | R-50 | 500 | 42.0 |
| Deformable DETR | R-50 | 50 | 43.8 |
| DINO | R-50 | 12 | 49.0 |
| DINO | R-50 | 36 | 50.9 |
| DINO | Swin-L | 36 | 58.5 |

## RT-DETR (2023)

**Real-Time DETR** - First real-time end-to-end detector.

### Key Innovations

1. **Efficient hybrid encoder**: CNN + transformer
2. **IoU-aware query selection**: Better initial queries
3. **Decoupled intra-scale attention**: Reduce computation

### Architecture

```
                    Input Image
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│            Efficient Hybrid Encoder                             │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │         CNN Backbone (HGNetv2)                           │  │
│   │         Fast, efficient feature extraction               │  │
│   └─────────────────────────────────────────────────────────┘  │
│                         │                                       │
│                         ▼                                       │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │      Attention-based Intra-scale Feature Interaction     │  │
│   │      (Self-attention within each scale, not across)      │  │
│   └─────────────────────────────────────────────────────────┘  │
│                         │                                       │
│                         ▼                                       │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │           CNN-based Cross-scale Fusion                   │  │
│   │           (Efficient alternative to multi-scale attn)    │  │
│   └─────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│            IoU-aware Query Selection                            │
│                                                                 │
│   Select top-K encoder features as initial queries             │
│   Use IoU prediction to weight selection                       │
└────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│                  Transformer Decoder                            │
│                      (6 layers)                                 │
└────────────────────────────────────────────────────────────────┘
                         │
                         ▼
                   Predictions
```

### Speed vs Accuracy

```
                    RT-DETR-R50
                         ●
        DINO-R50         │
              ●          │
                   ●─────┤  RT-DETR-R101
    Deformable DETR      │
              ●          │
                         │
    DETR ●               │
         │               │
         └───────────────┴────────────────────► Speed (FPS)
         10     20      40     60     80    100
```

| Model | Backbone | AP | FPS (T4) |
|-------|----------|-----|----------|
| YOLO-v8-L | CSPDarknet | 52.9 | 78 |
| RT-DETR-R50 | ResNet-50 | 53.1 | 108 |
| RT-DETR-R101 | ResNet-101 | 54.3 | 74 |

### Usage

```python
from ultralytics import RTDETR

# Load model
model = RTDETR('rtdetr-l.pt')

# Inference
results = model('image.jpg')

# Train
model.train(data='coco.yaml', epochs=100)

# Export
model.export(format='onnx')
```

## Hugging Face Transformers

```python
from transformers import DetrForObjectDetection, DetrImageProcessor
import torch

# Load model and processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Inference
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# Post-process
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(
    outputs, target_sizes=target_sizes, threshold=0.9
)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    print(f"{model.config.id2label[label.item()]}: {score.item():.2f} at {box.tolist()}")
```

### Available Models

```python
# DETR
"facebook/detr-resnet-50"
"facebook/detr-resnet-101"

# Deformable DETR
"SenseTime/deformable-detr"

# Conditional DETR
"microsoft/conditional-detr-resnet-50"

# DINO (when available)
# Check Hugging Face model hub
```

## Comparison Summary

| Model | Year | Key Innovation | AP | Speed | Epochs |
|-------|------|----------------|-----|-------|--------|
| DETR | 2020 | End-to-end, Hungarian | 42.0 | Slow | 500 |
| Deformable DETR | 2021 | Sparse attention | 43.8 | Medium | 50 |
| Conditional DETR | 2021 | Conditional cross-attention | 43.0 | Medium | 50 |
| DAB-DETR | 2022 | Dynamic anchor boxes | 45.7 | Medium | 50 |
| DINO | 2022 | Denoising, contrastive | 50.9 | Medium | 36 |
| RT-DETR | 2023 | Efficient encoder | 54.3 | Fast | 72 |

## When to Use DETR-family

| Scenario | Recommendation |
|----------|----------------|
| **Research/understanding** | Original DETR |
| **Production (accuracy)** | DINO |
| **Production (speed)** | RT-DETR |
| **Fast prototyping** | RT-DETR via ultralytics |
| **Small objects** | Deformable DETR or DINO |
| **Instance segmentation** | Mask DINO |
