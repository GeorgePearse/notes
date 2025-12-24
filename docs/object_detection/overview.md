# Object Detection Overview

A comprehensive guide to object detection architectures, their evolution, and when to use each.

## The Timeline

```
2014 ──────────────────────────────────────────────────────────────► 2024

R-CNN                                                              RT-DETR
  │                                                                    │
  ▼                                                                    ▼
┌─────┐   ┌─────┐   ┌──────┐   ┌─────┐   ┌─────┐   ┌──────┐   ┌──────┐
│R-CNN│──►│Fast │──►│Faster│──►│ SSD │──►│YOLO │──►│Retina│──►│ FCOS │
│2014 │   │R-CNN│   │R-CNN │   │2016 │   │v1-v3│   │Net   │   │ 2019 │
└─────┘   │2015 │   │ 2015 │   └─────┘   │2016-│   │ 2017 │   └──────┘
          └─────┘   └──────┘             │2018 │   └──────┘       │
                        │                └─────┘       │          │
                        ▼                    │         ▼          ▼
                   ┌────────┐           ┌────┴────┐  ┌─────┐  ┌──────┐
                   │  Mask  │           │YOLOv4-v8│  │DETR │  │Center│
                   │ R-CNN  │           │2020-2023│  │2020 │  │ Net  │
                   │  2017  │           └─────────┘  └─────┘  └──────┘
                   └────────┘                            │
                                                         ▼
                                                    ┌─────────┐
                                                    │RT-DETR  │
                                                    │DINO 2023│
                                                    └─────────┘
```

## Taxonomy

```
Object Detection
├── Two-Stage Detectors
│   ├── R-CNN (2014)
│   ├── Fast R-CNN (2015)
│   ├── Faster R-CNN (2015)
│   ├── Mask R-CNN (2017)
│   └── Cascade R-CNN (2018)
│
├── One-Stage Detectors (Anchor-Based)
│   ├── YOLO v1-v4 (2016-2020)
│   ├── SSD (2016)
│   ├── RetinaNet (2017)
│   └── EfficientDet (2020)
│
├── Anchor-Free Detectors
│   ├── CornerNet (2018)
│   ├── CenterNet (2019)
│   ├── FCOS (2019)
│   └── YOLO v8+ (2023)
│
└── Transformer-Based
    ├── DETR (2020)
    ├── Deformable DETR (2021)
    ├── DINO (2022)
    └── RT-DETR (2023)
```

## Two-Stage vs One-Stage

| Aspect | Two-Stage | One-Stage |
|--------|-----------|-----------|
| **Architecture** | Region proposal + Classification | Single pass prediction |
| **Speed** | Slower (100-300ms) | Fast (10-50ms) |
| **Accuracy** | Higher on small objects | Slightly lower |
| **Examples** | Faster R-CNN, Mask R-CNN | YOLO, SSD, RetinaNet |
| **Use Case** | When accuracy matters most | Real-time applications |

### Two-Stage Flow
```
Image → Backbone → Region Proposal Network → RoI Features → Classification + Regression
                           ↓
                   ~2000 proposals → ~100 detections
```

### One-Stage Flow
```
Image → Backbone → Feature Pyramid → Dense Predictions → NMS → Detections
                                           ↓
                           Predictions at every location/anchor
```

## Key Metrics

### mAP (Mean Average Precision)

The standard metric for object detection:

```python
# mAP calculation conceptually
def compute_map(predictions, ground_truth, iou_thresholds=[0.5, 0.55, ..., 0.95]):
    aps = []
    for iou_thresh in iou_thresholds:
        for class_id in classes:
            precision, recall = compute_pr_curve(predictions, ground_truth, class_id, iou_thresh)
            ap = compute_area_under_curve(precision, recall)
            aps.append(ap)
    return mean(aps)
```

| Metric | IoU Threshold | Description |
|--------|---------------|-------------|
| AP | 0.5 | PASCAL VOC style |
| AP50 | 0.5 | COCO at IoU=0.5 |
| AP75 | 0.75 | COCO at IoU=0.75 (stricter) |
| mAP | 0.5:0.95 | COCO primary metric |
| AP_S | 0.5:0.95 | Small objects (<32²px) |
| AP_M | 0.5:0.95 | Medium objects (32²-96²px) |
| AP_L | 0.5:0.95 | Large objects (>96²px) |

### Speed Metrics

| Metric | Description |
|--------|-------------|
| FPS | Frames per second |
| Latency | Time for single inference |
| Throughput | Images/second with batching |

## Dataset Landscape

### COCO (Common Objects in Context)
- **Size**: 330K images, 80 categories
- **Standard**: Primary benchmark since 2015
- **Challenges**: Small objects, crowded scenes

```python
# COCO categories
categories = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', ...
]
```

### PASCAL VOC
- **Size**: 11K images, 20 categories
- **Standard**: Pre-2015 benchmark
- **Simpler**: Fewer classes, less crowded

### Open Images
- **Size**: 9M images, 600 categories
- **Scale**: Largest public dataset
- **Challenge**: Long-tail distribution

### Comparison

| Dataset | Images | Classes | Annotations | Use |
|---------|--------|---------|-------------|-----|
| COCO | 330K | 80 | 1.5M | Primary benchmark |
| VOC | 11K | 20 | 27K | Quick experiments |
| Open Images | 9M | 600 | 16M | Large-scale training |
| Objects365 | 2M | 365 | 30M | Pretraining |

## When to Use What

### Decision Flowchart

```
                    Start
                      │
                      ▼
              Real-time needed?
              /              \
           Yes                No
            │                  │
            ▼                  ▼
    Latency budget?     Accuracy priority?
    /          \        /            \
  <10ms     10-50ms   Yes             No
    │          │        │              │
    ▼          ▼        ▼              ▼
 YOLOv8n   YOLOv8s   Cascade      Faster
 RT-DETR   RT-DETR   R-CNN        R-CNN
  tiny      base     DINO          YOLO
                                   medium
```

### Recommendations by Use Case

| Use Case | Recommended | Why |
|----------|-------------|-----|
| **Real-time video** | YOLOv8, RT-DETR | Speed + accuracy balance |
| **Autonomous driving** | YOLOv8-L, RT-DETR-X | Robust, well-tested |
| **Medical imaging** | Faster R-CNN, DINO | Accuracy on small objects |
| **Satellite imagery** | DINO, Cascade R-CNN | High resolution, small objects |
| **Edge devices** | YOLOv8n, YOLO-NAS | Optimized for mobile |
| **Instance segmentation** | Mask R-CNN, YOLO-seg | Built-in mask prediction |
| **Research baseline** | Faster R-CNN, DETR | Well-documented, reproducible |

## Speed vs Accuracy (COCO val2017)

```
mAP
 │
60┤                                    ● DINO-SwinL
  │                               ● DINO-R50
55┤                          ● RT-DETR-X
  │                     ● YOLOv8x
50┤                ● YOLOv8l    ● Cascade R-CNN
  │           ● YOLOv8m
45┤      ● YOLOv8s
  │ ● YOLOv8n
40┤
  │
35┤
  └────────────────────────────────────────────► FPS
     200   100    50    30    20    10     5

   Fast ◄─────────────────────────────────► Accurate
```

## Common Architecture Pattern

Most modern detectors follow this pattern:

```
┌─────────────────────────────────────────────────────────┐
│                      Input Image                         │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    Backbone                              │
│         (ResNet, CSPDarknet, Swin, etc.)                │
│                                                          │
│    ┌────┐    ┌────┐    ┌────┐    ┌────┐    ┌────┐      │
│    │ C1 │───►│ C2 │───►│ C3 │───►│ C4 │───►│ C5 │      │
│    └────┘    └────┘    └────┘    └────┘    └────┘      │
│                          │         │         │          │
└──────────────────────────┼─────────┼─────────┼──────────┘
                           │         │         │
                           ▼         ▼         ▼
┌─────────────────────────────────────────────────────────┐
│                      Neck (FPN)                          │
│                                                          │
│    ┌────┐    ┌────┐    ┌────┐                           │
│    │ P3 │◄───│ P4 │◄───│ P5 │                           │
│    └────┘    └────┘    └────┘                           │
│       │         │         │                              │
└───────┼─────────┼─────────┼──────────────────────────────┘
        │         │         │
        ▼         ▼         ▼
┌─────────────────────────────────────────────────────────┐
│                    Detection Head                        │
│                                                          │
│  Classification + Bounding Box Regression (+ Mask)      │
│                                                          │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              Post-Processing (NMS)                       │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
                     Detections
```

## Quick Start Examples

### YOLO (ultralytics)
```python
from ultralytics import YOLO

model = YOLO('yolov8m.pt')
results = model('image.jpg')
results[0].show()
```

### Faster R-CNN (torchvision)
```python
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
model.eval()

# Inference
predictions = model([image_tensor])
```

### DETR (transformers)
```python
from transformers import DetrForObjectDetection, DetrImageProcessor

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
```

## What's Next

Detailed deep-dives:
- [Common Components](common-components.md) - Backbones, FPN, NMS, losses
- [R-CNN Family](rcnn-family.md) - Two-stage detectors
- [YOLO](yolo.md) - The YOLO evolution
- [SSD & RetinaNet](ssd-retinanet.md) - One-stage pioneers
- [Anchor-Free](anchor-free.md) - FCOS, CenterNet
- [DETR Family](detr-family.md) - Transformer detectors
