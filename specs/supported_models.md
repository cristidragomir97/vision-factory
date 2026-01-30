# Supported Models Reference

## Detection & Open-Vocabulary

| Model | Description | Use Case | Link |
|-------|-------------|----------|------|
| **Grounding DINO** | Open-set object detection with text prompts | Detect anything by description | [GitHub](https://github.com/IDEA-Research/GroundingDINO) |
| **YOLO (v8/v9/v10/v11)** | Fast real-time object detection | General detection, tracking | [Ultralytics](https://github.com/ultralytics/ultralytics) |
| **OWL-ViT / OWLv2** | Open-vocabulary detection using CLIP | Text-guided detection | [HuggingFace](https://huggingface.co/google/owlv2-base-patch16-ensemble) |
| **RT-DETR** | Real-time Detection Transformer | Fast accurate detection | [GitHub](https://github.com/lyuwenyu/RT-DETR) |
| **DETIC** | Detect 20k+ categories | Large vocabulary detection | [GitHub](https://github.com/facebookresearch/Detic) |

## Depth Estimation

| Model | Description | Use Case | Link |
|-------|-------------|----------|------|
| **Depth Anything (v1/v2)** | Relative monocular depth | General depth perception | [GitHub](https://github.com/LiheYoung/Depth-Anything) |
| **ZoeDepth** | Metric depth estimation | Actual distance in meters | [GitHub](https://github.com/isl-org/ZoeDepth) |
| **UniDepth** | Camera-agnostic metric depth | Works without camera intrinsics | [GitHub](https://github.com/lpiccinelli-eth/UniDepth) |
| **Metric3D v2** | Metric depth + surface normals | Navigation, 3D reconstruction | [GitHub](https://github.com/YvanYin/Metric3D) |
| **MASt3R** | 3D reconstruction from image pairs | Mapping, localization | [GitHub](https://github.com/naver/mast3r) |
| **DUSt3R** | Dense 3D from unconstrained images | Scene reconstruction | [GitHub](https://github.com/naver/dust3r) |

## Segmentation

| Model | Description | Use Case | Link |
|-------|-------------|----------|------|
| **Segment Anything (SAM)** | Promptable segmentation | Interactive segmentation | [GitHub](https://github.com/facebookresearch/segment-anything) |
| **SAM 2** | Video segmentation + improved SAM | Video object segmentation | [GitHub](https://github.com/facebookresearch/segment-anything-2) |
| **FastSAM** | Fast SAM alternative | Real-time segmentation | [GitHub](https://github.com/CASIA-IVA-Lab/FastSAM) |
| **MobileSAM** | Lightweight SAM | Edge devices | [GitHub](https://github.com/ChaoningZhang/MobileSAM) |
| **Grounded-SAM** | DINO + SAM combined | Detect then segment | [GitHub](https://github.com/IDEA-Research/Grounded-Segment-Anything) |
| **OneFormer** | Unified segmentation | Panoptic/semantic/instance | [GitHub](https://github.com/SHI-Labs/OneFormer) |

## Pose Estimation

| Model | Description | Use Case | Link |
|-------|-------------|----------|------|
| **RTMPose** | Real-time multi-person pose | Human pose tracking | [GitHub](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) |
| **YOLO-Pose** | YOLO with keypoints | Fast pose detection | [Ultralytics](https://docs.ultralytics.com/tasks/pose/) |
| **ViTPose** | Vision Transformer pose | High accuracy pose | [GitHub](https://github.com/ViTAE-Transformer/ViTPose) |
| **MediaPipe Pose** | Lightweight pose | Mobile/embedded | [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) |

## Tracking

| Model | Description | Use Case | Link |
|-------|-------------|----------|------|
| **ByteTrack** | Multi-object tracking | Object tracking | [GitHub](https://github.com/ifzhang/ByteTrack) |
| **BoT-SORT** | Robust multi-object tracking | Occlusion handling | [GitHub](https://github.com/NirAharon/BoT-SORT) |
| **Co-Tracker** | Dense point tracking | Motion estimation | [GitHub](https://github.com/facebookresearch/co-tracker) |
| **TAPIR** | Point tracking any video | Long-term tracking | [GitHub](https://github.com/deepmind/tapnet) |
| **SAM 2** | Video object segmentation | Segment + track | [GitHub](https://github.com/facebookresearch/segment-anything-2) |

## Vision-Language Models

| Model | Description | Use Case | Link |
|-------|-------------|----------|------|
| **Florence-2** | Unified vision model | Detection, captioning, OCR | [HuggingFace](https://huggingface.co/microsoft/Florence-2-large) |
| **Qwen2.5-VL 3B** | Small VLM | Scene understanding, VQA | [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) |
| **CLIP** | Image-text similarity | Semantic search | [GitHub](https://github.com/openai/CLIP) |
| **SigLIP** | Improved CLIP | Better zero-shot | [HuggingFace](https://huggingface.co/google/siglip-base-patch16-224) |
| **PaliGemma** | Gemma-based VLM | Detection, segmentation, VQA | [HuggingFace](https://huggingface.co/google/paligemma-3b-pt-224) |

## Robotics-Specific

| Model | Description | Use Case | Link |
|-------|-------------|----------|------|
| **AnyGrasp** | 6-DoF grasp detection | Pick and place | [GitHub](https://github.com/graspnet/anygrasp_sdk) |
| **GraspNet** | Grasp pose prediction | Manipulation | [GitHub](https://github.com/graspnet/graspnet-baseline) |
| **Contact-GraspNet** | Contact-based grasping | Dense clutter | [GitHub](https://github.com/NVlabs/contact_graspnet) |
| **ClearGrasp** | Transparent object depth | Grasping glass/plastic | [GitHub](https://github.com/Shreeyak/cleargrasp) |
| **Dex-Net** | Analytic grasp quality | Bin picking | [GitHub](https://github.com/BerkeleyAutomation/dex-net) |

## OCR & Document

| Model | Description | Use Case | Link |
|-------|-------------|----------|------|
| **PaddleOCR** | Multi-language OCR | Text detection/recognition | [GitHub](https://github.com/PaddlePaddle/PaddleOCR) |
| **EasyOCR** | Ready-to-use OCR | Simple text extraction | [GitHub](https://github.com/JaidedAI/EasyOCR) |
| **TrOCR** | Transformer OCR | High accuracy OCR | [HuggingFace](https://huggingface.co/microsoft/trocr-base-printed) |
| **DocTR** | Document text recognition | Structured documents | [GitHub](https://github.com/mindee/doctr) |

## Face & Identity

| Model | Description | Use Case | Link |
|-------|-------------|----------|------|
| **InsightFace** | Face detection + recognition | Identity verification | [GitHub](https://github.com/deepinsight/insightface) |
| **RetinaFace** | Fast face detection | Face localization | [GitHub](https://github.com/serengil/retinaface) |
| **ArcFace** | Face embeddings | Face matching | [GitHub](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) |

---

## Priority for Implementation

### Phase 1 (Core)
1. YOLO v8/v11 - Fast general detection
2. Depth Anything v2 - Relative depth
3. Grounding DINO - Open-set detection
4. SAM 2 - Segmentation

### Phase 2 (Extended)
5. ZoeDepth - Metric depth
6. RTMPose - Human pose
7. ByteTrack - Object tracking
8. Florence-2 - Multi-task

### Phase 3 (Robotics)
9. AnyGrasp - Grasp detection
10. Grounded-SAM - Detect + segment
11. Co-Tracker - Point tracking
12. Qwen2.5-VL - Scene understanding
