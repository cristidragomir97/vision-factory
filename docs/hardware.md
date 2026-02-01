# Hardware Platforms

Different robots and edge devices ship with different hardware. The backend you choose depends on what's on the board.

## NVIDIA GPUs (desktop, workstation, data center)

Desktop and workstation GPUs (RTX 3000/4000/5000 series, A-series, etc.) have the broadest support. Use `--backend cuda` for development and `--backend tensorrt` for optimized deployment. All models work with CUDA. TensorRT currently supports YOLO.

```bash
# Development — PyTorch with torch.compile
vision-factory generate --model yolo --backend cuda --variant yolo_v8s --package-name my_detector

# Production — TensorRT compiled engine
vision-factory generate --model yolo --backend tensorrt --variant yolo_v8s --package-name my_detector
```

## NVIDIA Jetson (Orin, Xavier, Nano)

Jetson devices run the same CUDA and TensorRT stacks as desktop NVIDIA GPUs but with ARM CPUs and shared GPU memory. The generated packages work without modification — use `--backend cuda` for flexibility or `--backend tensorrt` for the best inference throughput on Jetson's constrained power budget. TensorRT is particularly valuable here since Jetson devices are typically deployed with strict latency requirements.

```bash
# Jetson deployment — TensorRT for best perf/watt
vision-factory generate --model yolo --backend tensorrt --variant yolo_v8n --package-name jetson_detector
```

## AMD GPUs (Radeon, Instinct)

AMD GPUs are supported via PyTorch's ROCm backend. The generated code is identical to CUDA — PyTorch abstracts both under the same `cuda` device API. The difference is the pip dependencies: the package installs `torch` from the `rocm6.2` index instead of `cu124`. ROCm support is less mature than CUDA in the broader ML ecosystem, but well-supported models work without issues.

```bash
vision-factory generate --model depth_anything --backend rocm --variant depth_anything_v2_vitb --package-name depth_node
```

## Intel CPUs and iGPUs (NUC, industrial PCs)

For x86 deployments without a discrete GPU, use `--backend openvino`. OpenVINO applies Intel-specific optimizations (layer fusion, INT8 quantization) that run significantly faster than naive CPU inference. This is the right choice for Intel NUCs, industrial PCs, and any x86 edge device where no NVIDIA or AMD GPU is available.

```bash
vision-factory generate --model yolo --backend openvino --variant yolo_v8n --package-name edge_detector
```

## NPUs (Intel Core Ultra, AMD Ryzen AI, Qualcomm Snapdragon X)

Recent laptop and desktop processors ship with dedicated neural processing units — Intel's NPU in Core Ultra 2/3 series, AMD's XDNA in Ryzen AI (HX 370, etc.), and Qualcomm's Hexagon in Snapdragon X Elite. These are the "Copilot+ PC" chips. Vision-factory can target them through existing backends, no special NPU backend needed.

**Intel Core Ultra NPUs** are supported through OpenVINO. The OpenVINO runtime auto-detects the NPU and can offload supported models to it. Generate with `--backend openvino` and set the device to NPU at launch time.

```bash
vision-factory generate --model yolo --backend openvino --variant yolo_v8n --package-name npu_detector

# At launch, target the NPU explicitly:
ros2 launch npu_detector vision.launch.py device:=NPU
```

**AMD Ryzen AI (XDNA) NPUs** are supported through ONNX Runtime with AMD's Vitis AI execution provider. Generate with `--backend onnx` and select `VitisAIExecutionProvider` at launch. Requires the Ryzen AI SDK to be installed on the target machine.

```bash
vision-factory generate --model yolo --backend onnx --variant yolo_v8n --package-name ryzen_detector

ros2 launch ryzen_detector vision.launch.py execution_provider:=VitisAIExecutionProvider
```

**Qualcomm Snapdragon X** NPUs are also supported through ONNX Runtime, via the QNN execution provider. Generate with `--backend onnx` and select `QNNExecutionProvider` at launch. Requires the Qualcomm AI Engine Direct SDK.

```bash
vision-factory generate --model yolo --backend onnx --variant yolo_v8n --package-name snapdragon_detector

ros2 launch snapdragon_detector vision.launch.py execution_provider:=QNNExecutionProvider
```

**Caveats:** NPU support is untested. Operator coverage on NPUs is narrower than on CPUs and GPUs — simple models like YOLO will likely work, but complex attention-heavy architectures (Grounding DINO, Florence) may hit unsupported ops that silently fall back to CPU. Verify actual NPU utilization with vendor profiling tools (Intel `vtune`, AMD `amdprofsys`, Qualcomm Profiler) before assuming your model is running on the NPU.

## Mixed or unknown hardware

If you need one model artifact that runs on any hardware, use `--backend onnx`. ONNX Runtime supports multiple execution providers selectable at launch time via a ROS parameter — the same package runs on NVIDIA, AMD, Intel, or plain CPU without re-exporting.

```bash
vision-factory generate --model yolo --backend onnx --variant yolo_v8s --package-name portable_detector

# Then at launch time, pick the execution provider:
ros2 launch portable_detector vision.launch.py execution_provider:=CUDAExecutionProvider
ros2 launch portable_detector vision.launch.py execution_provider:=CPUExecutionProvider
```
