# Backends

| Backend | Device | Torch Index |
|---------|--------|-------------|
| `cuda` | NVIDIA GPU | `cu124` |
| `rocm` | AMD GPU | `rocm6.2` |
| `openvino` | Intel CPU/iGPU | - |
| `tensorrt` | NVIDIA GPU (optimized) | `cu124` |
| `onnx` | CPU / any EP | - |

Not every model supports every backend — each model's `manifest.yaml` declares its supported backends. The generator will reject unsupported combinations.

## cuda

Models run through PyTorch on NVIDIA GPUs with `torch.compile` enabled by default for non-Ultralytics models (Depth Anything, Grounding DINO, etc.). `torch.compile` traces the forward pass and compiles it into fused GPU kernels via TorchInductor, eliminating Python overhead and fusing operations — typically 1.3-2x faster than eager execution after warmup. The first inference call triggers compilation and will be slow (several seconds); subsequent calls run at full speed. For Ultralytics models (YOLO), `torch.compile` is skipped because Ultralytics manages its own inference pipeline internally. You can disable compilation at launch time by setting the `use_torch_compile` ROS parameter to `false`, which is useful for debugging or if a model hits a compilation edge case.

## rocm

The AMD equivalent of the CUDA backend. It runs the same PyTorch code paths (including `torch.compile` for non-Ultralytics models) but targets AMD GPUs via the ROCm stack. The generated package is nearly identical to CUDA — the difference is the pip index URL (`rocm6.2` instead of `cu124`) and the device string. ROCm support in the broader ML ecosystem is less mature than CUDA, so expect occasional rough edges with newer model architectures, but for well-supported models like YOLO and Depth Anything it works without modification.

## tensorrt

Compiles the model into an optimized GPU engine using NVIDIA's TensorRT library. On first run, the model is exported from PyTorch to a TensorRT engine file and cached on disk; subsequent runs load the cached engine directly, skipping the export step. The compiled engine fuses operations, quantizes where possible, and runs inference significantly faster than PyTorch — typically 2-5x depending on the model and GPU. The engine is compiled for a specific GPU architecture, batch size, and input shape, so it's not portable across different GPU models. TensorRT only supports NVIDIA GPUs and currently only works with YOLO (via Ultralytics' native export path).

## onnx

Exports the model to the ONNX interchange format and runs it through Microsoft's ONNX Runtime. Like TensorRT, the first run exports and caches the `.onnx` file; subsequent runs load from cache. The key advantage is portability: ONNX Runtime supports multiple execution providers (`CPUExecutionProvider`, `CUDAExecutionProvider`, `TensorrtExecutionProvider`, `ROCMExecutionProvider`, `OpenVINOExecutionProvider`, etc.), selectable via the `execution_provider` ROS parameter at launch time. A single exported model can run on NVIDIA GPUs, AMD GPUs, Intel hardware, or plain CPU without re-exporting. Not all PyTorch operations have clean ONNX equivalents, so complex models may need custom export logic. Currently supports YOLO via Ultralytics' native ONNX export.

## openvino

Targets Intel hardware — CPUs, integrated GPUs, and VPUs. Runs inference without CUDA or any discrete GPU. OpenVINO applies Intel-specific optimizations (layer fusion, INT8 quantization on supported hardware) that make CPU inference considerably faster than naive PyTorch CPU mode. Only benefits Intel hardware — on AMD or ARM CPUs, use `onnx` with `CPUExecutionProvider` instead.
