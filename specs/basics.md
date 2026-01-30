### the-vision-node
An extremely modular vision node for ROS, allowing users to run vision models on device with acceleration, ranges from simple orin class devices to AMD Halo Strix, NVidia Thor, and last gen Intel OpenVino devices. 

## Basics:
- should be extremely modular and scallable
- should support CUDA, ROCm and OpenVino acceleration (starting with cuda)
- should plug into ROS, basically a ROS package
- should support preloading models in VRAM or dynamic loading
- lifecycle support (later on)
- for now support for:
    * grounding-dino
    * yolo-models 
    * depth-anything
    * segment-anything
    * maybe florence 
    * maybe qwen2.5 vl 3B 
