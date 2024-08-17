
## Prerequisites
- git-lfs: `sudo apt install -y git-lfs`
- python3: https://github.com/tdr2d/cloud-native-cheat/blob/master/sys/unix/install_python.sh
- CUDA: https://gist.github.com/denguir/b21aa66ae7fb1089655dd9de8351a202 (official doc: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu)
- CUDA HPC SDK https://developer.nvidia.com/hpc-sdk-downloads
- CUDA postinstall:
```bash
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.7/cuda/12.5/targets/x86_64-linux/lib/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
- onnxruntime-gpu latest pip package: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- if a lib can't be found: "libcudart.so.12: cannot open shared object file: No such file or directory" look for the lib location with this command : `sudo find / -name "*libcudart.so*"` and update LD_LIBRARY_PATH to the directory path containing the libs.


TensorRT runtime is too slow +1min compared to CUDAExecutionProvider
- tensorrt: 
```bash
# Download urls: https://developer.nvidia.com/tensorrt/download/
# Documentation: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-debian
sudo python3 -m pip install --upgrade tensorrt
# update LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/tensorrt_libs/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

# Install 
Follow install instructions https://github.com/haofanwang/inswapper/tree/main


# Usage
```bash
python swapper.py --source_img="./data/valou.jpeg" --target_img "./data/model_1.png" \
--face_restore --background_enhance --face_upsample --upscale=2 --codeformer_fidelity=0.5
```


# Notes
CPUExecutionProvider is actually faster than CUDA for roughtly the same CPU usage