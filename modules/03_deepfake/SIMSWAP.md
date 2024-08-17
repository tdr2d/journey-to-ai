
# SIMSWAP
https://github.com/neuralchen/SimSwap/blob/main/docs/guidance/preparation.md

## Prerequisites
- python 3.12.4+
- cuda

## Install dependencies
```sh
pip install torch torchvision torchaudio imageio insightface==0.2.1 
pip install onnxruntime moviepy
pip install onnxruntime-gpu
```

## Install pretrained models
```sh
curdir=$(pwd)

# Insightface alignment model
cd insightface_func/models
curl -LO "https://tdr2d.s3.gra.io.cloud.ovh.net/antelope.zip"
unzip antelope.zip

# face-parsing.PyTorch
cd $curdir && mkdir -p parsing_model/checkpoint && cd parsing_model/checkpoint
curl -LO "https://tdr2d.s3.gra.io.cloud.ovh.net/79999_iter.pth"

https://github.com/neuralchen/SimSwap/releases/download/512_beta/512.zip


# Simswap Pretrained model
cd $curdir
curl -LO "https://tdr2d.s3.gra.io.cloud.ovh.net/checkpoints.zip"
unzip checkpoints.zip

cd $curdir && mkdir -p arcface_model && cd arcface_model
[[ ! -e arcface_checkpoint.tar ]] && curl -LO "https://tdr2d.s3.gra.io.cloud.ovh.net/arcface_checkpoint.tar"

# Replace deprecated usage of np.float by float
find . -name "*.py" | xargs sed -i 's/np.float)/float)/g'
```


## Test Run
Your need to add the argument --crop_size 224 or --crop_size 512

```sh
# Simple run with already aligned images
python test_one_image.py --crop_size 224 --name people --Arc_path arcface_model/arcface_checkpoint.tar --pic_a_path crop_224/6.jpg --pic_b_path crop_224/ds.jpg --output_path output/

# valou
python test_wholeimage_swapsingle.py --crop_size 224 --use_mask  --name people --Arc_path arcface_model/arcface_checkpoint.tar --pic_a_path ./demo_file/valou.jpeg --pic_b_path ./demo_file/model_1.png --output_path ./output/ 


```
