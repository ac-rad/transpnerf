# TranspNerf: Neural Radiance Fields for Transparent Objects
Engineering Science Thesis, April 2024
Nicole Streltsov

## Brief Overview
TODO

## Quickstart

### 1. Installing Nerfstudio

- Requirements: NVIDIA GPU with CUDA installed. As well, install the following [NVIDIA container toolkit.](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- I suggest following Nerfstudio's [Docker image Installation](https://github.com/nerfstudio-project/nerfstudio/blob/main/docs/quickstart/installation.md#use-docker-image) method as there are fewer dependency conflicts.
   - I found that using docker run with docker exec worked best for me. Here are the commands I used for the `main` docker image:
    ```
    sudo docker run --gpus all -v /home/nicole/nerfstudio:/workspace/    -p 7007:7007  -it  -d --shm-size=12gb dromni/nerfstudio:main
    docker ps 
    docker exec -it <docker process id> /bin/bash
    ```
- To download this repository to be used with the NerfStudio framework:
```
cd nerfstudio/
git clone git@github.com:NicoleStrel/transpnerf.git
cd transpnerf/
pip install -e .
```
### 2. Data 

Two datasets were used: a synthetic dataset made with Blender and a real dataset made with iPhone image frame captures. They can be found below. 

- [Synthetic Blender Dataset](google.com)
   - Created by the script in `scripts/blender_dataset.py`
   - Train: 40 images, Test: 40 images
   - transforms.json contains the camera angle, poses, and image files
   - _depth and _normal suffixes define the depth (grayscale) and normals respectively. 
- [Real Dataset]()
  - 40 images to be used for both test/train
  - Used the nerfstudio command: `ns-process-data images --data {DATA_PATH} --output-dir {PROCESSED_DATA_DIR}` to run [COLMAP](https://github.com/colmap/colmap) to generate poses.
  - Generated depths using [Depth-Anything](https://github.com/LiheYoung/Depth-Anything) using the command:
      - `python run.py --encoder vitl --img-path <img folder> --outdir <depth folder> --grayscale --pred-only`

As well, Nerfstudio has the option to create and use any dataset with instructions [here](https://docs.nerf.studio/quickstart/custom_dataset.html). 

### 3. Running Transpnerf & Nerfacto

TODO

### 4. Running the evaluation script

TODO
