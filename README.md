# TranspNeRF: Neural Radiance Fields for Transparent Objects
University of Toronto Engineering Science Thesis, April 2024
Nicole Streltsov

This project is built on [NeRFStudio](https://github.com/nerfstudio-project/nerfstudio).

## Brief Overview
TODO

## Quickstart

### 1. Installing Nerfstudio

- Requirements: NVIDIA GPU with CUDA installed. As well, install the following [NVIDIA container toolkit.](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- I suggest following Nerfstudio's [Docker image Installation](https://github.com/nerfstudio-project/nerfstudio/blob/main/docs/quickstart/installation.md#use-docker-image) method as there are fewer dependency conflicts.
   - I found that using docker run with docker exec worked best for me. Here are the commands I used after pulling the `main` docker image:
    ```
    sudo docker run --gpus all -v /home/nicole/nerfstudio:/workspace/    -p 7007:7007  -it  -d --shm-size=12gb dromni/nerfstudio:main
    docker ps 
    docker exec -it <docker process id> /bin/bash
    ```
- To download this repository to be used with the NerfStudio framework:
   ```
   cd <the same directory as nerfstudio's pyproject.toml>
   git clone git@github.com:NicoleStrel/transpnerf.git
   cd transpnerf/
   pip install -e .
   ```
### 2. Data 

Two datasets were used: a synthetic dataset made with Blender and a real dataset made with iPhone image frame captures.

- **data/synthetic-blender-dataset**
   - Created by the script in `scripts/blender_dataset.py`, which can also be found in the '.blend' files. 
   - Train: 40 images, Test: 40 images
   - transforms_*.json contains the camera angle, poses, and image files
   - _depth and _normal suffixes define the depth (grayscale) and normals respectively.
   - To remove the blender generated id's from the script, run inside the generated folder:
        - `sudo find . -type f -name '*_0000*' -exec sh -c 'mv "$1" "$(echo "$1" | sed "s/_0000//")"' sh {} \;`
   - Note: The hotdog data was taken from the original [NeRF blender dataset](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)
   
- **data/real-capture-dataset**
  - 40 images used for both test/train
  - Used the nerfstudio command: `ns-process-data images --data {DATA_PATH} --output-dir {PROCESSED_DATA_DIR}` to run [COLMAP](https://github.com/colmap/colmap) to generate poses.
  - Generated depths using [Depth-Anything](https://github.com/LiheYoung/Depth-Anything) using the command:
      - `python run.py --encoder vitl --img-path <img folder> --outdir <depth folder> --grayscale --pred-only`

As well, Nerfstudio has the option to create and use any dataset with instructions [here](https://docs.nerf.studio/quickstart/custom_dataset.html). 

### 3. Running TranspNeRF

To run TranspNeRF on the synthetic dataset: 

`ns-train transpnerf --pipeline.model.background-color white --pipeline.model.disable-scene-contraction True --pipeline.model.proposal-initial-sampler uniform --pipeline.model.near-plane 2. --pipeline.model.far-plane 6. --pipeline.model.use-average-appearance-embedding False --pipeline.model.distortion-loss-mult 0 --data {dataset folder path}/transforms.json`

Note: the near and far planes for the wine glass work best if set to 6 and 9 respectively. 

To run TranspNeRF on the real dataset: 

`ns-train transpnerf --data {dataset folder path}`

### 4. Running the evaluation script

The evaluation procedure runs the training, evaluation metric script (`ns-eval`), creates the output depths from the test image dataset, and output point clouds. 

- To run: `./transpnerf/scripts/train_and_eval_master.sh {dataset type}` - dataset type is either `synthetic` or `real`
- To run as a background task: `nohup ./transpnerf/scripts/train_and_eval_master.sh synthetic &`
    -  check status: `ps aux | grep ./transpnerf/scripts/train_and_eval_master.sh`
    -  view logging: `cat nohup.out`
The python script `get_eval_results.py` called in this master shell script will create an excel file with the metrics: psnr, ssim, lpips, number of rays per second, and the average depth error in meters. 
