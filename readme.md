# Detectron2 Example

## DKube Repos 
Add dataset **baloon**
  - Source: pub_url
  - URL: https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip

Add project **detectron2**
  - Source: Git
  - URL: https://github.com/riteshkarvaloc/detectron2.git 

Add Model **detectron2**
  - Source: None

## Training Job:
  - Project: **detectron2**
  - Dataset: **baloon** with mount point **/opt/dkube/input**
  - Model **detectron2** with mount point **opt/dkube/output**
  - Add GPU from configutation and submit

## Jupyterlab notebook example

From IDE section launch Jupyter lab with custom framework and image **ocdr/detectron2:gpu**, with project **detectron2** and dataset **baloon** with mount point **/opt/dkube/input/**. Add GPU from the configuration. 
  - Open Jupyterlab
  - Go to **workspace/detectron2**
  - Run **detectron2_example.ipynb**

## Nvidia Triton Serving
  - Copy model files to directory **model** in home dir
  -   > sudo docker pull nvcr.io/nvidia/tritonserver:20.06-py3
  -   > sudo docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v /home/<user>/model:/models nvcr.io/nvidia/tritonserver:20.06-py3 tritonserver --model-repository=/models
  -   Use --gpus=1 for gpu in the above command, GPU driver must be present. 
  -   Run **curl -v localhost:8000/v2/health/ready** to check status
  