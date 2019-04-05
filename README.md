# imagenet-benchmark

### This repository holds code for benchmarking Imagenet models hosted on modelhub.

#### Prerequisites

The following prerequisites are needed:

- python 2.7 or 3.6 installed
- docker installed
- Imagenet validation images and ground truth. These can be downloaded from the Imagenet [website](http://image-net.org/download). Unzip both the "ILSVRC2012_img_val" and " ILSVRC2012_devkit_t12" folders and place them under /data.

#### Workflow

1. Run the model(s) locally. We will be interacting with the model through REST API.

- Run `python start.py *model_name*`. Imagenet models currently hosted on modelhub include: squeezenet, googlenet, inception-v3, vgg-19, xception, alexnet, densenet, resnet-50, and mobilenet. Visit the modelhub [app](http://app.modelhub.ai/) for a full list of models. The model should now be running on your host machine on port 80. Try http://localhost/api/get_config in your browser to confirm.
- You can run multiple models simultaneously. However, make sure to pass a different port to each one. For running squeezenet and alexnet for example, run `python start.py squeezenet -ap 80` and `python start.py alexnet -ap 81` in two different terminals.

2. Run the benchmarking analysis docker.

- Build the docker
  `docker build -f dockerfile-imagenet-benchmark -t dockerfile-imagenet-benchmark .`
- Run the docker
  `docker run -it --net=host -v $PWD/data:/data -v $PWD/files:/files -v $PWD/output:/output dockerfile-imagenet-benchmark /bin/bash`
- Start the jupyter notebook
  `jupyter notebook --allow-root --ip=0.0.0.0`
- Run `/files/benchmark.ipynb` to validate the model on the Imagenet data and `/files/plot.ipynb` to plot the results.
