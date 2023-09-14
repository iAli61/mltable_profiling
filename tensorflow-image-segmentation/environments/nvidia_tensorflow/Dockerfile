# check release notes https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/index.html
FROM nvcr.io/nvidia/tensorflow:23.08-tf2-py3

# Install dependencies missing in this container
# NOTE: container already has matplotlib==3.5.1 tqdm==4.62.0
RUN python -m pip install   azureml-defaults \
                            mlflow \
                            azureml-mlflow \
                            psutil \
                            pynvml 
