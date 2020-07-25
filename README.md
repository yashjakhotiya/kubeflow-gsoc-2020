# Samples for Notebook to Kubeflow Deployment using TensorFlow 2.0 Keras
This project aims at building Kubeflow use case samples with Tensorflow 2.0 Keras training code demonstrating 'Customer User Journey' (CUJ) in the process.
The `kubeflow-config` directory holds `.yaml` config files required for and generated in the process of Kubeflow Deployment.

The samples hosted are as follows - 
### Text Classification 
Based on a [Tensorflow tutorial](https://www.tensorflow.org/tutorials/keras/text_classification_with_hub), this machine learning task aims at classifying IMDB movie 
reviews. Building on this, this sample hosts the following,
1. `text_classification_with_rnn.py` - The core training code upon which all subsequent examples showing Kubeflow functionalities are based.

2. `distributed_text_classification_with_rnn.py` - Same as above, but uses Tensorflow's distributed (training) strategy.

3. `Dockerfile` - Dockerfile used to build Docker image of the training code as required by some Kubeflow functionalities.

4. `fairing-with-python-sdk.ipynb` - Jupyter Notebook which deploys a model training task on cloud using [Kubeflow Fairing](https://www.kubeflow.org/docs/fairing/fairing-overview/). 
Fairing does not require you to build a Docker image of the training code first. Hence, its training code resides in the same notebook.

5. `katib-with-python-sdk.ipynb` - Jupyter Notebook which launches [Katib](https://www.kubeflow.org/docs/components/hyperparameter-tuning/hyperparameter/) 
hyperparameter tuning experiments using its [Python SDK](https://github.com/kubeflow/katib/tree/master/sdk/python). Katib requires you to build and host a
Docker image of your training code in a container registry. This sample has used `gcloud builds` to build the required Docker image of the training code along with
the training data and hosts it on [gcr.io](gcr.io).

6. `tfjob-with-python-sdk.ipynb` - Jupyter Notebook which launches a [TFJob](https://www.kubeflow.org/docs/components/training/tftraining/). TFJobs are used 
for distributed training over a Kuberenetes backend. This notebook uses the Docker image built from the distributed version of our core training code.