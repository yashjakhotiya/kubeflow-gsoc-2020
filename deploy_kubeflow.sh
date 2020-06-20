export PATH=$PATH:/snap/bin:/home/yash_jakhotiya/gsoc
# gcloud auth login
# gcloud auth application-default login

export PROJECT=gsoc-kf-example
export ZONE=asia-east1-b
gcloud config set project ${PROJECT}
gcloud config set compute/zone ${ZONE}

export CONFIG_URI="https://raw.githubusercontent.com/kubeflow/manifests/v1.0-branch/kfdef/kfctl_gcp_iap.v1.0.2.yaml"
export CLIENT_ID="redacted"
export CLIENT_SECRET="redacted"
export KF_NAME=kf-keras-nlp
export BASE_DIR=/home/yash_jakhotiya/gsoc/
export KF_DIR=${BASE_DIR}/${KF_NAME}

mkdir -p ${KF_DIR}
cd ${KF_DIR}
kfctl apply -V -f ${CONFIG_URI}