sh build.sh $1
#gcloud beta compute instances create-with-container $1 --zone europe-north1-a --container-image=eu.gcr.io/gustaf-dd2412/$1 --container-mount-host-path host-path=model_outputs,mount-path=model_outputs
# 