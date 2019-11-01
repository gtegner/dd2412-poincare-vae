zip -r gcp.zip . -x *.git* checkpoints saved_csv saved_models data

gcloud compute scp gcp.zip 