#!/usr/bin/env sh

# Abort on any non-zero exitcode
set -e
DOCKER_REGISTRY="eu.gcr.io"
PROJECT_NAME="gustaf-dd2412"
BIN_NAME=$1

finish () {
    if [ $DONE = "true" ]
    then
        echo "Build done successfully, exiting"
    else
        echo "Build has exited with errors, aborting"
    fi
}
trap finish EXIT

export DONE="false"

echo "Building docker image"
docker build -f Dockerfile --build-arg git_hash=$(git describe --match=NeVeRmAtCh --always --abbrev=40 --dirty) --build-arg experiment=$1 -t $DOCKER_REGISTRY/$PROJECT_NAME/$BIN_NAME .

echo "Pushing docker image"
docker push $DOCKER_REGISTRY/$PROJECT_NAME/$BIN_NAME

DONE="true"
