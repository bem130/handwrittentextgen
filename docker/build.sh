#!/bin/bash
cd $(cd $(dirname $0); pwd)/..

args=(
  --build-arg UID=$(id -u $USER)
  --build-arg GID=$(id -g $USER)
)
docker build -t handwrittentextgen -f docker/Dockerfile "${args[@]}" .
