#!/bin/bash
cd $(cd $(dirname $0)&& pwd)/..

WORK_DIR=/workspace/handwrittentextgen
args=(
  --gpus all
  --shm-size=8g
  -it
  --rm
  -u=$(id -u $USER):$(id -g $USER)
  -v ~/.gitconfig:/home/developer/.gitconfig
  -v ~/.ssh:/home/developer/.ssh:ro
  -v $(pwd):$WORK_DIR
)
docker run "${args[@]}" handwrittentextgen
