#!/usr/bin/env bash

set -e

pushd ItemSwap

python convert_face_image.py $1 $2

popd
