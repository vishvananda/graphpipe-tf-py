#!/bin/bash
#
# Copyright (c) 2018, Oracle and/or its affiliates. All rights reserved.
#
# Licensed under the Universal Permissive License v 1.0 as shown at
# http://oss.oracle.com/licenses/upl.

set -ex

brew install curl openssl

TF_DIR=/usr/local
TF_TYPE=cpu

if [ ! -e "/usr/local/lib/libtensorflow.dylib" ]; then
    curl -L "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-${TF_TYPE}-darwin-x86_64-2.4.1.tar.gz" | sudo tar -C ${TF_DIR} -xz
fi

pip3 install -r requirements.txt

pushd remote_op

FB_VERSION=1.12.0
if [ ! -d "flatbuffers" ]; then
    curl -L https://github.com/google/flatbuffers/archive/v${FB_VERSION}.tar.gz | tar -xz
    mv flatbuffers-${FB_VERSION} flatbuffers
fi


export EXTRA_FLAGS="-L/usr/local/opt/openssl/lib -I/usr/local/opt/openssl/include"
make clean; make

popd

pip3 install -r test-requirements.txt

python3 setup.py bdist_wheel
