#!/bin/bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
pushd $DIR
python -m grpc_tools.protoc -I. --protobuf-to-pydantic_out=. --python_out=. --pyi_out=. comm.proto

