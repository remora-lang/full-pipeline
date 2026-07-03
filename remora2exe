#!/bin/sh
#
# Proof of concept of the full pipeline. The existence of this script does not
# imply that shell script is a good language for writing compiler drivers.

set -ex

if [ $# -ne 1 ]; then
    exit "Usage: $0 FILE"
fi

REMORA_FILE=$1
FUTHARK_FILE=${REMORA_FILE/.remora/.fut_soacs}
MLIR_FILE=${REMORA_FILE/.remora/.mlir}
LLVM_FILE=${REMORA_FILE/.remora/.ll}
EXE_FILE=${REMORA_FILE/.remora/}

remora futhark < "$REMORA_FILE" > "$FUTHARK_FILE"
mlir-backend "$FUTHARK_FILE" > "$MLIR_FILE"
