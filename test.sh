#!/usr/bin/env bash
#
# Tests invoked by CI. Builds each example into a library, links the example
# scaffolding against it, and runs the result.

set -e

# Download data files if they are not already here

if ! [ -f yolov4.weights ]; then
    curl https://sigkill.dk/junk/yolov4.weights -O
fi
if ! [ -f input.bin ]; then
    curl https://sigkill.dk/junk/input.bin -O
fi


for prog in examples/*.remora; do
    echo "# $prog"
    ./remora2exe "$prog"
    "./build/$(basename "${prog%.remora}")"
done
