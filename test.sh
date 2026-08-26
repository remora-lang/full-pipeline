#!/usr/bin/env bash
#
# Tests invoked by CI. Builds each example into a library, links the example
# scaffolding against it, and runs the result.

set -e

for prog in examples/*.remora; do
    echo "# $prog"
    ./remora2exe "$prog"
    "./build/$(basename "${prog%.remora}")"
done
