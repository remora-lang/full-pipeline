#!/usr/bin/env bash
#
# Tests invoked by CI.

set -e

for prog in examples/*.remora; do
    echo "# $prog"
    ./remora2exe "$prog"
    "${prog/.remora/}"
done
