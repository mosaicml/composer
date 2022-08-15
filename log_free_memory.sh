#!/bin/bash -e

echo "      date     time $(free -g | grep total | sed -E 's/^    (.*)/\1/g')"
while true; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') $(free -g | grep Mem: | sed 's/Mem://g')"
    sleep 1
done
