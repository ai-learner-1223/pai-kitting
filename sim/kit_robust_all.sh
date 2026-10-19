#!/bin/bash
cd "$(dirname "$0")"
for c in base pos10 pos15 yaw30 yaw45 shift_near shift_far err2 err4 err6 err8; do
  python kit_robust.py $c ${N:-10} >> kit_data/robust.log 2>&1
done
echo done >> kit_data/robust.log
