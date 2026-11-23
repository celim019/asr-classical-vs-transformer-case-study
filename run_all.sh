#!/bin/bash
# Runs every system sequentially (never in parallel) so latency numbers are not disturbed.
export SANDBOX_SHERPA_ONNX_WHISPER_FILES=0   # measure the original published model files
export OMP_NUM_THREADS=1
cd "$(dirname "$0")"
for s in pocketsphinx whisper-tiny.en-int8 whisper-tiny.en-fp32 whisper-base.en-int8 whisper-base.en-fp32; do
  python3 run_experiment.py --system $s
done
