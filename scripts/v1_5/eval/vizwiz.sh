#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/llava-v1.5-13b_debug \
    --question-file ./data/data/eval/viswiz/test.json \
    --image-folder ./data/data/eval/viswiz/test \
    --answers-file ./playground/eval/vizwiz/answers/llava-v1.5-13b-Moe.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./data/data/eval/viswiz/test.json \
    --result-file ./playground/eval/vizwiz/answers/llava-v1.5-13b-Moe.jsonl \
    --result-upload-file ./playground/eval/vizwiz/answers_upload/llava-v1.5-13b-Moe.json
