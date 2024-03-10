#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/llava-v1.5-13b \
    --question-file ./data/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --image-folder ./data/data/eval/textvqa/train_images \
    --answers-file ./playground/eval/textvqa/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ./data/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/eval/textvqa/answers/llava-v1.5-13b.jsonl
