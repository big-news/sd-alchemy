#!/bin/bash

####################### Parameters #######################
# Input
is_v2_model="false"
src_model="./models/model.ckpt"
src_lora="./lora/lora.safetensors" # can be a list of lora separated by blank space

# Output
dst_model="./output/model_lora.safetensors"

# Training parameters
ratios="0.8" # a list correpsonding to $src_lora
precision="bf16"
save_precision="bf16"


################# Logic (DON'T TOUCH ME) #################

extra_args=""

if [[ ${is_v2_model} = true ]]; then
    extra_args="${extra_args} --v2"
fi

if [[ ${src_model} != "" ]]; then
    extra_args="${extra_args} --sd_model ${src_model}"
fi

if [[ ${precision} != "" ]]; then
    extra_args="${extra_args} --precision ${precision}"
fi
if [[ ${save_precision} != "" ]]; then
    extra_args="${extra_args} --save_precision ${save_precision}"
fi

source venv/bin/activate
python sd-scripts/networks/merge_lora.py \
    --models "${src_lora}" \
    --ratios "${ratios}" \
    --save_to "${dst_model}" \
    ${extra_args}