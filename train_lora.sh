#!/bin/bash

####################### Parameters #######################

# Input
is_v2_model="false"
train_dataset="./dataset/him"
reg_dataset=""
pretrained_model="./models/model.ckpt"
pretrained_lora=""

# Output
output_name="him"
output_extension="safetensors"
output_dir="output"
save_every_n_epochs="2"

# Network structure
network_dim="128"
network_alpha="32"
clip_skip="2"
max_token_length="225"

# Training
resolution="512,512"
max_train_epochs="20"
batch_size="1"
lr="1e-4"
unet_lr="1e-4"
text_encoder_lr="1e-4"
optimizer_type="Lion"
optimizer_args=""
lr_scheduler="cosine_with_restarts"
lr_warmup_steps="0"
lr_restart_cycles="1"
lr_scheduler_power="0"
train_unet_only="false"
train_text_encoder_only="false"
prior_loss_weight="1"

# Efficiency
num_cpu_threads_per_process="8"
max_data_loader_n_workers="8"
mixed_precision="bf16"

# Conv2d
enable_conv2d="false"
conv_dim="4"
conv_alpha="2"

# Logging
enable_logging="true"
logging_dir="logs"
log_prefix="${output_name}"

# Bucket
enable_bucket="true"
min_bucket_reso="256"
max_bucket_reso="1024"

# Misc
seed="1337"
noise_offset=""
enable_shuffle_caption="false"
enable_cache_latents="true"

################# Logic (DON'T TOUCH ME) #################
source venv/bin/activate

supported_optimizer_types=("AdamW" "AdamW8bit" "Lion" "SGDNesterov" "SGDNesterov8bit" "DAdaptation" "AdaFactor")

supported_lr_schedulers=("linear" "cosine" "cosine_with_restarts" "polynomial" "constant" "constant_with_warmup")

extra_args=""

# based on stable-diffusion v1 or v2
if [[ "${is_v2_model}" = "true" ]]; then
    extra_args="${extra_args} --v2"
else
    extra_args="${extra_args} --clip_skip=${clip_skip}"
fi

if [[ "${reg_dataset}" != "" ]]; then
    extra_args="${extra_args} --reg_data_dir=${reg_dataset}"
fi

if [[ "${pretrained_lora}" != "" ]]; then
    extra_args="${extra_args} --network_weights=${pretrained_lora}"
fi

# train unet only / train text encoder only
if [[ "${train_unet_only}" = "true" && "${train_text_encoder_only}" = "true" ]]; then
    echo "train_unet_only and train_text_encoder_only cannot be true at the same time."
    exit 1
fi
if [[ "${train_unet_only}" = "true" ]]; then
    extra_args="${extra_args} --network_train_unet_only"
fi
if [[ "${train_text_encoder_only}" = "true" ]]; then
    extra_args="${extra_args} --network_train_text_encoder_only"
fi

# optimizer
if [[ ! "${supported_optimizer_types}" =~ "${optimizer_type}" ]]; then
    echo "Unsupported optimizer type: ${optimizer_type}"
    exit 1
fi
if [[ "${optimizer_args}" != "" ]]; then
    extra_args="${extra_args} --optimizer_args ${optimizer_args}"
fi

# lr scheduler
if [[ ! "${supported_lr_schedulers}" =~ "${lr_scheduler}" ]]; then
    echo "Unsupported lr_scheduler: ${lr_scheduler}"
    exit 1
fi
if [[ "${lr_scheduler}" = "constant_with_warmup" ]]; then
    extra_args="${extra_args} --lr_warmup_steps=${lr_warmup_steps}"
elif [[ "${lr_scheduler}" = "cosine_with_restarts" ]]; then
    extra_args="${extra_args} --lr_scheduler_num_cycles=${lr_restart_cycles}"
elif [[ "${lr_scheduler}" = "polynomial" ]]; then
    extra_args="${extra_args} --lr_scheduler_power=${lr_scheduler_power}"
fi

# conv2d
if [[ "${enable_conv2d}" = "true" ]]; then
    extra_args="${extra_args} --network_args conv_dim=${conv_dim} conv_alpha=${conv_alpha}"
fi

# logging
if [[ "${enable_logging}" = "true" ]]; then
    extra_args="${extra_args} --logging_dir=${logging_dir} --log_prefix=${log_prefix}"
fi

# bucket
if [[ "${enable_bucket}" = "true" ]]; then
    extra_args="${extra_args} --enable_bucket --min_bucket_reso=${min_bucket_reso} --max_bucket_reso=${max_bucket_reso}"
fi

# noise offset
if [[ "noise_offset" != "" ]]; then
    extra_args="${extra_args} --noise_offset=${noise_offset}"
fi

if [[ "${enable_shuffle_caption}" = "true" ]]; then
    extra_args="${extra_args} --shuffle_caption"
fi

# cache_latents
if [[ "${enable_cache_latents}" = "true" ]]; then
    extra_args="${extra_args} --cache_latents"
fi


accelerate launch \
    --num_cpu_threads_per_process "${num_cpu_threads_per_process}" \
    ./sd-scripts/train_network.py \
    --pretrained_model_name_or_path="${pretrained_model}" \
    --train_data_dir="${train_dataset}" \
    --resolution="${resolution}" \
    --output_dir="${output_dir}" \
    --output_name="${output_name}" \
    --save_model_as="${output_extension}" \
    --save_every_n_epochs="${save_every_n_epochs}" \ 
    --max_train_epochs="${max_train_epochs}" \
    --batch_size="${batch_size}" \
    --network_dim="${network_dim}" \
    --network_alpha="${network_alpha}" \
    --learning_rate="${lr}" \
    --unet_lr="${unet_lr}" \
    --text_encoder_lr="${text_encoder_lr}" \
    --optimizer_type="${optimizer_type}" \
    --lr_scheduler="${lr_scheduler}" \
    --max_token_length="${max_token_length}" \
    --mixed_precision="${mixed_precision}" \
    --seed="${seed}" \
    --prior_loss_weight="${prior_loss_weight}" \
    --xformers \
    --gradient_checkpointing \
    --max_data_loader_n_workers="${max_data_loader_n_workers}" \
    --network_module=networks.lora \
    ${extra_args}