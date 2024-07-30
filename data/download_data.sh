#!/bin/bash
PATH_DATA='/mnt/sd1tb/tinydiffusion/'
NAME_DATASET='dataset_v1'
NUM_LAION_PARTITIONS=1
NUM_YE_POP_PARTITIONS=11
DIFFUSION_DB_PARTITIONS=10

mkdir -p $PATH_DATA$NAME_DATASET
mkdir -p "$PATH_DATA$NAME_DATASET/ye-pop"
mkdir -p "$PATH_DATA$NAME_DATASET/diffusiondb"

# LAION
# echo "Downloading [EXTERNAL] data [LAION], num_partitions: $NUM_LAION_PARTITIONS"
# for i in $(seq 0 $(($NUM_LAION_PARTITIONS - 1)))
# do
#     wget --no-check-certificate -nc -O "$PATH_DATA$NAME_DATASET/laion_$i.parquet" "https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/dataset/part-$(printf '%05d' $i)-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet"
# done

# Ye-POP
# for i in $(seq 1 $(($NUM_YE_POP_PARTITIONS)))
# do
#     curl -L -X GET "https://huggingface.co/datasets/Ejafa/ye-pop/resolve/main/images/chunk_${i}/chunk_${i}.0.json?download=true" -o "${PATH_DATA}${NAME_DATASET}/ye-pop/ye-pop-${i}.json"
# done

# Diffusion-DB
start_time=$(date +%s)
out_dir="${PATH_DATA}${NAME_DATASET}/diffusiondb/"
base_url="https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/images/part-"
for i in $(seq -f "%06g" 1 $(($DIFFUSION_DB_PARTITIONS)))
do
    url="${base_url}${i}.zip"
    output_file="${out_dir}part-${i}.zip"
    curl -L -X GET "${url}" -o "${output_file}"
    unzip -o "${output_file}" -d "${out_dir}part-${i}" > /dev/null 2>&1
    rm "${output_file}"
done

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "EElapsed time: $elapsed_time"