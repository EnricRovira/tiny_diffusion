PATH_DATA='/mnt/sd1tb/tinydiffusion/'
NAME_DATASET='dataset_v0'
NUM_LAION_PARTITIONS=1

mkdir -p $PATH_DATA$NAME_DATASET

echo "Downloading [EXTERNAL] data [LAION], num_partitions: $NUM_LAION_PARTITIONS"
for i in $(seq 0 $(($NUM_LAION_PARTITIONS - 1)))
do
    wget --no-check-certificate -nc -O "$PATH_DATA$NAME_DATASET/laion_$i.parquet" "https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/dataset/part-$(printf '%05d' $i)-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet"
done