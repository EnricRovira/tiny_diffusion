"""
Prepare data
"""

import os
import logging
import json
from tqdm import tqdm
import polars as pl

pl.Config.set_fmt_str_lengths(200)

#####################################################33

PATH = '/mnt/sd1tb/tinydiffusion/dataset_v1/'
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def build_real_images_consolidated_data(path):
    path_external_imgs = path + 'imgs/'
    dataframes = []
    for filename in os.listdir(path_external_imgs):
        if filename.endswith(".parquet"):
            parquet_path = os.path.join(path_external_imgs, filename)
            path_shard = parquet_path.split('.')[:-1][0].replace('dataset_v0', 'dataset_v1')
            df = pl.read_parquet(parquet_path)
            df = df.with_columns(
                pl.col('key').map_elements(
                    lambda x: os.path.join(path_shard, f'{x}.jpg'),
                    return_dtype=pl.Utf8
                ).alias("path")
            )
            dataframes.append(df)
    df_final = pl.concat(dataframes).filter(pl.col('status')=='success')
    return df_final

def build_synthetic_images_consolidated_data(path):
    path_data = path + 'diffusiondb/'
    parts = os.listdir(path_data)
    list_data = []
    for part in tqdm(parts):
        data = json.load(open(f'{path_data}{part}/{part}.json'))
        for record in data:
            list_data.append({
                'id': record,
                'path': f'{path_data}{part}/{record}',
                'caption': data[record]['p']
            })
    df = pl.from_dicts(list_data).unique('caption')
    return df

def main():
    logging.info('Starting...')

    df_real_images = build_real_images_consolidated_data(PATH)
    df_synthetic_images = build_synthetic_images_consolidated_data(PATH)
    logging.info(
        f'Step 1 - Loaded data - Real data: {len(df_real_images)} - '
        f'Synth data: {len(df_synthetic_images)}' 
    )

    df_real_images = df_real_images.with_columns(pl.lit('real').alias('source'))
    df_synthetic_images = df_synthetic_images.with_columns(pl.lit('synthetic').alias('source'))
    df = pl.concat([
        df_real_images.select(['id', 'path', 'caption', 'source']),
        df_synthetic_images.select(['id', 'path', 'caption', 'source'])
    ])
    logging.info(
        f'Step 1 - Concatenated data - {len(df)}'
    )
    df.write_parquet(f"{PATH}/dataset_gold.parquet")


if __name__ == "__main__":
    main()

