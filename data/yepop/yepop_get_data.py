"""
Prepare data
"""

import os
import logging
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from PIL import Image
from tqdm import tqdm
# from streaming import MDSWriter, LocalDataset
import polars as pl
import spacy
from img2dataset import download

pl.Config.set_fmt_str_lengths(200)

#####################################################33

PATH = '/mnt/sd1tb/tinydiffusion/dataset_v0/'
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def clean_sizes(df, ) -> pl.DataFrame:
    return (
        df.filter(
            (pl.col('height')>=256) &
            (pl.col('width')>=256) &
            (pl.col('height')<=3000) &
            (pl.col('width')<=3000)
        )
        .filter(
            ~((pl.col('height') >= 2.4*pl.col('width')) | (pl.col('width') >= 2.4*pl.col('height')))
        )
        # .filter(pl.col('NSFW').is_in({'UNLIKELY', 'UNSURE', 'False'}))
        # .rename({'SAMPLE_ID': 'id', 'URL': 'url', 'TEXT': 'caption'})
        .with_columns([
            pl.col('filename').cast(pl.Utf8).alias('id')
        ])
        .unique(subset=['url'])
    )

def clean_columns(df) -> pl.DataFrame:
    return df.select(
        'id', 'url', 'llava_caption', 'width', 'height', 'original_width', 'original_height'	
    ).rename({"llava_caption": "caption"})


def download_imgs(df):
    path_out = f"{PATH}imgs"
    os.makedirs(f"{PATH}imgs", exist_ok=True)
    download(
        processes_count=12,
        thread_count=32,
        url_list=f"{PATH}dataset_raw.parquet",
        image_size=256,
        min_image_size=256,
        resize_only_if_bigger=True,
        resize_mode="no",
        output_folder=path_out,
        output_format="files",
        input_format="parquet",
        url_col="url",
        caption_col="caption",
        save_additional_columns=['id'],
        enable_wandb=False,
        number_sample_per_shard=5_000,
        encode_format='jpg',
        encode_quality=95,
        timeout=20
    )


def build_consolidated_data():
    path_external_imgs = PATH + 'imgs/'
    dataframes = []
    for filename in os.listdir(path_external_imgs):
        if filename.endswith(".parquet"):
            parquet_path = os.path.join(path_external_imgs, filename)
            path_shard = parquet_path.split('.')[:-1][0]
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


def load_partitions_into_df(base_path) -> pl.DataFrame:
    df_all = pl.DataFrame()
    for i in tqdm(range(1, 10+1)):
        list_dicts = pl.read_json(
            base_path + f'ye-pop/ye-pop-{i}.json'
        ).transpose()['column_0'].to_list()
        df_yepop = pl.from_dicts(list_dicts)
        df_all = pl.concat([df_all, df_yepop])
    return df_all

def main():
    logging.info('Starting...')

    df = load_partitions_into_df(PATH)
    logging.info(f'Step 1 - Loading data - Num records: {len(df)}')

    df = clean_sizes(df)
    logging.info(f'Step 2 - Clean sizes- Num records: {len(df)}')

    df = clean_columns(df) 
    logging.info(f'Step 3 - Clean columns - Num records: {len(df)}')

    df.write_parquet(f"{PATH}/dataset_raw.parquet")

    logging.info('Step 4 - Downloading imgs...')
    download_imgs(df)

    df = build_consolidated_data()
    logging.info(f'Step 5 - consolidating data - Num records: {len(df)}')
    df.write_parquet(f"{PATH}/dataset_gold.parquet")


if __name__ == "__main__":
    main()

