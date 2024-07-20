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
            (pl.col('HEIGHT')>=256) &
            (pl.col('WIDTH')>=256)
        )
        .filter(
            ~((pl.col('HEIGHT') >= 2.4*pl.col('WIDTH')) | (pl.col('WIDTH') >= 2.4*pl.col('HEIGHT')))
        )
        .filter(pl.col('NSFW').is_in({'UNLIKELY', 'UNSURE', 'False'}))
        .rename({'SAMPLE_ID': 'id', 'URL': 'url', 'TEXT': 'caption'})
        .with_columns([
            pl.col('id').cast(pl.Utf8).alias('id')
        ])
        .unique(subset=['url'])
    )


def clean_text(df: pl.DataFrame):
    def remove_dot_at_end(text) -> str:
        if text.endswith('.'):
            return text[:-1]
        return text
    
    def starts_with_alnum(text) -> bool:
        return bool(re.match(r'^[A-Za-z0-9]', text))

    df = df.with_columns(
        pl.col('caption')
        .map_elements(remove_dot_at_end, return_dtype=pl.Utf8),
    )
    df = df.filter(
        (pl.col('caption').str.split(' ').list.len() >= 10) & 
        (pl.col('caption').map_elements(starts_with_alnum, return_dtype=pl.Boolean))
    ).unique(subset=['caption'])
    return df


def clean_nsfw(df: pl.DataFrame):
    df = (
        df.filter(
            ~pl.col('caption').str.contains('fuck|shit|ass|bitch|cunt|dick|hell|pussy|tits|whore|motherfucker|nigger|bastard|slut') &
            ~pl.col('caption').str.contains(
                'sex|orgasm|masturbation|dick|vagina|naked|erotic|nude|penis|pornography|BDSM|fetish|blowjob|handjob|anal|threesome|orgy|gangbang|nudist|exhibitionist'
            ) &
            ~pl.col('caption').str.contains('weapon|gun|murder|violence|blood|gore|mutilation|torture|rape|decapitation|massacre|genocid|shooting|assault') &
            ~pl.col('caption').str.contains('cocaine|heroin|marijuana|LSD|ecstasy|methamphetamine|crack|opium|ketamine|amphetamine|hallucinogen') &
            ~pl.col('caption').str.contains('racist|sexist|homophobic|xenophobic|nazi|supremacist|intolerant|discrimination|hate speech') &
            ~pl.col('caption').str.contains(
                'suicide|self-harm|bulimia|anorexia|child abuse|domestic violence|human trafficking|incest|bestiality|necrophilia|sadomasochism|sexual assaul'
            )
        )
    ).unique(subset=['caption'])
    return df

def filter_top_k_percent(
    df: pl.DataFrame,
    column: str,
    k: float=0.5
) -> pl.DataFrame:
    k = 1 - k
    limit_value = df[column].quantile(k)
    df = df.filter(df[column] >= limit_value)
    return df



def pos_sanity_text_doc(doc):
    try:
        # Verificar si el primer token es un verbo
        if doc[0].pos_ == 'VERB':
            return False

        # Tiene nombre
        list_pos = [token.pos_ for token in doc]
        tiene_nombre = any(t in {'NOUN'} for t in list_pos)
        if not tiene_nombre:
            return False

        # Verificar la regla del adjetivo con ventana
        cumple_regla_adj_nombre = True
        # Verificar repetición de pronombres, adverbios, determinantes, etc.
        cumple_regla_no_repetidos = True
        tipos_prohibidos = {'PRON', 'ADV', 'DET'}
        tipo_anterior = None
        for i, token in enumerate(doc):
            # Verificar la regla de adjetivo con ventana
            if token.pos_ == 'ADJ':
                ventana = doc[max(i-2, 0):min(i+3, len(doc))]
                if not any(t.pos_ == 'NOUN' for t in ventana):
                    cumple_regla_adj_nombre = False

            # Verificar la repetición de tipos de palabras
            if token.pos_ in tipos_prohibidos and token.pos_ == tipo_anterior:
                cumple_regla_no_repetidos = False
                break
            tipo_anterior = token.pos_

        if not cumple_regla_adj_nombre or not cumple_regla_no_repetidos:
            return False

        return True
    except Exception as e:
        return False



def pos_process_batch(batch, pos_model):
    model = spacy.load(pos_model, disable=["lemmatizer", "senter", "ner"])
    docs = model.pipe(batch)
    return [(doc.text, pos_sanity_text_doc(doc)) for doc in docs]


def pos_filter(
    df: pl.DataFrame,
    pos_bs=10,
    pos_max_workers=2,
    pos_model='en_core_web_sm'
):
    texts = df['caption'].to_list()
    batches = [texts[i:i + pos_bs] for i in range(0, len(texts), pos_bs)]

    process_with_args = partial(pos_process_batch, pos_model=pos_model)
    with ProcessPoolExecutor(pos_max_workers) as executor:
        results = executor.map(process_with_args, batches)

    processed_data = [item[-1] for sublist in results for item in sublist]
    df = (
        df.with_columns(
            pl.Series('flag_pos', processed_data)
        ).filter(
            pl.col('flag_pos') == True
        ).drop('flag_pos')
    )
    return df

def download_imgs(df):
    path_out = f"{PATH}imgs"
    os.makedirs(f"{PATH}imgs", exist_ok=True)
    download(
        processes_count=20,
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
        number_sample_per_shard=2_000,
        encode_format='jpg',
        encode_quality=95,
        timeout=7
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



def main():
    df = pl.read_parquet(
        PATH + "laion_0.parquet"
    ).head(5_000_000)
    logging.info(f'Step 1 - Loading data - Num records: {len(df)}')

    df = clean_sizes(df)
    logging.info(f'Step 2 - Clean sizes- Num records: {len(df)}')

    df = clean_text(df)
    logging.info(f'Step 3 - Clean text - Num records: {len(df)}')

    df = clean_nsfw(df)
    logging.info(f'Step 4 - Clean NSFW - Num records: {len(df)}')

    df = filter_top_k_percent(df, 'similarity', 0.4)
    logging.info(f'Step 5 - Get top K percent - Num records: {len(df)}')

    df = pos_filter(df, pos_bs=20_000, pos_max_workers=16)
    logging.info(f'Step 6 - POS filter - Num records: {len(df)}')

    df.write_parquet(f"{PATH}/dataset_raw.parquet")

    logging.info('Step 7- Downloading imgs...')
    download_imgs(df)

    df = build_consolidated_data()
    logging.info(f'Step 8 - consolidating data - Num records: {len(df)}')
    df.write_parquet(f"{PATH}/dataset_gold.parquet")


if __name__ == "__main__":
    main()

