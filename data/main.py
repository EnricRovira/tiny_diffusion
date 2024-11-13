
import click
import logging
import huggingface_hub
import polars as pl
import img2dataset
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def download_data_from_hf(
    local_dir: str,
    num_files_to_download: int
) -> pl.DataFrame:
    # Download raw data
    df_final = None
    for i in range(num_files_to_download):
        path_file = f"{local_dir}/metadata/pd3m.{i:02d}.parquet"
        huggingface_hub.hf_hub_download(
            repo_id="Spawning/PD3M",
            repo_type="dataset",
            filename=f"metadata/pd3m.{i:02d}.parquet",
            local_dir=local_dir,
        )
        if i==0:
            df_final = pl.read_parquet(path_file)
        else:
            df_final = pl.concat([df_final, pl.read_parquet(path_file)])
    return df_final



@click.command()
@click.option("--local_dir", "-i", type=str, help="Output directory")
@click.option("--num_files_to_download", "-i", type=int, default=1, help="Number of files to download")
@click.option("--num_workers", "-i", type=int, default=16, help="Number of workers")
@click.option("--num_threads", "-i", type=int, default=16, help="Number of threads")
@click.option("--url_col", "-i", type=str, default="url", help="URL column")
@click.option("--caption_col", "-i", type=str, default="caption", help="Caption column")
def run(
    local_dir: str,
    num_files_to_download: int,
    num_workers: int,
    num_threads: int,
    url_col: str,
    caption_col: str
):
    # Download from source
    logging.info(f"1. Downloading {num_files_to_download} files from source")
    df_final = download_data_from_hf(local_dir, num_files_to_download)
    df_final.write_parquet(f"{local_dir}/metadata/pd3m_all.parquet")

    # Dwonload imgs
    logging.info(f"2. Downloading {len(df_final)} imgs on {str(Path(local_dir) / 'metadata' / 'pd3m_all.parquet')}")
    img2dataset.download(
        url_list=str(Path(local_dir) / 'metadata' / 'pd3m_all.parquet'),
        output_folder=str(Path(local_dir) / 'images'),
        processes_count=num_workers,
        thread_count=num_threads,
        output_format="webdataset",
        input_format="parquet",
        url_col=url_col,
        caption_col=caption_col,
        save_additional_columns=["id", 'source'],
        timeout=8,
        resize_only_if_bigger=True,
        image_size=256,
        resize_mode="keep_ratio",
        encode_format="jpg",
        encode_quality=95,
        skip_reencode=True,
        number_sample_per_shard=10_000 
    )


if __name__ == "__main__":
    run()