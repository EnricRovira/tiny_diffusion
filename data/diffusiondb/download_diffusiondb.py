
import os
import requests
import httpx
import asyncio
import time
from zipfile import ZipFile
import shutil
from concurrent.futures import ThreadPoolExecutor

# Variables
base_url = "https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/images/part-"
path_data='/mnt/sd1tb/tinydiffusion/'
name_dataset='dataset_v1'
diffusion_db_partitions = 800
out_dir = os.path.join(path_data, name_dataset, "diffusiondb")
parallel_downloads = 8

os.makedirs(out_dir, exist_ok=True)


def download_and_unzip(part_id):
    url = f"{base_url}{part_id:06}.zip"
    output_file = os.path.join(out_dir, f"part-{part_id:06}.zip")
    extract_dir = os.path.join(out_dir, f"part-{part_id:06}")

    response = requests.get(url, timeout=10)
    response.raise_for_status() 
    with open(output_file, 'wb') as f:
        f.write(response.content)

    # Descomprimir el archivo
    with ZipFile(output_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    os.remove(output_file)


def main():
    start = time.time()
    part_ids = range(1, diffusion_db_partitions + 1)
    with ThreadPoolExecutor(max_workers=parallel_downloads) as executor:
        executor.map(download_and_unzip, part_ids)
    elapsed = time.time() - start
    print(f"Elapsed: {elapsed}")

if __name__=='__main__':
    main()
