import os
import subprocess
import time
from src.core.config import Constants

def parser():

    bucket_path = Constants.GCP_BUCKET_PROCESSED_JSON_PATH
    local_path = Constants.PROCESSED_DATA_DIR
    
    # 1. cria diretório
    os.makedirs(local_path, exist_ok=True)

    # 2. comando gcloud

    cmd = f'gcloud storage rsync "{bucket_path}" "{local_path}" --recursive'

    # 3. mede tempo
    start = time.time()

    subprocess.run(cmd, shell=True, check=True)

    end = time.time()

    print(f"Tempo de execução: {end - start:.2f}s")
