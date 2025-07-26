import numpy as np
import boto3
import os

def upload_to_s3(file: np.ndarray, s3_dir: str):
    bucket_name = os.getenv("BUCKET_NAME")

    # print(f"Uploading {local_file_name} to s3://{bucket_name}/{s3_path}")
    s3_client = boto3.client("s3")

    try:
        s3_client.upload_file(file, bucket_name, f"{s3_dir}/{file}")
        # print(f"Upload success")
    except Exception as e:
        # print(f"Upload failed: {e}")
        raise e
    
    if os.path.exists(file):
        os.remove(file)
        # print(f"Removed {file}")

def upload_flag_to_s3(N: int, z: int, simulation_title: str, time: str, params: str):
    bucket_name = os.getenv("BUCKET_NAME")
    s3_path = f"hawkes_N={N}_z={z}/exp_001_done.flag"

    with open("exp_001_done.flag", "w") as f:
        f.write(f"{simulation_title} done.\n")
        f.write(f"{time}.\n")
        f.write(f"{params}.\n")

    # print(f"Uploading {simulation_title} to s3://{bucket_name}/{s3_path}")
    s3_client = boto3.client("s3")

    try:
        s3_client.upload_file("exp_001_done.flag", bucket_name, s3_path)
        # print(f"Upload success: {simulation_title}")
    except Exception as e:
        # print(f"Upload failed: {simulation_title}")
        raise e
