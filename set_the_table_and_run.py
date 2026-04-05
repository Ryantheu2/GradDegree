import os
import boto3
import shutil
import subprocess

s3 = boto3.client('s3')

def download_model_data_and_unzip(data_dir):
    try:
        zip_path = os.path.join(data_dir, 'ml_files.zip')
        s3.download_file('aso-misc-or', 'ryan/ml_work/ml_files.zip', zip_path)
        print(f"Downloaded ml_files.zip")
        shutil.unpack_archive(zip_path, extract_dir=data_dir)
        os.remove(zip_path)
        print(f"Unzipped ml_files.zip to {data_dir}")
    except Exception as e:
        raise RuntimeError("Error downloading model data:", e)

def run_ml_script(script_path, bucket, checkpoint_prefix, data_dir, output_data_prefix):
    command = f"python {script_path}"
    command += f" --bucket {bucket}"
    command += f" --checkpoint_prefix {checkpoint_prefix}"
    command += f" --data_dir {data_dir}"
    command += f" --output_data_prefix {output_data_prefix}"
    subprocess.run(command, shell=True, check=True)


if __name__ == "__main__":
    bucket_name         = 'aso-misc-or'
    checkpoint_prefix   = 'ryan/ml_work/ml_checkpoint/'
    output_data_prefix  = 'ryan/ml_work/ml_output/'

    data_dir = './data/'
    os.makedirs(data_dir, exist_ok=True)

    download_model_data_and_unzip(data_dir)

    run_ml_script('GEOG_ML.py', bucket_name, checkpoint_prefix, data_dir, output_data_prefix)
