''' infer.py for runpod worker '''
import os
import subprocess
import sys

import boto3
import argparse
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils.rp_upload import upload_file_to_bucket
from runpod.serverless.utils import rp_download, rp_cleanup
from io import BytesIO
from rp_schema import INPUT_SCHEMA
from utils.helper import download_image_from_s3
from utils.logger import logger


def encode_key(text, shift):
    """
    Encode an alphanumeric key by shifting characters.

    :param text: The alphanumeric key to encode.
    :param shift: The number of positions to shift each character.
    :return: Encoded key as a string.
    """
    encoded = []
    for char in text:
        if char.isalpha():
            # Shift alphabetic characters (wrap within 'a-z' or 'A-Z')
            base = ord('a') if char.islower() else ord('A')
            encoded.append(chr((ord(char) - base + shift) % 26 + base))
        elif char.isdigit():
            # Shift numeric characters (wrap within '0-9')
            encoded.append(chr((ord(char) - ord('0') + shift) % 10 + ord('0')))
        else:
            # Keep other characters unchanged
            encoded.append(char)
    return ''.join(encoded)


ACCESS_KEY_ID = encode_key("EOMECW6RXDRLERXEEI1Q", -4)
SECRET_ACCESS_KEY = encode_key("Wwcb3I78eyTDReRCOnTvQk5qToGAsncZTE1rXITy", -4)
BUCKET_NAME = "rekogniz-training-data"


def save_file(directory, s3_prefix):
    """Uploads an image to an S3 bucket"""
    try:
        s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY_ID, aws_secret_access_key=SECRET_ACCESS_KEY)

        for root, _, files in os.walk(directory):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, directory)
                s3_path = os.path.join(s3_prefix, relative_path).replace("\\", "/")  # S3 uses forward slashes


                # Upload file to S3
                s3.upload_file(local_path, BUCKET_NAME, s3_path)
                print(f"Uploaded: {local_path} to s3://{BUCKET_NAME}/{s3_path}")


    except Exception as e:
        return e
    return True


def download_images(s3_url, local_dir):
    """download images from the cloud folder"""
    s3_resource = boto3.resource(
        's3',
        aws_access_key_id=ACCESS_KEY_ID,
        aws_secret_access_key=SECRET_ACCESS_KEY
    )
    dir1, _ = download_image_from_s3(s3_resource, s3_url, local_dir)

    return dir1


def run(job):
    '''
    Run inference on the model.
    Returns output path, width the seed used to generate the image.
    '''
    try:
        job_input = job['input']
        cwd_training = os.getcwd()
        # Input validation
        validated_input = validate(job_input, INPUT_SCHEMA)


        if 'errors' in validated_input:
            return {"error": validated_input['errors']}
        validated_input = validated_input['validated_input']
        local_dir = os.getcwd() + f"/{validated_input['id']}" + f"/{validated_input['training_id']}/"
        instance_dir  = download_images(validated_input['s3_url'], local_dir)
        job_output = []
        returncode = subprocess.call([sys.executable,
                                      'flux_lora_training.py',
                                      '--pretrained_model_name_or_path=/workspace/flux-dev-1',
                                      f'--instance_data_dir={instance_dir}',
                                      f'--output_dir=./logs/{validated_input["id"]}/{validated_input["training_id"]}',
                                      '--mixed_precision=bf16',
                                      f'--instance_prompt={validated_input["instance_prompt"]}',
                                      f'--resolution=512',
                                      '--train_batch_size=1',
                                      '--gradient_accumulation_steps=4',
                                      '--optimizer="prodigy"',
                                      '--learning_rate=1e-4',
                                      '--lr_scheduler="constant"',
                                      '--lr_warmup_steps=0',
                                      '--max_train_steps=500',
                                      f'--seed={"0"}'],
                                     cwd=cwd_training,
                                     stdout=subprocess.PIPE)
        save_file(f'./logs/{validated_input["id"]}/{validated_input["training_id"]}', f'trainings/{validated_input["id"]}/{validated_input["training_id"]}')
        job_output.append({"Training_success": True,
                           "Training_id": validated_input["training_id"]})
        # Remove downloaded input objects
        rp_cleanup.clean(['input_objects'])
        return job_output
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error(exc_type, fname, exc_tb.tb_lineno)
        logger.error(f"error occured due to {e}")


if __name__ == "__main__":
    logger.info("starting training ...")
    runpod.serverless.start({"handler": run})
