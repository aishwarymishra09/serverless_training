import os
import boto3
from urllib.parse import urlparse



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
def download_directory_from_s3_uri(s3_uri, local_directory):
    """
    Downloads all files from a given S3 URI to a local directory without using a paginator.

    Args:
    - s3_uri (str): The S3 URI of the directory (e.g., 's3://bucket-name/prefix/').
    - local_directory (str): The local directory where the files will be downloaded.
    """

    # Parse the S3 URI
    parsed_url = urlparse(s3_uri)
    bucket_name = parsed_url.netloc
    s3_prefix = parsed_url.path.lstrip('/')  # Remove the leading '/'

    # Initialize the S3 client
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY_ID,
                      aws_secret_access_key=SECRET_ACCESS_KEY)

    # Ensure the local directory exists
    if not os.path.exists(local_directory):
        os.makedirs(local_directory)

    # List all objects in the specified S3 directory (prefix)
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_prefix)

    if 'Contents' in response:

        for obj in response['Contents']:
            s3_file_path = obj['Key']
            local_file_path = os.path.join(local_directory, os.path.relpath(s3_file_path, s3_prefix))

            # Ensure the local directory for the file exists
            local_file_dir = os.path.dirname(local_file_path)
            if not os.path.exists(local_file_dir):
                os.makedirs(local_file_dir)

            # Download the file from S3
            print(f"Downloading {s3_file_path} to {local_file_path}")
            s3.download_file(bucket_name, s3_file_path, local_file_path)
    else:
        print("No files found in the specified S3 directory.")


if __name__ == '__main__':
    s3_uri = "s3://research-novelty-data/runpod-machine-flux-lora-A100/dreambooth/trained-flux-lora_headphones/"
    local_directory = '/workspace/dreambooth/trained-flux-lora_headphones'

    download_directory_from_s3_uri(s3_uri, local_directory)


