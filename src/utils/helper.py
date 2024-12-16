import os
from urllib.parse import urlparse
from botocore.exceptions import ClientError
from src.utils.logger import logger


def download_image_from_s3(s3, s3_url, local_dir):
    #
    parsed_url = urlparse(s3_url)
    bucket_name = parsed_url.netloc
    directory_prefix = parsed_url.path.strip('/')
    files = []
    # Get bucket object
    bucket = s3.Bucket(bucket_name)
    try:
        # List objects in the specified directory
        for obj in bucket.objects.filter(Prefix=directory_prefix):
            key = obj.key
            file_name = os.path.basename(key)
            if not file_name:
                continue
            local_path = os.path.join(local_dir, file_name)
            files.append(local_path)
            # Skip directories


            # Create directories if they don't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            # Download file from S3
            bucket.download_file(key, local_path)
            logger.info(f"File downloaded successfully to: {local_path}")
        return local_dir, files
    except ClientError as e:
        print(e)
        logger.error(f"#### Error downloading image. Error {e} ####")