from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from logging import getLogger

logger = getLogger(__name__)

s3 = boto3.client("s3")

AUDIO_EXTS: set[str] = {".mp3", ".wav", ".m4a", ".mp4", ".aac", ".flac"}
DATA_EXTS : set[str] = {".csv", ".parquet", ".xls", ".xlsx"}

BUCKET_NAME = "ai-agent-data-storage-bucket"

def _upload_file(file_name: str, key: str):
    s3.upload_file(
        Filename=file_name,
        Bucket=BUCKET_NAME,
        Key=key,
    )

def perform_s3_upload(file: str | Path) -> str | None:
    """Upload a file to an S3 bucket"""
    file = Path(file)
    file_ext  = file.suffix

    if file_ext in AUDIO_EXTS:
        key = f"uploads/audio/{str(file)}"
    elif file_ext in DATA_EXTS:
        key = f"uploads/data/{str(file)}"
    else:
        logger.exception(f"Unsupported file type: {file_ext}")
        return None

    try:
        _upload_file(file, key)
        logger.info(f"Artifact uploaded to S3 bucket: s3://{BUCKET_NAME}/{key}")
        s3_uri = f"s3://{BUCKET_NAME}/{key}"
        return s3_uri
    except ClientError as e:
        logger.exception(e)
        return None
