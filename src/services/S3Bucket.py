import threading

import boto3
from pathlib import Path
from typing import Callable

import config
from services.Logging import LoggingService


class S3Bucket:

    logger = LoggingService.get_logger("S3Bucket")

    def __init__(self, bucket_name: str, region_name: str = "eu-west-1"):
        self.bucket_name = bucket_name
        self.region_name = region_name
        self._local = threading.local()

    def _get_client(self):
        if not hasattr(self._local, "s3_client"):
            self._local.s3_client = boto3.client('s3', region_name=self.region_name)
        return self._local.s3_client

    def download(self, key: str) -> str:
        try:
            return self.download_bytes(key).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Failed to download s3://{self.bucket_name}/{key}: {e}")
            raise

    def download_bytes(self, key: str) -> bytes:
        response = self._get_client().get_object(Bucket=self.bucket_name, Key=key)
        return response['Body'].read()

    def download_to_file(
        self,
        key: str,
        destination_path: Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> int:
        client = self._get_client()
        metadata = client.head_object(Bucket=self.bucket_name, Key=key)
        total_bytes = int(metadata.get("ContentLength", 0))
        transferred = 0

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        with destination_path.open("wb") as output_file:
            def _on_progress(bytes_amount: int):
                nonlocal transferred
                transferred += bytes_amount
                if progress_callback is not None:
                    progress_callback(transferred, total_bytes)

            client.download_fileobj(
                self.bucket_name,
                key,
                output_file,
                Callback=_on_progress,
            )

        return total_bytes


minimized_pages_bucket = S3Bucket(bucket_name=config.get_minimized_pages_bucket_name())