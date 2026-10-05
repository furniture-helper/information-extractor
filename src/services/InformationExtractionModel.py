import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import MarkupLMForTokenClassification, AutoTokenizer

import config
from config import get_models_dir
from helpers.decode_helpers import decode_by_xpath
from services.Logging import LoggingService
from services.S3Bucket import S3Bucket


@dataclass
class PricePrediction:
    price: float
    confidence: float
    distance_from_title: Optional[int]


@dataclass
class ExtractionResult:
    title: str
    title_confidence: float
    price_candidates: list[PricePrediction]
    image_url: Optional[str]
    image_confidence: Optional[float]
    stock_status: Optional[str]
    stock_confidence: Optional[float]


class InformationExtractionModel:
    logger = LoggingService.get_logger("InformationExtractionModel")

    def __init__(self, model_name: str):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dir = Path(get_models_dir()).resolve() / model_name

        if not model_dir.exists():
            self.logger.info(f"Model not found at {model_dir}. Downloading from S3.")
            self._download_model_from_s3(model_name)
        else:
            self.logger.info(f"Using local model from {model_dir}")

        model_load_kwargs = {}
        if device.type == "cuda":
            model_load_kwargs["torch_dtype"] = torch.float16

        self.model = MarkupLMForTokenClassification.from_pretrained(str(model_dir), **model_load_kwargs)
        self.model.to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.logger.info(f"Model on {device} | id2label: {self.model.config.id2label}")

    def _download_model_from_s3(self, model_name: str):
        model_bucket = S3Bucket("kaneel-sagemaker-testing")
        key = f"ie-model-artifacts/{model_name}/output/model.tar.gz"

        model_base_dir = Path(get_models_dir()).resolve()
        model_base_dir.mkdir(parents=True, exist_ok=True)

        model_dir = model_base_dir / model_name
        archive_path = model_base_dir / f"{model_name}.tar.gz"

        self.logger.info(f"Downloading model artifact s3://{model_bucket.bucket_name}/{key}")
        next_percent_to_log = 10

        def _log_progress(downloaded_bytes: int, total_bytes: int):
            nonlocal next_percent_to_log
            if total_bytes <= 0:
                return

            current_percent = int((downloaded_bytes * 100) / total_bytes)
            while current_percent >= next_percent_to_log and next_percent_to_log <= 100:
                self.logger.info(
                    f"Downloading {model_name}: {next_percent_to_log}% "
                    f"({downloaded_bytes}/{total_bytes} bytes)"
                )
                next_percent_to_log += 10

        total_bytes = model_bucket.download_to_file(key, archive_path, _log_progress)
        if total_bytes > 0 and next_percent_to_log <= 100:
            downloaded_bytes = archive_path.stat().st_size
            self.logger.info(
                f"Downloading {model_name}: 100% ({downloaded_bytes}/{total_bytes} bytes)"
            )

        if model_dir.exists():
            shutil.rmtree(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        with tarfile.open(archive_path, "r:gz") as tar:
            # Guard against path traversal in tar members.
            for member in tar.getmembers():
                target_path = (model_dir / member.name).resolve()
                if target_path != model_dir and model_dir not in target_path.parents:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
            tar.extractall(model_dir)

        archive_path.unlink(missing_ok=True)
        self.logger.info(f"Downloaded and extracted model to {model_dir}")
        return str(model_dir)

    def extract(self, encoding) -> ExtractionResult:
        device = next(self.model.parameters()).device
        model_inputs = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**model_inputs)

        logits = outputs.logits  # (batch, seq_len, num_labels)
        pred_labels = logits.argmax(dim=-1)  # (batch, seq_len)

        title, title_score, predicted_prices, image_url, image_score, stock_status, stock_score, stock_evidence = decode_by_xpath(
            tokenizer=self.tokenizer,
            input_ids=encoding["input_ids"].cpu().tolist()[0],
            xpath_tags_seq=encoding["xpath_tags_seq"].cpu().tolist()[0],
            xpath_subs_seq=encoding["xpath_subs_seq"].cpu().tolist()[0],
            pred_labels=pred_labels.cpu().tolist()[0],
            logits=logits.cpu().tolist()[0],
            label2id=self.model.config.label2id
        )

        return ExtractionResult(
            title=title,
            title_confidence=title_score,
            price_candidates=[
                PricePrediction(
                    price=float(candidate["price"]),
                    confidence=float(candidate["confidence"]),
                    distance_from_title=candidate.get("distance_from_title"),
                )
                for candidate in predicted_prices
            ],
            image_url=image_url,
            image_confidence=image_score,
            stock_status=stock_status,
            stock_confidence=stock_score
        )


ie_model = InformationExtractionModel(config.get_ie_model_id())
