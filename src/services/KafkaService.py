import asyncio
import json
import os
from dataclasses import dataclass, asdict
from typing import Optional

from kafka import KafkaProducer as KafkaProducerClient

from helpers.ecs_helpers import get_host, get_region
from helpers.url_helpers import get_domain_from_url
from services.InformationExtractionModel import ExtractionResult
from services.PredictionFilter import FilteredResult


class KafkaService:

    def __init__(self, bootstrap_servers: str, success_topic: str, failed_topic: str):
        self.producer = KafkaProducerClient(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        )
        self.success_topic = success_topic
        self.failed_topic = failed_topic
        self.metadata = {}

    def send_success_message(self, message: dict):
        if not self.metadata:
            self.metadata = {
                "host": asyncio.run(get_host()),
                "region": asyncio.run(get_region())
            }

        message_with_metadata = {**message, **self.metadata}
        self.producer.send(self.success_topic, value=message_with_metadata)

    def send_failed_message(self, message: dict):
        if not self.metadata:
            self.metadata = {
                "host": asyncio.run(get_host()),
                "region": asyncio.run(get_region())
            }

            message_with_metadata = {**message, **self.metadata}
            self.producer.send(self.failed_topic, value=message_with_metadata)

    def close(self):
        self.producer.flush()
        self.producer.close()


@dataclass
class ExtractionEvent:
    url: str
    prediction: ExtractionResult
    filtered: FilteredResult

    def to_dict(self):
        return {
            "url": self.url,
            "domain": get_domain_from_url(self.url),
            "prediction": asdict(self.prediction),
            **asdict(self.filtered),
            "source": "ecs_extractor"
        }


@dataclass
class FailedExtractionEvent:
    url: str
    prediction: Optional[ExtractionResult]
    filtered: Optional[FilteredResult]
    error: str

    def to_dict(self):
        return {
            "url": self.url,
            "domain": get_domain_from_url(self.url),
            "prediction": asdict(self.prediction) if self.prediction else None,
            "filtered": asdict(self.filtered) if self.filtered else {},
            "source": "ecs_extractor"
        }


kafka_service = KafkaService(os.getenv("KAFKA_BROKER_URLS"), "extraction-events", "failed-extraction-events")
