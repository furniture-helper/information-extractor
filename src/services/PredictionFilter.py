from dataclasses import dataclass
from typing import Optional

import config
from services.InformationExtractionModel import ExtractionResult, PricePrediction


@dataclass
class FilteredResult:
    title: str
    price: float
    image: Optional[str]
    in_stock: Optional[bool]


class PredictionFilter:

    def __init__(self, prediction: ExtractionResult):
        self.prediction = prediction

    def filter(self) -> FilteredResult:
        return FilteredResult(
            title=self._determine_title(),
            price=self._determine_price(),
            image=self._determine_image(),
            in_stock=self._determine_in_stock(),
        )

    def _normalize_price_candidates(self) -> None:
        normalized = []
        for candidate in self.prediction.price_candidates:
            if isinstance(candidate, PricePrediction):
                normalized.append(candidate)
                continue

            if isinstance(candidate, dict):
                normalized.append(PricePrediction(
                    price=float(candidate.get("price", 0.0)),
                    confidence=float(candidate.get("confidence", 0.0)),
                    distance_from_title=candidate.get("distance_from_title"),
                ))

        self.prediction.price_candidates = normalized

    def _determine_title(self) -> str:
        if self.prediction.title is None:
            raise ValueError("Title cannot be None")

        if self.prediction.title_confidence < config.get_title_confidence_threshold():
            raise ValueError(f"Title confidence {self.prediction.title_confidence} is below the threshold "
                             f"{config.get_title_confidence_threshold()}")

        return self.prediction.title

    def _determine_price(self) -> float:
        self._normalize_price_candidates()
        if not self.prediction.price_candidates:
            raise ValueError("Price cannot be found")

        self._eliminate_low_confidence_predictions()
        if not self.prediction.price_candidates:
            raise ValueError("Price cannot be found")

        self._filter_price_candidates_by_distance()
        if not self.prediction.price_candidates:
            raise ValueError("Price cannot be found")

        self._filter_price_by_fractions()

        final_price = self._find_lowest_price_prediction()
        if final_price is None:
            raise ValueError("Price cannot be found")

        return final_price.price

    def _determine_image(self) -> Optional[str]:
        if self.prediction.image_url is None:
            return None

        if self.prediction.image_confidence is None or self.prediction.image_confidence < config.get_image_confidence_threshold():
            return None

        return self.prediction.image_url

    def _determine_in_stock(self) -> bool:
        if self.prediction.stock_status is None:
            return True

        if self.prediction.stock_confidence is None or self.prediction.stock_confidence < config.get_stock_confidence_threshold():
            return True

        return self.prediction.stock_status.lower() == "in_stock"

    def _eliminate_low_confidence_predictions(self) -> None:
        self.prediction.price_candidates = [p for p in self.prediction.price_candidates if
                                            p.confidence >= config.get_price_confidence_threshold()]

    def _get_closest_price_prediction_to_title(self) -> PricePrediction | None:
        if not self.prediction.price_candidates:
            return None
        return min(self.prediction.price_candidates, key=lambda p: p.distance_from_title)

    def _filter_price_candidates_by_distance(self):
        anchor_price_prediction = self._get_closest_price_prediction_to_title()
        if anchor_price_prediction is None or anchor_price_prediction.distance_from_title is None:
            return
        threshold_distance = anchor_price_prediction.distance_from_title * 5
        self.prediction.price_candidates = [
            prediction for prediction in self.prediction.price_candidates
            if prediction.distance_from_title is None or prediction.distance_from_title <= threshold_distance
        ]

    def _filter_price_by_fractions(self):
        anchor_price_prediction = self._get_closest_price_prediction_to_title()
        if anchor_price_prediction is None or anchor_price_prediction.price == 0:
            return
        self.price_scores = [
            prediction for prediction in self.prediction.price_candidates
            if prediction.price >= anchor_price_prediction.price * 0.5
        ]

    def _find_lowest_price_prediction(self) -> PricePrediction | None:
        return min(self.prediction.price_candidates,
                   key=lambda p: p.price) if self.prediction.price_candidates else None
