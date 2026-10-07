import os

DEFAULT_PAGE_RETRIEVAL = 1
DEFAULT_TITLE_CONFIDENCE_THRESHOLD = 0.95
DEFAULT_PRICE_CONFIDENCE_THRESHOLD = 0.80
DEFAULT_IMAGE_CONFIDENCE_THRESHOLD = 0.90
DEFAULT_STOCK_CONFIDENCE_THRESHOLD = 0.80


def get_page_retrieval_count() -> int:
    count = os.environ.get("PAGE_RETRIEVAL_COUNT")
    if count is not None:
        try:
            return int(count)
        except ValueError:
            raise ValueError(f"Invalid PAGE_RETRIEVAL_COUNT value: {count}. Must be an integer.")
    return DEFAULT_PAGE_RETRIEVAL


def get_minimized_pages_bucket_name() -> str:
    minimized_pages_bucket_name = os.environ.get("MINIMIZED_PAGES_BUCKET_NAME")
    if minimized_pages_bucket_name is None:
        raise ValueError("MINIMIZED_PAGES_BUCKET_NAME environment variable is not set.")
    return minimized_pages_bucket_name


def get_processor_model_id() -> str:
    processor_model_id = os.environ.get("PROCESSOR_MODEL_ID")
    if processor_model_id is None:
        raise ValueError("PROCESSOR_MODEL_ID environment variable is not set.")
    return processor_model_id


def get_models_dir() -> str:
    return "../.models"


def get_ie_model_id() -> str:
    classification_model_id = os.environ.get("IE_MODEL_ID")
    if classification_model_id is None:
        raise ValueError("IE_MODEL_ID environment variable is not set.")
    return classification_model_id


def get_title_confidence_threshold() -> float:
    title_confidence_threshold = os.environ.get("TITLE_CONFIDENCE_THRESHOLD")
    if title_confidence_threshold is None:
        return DEFAULT_TITLE_CONFIDENCE_THRESHOLD
    return float(title_confidence_threshold)


def get_price_confidence_threshold() -> float:
    price_confidence_threshold = os.environ.get("PRICE_CONFIDENCE_THRESHOLD")
    if price_confidence_threshold is None:
        return DEFAULT_PRICE_CONFIDENCE_THRESHOLD
    return float(price_confidence_threshold)


def get_image_confidence_threshold() -> float:
    image_confidence_threshold = os.environ.get("IMAGE_CONFIDENCE_THRESHOLD")
    if image_confidence_threshold is None:
        return DEFAULT_IMAGE_CONFIDENCE_THRESHOLD
    return float(image_confidence_threshold)


def get_stock_confidence_threshold() -> float:
    stock_confidence_threshold = os.environ.get("STOCK_CONFIDENCE_THRESHOLD")
    if stock_confidence_threshold is None:
        return DEFAULT_STOCK_CONFIDENCE_THRESHOLD
    return float(stock_confidence_threshold)
