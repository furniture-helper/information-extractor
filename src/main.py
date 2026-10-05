from concurrent.futures import ThreadPoolExecutor, as_completed

# from services.InformationExtractionModel import ie_model
from helpers.page_helpers import get_unextracted_pages, update_extracted_page_details
from models.Page import Page
from services.HtmlCleanerService import cleaner
from services.InformationExtractionModel import ie_model
from services.KafkaService import ExtractionEvent, kafka_service
from services.Logging import LoggingService
from services.PostgresConnector import postgres_connector
from services.PredictionFilter import PredictionFilter
from services.Processor import processor
from services.S3Bucket import minimized_pages_bucket

logger = LoggingService.get_logger("main")


def main():
    logger.info("Starting information extractor...")

    # processor = Processor(config.get_processor_model_id())
    # model = ClassificationModel(config.get_classification_model_id())
    # dataset = PageClassificationDataset(minimized_pages_bucket, processor, model)

    pages = get_unextracted_pages(postgres_connector)
    total = len(pages)
    logger.info("Found %d pages to process.", total)
    max_workers = min(8, max(1, total))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_page, page): page for page in pages}
        for completed, future in enumerate(as_completed(futures), start=1):
            page = futures[future]
            try:
                future.result()
                logger.info("Progress: %d/%d pages completed (last: %s)", completed, total, page.url)
            except Exception:
                logger.exception("Failed to process page: %s", page.url)
                logger.info("Progress: %d/%d pages completed (last failed: %s)", completed, total, page.url)


def process_page(page: Page):
    logger.info("Processing page: %s", page.url)

    try:
        content = minimized_pages_bucket.download(page.s3_key)
        cleaned = cleaner.clean(content)
        encoding = processor.encode(cleaned)
        result = ie_model.extract(encoding)
        logger.debug("Extracted result: %s", result)

        prediction_filter = PredictionFilter(result)
        filtered_result = prediction_filter.filter()
        logger.debug("Filtered result: %s", filtered_result)

        update_extracted_page_details(postgres_connector, page.url, filtered_result)

        extraction_event = ExtractionEvent(
            url=page.url,
            prediction=result,
            filtered=filtered_result
        )

        kafka_service.send_message(extraction_event.to_dict())


    except Exception:
        logger.exception("Failed to process page: %s", page.url)
        


if __name__ == "__main__":
    main()
