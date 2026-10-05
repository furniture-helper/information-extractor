import config
from models.Page import Page
from services.PostgresConnector import PostgresConnector
from services.PredictionFilter import FilteredResult


def get_unextracted_pages(postgres: PostgresConnector) -> list[Page]:
    query = """
            SELECT mp.url, mp.s3_key
            FROM minimized_pages mp
                     LEFT JOIN page_inferred_labels pil ON pil.url = mp.url
                     INNER JOIN page_classifications pc ON pc.url = mp.url
                     INNER JOIN pages p ON p.url = mp.url
            WHERE pc.type = 'product'
              AND (
                (pil.last_inferred_at < mp.last_minimized_at AND
                 pil.last_inferred_at < p.last_crawled_at - INTERVAL '5 minutes') OR
                pil.last_inferred_at IS NULL
                )
            ORDER BY pil.last_inferred_at ASC
                LIMIT %s \
            """

    result = postgres.run_query(query, (config.get_page_retrieval_count(),))

    pages: list[Page] = []
    for row in result:
        pages.append(Page(row[0], row[1]))

    return pages


def update_extracted_page_details(postgres_connector: PostgresConnector, url: str, result: FilteredResult) -> None:
    query = """
            INSERT INTO page_inferred_labels (url,
                                              product_title,
                                              product_price,
                                              product_image_url,
                                              in_stock,
                                              last_inferred_at,
                                              created_at,
                                              updated_at)
            VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) ON CONFLICT (url) DO
            UPDATE SET
                product_title = EXCLUDED.product_title,
                product_price = EXCLUDED.product_price,
                product_image_url = EXCLUDED.product_image_url,
                in_stock = EXCLUDED.in_stock,
                last_inferred_at = EXCLUDED.last_inferred_at,
                updated_at = CURRENT_TIMESTAMP \
            """

    postgres_connector.execute(
        query,
        (
            url,
            result.title,
            result.price,
            result.image,
            result.in_stock,
        ),
    )


def touch_page(postgres_connector: PostgresConnector, url: str) -> None:
    upsert_inferred_timestamps(postgres_connector, url)


def upsert_inferred_timestamps(postgres_connector: PostgresConnector, url: str) -> None:
    query = """
            INSERT INTO page_inferred_labels (url,
                                              last_inferred_at,
                                              created_at,
                                              updated_at)
            VALUES (%s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT (url) DO UPDATE
            SET last_inferred_at = EXCLUDED.last_inferred_at,
                updated_at = CURRENT_TIMESTAMP
            """

    postgres_connector.execute(query, (url,))
