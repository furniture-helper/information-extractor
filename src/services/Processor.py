from pathlib import Path
from threading import Lock

from lxml import html as lxml_html
from transformers import MarkupLMProcessor

import config
from services.Logging import LoggingService


class Processor:
    logger = LoggingService.get_logger("Processor")

    def __init__(self, model_id: str):
        local_path = Path(config.get_models_dir()) / model_id
        if local_path.exists():
            self.logger.info(f"Loading model from {local_path}")
            self.processor = MarkupLMProcessor.from_pretrained(local_path, local_files_only=True)
        else:
            self.logger.info(f"Loading model from Hugging Face Hub: {model_id}")
            self.processor = MarkupLMProcessor.from_pretrained(model_id)
        self._lock = Lock()

    @staticmethod
    def _html_to_nodes_and_xpaths(html_string: str) -> tuple[list[str], list[str]]:
        try:
            root = lxml_html.fromstring(html_string)
            tree = root.getroottree()
        except Exception:
            return ["empty"], ["/html/body"]

        nodes: list[str] = []
        xpaths: list[str] = []

        for element in root.iter():
            # Emit img src as a synthetic text node (same as training time).
            if element.tag == "img":
                src = element.get("src", "").strip()
                if src:
                    nodes.append(src)
                    xpaths.append(tree.getpath(element))

            if element.text and element.text.strip():
                nodes.append(element.text.strip())
                xpaths.append(tree.getpath(element))

            if element.tail and element.tail.strip():
                parent = element.getparent()
                xpath = tree.getpath(parent) if parent is not None else tree.getpath(element)
                nodes.append(element.tail.strip())
                xpaths.append(xpath)

        if not nodes:
            return ["empty"], ["/html/body"]

        return nodes, xpaths

    def encode(self, html_content: str):
        nodes, xpaths = self._html_to_nodes_and_xpaths(html_content)
        with self._lock:
            self.processor.parse_html = False
            encoding = self.processor(
                nodes=nodes,
                xpaths=xpaths,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
        return encoding


processor = Processor(config.get_processor_model_id())
