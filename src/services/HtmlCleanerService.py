import re


class HTMLCleanerService():
    _TITLE_TAG_PATTERN = re.compile(r"<title\b[^>]*>.*?</title>", re.IGNORECASE | re.DOTALL)
    _NOISE_TAG_PATTERN = re.compile(
        r"<(script|style|noscript|template|svg)\b[^>]*>.*?</\1>",
        re.IGNORECASE | re.DOTALL,
    )
    _HTML_COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)

    def clean(self, html_content: str) -> str:
        if not html_content:
            return html_content

        # Preserve tags/xpaths for MarkupLM; only strip known noisy sections.
        html_content = self._remove_title_tags(html_content)
        html_content = self._remove_noise_tags(html_content)
        html_content = self._remove_comments(html_content)
        return html_content

    def _remove_title_tags(self, html_content: str) -> str:
        return self._TITLE_TAG_PATTERN.sub("", html_content)

    def _remove_noise_tags(self, html_content: str) -> str:
        return self._NOISE_TAG_PATTERN.sub("", html_content)

    def _remove_comments(self, html_content: str) -> str:
        return self._HTML_COMMENT_PATTERN.sub("", html_content)


cleaner = HTMLCleanerService()
