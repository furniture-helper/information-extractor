import re

import torch


def _decode_title(
        tokenizer,
        tokens,
        xpath_tags_seq,
        xpath_subs_seq,
        pred_labels,
        logits,
        title_b_label_id,
        title_i_label_id,
):
    if title_b_label_id is None and title_i_label_id is None:
        return "", 0.0, -1

    spans = []
    current_start = None
    current_scores = []

    for i, pred in enumerate(pred_labels):
        if pred in {title_b_label_id, title_i_label_id}:
            if current_start is None:
                current_start = i
            current_scores.append(_entity_score(logits[i], title_b_label_id, title_i_label_id))
            continue

        if current_start is not None:
            spans.append((current_start, i - 1, current_scores))
            current_start = None
            current_scores = []

    if current_start is not None:
        spans.append((current_start, len(pred_labels) - 1, current_scores))

    best = None
    for start, end, scores in spans:
        text = _extract_text_for_range(tokenizer, tokens, start, end)
        text = _normalize_title_text(text)
        if not text:
            continue

        quality = _title_quality_score(text)
        if quality <= 0.0:
            continue

        model_score = float(max(scores)) if scores else 0.0
        blended_score = model_score * quality
        candidate = {
            "text": text,
            "model_score": model_score,
            "blended_score": blended_score,
            "anchor_idx": start,
        }
        if best is None or candidate["blended_score"] > best["blended_score"]:
            best = candidate

    if best is None:
        return "", 0.0, -1

    return best["text"], best["model_score"], best["anchor_idx"]


def _decode_image(
        tokenizer,
        tokens,
        xpath_tags_seq,
        xpath_subs_seq,
        pred_labels,
        logits,
        image_b_label_id,
        image_i_label_id,
):
    if image_b_label_id is None and image_i_label_id is None:
        return "", 0.0

    best_idx = -1
    best_score = -float("inf")

    for i, pred in enumerate(pred_labels):
        if pred in {image_b_label_id, image_i_label_id}:
            score = _entity_score(logits[i], image_b_label_id, image_i_label_id)
            if score > best_score:
                best_score = score
                best_idx = i

    if best_idx == -1:
        return "", 0.0

    xpath_key = (tuple(xpath_tags_seq[best_idx]), tuple(xpath_subs_seq[best_idx]))
    image_text = _extract_text_for_xpath(
        tokenizer=tokenizer,
        tokens=tokens,
        xpath_tags_seq=xpath_tags_seq,
        xpath_subs_seq=xpath_subs_seq,
        xpath_key=xpath_key,
    )
    return image_text, float(_entity_score(logits[best_idx], image_b_label_id, image_i_label_id))


def _decode_stock(
        tokenizer,
        tokens,
        xpath_tags_seq,
        xpath_subs_seq,
        pred_labels,
        logits,
        out_stock_b_label_id,
        out_stock_i_label_id,
):
    best = {"score": 0.0, "idx": -1, "xpath_key": None}

    for i, pred in enumerate(pred_labels):
        if pred not in {out_stock_b_label_id, out_stock_i_label_id}:
            continue
        score = _entity_score(logits[i], out_stock_b_label_id, out_stock_i_label_id)
        if score > best["score"]:
            best = {
                "score": float(score),
                "idx": i,
                "xpath_key": (tuple(xpath_tags_seq[i]), tuple(xpath_subs_seq[i])),
            }

    if best["idx"] == -1:
        return "in_stock", 0.0, ""

    evidence = _extract_text_for_xpath(
        tokenizer=tokenizer,
        tokens=tokens,
        xpath_tags_seq=xpath_tags_seq,
        xpath_subs_seq=xpath_subs_seq,
        xpath_key=best["xpath_key"],
    )
    return "out_of_stock", best["score"], evidence


def _decode_price_candidates(
        tokenizer,
        tokens,
        xpath_tags_seq,
        xpath_subs_seq,
        pred_labels,
        logits,
        price_b_label_id,
        price_i_label_id,
        title_idx,
):
    if price_b_label_id is None and price_i_label_id is None:
        return []

    best_by_xpath = {}

    for i, pred in enumerate(pred_labels):
        if pred not in {price_b_label_id, price_i_label_id}:
            continue

        confidence = _entity_score(logits[i], price_b_label_id, price_i_label_id)

        xpath_key = (tuple(xpath_tags_seq[i]), tuple(xpath_subs_seq[i]))
        existing = best_by_xpath.get(xpath_key)
        if existing is None or confidence > existing["confidence"]:
            best_by_xpath[xpath_key] = {
                "confidence": confidence,
                "token_idx": i,
            }

    candidates = []
    for xpath_key, meta in best_by_xpath.items():
        price_text = _extract_text_for_xpath(
            tokenizer=tokenizer,
            tokens=tokens,
            xpath_tags_seq=xpath_tags_seq,
            xpath_subs_seq=xpath_subs_seq,
            xpath_key=xpath_key,
        )
        if not price_text:
            continue

        numeric_price = _parse_price_value(price_text)
        if numeric_price is None:
            continue

        candidates.append(
            {
                "price": numeric_price,
                "confidence": float(meta["confidence"]),
                "distance_from_title": (
                    None if title_idx == -1 else abs(int(meta["token_idx"]) - int(title_idx))
                ),
            }
        )

    # Keep one score per numeric price and return highest-confidence first.
    deduped = {}
    for candidate in candidates:
        key = candidate["price"]
        current = deduped.get(key)
        if current is None or candidate["confidence"] > current["confidence"]:
            deduped[key] = candidate

    return sorted(deduped.values(), key=lambda c: c["confidence"], reverse=True)


def decode_by_xpath(
        tokenizer,
        input_ids,
        xpath_tags_seq,
        xpath_subs_seq,
        pred_labels,
        logits,
        label2id,
):
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    title_b_label_id = label2id.get("B-TITLE")
    title_i_label_id = label2id.get("I-TITLE")
    price_b_label_id = label2id.get("B-PRICE")
    price_i_label_id = label2id.get("I-PRICE")
    image_b_label_id = label2id.get("B-IMAGE")
    image_i_label_id = label2id.get("I-IMAGE")
    out_stock_b_label_id = label2id.get("B-OUT_OF_STOCK")
    out_stock_i_label_id = label2id.get("I-OUT_OF_STOCK")

    title_text, title_score, title_idx = _decode_title(
        tokenizer=tokenizer,
        tokens=tokens,
        xpath_tags_seq=xpath_tags_seq,
        xpath_subs_seq=xpath_subs_seq,
        pred_labels=pred_labels,
        logits=logits,
        title_b_label_id=title_b_label_id,
        title_i_label_id=title_i_label_id,
    )

    predicted_prices = _decode_price_candidates(
        tokenizer=tokenizer,
        tokens=tokens,
        xpath_tags_seq=xpath_tags_seq,
        xpath_subs_seq=xpath_subs_seq,
        pred_labels=pred_labels,
        logits=logits,
        price_b_label_id=price_b_label_id,
        price_i_label_id=price_i_label_id,
        title_idx=title_idx,
    )

    image_url, image_score = _decode_image(
        tokenizer=tokenizer,
        tokens=tokens,
        xpath_tags_seq=xpath_tags_seq,
        xpath_subs_seq=xpath_subs_seq,
        pred_labels=pred_labels,
        logits=logits,
        image_b_label_id=image_b_label_id,
        image_i_label_id=image_i_label_id,
    )

    stock_status, stock_score, stock_evidence = _decode_stock(
        tokenizer=tokenizer,
        tokens=tokens,
        xpath_tags_seq=xpath_tags_seq,
        xpath_subs_seq=xpath_subs_seq,
        pred_labels=pred_labels,
        logits=logits,
        out_stock_b_label_id=out_stock_b_label_id,
        out_stock_i_label_id=out_stock_i_label_id,
    )

    return title_text, title_score, predicted_prices, image_url, image_score, stock_status, stock_score, stock_evidence


def _parse_price_value(price_text: str):
    cleaned = price_text.replace(",", "")
    matches = re.findall(r"\d+(?:\.\d+)?", cleaned)
    if not matches:
        return None
    try:
        return float(matches[0])
    except ValueError:
        return None


def _label_score(logit_row, label_id: int) -> float:
    tensor_row = torch.tensor(logit_row)
    return float(torch.softmax(tensor_row, dim=-1)[label_id].item())


def _entity_score(logit_row, b_label_id, i_label_id) -> float:
    label_ids = [lid for lid in (b_label_id, i_label_id) if lid is not None]
    if not label_ids:
        return 0.0
    return max(_label_score(logit_row, lid) for lid in label_ids)


def _extract_text_for_xpath(tokenizer, tokens, xpath_tags_seq, xpath_subs_seq, xpath_key) -> str:
    text_tokens = []
    for i, token in enumerate(tokens):
        if token in {tokenizer.pad_token, "[PAD]"}:
            continue
        key_i = (tuple(xpath_tags_seq[i]), tuple(xpath_subs_seq[i]))
        if key_i == xpath_key:
            text_tokens.append(token)
    return tokenizer.convert_tokens_to_string(text_tokens).strip()


def _extract_text_for_range(tokenizer, tokens, start_idx: int, end_idx: int) -> str:
    text_tokens = []
    for token in tokens[start_idx:end_idx + 1]:
        if token in {tokenizer.pad_token, "[PAD]"}:
            continue
        if token in getattr(tokenizer, "all_special_tokens", []):
            continue
        text_tokens.append(token)
    return tokenizer.convert_tokens_to_string(text_tokens).strip()


def _normalize_title_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip(" -|:\u00a0")
    return text


def _title_quality_score(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0

    length_penalty = 1.0
    if len(words) > 22 or len(text) > 180:
        length_penalty = 0.1
    elif len(words) > 16 or len(text) > 130:
        length_penalty = 0.6

    if re.search(r"https?://|www\.", text, flags=re.IGNORECASE):
        return 0.0

    # A strong stop-phrase for spec-table blobs that should not become titles.
    if re.search(r"\btech\s*specs?\b", text, flags=re.IGNORECASE):
        length_penalty *= 0.2

    return max(0.0, min(1.0, length_penalty))
