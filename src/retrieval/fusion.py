from __future__ import annotations


def to_similarity(distance: float) -> float:
    return 1.0 - float(distance)


def minmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    if high == low:
        return [1.0] * len(scores)
    span = high - low
    return [(score - low) / span for score in scores]


def _normalized(results: list[dict]) -> list[tuple[dict, float]]:
    similarities = [to_similarity(item["distance"]) for item in results]
    return list(zip(results, minmax(similarities)))


def fuse_rank(
    text_results: list[dict],
    image_results: list[dict],
    *,
    text_weight: float = 1.0,
    image_weight: float = 1.0,
) -> list[dict]:
    if text_weight < 0 or image_weight < 0:
        raise ValueError("fusion weights must be nonnegative")
    merged: dict[str, dict] = {}
    first_seen: dict[str, int] = {}
    counter = 0

    for modality, results, weight in (
        ("text", text_results, text_weight),
        ("image", image_results, image_weight),
    ):
        for item, score in _normalized(results):
            item_id = str(item["id"])
            if item_id not in merged:
                merged[item_id] = {
                    **item,
                    "fusion_score": 0.0,
                    "modality_scores": {},
                }
                first_seen[item_id] = counter
                counter += 1
            merged[item_id]["modality_scores"][modality] = score
            merged[item_id]["fusion_score"] += weight * score

    return sorted(
        merged.values(),
        key=lambda item: (-item["fusion_score"], first_seen[str(item["id"])]),
    )
