from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class PlaceResult:
    provider_id: str
    name: str
    address: str | None
    categories: list[str]
    price_level: str | None
    rating: float | None
    review_count: int | None
    website: str | None
    opening_hours: dict | None
    photo_refs: list[str]
    source: str = "google_places"


class GooglePlacesClient:
    endpoint = "https://places.googleapis.com/v1/places:searchText"
    field_mask = ",".join(
        [
            "places.id",
            "places.displayName",
            "places.formattedAddress",
            "places.types",
            "places.priceLevel",
            "places.rating",
            "places.userRatingCount",
            "places.websiteUri",
            "places.regularOpeningHours",
            "places.photos",
        ]
    )

    def __init__(
        self,
        api_key: str,
        *,
        post: Callable | None = None,
        timeout: float = 10.0,
    ) -> None:
        api_key = api_key.strip()
        if not api_key:
            raise ValueError("GOOGLE_PLACES_API_KEY is required to construct GooglePlacesClient")
        self.api_key = api_key
        self.timeout = timeout
        if post is None:
            import requests

            post = requests.post
        self._post = post

    def search(self, query: str, *, max_results: int = 5) -> list[PlaceResult]:
        response = self._post(
            self.endpoint,
            headers={
                "X-Goog-Api-Key": self.api_key,
                "X-Goog-FieldMask": self.field_mask,
            },
            json={"textQuery": query, "maxResultCount": max_results},
            timeout=self.timeout,
        )
        payload = response.json() if hasattr(response, "json") else response
        output: list[PlaceResult] = []
        for item in payload.get("places", [])[:max_results]:
            display_name = item.get("displayName") or {}
            output.append(
                PlaceResult(
                    provider_id=str(item.get("id") or ""),
                    name=str(display_name.get("text") or ""),
                    address=item.get("formattedAddress"),
                    categories=list(item.get("types") or []),
                    price_level=item.get("priceLevel"),
                    rating=item.get("rating"),
                    review_count=item.get("userRatingCount"),
                    website=item.get("websiteUri"),
                    opening_hours=item.get("regularOpeningHours"),
                    photo_refs=[
                        str(photo.get("name"))
                        for photo in item.get("photos", [])
                        if photo.get("name")
                    ],
                )
            )
        return output
