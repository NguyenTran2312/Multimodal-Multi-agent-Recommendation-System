from src.tools.places import GooglePlacesClient


def test_places_normalizes_sparse_fields_without_invention():
    def post(url, *, headers, json, timeout):
        return {
            "places": [
                {
                    "id": "p1",
                    "displayName": {"text": "Pho House"},
                    "formattedAddress": "1 Main St",
                    "types": ["restaurant", "vietnamese_restaurant"],
                    "rating": 4.5,
                    "userRatingCount": 100,
                    "websiteUri": "https://p",
                    "regularOpeningHours": {"openNow": True},
                    "photos": [{"name": "photo/1"}],
                },
                {"id": "p2", "displayName": {"text": "Mystery"}},
            ]
        }

    rows = GooglePlacesClient("key", post=post).search("pho", max_results=5)
    assert rows[0].provider_id == "p1"
    assert rows[0].categories == ["restaurant", "vietnamese_restaurant"]
    assert rows[0].price_level is None
    assert rows[1].address is None
    assert rows[1].categories == []
