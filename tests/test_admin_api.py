"""Tests for administrative endpoints including /admin/reindex."""

import os
from unittest.mock import patch
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_admin_reindex_unauthorized():
    """Verify /admin/reindex rejects requests with missing or invalid key."""
    response = client.post("/admin/reindex")
    assert response.status_code == 401
    assert "Invalid or missing" in response.json()["detail"]

    response = client.post("/admin/reindex", headers={"X-Admin-Key": "wrong-key"})
    assert response.status_code == 401


def test_admin_reindex_authorized():
    """Verify /admin/reindex accepts valid admin key and returns summary."""
    with patch.dict(os.environ, {"ADMIN_API_KEY": "test-admin-secret"}), \
         patch("knowledge_base.ingest.ingest") as mock_ingest:

        mock_ingest.return_value = {
            "status": "success",
            "added": 0,
            "updated": 0,
            "deleted": 0,
            "unchanged": 125,
            "total_articles": 125,
            "total_chunks": 1114,
        }

        response = client.post("/admin/reindex", headers={"X-Admin-Key": "test-admin-secret"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["summary"]["total_articles"] == 125
        mock_ingest.assert_called_once_with(force=False)
