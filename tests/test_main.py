from unittest.mock import patch

import pytest

from app.main import get_app


class TestGetApp:
    @patch("app.main.get_lifespan_handler")
    @patch("app.main.get_api_router")
    @patch("app.main.load_config")
    def test_get_app_configuration(self, mock_config, mock_router, mock_lifespan):
        """Test that get_app() creates properly configured FastAPI instance"""
        mock_config.return_value = {
            "app_name": "Payment Classifier",
            "app_version": "0.1",
            "app_description": "Payment classification API service using LLM models",
        }

        app = get_app()

        assert app.title == "Payment Classifier"
        assert app.version == "0.1"
        assert app.description == "Payment classification API service using LLM models"

        # Verify router and lifespan are set up
        mock_router.assert_called_once()
        mock_lifespan.assert_called_once()

    @patch("app.main.get_lifespan_handler")
    @patch("app.main.get_api_router")
    def test_app_has_required_routes(self, mock_router, mock_lifespan):
        """Test that app includes required routes"""
        app = get_app()

        # Check that routes are present
        route_paths = [
            getattr(route, "path", None)
            for route in app.routes
            if hasattr(route, "path")
        ]
        assert "/" in route_paths  # Root endpoint

        # Verify router inclusion was called
        mock_router.assert_called_once()
