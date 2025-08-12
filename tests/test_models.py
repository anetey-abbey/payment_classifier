from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app.models.classification import (
    ActionType,
    ClassificationRequest,
    ModelType,
    PaymentClassification,
    PaymentDecision,
)


class TestModelType:
    def test_model_type_values(self):
        assert ModelType.LOCAL == "local"
        assert ModelType.CLOUD == "cloud"


class TestActionType:
    def test_action_type_values(self):
        assert ActionType.CLASSIFY == "classify"
        assert ActionType.SEARCH == "search"


class TestPaymentClassification:
    def test_valid_payment_classification(self):
        classification = PaymentClassification(
            category="groceries", reasoning="Purchase at supermarket"
        )
        assert classification.category == "groceries"
        assert classification.reasoning == "Purchase at supermarket"
        assert classification.search_used is False

    def test_payment_classification_with_search_used(self):
        classification = PaymentClassification(
            category="transport", reasoning="Gas station purchase", search_used=True
        )
        assert classification.search_used is True


class TestPaymentDecision:
    def test_payment_decision_classify_action(self):
        decision = PaymentDecision(
            action=ActionType.CLASSIFY,
            category="entertainment",
            reasoning="Movie ticket purchase",
        )
        assert decision.action == ActionType.CLASSIFY
        assert decision.category == "entertainment"
        assert decision.search_query is None

    def test_payment_decision_search_action(self):
        decision = PaymentDecision(
            action=ActionType.SEARCH,
            search_query="restaurant reviews",
            reasoning="Need more context for classification",
        )
        assert decision.action == ActionType.SEARCH
        assert decision.category is None
        assert decision.search_query == "restaurant reviews"


class TestClassificationRequest:
    @pytest.fixture
    def mock_config(self):
        return {
            "valid_models": {
                "local": ["qwen2.5:1.5b", "qwen2.5:3b"],
                "cloud": ["gemini-2.5-flash", "gemini-pro"],
            }
        }

    def test_valid_classification_request_local(self, mock_config):
        with patch("app.models.classification.load_config", return_value=mock_config):
            request = ClassificationRequest(
                payment_text="Coffee at Starbucks",
                categories=["food", "entertainment"],
                model_type=ModelType.LOCAL,
                model_name="qwen2.5:1.5b",
            )
            assert request.payment_text == "Coffee at Starbucks"
            assert request.categories == ["food", "entertainment"]
            assert request.model_type == ModelType.LOCAL
            assert request.model_name == "qwen2.5:1.5b"
            assert request.use_search is False

    def test_valid_classification_request_cloud(self, mock_config):
        with patch("app.models.classification.load_config", return_value=mock_config):
            request = ClassificationRequest(
                payment_text="Grocery shopping",
                categories=["groceries", "household"],
                model_type=ModelType.CLOUD,
                model_name="gemini-2.5-flash",
            )
            assert request.model_type == ModelType.CLOUD
            assert request.model_name == "gemini-2.5-flash"

    def test_categories_validation_removes_duplicates(self, mock_config):
        with patch("app.models.classification.load_config", return_value=mock_config):
            request = ClassificationRequest(
                payment_text="Test payment",
                categories=["food", "FOOD", "food", "entertainment"],
                model_type=ModelType.LOCAL,
                model_name="qwen2.5:1.5b",
            )
            # Should remove duplicates (case-insensitive)
            assert len(request.categories) == 2
            assert "food" in request.categories
            assert "entertainment" in request.categories

    def test_categories_validation_strips_whitespace(self, mock_config):
        with patch("app.models.classification.load_config", return_value=mock_config):
            request = ClassificationRequest(
                payment_text="Test payment",
                categories=[" food ", "  entertainment  ", "transport"],
                model_type=ModelType.LOCAL,
                model_name="qwen2.5:1.5b",
            )
            assert "food" in request.categories
            assert "entertainment" in request.categories
            assert "transport" in request.categories

    def test_empty_categories_raises_error(self, mock_config):
        with patch("app.models.classification.load_config", return_value=mock_config):
            with pytest.raises(ValidationError):
                ClassificationRequest(
                    payment_text="Test payment",
                    categories=[],
                    model_type=ModelType.LOCAL,
                    model_name="qwen2.5:1.5b",
                )

    def test_categories_only_whitespace_raises_error(self, mock_config):
        with patch("app.models.classification.load_config", return_value=mock_config):
            with pytest.raises(
                ValueError, match="At least one valid category must be provided"
            ):
                ClassificationRequest(
                    payment_text="Test payment",
                    categories=["   ", "\t", ""],
                    model_type=ModelType.LOCAL,
                    model_name="qwen2.5:1.5b",
                )

    def test_too_many_categories_raises_error(self, mock_config):
        with patch("app.models.classification.load_config", return_value=mock_config):
            categories = [f"category{i}" for i in range(21)]  # 21 categories
            with pytest.raises(ValueError, match="Maximum 20 categories allowed"):
                ClassificationRequest(
                    payment_text="Test payment",
                    categories=categories,
                    model_type=ModelType.LOCAL,
                    model_name="qwen2.5:1.5b",
                )

    def test_empty_model_name_raises_error(self, mock_config):
        with patch("app.models.classification.load_config", return_value=mock_config):
            with pytest.raises(ValidationError):
                ClassificationRequest(
                    payment_text="Test payment",
                    categories=["food"],
                    model_type=ModelType.LOCAL,
                    model_name="",
                )

    def test_invalid_local_model_name_raises_error(self, mock_config):
        with patch("app.models.classification.load_config", return_value=mock_config):
            with pytest.raises(
                ValueError, match="Invalid model 'invalid-model' for type 'local'"
            ):
                ClassificationRequest(
                    payment_text="Test payment",
                    categories=["food"],
                    model_type=ModelType.LOCAL,
                    model_name="invalid-model",
                )

    def test_invalid_cloud_model_name_raises_error(self, mock_config):
        with patch("app.models.classification.load_config", return_value=mock_config):
            with pytest.raises(
                ValueError, match="Invalid model 'invalid-model' for type 'cloud'"
            ):
                ClassificationRequest(
                    payment_text="Test payment",
                    categories=["food"],
                    model_type=ModelType.CLOUD,
                    model_name="invalid-model",
                )

    def test_use_search_with_cloud_model_raises_error(self, mock_config):
        with patch("app.models.classification.load_config", return_value=mock_config):
            with pytest.raises(
                ValueError, match="use_search is not supported for cloud models"
            ):
                ClassificationRequest(
                    payment_text="Test payment",
                    categories=["food"],
                    model_type=ModelType.CLOUD,
                    model_name="gemini-2.5-flash",
                    use_search=True,
                )

    def test_use_search_with_local_model_allowed(self, mock_config):
        with patch("app.models.classification.load_config", return_value=mock_config):
            request = ClassificationRequest(
                payment_text="Test payment",
                categories=["food"],
                model_type=ModelType.LOCAL,
                model_name="qwen2.5:1.5b",
                use_search=True,
            )
            assert request.use_search is True

    def test_model_name_whitespace_stripped(self, mock_config):
        # Test that model name validation happens before stripping
        with patch("app.models.classification.load_config", return_value=mock_config):
            with pytest.raises(ValueError, match="Invalid model"):
                ClassificationRequest(
                    payment_text="Test payment",
                    categories=["food"],
                    model_type=ModelType.LOCAL,
                    model_name="  qwen2.5:1.5b  ",
                )

    def test_empty_payment_text_raises_error(self, mock_config):
        with patch("app.models.classification.load_config", return_value=mock_config):
            with pytest.raises(ValueError):
                ClassificationRequest(
                    payment_text="",
                    categories=["food"],
                    model_type=ModelType.LOCAL,
                    model_name="qwen2.5:1.5b",
                )
