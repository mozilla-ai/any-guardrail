import functools
import os
from collections.abc import Callable
from typing import Any, ClassVar

from azure.ai.contentsafety import BlocklistClient, ContentSafetyClient
from azure.ai.contentsafety.models import (
    AddOrUpdateTextBlocklistItemsOptions,
    AnalyzeImageOptions,
    AnalyzeTextOptions,
    ImageData,
    RemoveTextBlocklistItemsOptions,
    TextBlocklist,
    TextBlocklistItem,
    TextCategory,
)
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

from any_guardrail.base import Guardrail, GuardrailOutput


def error_message(message: str) -> Any:
    """Handle exceptions for Azure Content Safety operations."""

    def error_handler_decorator(func: Callable) -> Any:  # type: ignore [type-arg]
        """Handle exceptions for the wrapped function."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except HttpResponseError as e:
                raise RuntimeError(message + f" Details: {e!s}") from e

        return wrapper

    return error_handler_decorator


class AzureContentSafety(Guardrail):
    """Guardrail implementation using Azure Content Safety service."""

    SUPPORTED_MODELS: ClassVar = ["azure-content-safety"]

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        threshold: int = 2,
        score_type: str = "max",
        blocklist_name: list[str] | None = None,
    ) -> None:
        """Initialize Azure Content Safety client.

        Args:
            endpoint (str): The endpoint URL for the Azure Content Safety service.
            api_key (str): The API key for authenticating with the service.
            threshold (int): The threshold for determining if content is unsafe.
            score_type (str): The type of score to use ("max" or "avg").
            blocklist_name (List[str] | None): List of blocklist names to use for content evaluation.

        """
        if api_key:
            credential = AzureKeyCredential(api_key)
        else:
            try:
                credential = AzureKeyCredential(os.environ["CONTENT_SAFETY_KEY"])
            except KeyError as e:
                msg = "CONTENT_SAFETY_KEY environment variable is not set. Either provide an api_key or set the environment variable."
                raise KeyError(msg) from e
        if not endpoint:
            try:
                endpoint = os.environ["CONTENT_SAFETY_ENDPOINT"]
            except KeyError as e:
                msg = "CONTENT_SAFETY_ENDPOINT environment variable is not set. Either provide an endpoint or set the environment variable."
                raise KeyError(msg) from e

        self.client = ContentSafetyClient(endpoint=endpoint, credential=credential)
        self.blocklist_client = BlocklistClient(endpoint=endpoint, credential=credential)
        self.threshold = threshold

        if score_type not in ["max", "avg"]:
            msg = "score_type must be either 'max' or 'avg'"
            raise ValueError(msg)
        self.score_type = score_type

        if blocklist_name:
            if not isinstance(blocklist_name, list):
                msg = "blocklist_name must be a list of strings"
                raise ValueError(msg)
            for name in blocklist_name:
                if not isinstance(name, str):
                    msg = "blocklist_name must be a list of strings"
                    raise ValueError(msg)
        self.blocklist_name = blocklist_name

    def validate(self, content: str) -> GuardrailOutput:
        """Validate content using Azure Content Safety.

        Args:
            content (str): The content to be evaluated.

        Returns:
            GuardrailOutput: The result of the guardrail evaluation.

        """
        model_inputs = self._pre_processing(content)
        model_outputs = self._inference(model_inputs)
        return self._post_processing(model_outputs)

    @error_message("Was unable to create or update blocklist.")  # type: ignore [untyped-decorator]
    def create_or_update_blocklist(self, blocklist_name: str, blocklist_description: str) -> None:
        """Create or update a blocklist in Azure Content Safety.

        Args:
            blocklist_name (str): The name of the blocklist.
            blocklist_description (str): The description of the blocklist.

        """
        self.blocklist_client.create_or_update_text_blocklist(
            blocklist_name=blocklist_name,
            options=TextBlocklist(blocklist_name=blocklist_name, description=blocklist_description),
        )

    @error_message("Was unable to add blocklist items.")  # type: ignore [untyped-decorator]
    def add_blocklist_items(self, blocklist_name: str, blocklist_terms: list[str]) -> None:
        """Add items to a blocklist.

        Args:
            blocklist_name (str): The name of the blocklist.
            blocklist_terms (List[str]): The terms to add to the blocklist.

        """
        blocklist_items = []
        for term in blocklist_terms:
            blocklist_items.append(TextBlocklistItem(text=term))

        self.blocklist_client.add_or_update_blocklist_items(
            blocklist_name=blocklist_name,
            options=AddOrUpdateTextBlocklistItemsOptions(blocklist_items=blocklist_items),
        )

    @error_message("Was unable to list blocklists.")  # type: ignore [untyped-decorator]
    def list_blocklists(self) -> list[dict[str, str | None]]:
        """List all blocklists in Azure Content Safety.

        Returns:
            List[Dict[str, str]]: A list of blocklist details.

        """
        blocklists = self.blocklist_client.list_text_blocklists()
        return [{"name": blocklist.blocklist_name, "description": blocklist.description} for blocklist in blocklists]

    @error_message("Was unable to list blocklist items.")  # type: ignore [untyped-decorator]
    def list_blocklist_items(self, blocklist_name: str) -> list[dict[str, str | None]]:
        """List items in a blocklist.

        Args:
            blocklist_name (str): The name of the blocklist.

        Returns:
            List[Dict[str, str]]: The list of blocklist items.

        """
        blocklist_items = self.blocklist_client.list_text_blocklist_items(blocklist_name=blocklist_name)
        return [
            {"id": item.blocklist_item_id, "text": item.text, "description": item.description}
            for item in blocklist_items
        ]

    @error_message("Was unable to get blocklist.")  # type: ignore [untyped-decorator]
    def get_blocklist(self, blocklist_name: str) -> dict[str, str | None]:
        """Get a blocklist by name.

        Args:
            blocklist_name (str): The name of the blocklist.

        Returns:
            dict[str, str]: The blocklist details.

        """
        blocklist = self.blocklist_client.get_text_blocklist(blocklist_name=blocklist_name)
        return {"name": blocklist.blocklist_name, "description": blocklist.description}

    @error_message("Was unable to get blocklist item.")  # type: ignore [untyped-decorator]
    def get_blocklist_item(self, blocklist_name: str, item_id: str) -> dict[str, str | None]:
        """Get a blocklist item by ID.

        Args:
            blocklist_name (str): The name of the blocklist.
            item_id (str): The ID of the blocklist item.

        Returns:
            dict[str, str]: The blocklist item details.

        """
        item = self.blocklist_client.get_text_blocklist_item(blocklist_name=blocklist_name, blocklist_item_id=item_id)
        return {"id": item.blocklist_item_id, "text": item.text, "description": item.description}

    @error_message("Was unable to delete blocklist.")  # type: ignore [untyped-decorator]
    def delete_blocklist(self, blocklist_name: str) -> None:
        """Delete a blocklist by name.

        Args:
            blocklist_name (str): The name of the blocklist.

        """
        self.blocklist_client.delete_text_blocklist(blocklist_name=blocklist_name)

    @error_message("Was unable to delete blocklist item.")  # type: ignore [untyped-decorator]
    def delete_blocklist_items(self, blocklist_name: str, item_ids: list[str]) -> None:
        """Delete a blocklist item by ID.

        Args:
            blocklist_name (str): The name of the blocklist.
            item_ids (List[str]): The IDs of the blocklist items.

        """
        self.blocklist_client.remove_blocklist_items(  # type: ignore [call-overload]
            blocklist_name=blocklist_name,
            blocklist_item_id=RemoveTextBlocklistItemsOptions(blocklist_item_ids=item_ids),
        )

    def _pre_processing(self, text: str) -> AnalyzeTextOptions | AnalyzeImageOptions:
        if self._is_existing_path(text):
            try:
                with open(text, "rb") as file:
                    return AnalyzeImageOptions(image=ImageData(content=file.read()))
            except ValueError as e:
                msg = "Must provide a file path to an image file."
                raise ValueError(msg) from e
        else:
            if self.blocklist_name:
                return AnalyzeTextOptions(text=text, blocklist_names=self.blocklist_name, halt_on_blocklist_hit=False)
            return AnalyzeTextOptions(text=text)

    @error_message("Was unable to analyze text or image.")  # type: ignore [untyped-decorator]
    def _inference(self, model_inputs: AnalyzeTextOptions | AnalyzeImageOptions) -> Any:
        if isinstance(model_inputs, AnalyzeTextOptions):
            response = self.client.analyze_text(model_inputs)
        else:
            response = self.client.analyze_image(model_inputs)  # type: ignore [assignment]
        return response

    def _post_processing(self, model_outputs: Any) -> GuardrailOutput:
        results_dict = {
            "hate": next(item for item in model_outputs.categories_analysis if item.category == TextCategory.HATE),
            "self_harm": next(
                item for item in model_outputs.categories_analysis if item.category == TextCategory.SELF_HARM
            ),
            "sexual": next(item for item in model_outputs.categories_analysis if item.category == TextCategory.SEXUAL),
            "violence": next(
                item for item in model_outputs.categories_analysis if item.category == TextCategory.VIOLENCE
            ),
        }

        explanation = {key: result.severity for key, result in results_dict.items() if result is not None}

        if self.score_type == "max":
            score = max(
                explanation_score for explanation_score in explanation.values() if explanation_score is not None
            )
        else:
            score = sum(
                explanation_score for explanation_score in explanation.values() if explanation_score is not None
            ) / sum(1 for explanation_score in explanation.values() if explanation_score is not None)

        explanation["blocklist"] = model_outputs.blocklists_match if self.blocklist_name else None

        valid = score < self.threshold
        if valid and explanation.get("blocklist"):
            valid = False

        return GuardrailOutput(valid=valid, explanation=explanation, score=score)

    def _is_existing_path(self, text: str) -> bool:
        return os.path.exists(text)
