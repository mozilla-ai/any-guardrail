import os
from typing import Any

import requests

from any_guardrail.base import Guardrail, GuardrailOutput


class Alinia(Guardrail[bool, dict[str, dict[str, float | bool | str]], dict[str, dict[str, float]]]):
    """Wraps the Alinia API for content moderation and safety detection.

    This wrapper allows you to send conversations or text inputs to the Alinia API. You must get an API key from Alinia
    and either set it to the ALINIA_API_KEY environment variable or pass it directly to the constructor. From Alinia, you'll also
    be able to get the proper endpoint URL as well.

    Args:
        endpoint (str): The Alinia API endpoint URL.
        detection_config (str | dict): The detection configuration ID or a dictionary specifying detection parameters.
        api_key (str | None): The API key for authenticating with the Alinia API. If not provided, it will be read from the ALINIA_API_KEY environment variable.
        metadata (dict | None): Optional metadata to include with the request.
        blocked_response (dict | None): Optional response to return if content is blocked.
        stream (bool): Whether to use streaming for the API response.

    """

    def __init__(
        self,
        detection_config: str | dict[str, float | bool] | dict[str, dict[str, float | bool | str]],
        api_key: str | None = None,
        endpoint: str | None = None,
        metadata: dict[str, Any] | None = None,
        blocked_response: dict[str, str] | None = None,
        stream: bool = False,
    ):
        """Initialize the Alinia guardrail with the provided configuration."""
        if api_key:
            self.api_key = api_key
        elif os.getenv("ALINIA_API_KEY"):
            self.api_key = os.getenv("ALINIA_API_KEY")  # type: ignore[assignment]
        else:
            msg = "API key must be provided either as a parameter or through the ALINIA_API_KEY environment variable."
            raise ValueError(msg)

        if endpoint:
            self.endpoint = endpoint
        elif os.getenv("ALINIA_ENDPOINT"):
            self.endpoint = os.getenv("ALINIA_ENDPOINT")  # type: ignore[assignment]
        else:
            msg = "Endpoint URL must be provided either as a parameter or through the ALINIA_ENDPOINT environment variable."
            raise ValueError(msg)

        self.detection_config = detection_config
        self.metadata = metadata
        self.blocked_response = blocked_response
        self.stream = stream

    def validate(
        self,
        conversation: str | list[dict[str, str]],
        output: str | None = None,
        context_documents: list[str] | None = None,
    ) -> GuardrailOutput[bool, dict[str, dict[str, float | bool | str]], dict[str, dict[str, float]]]:
        """Validate conversation or text input using the Alinia API.

        This can be used for validation using any of the API endpoints provided by Alinia. If using sensitive information endpoint,
        use the explanation from the GuardrailOutput to grab the recommended action text.

        Args:
            conversation (str | list[dict[str, str]]): The conversation or text input to validate.
            output (str | None): Optional expected output to validate against.
            context_documents (list[str] | None): Optional context documents to provide additional context for validation

        """
        params = self._pre_processing(conversation, output, context_documents)
        response = self._inference(params)
        return self._post_processing(response)

    def _pre_processing(
        self,
        conversation: str | list[dict[str, str]],
        output: str | None = None,
        context_documents: list[str] | None = None,
    ) -> dict[str, Any]:
        initial_json = {}

        if isinstance(conversation, str):
            initial_json["input"] = conversation
        elif isinstance(conversation, list):
            initial_json["messages"] = conversation  # type: ignore[assignment]
        else:
            msg = "Conversation must be either a string or a list of message dictionaries."
            raise ValueError(msg)

        if isinstance(self.detection_config, dict):
            initial_json["detection_config"] = self.detection_config  # type: ignore[assignment]
        elif isinstance(self.detection_config, str):
            initial_json["detection_config_id"] = self.detection_config
        else:
            msg = "Detection configuration must be either a string ID or a dictionary."
            raise ValueError(msg)

        if self.metadata:
            initial_json["metadata"] = self.metadata  # type: ignore[assignment]
        if self.stream:
            initial_json["stream"] = self.stream  # type: ignore[assignment]
        if context_documents:
            initial_json["context_documents"] = context_documents  # type: ignore[assignment]
        if output:
            initial_json["output"] = output
        if self.blocked_response:
            initial_json["blocked_response"] = self.blocked_response  # type: ignore[assignment]

        return initial_json

    def _inference(self, params: dict[str, Any]) -> requests.Response:  # type: ignore[name-defined]
        response = requests.post(  # type: ignore[attr-defined, no-untyped-call]
            self.endpoint,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=params,
        )
        if response.status_code != 200:
            msg = f"Request to Alinia API failed with status code {response.status_code}: {response.text}"
            raise ValueError(msg)
        return response

    def _post_processing(
        self,
        response: requests.Response,  # type: ignore[name-defined]
    ) -> GuardrailOutput[bool, dict[str, dict[str, float | bool | str]], dict[str, dict[str, float]]]:
        explanation = response.json()
        valid = not explanation.get("result").get("flagged")
        score = explanation.get("result").get("category_details")
        return GuardrailOutput(
            valid=valid,
            explanation=explanation,
            score=score,
        )
