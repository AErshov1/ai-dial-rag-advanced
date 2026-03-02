import json
import requests
import http

from task._constants import API_KEY

DIAL_EMBEDDINGS = 'https://ai-proxy.lab.epam.com/openai/deployments/{model}/embeddings?api-version=2023-12-01-preview'


# ---
# https://dialx.ai/dial_api#operation/sendEmbeddingsRequest
# ---
# Implement DialEmbeddingsClient:
# - constructor should apply deployment name and api key
# - create method `get_embeddings` that will generate embeddings for input list (don't forget about dimensions)
#   with Embedding model and return back a dict with indexed embeddings (key is index from input list and value vector list)

# Hint:
#  Response JSON:
#  {
#     "data": [
#         {
#             "embedding": [
#                 0.19686688482761383,
#                 ...
#             ],
#             "index": 0,
#             "object": "embedding"
#         }
#     ],
#     ...
#  }

class DialEmbeddingsClient:
    def __init__(self, deployment_name: str, api_key: str = API_KEY):
        self.deployment_name = deployment_name
        self.api_key = api_key

    def get_embeddings(self, input_list: list[str], dimensions: int = 1536) -> dict[int, list[float]]:
        """
        Get embeddings for input list of strings.

        :param input_list: List of strings to generate embeddings for.
        :return: Dict with indexed embeddings (key is index from input list and value vector list).
        """

        headers = {
              "api-key": self.api_key,
              "Content-Type": "application/json"
        }
        request_data = {
            "input": input_list,
            "dimensions": dimensions
        }

        # text-embedding-ada-002
        response = requests.post(url=DIAL_EMBEDDINGS.format(model=self.deployment_name),
                                headers=headers,
                                json=request_data,
                                timeout=60)

        if response.status_code != http.HTTPStatus.OK:
            raise Exception(f"Failed to get embeddings: HTTP {response.status_code} - {response.text}")

        data = response.json()
        embeddings_dict = {}
        for item in data.get("data", []):
            index = item.get("index")
            embedding = item.get("embedding")
            if index is not None and embedding is not None:
                embeddings_dict[index] = embedding

        return embeddings_dict

