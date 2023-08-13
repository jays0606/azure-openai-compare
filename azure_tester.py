import os
import openai
import time
from dotenv import load_dotenv
from typing import Tuple


class AzureTester:
    chat_models = ["gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0613", "gpt-4-0613"]

    def __init__(self):
        load_dotenv()
        openai.api_type = "azure"
        openai.api_version = "2023-05-15"
        openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        openai.api_base = os.getenv("AZURE_OPENAI_API_BASE")

        self.deployment_id_dict = {
            "gpt-3.5-turbo-16k-0613": os.getenv("AZURE_GPT-3.5-16k_DEPLOYMENT_ID"),
            "gpt-3.5-turbo-0613": os.getenv("AZURE_GPT-3.5_DEPLOYMENT_ID"),
            "gpt-4-0613": os.getenv("AZURE_GPT-4_DEPLOYMENT_ID"),
            "embedding": os.getenv("AZURE_EMBEDDING_DEPLOYMENT_ID"),
        }

    def chat(self, model: str, input_text: str) -> dict[str, float]:
        start_time = time.time()
        _ = openai.ChatCompletion.create(
            messages=[{"role": "system", "content": input_text}],
            deployment_id=self.deployment_id_dict[model],
            temperature=0,
        )
        # print(_["choices"][0]["message"]["content"])
        elapsed_time = time.time() - start_time
        return elapsed_time

    def embedding(self, input_text: str) -> float:
        start_time = time.time()
        _ = openai.Embedding.create(
            deployment_id=self.deployment_id_dict["embedding"], input=input_text
        )
        return time.time() - start_time

    def streaming(self, model: str, input_text: str) -> Tuple[float, float, float]:
        start_time = time.time()
        response = openai.ChatCompletion.create(
            deployment_id=self.deployment_id_dict[model],
            messages=[
                {
                    "role": "user",
                    "content": input_text,
                }
            ],
            temperature=0,
            stream=True,
        )

        chunk_times = [time.time() - start_time]
        for _ in response:
            if (
                "content" in _["choices"][0]["delta"]
            ):  # first response is always "role": "assistant"
                chunk_times.append(time.time() - start_time)
                # print(_["choices"][0]["delta"]["content"])

        first_response_latency = chunk_times[1] - chunk_times[0]
        avg_step_interval = sum(
            chunk_times[i + 1] - chunk_times[i] for i in range(len(chunk_times) - 1)
        ) / len(chunk_times)
        final_response_latency = chunk_times[-1]

        return first_response_latency, avg_step_interval, final_response_latency
