import os
import openai
import time
from dotenv import load_dotenv
from utils import extract_text_from_pdf
from typing import Tuple

load_dotenv()


class OpenAITester:
    chat_models = ["gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0613", "gpt-4-0613"]
    embedding_model = "text-embedding-ada-002"

    def __init__(self):
        load_dotenv()

        openai.api_key = os.getenv("OPENAI_API_KEY")

    def chat(self, model: str, input_text: str) -> dict[str, float]:
        start_time = time.time()
        _ = openai.ChatCompletion.create(
            messages=[{"role": "system", "content": input_text}], model=model
        )

        elapsed_time = time.time() - start_time
        return elapsed_time

    def embedding(self, input_text: str) -> float:
        start_time = time.time()
        _ = openai.Embedding.create(model=self.embedding_model, input=input_text)
        return time.time() - start_time

    def streaming(self, model: str, input_text: str) -> Tuple[float, float, float]:
        start_time = time.time()
        response = openai.ChatCompletion.create(
            model=model,
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

        first_response_latency = chunk_times[1] - chunk_times[0]
        avg_step_interval = sum(
            chunk_times[i + 1] - chunk_times[i] for i in range(len(chunk_times) - 1)
        ) / len(chunk_times)
        final_response_latency = chunk_times[-1]

        return first_response_latency, avg_step_interval, final_response_latency


if __name__ == "__main__":
    tester = OpenAITester()

    full_text = extract_text_from_pdf("data/gpt4.pdf")
    chunk_size = 1000

    chat_results = {
        "gpt-3.5-turbo-16k-0613": [],
        "gpt-3.5-turbo-0613": [],
        "gpt-4-0613": [],
    }
    embedding_result = []
    streaming_results = {
        "gpt-3.5-turbo-16k-0613": {
            "first_response": [],
            "avg_step_interval": [],
            "final_response": [],
        },
        "gpt-3.5-turbo-0613": {
            "first_response": [],
            "avg_step_interval": [],
            "final_response": [],
        },
        "gpt-4-0613": {
            "first_response": [],
            "avg_step_interval": [],
            "final_response": [],
        },
    }

    current_index = 0
    while current_index + chunk_size <= len(full_text):
        input_text = full_text[current_index : current_index + chunk_size]
        current_index += chunk_size

        for model in OpenAITester.chat_models:
            latency = tester.chat(model, input_text)
            chat_results[model].append(latency)

        embedding_latency = tester.embedding(input_text)
        embedding_result.append(embedding_latency)

        for model in OpenAITester.chat_models:
            (
                first_response_latency,
                avg_step_interval,
                final_response_latency,
            ) = tester.streaming(model, input_text)
            streaming_results[model]["first_response"].append(first_response_latency)
            streaming_results[model]["avg_step_interval"].append(avg_step_interval)
            streaming_results[model]["final_response"].append(final_response_latency)

    print("\nChat Results:")
    for model, latencies in chat_results.items():
        avg_latency = sum(latencies) / len(latencies)
        print(f"Model {model}: Average Latency {avg_latency:.4f} seconds")

    embedding_avg_latency = sum(embedding_result) / len(embedding_result)
    print(f"\nEmbedding Average Latency: {embedding_avg_latency:.4f} seconds")

    print("\nStreaming Results:")
    for model, results in streaming_results.items():
        print(f"\nModel: {model}")
        for test_type, latencies in results.items():
            avg_latency = sum(latencies) / len(latencies)
            print(f"{test_type}: Average Latency {avg_latency:.4f} seconds")
