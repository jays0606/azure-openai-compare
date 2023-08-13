import matplotlib.pyplot as plt
import numpy as np

from typing import List
from openai_tester import OpenAITester
from azure_tester import AzureTester


def run_tests(tester, text_list: List[str]):
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

    for input_text in text_list:
        for model in tester.chat_models:
            latency = tester.chat(model, input_text)
            chat_results[model].append(latency)

        embedding_latency = tester.embedding(input_text)
        embedding_result.append(embedding_latency)

        for model in tester.chat_models:
            (
                first_response_latency,
                avg_step_interval,
                final_response_latency,
            ) = tester.streaming(model, input_text)
            streaming_results[model]["first_response"].append(first_response_latency)
            streaming_results[model]["avg_step_interval"].append(avg_step_interval)
            streaming_results[model]["final_response"].append(final_response_latency)

    print(f"\nResults for {tester.__class__.__name__}:")
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

    return chat_results, embedding_result, streaming_results


def plot_comparison(openai_results, azure_results, test_type):
    labels = list(openai_results.keys())
    openai_latencies = [np.mean(openai_results[key]) for key in labels]
    azure_latencies = [np.mean(azure_results[key]) for key in labels]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, openai_latencies, width, label="OpenAI")
    rects2 = ax.bar(x + width / 2, azure_latencies, width, label="Azure")

    ax.set_ylabel("Latency (seconds)")
    ax.set_title(f"Latencies by model and platform for {test_type}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()
    plt.show()
    plt.savefig(f"data/{test_type}.png")
    plt.clf()


if __name__ == "__main__":
    azure_tester = AzureTester()
    openai_tester = OpenAITester()

    with open("data/query_sample.txt") as f:
        full_text = f.readlines()[:2]

    openai_chat_results, openai_embedding_result, openai_streaming_results = run_tests(
        openai_tester, full_text
    )

    azure_chat_results, azure_embedding_result, azure_streaming_results = run_tests(
        azure_tester, full_text
    )

    plot_comparison(openai_chat_results, azure_chat_results, "Chat")
    plot_comparison(openai_embedding_result, azure_embedding_result, "Embedding")
    plot_comparison(openai_streaming_results, azure_streaming_results, "Streaming")
