import matplotlib.pyplot as plt
import numpy as np

from typing import List, Dict
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
    chat_avg_results = {}
    for model, latencies in chat_results.items():
        avg_latency = sum(latencies) / len(latencies)
        chat_avg_results[model] = avg_latency
        print(f"Model {model}: Average Latency {avg_latency:.4f} seconds")

    embedding_avg_latency = sum(embedding_result) / len(embedding_result)
    print(f"\nEmbedding Average Latency: {embedding_avg_latency:.4f} seconds")

    streaming_avg_results = {}
    for model, results in streaming_results.items():
        model_avg = {}
        print("{model}: ")
        for test_type, latencies in results.items():
            avg_latency = sum(latencies) / len(latencies)
            model_avg[test_type] = avg_latency
            print(f"{test_type}: Average Latency {avg_latency:.4f} seconds")
        streaming_avg_results[model] = model_avg

    return chat_avg_results, embedding_avg_latency, streaming_avg_results


def plot_comparison_chat(
    openai_avg_results: Dict[str, float], azure_avg_results: Dict[str, float]
):
    labels = list(openai_avg_results.keys())
    openai_latencies = list(openai_avg_results.values())
    azure_latencies = list(azure_avg_results.values())

    plot_bars(labels, openai_latencies, azure_latencies, "Chat")


def plot_comparison_embedding(openai_avg_result: float, azure_avg_result: float):
    labels = ["Embedding"]
    openai_latencies = [openai_avg_result]
    azure_latencies = [azure_avg_result]

    plot_bars(labels, openai_latencies, azure_latencies, "Embedding")


def plot_comparison_streaming(
    openai_avg_results: Dict[str, Dict[str, float]],
    azure_avg_results: Dict[str, Dict[str, float]],
):
    metrics = list(next(iter(openai_avg_results.values())).keys())
    for metric in metrics:
        labels = list(openai_avg_results.keys())
        openai_latencies = [openai_avg_results[model][metric] for model in labels]
        azure_latencies = [azure_avg_results[model][metric] for model in labels]

        plot_bars(labels, openai_latencies, azure_latencies, f"Streaming - {metric}")


def plot_bars(labels, openai_latencies, azure_latencies, title):
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width / 2, openai_latencies, width, label="OpenAI")
    rects2 = ax.bar(x + width / 2, azure_latencies, width, label="Azure")

    ax.set_ylabel("Latency (seconds)")
    ax.set_title(f"Latencies by model and platform for {title}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    plt.savefig(f"data/plot/{title.replace(' ', '_').lower()}_comparison.png")
    plt.clf()


if __name__ == "__main__":
    with open("data/query_simple.txt") as f:
        full_text = f.readlines()

    azure_tester = AzureTester()
    azure_chat_results, azure_embedding_result, azure_streaming_results = run_tests(
        azure_tester, full_text
    )

    openai_tester = OpenAITester()
    openai_chat_results, openai_embedding_result, openai_streaming_results = run_tests(
        openai_tester, full_text
    )

    plot_comparison_chat(openai_chat_results, azure_chat_results)
    plot_comparison_embedding(openai_embedding_result, azure_embedding_result)
    plot_comparison_streaming(openai_streaming_results, azure_streaming_results)
