### 1st-Party
import argparse
import atexit
import glob
import hashlib
import json
import logging
import os
import psutil
import re
import signal
import subprocess
import sys
import time

# 3rd Party-libraries
import requests
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import pandas as pd
from packaging.metadata import Metadata
from transformers import AutoTokenizer
from tqdm import tqdm

# Local libraries
from App_Function_Libs.llm import *
from App_Function_Libs.prompting import create_metadata_repository
from App_Function_Libs.llm import initialize_llm_service
from App_Function_Libs.utils import *

#
#
##########################################################################################################
#
#

def create_hash_cache(path: str) -> Dict[str, Any]:
    cache = {}
    if os.path.exists(path):
        with open(path, "r") as f:
            cache = json.load(f)
    return {"path": path, "cache": cache}


# Global variables
server = None
server_log = None
hash_cache = create_hash_cache(".hashcache.json")
stats = {"cache_hits": 0, "cache_misses": 0}


def prepare_parser(parser, defaults=None):
    if defaults is None:
        defaults = {}

    parser.add_argument("--llamafile", default="./llamafile", help="Path to the llamafile executable")
    parser.add_argument("--contextsize", type=int, default=8192,
                        help="The maximum context size to use for the LLM model.")
    parser.add_argument("--llm", default="./llm.gguf", help="The LLM model to use.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output.")
    parser.add_argument("--debug", action="store_true", help="Print debug output.")
    parser.add_argument("--seed", type=int, default=12318731, help="Seed for random number generator.")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM usage.")

    def add_arg(name, type_, default, help_):
        parser.add_argument(f"--{name}", type=type_, default=defaults.get(name, default), help=help_)

    add_arg("ngl", int, 90, "Number of layers to load on GPU.")
    add_arg("temperature", float, 0.2, "Temperature for sampling.")
    add_arg("mirostat", int, 2, "0-2, where 0 is disabled.")
    add_arg("mirostat-tau", float, 5.0, "Mirostat tau.")
    add_arg("mirostat-eta", float, 0.1, "Mirostat eta.")
    add_arg("top-a", float, 0, "Top-k sampling parameter.")
    add_arg("top-p", float, 1, "Top-p sampling parameter.")
    add_arg("top-k", float, 0, "Top-k sampling parameter.")
    add_arg("rep-pen", float, 1.01, "Repetition penalty.")
    add_arg("rep-pen-range", int, 1024, "Repetition penalty range.")
    add_arg("min-p", float, 0.05, "Minimum probability.")
    add_arg("smoothing-factor", float, 0.3, "Smoothing factor.")
    add_arg("predict", int, 512, "Number of tokens to predict.")
    add_arg("quantkv", int, 0, "KV quantization level (0=f16, 1=q8, 2=q4)")


#
#
#######################################################################################################################
#
# Run MMLU-Pro benchmarks against a GGUF quant
#

choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]


def save_partial_results(results, filename="partial_results.json"):
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} results to {filename}")


def load_partial_results(filename="partial_results.json"):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return []


def save_results(results, filename="results.json"):
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} results to {filename}")


def load_results(filename="results.json"):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return []


def extract_answer(text):
    pattern = r"Answer:\s*([A-J])[:.]"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        print("Failed to extract answer using primary pattern. Trying secondary pattern.")
        secondary_pattern = r"([A-J]):\s"
        secondary_match = re.search(secondary_pattern, text)
        if secondary_match:
            return secondary_match.group(1)
        else:
            print("Failed to extract answer from:\n", text)
            return None


def tally_score(response, correct):
    return 1 if response is not None and response == correct else 0


def save_results(results, filename="results.json"):
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} results to {filename}")


def load_results(filename="results.json"):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return []


def main():
    parser = argparse.ArgumentParser(description="Run MMLU-Pro benchmarks against a GGUF quant")
    parser.add_argument("modelformat", type=str, help="The model format to use (e.g. 'llama3', 'mistral', etc.)")
    prepare_parser(parser, defaults={
        "temperature": 0,
        "smoothing-factor": 0,
        "predict": 1024,
    })
    args = parser.parse_args()

    print("Initializing LLM service...")
    llm_service = initialize_llm_service(args.llm, args)
    llm_service["llamafile"] = args.llamafile

    print("Getting tokenizer...")
    tokenizer = get_tokenizer(args.llm)

    print("Creating prompting service...")
    prompting_service = create_prompting_service_from_name(tokenizer, args.modelformat, "user", None)

    preamble_template = ("The following are multiple choice questions (with answers) about {category}. Think step by "
                         "step and then finish your answer with \"the answer is (X)\" where X is the correct letter "
                         "choice.\n\n\n")

    print("Loading dataset...")
    df = pd.read_parquet("validation-00000-of-00001.parquet")
    print(f"Loaded {len(df)} questions.")

    metadata = create_metadata_repository()

    stoppers = [
        create_ngram_tail_stopper(6, 11),
        create_regex_stopper(r"answer is \(?([ABCDEFGHIJ])\)?"),
        create_regex_stopper(r".*[aA]nswer:\s*([A-J])"),
    ]

    print("Loading previous results...")
    results = load_results()
    start_index = len(results)
    corrects = sum(1 for result in results if result['is_correct'])
    total = len(results)
    print(f"Loaded {len(results)} previous results. Starting from question {start_index + 1}.")

    try:
        for index, row in tqdm(df.iloc[start_index:].iterrows(), total=len(df) - start_index):
            print(f"\nProcessing question {index + 1}/{len(df)}")
            print(f"Category: {row['category']}")
            print(f"Question: {row['question']}")
            print(f"Correct Answer: {row['answer']}")

            prompting_service["preamble"] = preamble_template.format(category=row['category'])
            prompt = format_prompt(prompting_service, row['question'], row['options'], row['answer'],
                                   row['answer_index'], row['cot_content'], row['category'], row['src'])

            try:
                print("Generating completion...")
                response = cachable_finish_completion(
                    prompt,
                    llm_service,
                    get_child_metadata_repository(metadata, row['question']),
                    echo=True,
                    stop_sequence_additional=["Note:", "I apologize", "Please let me", ".assistant", "Seriously,",
                                              "For real", "P.S.", "(Sorry", "I'll stop now", "Really, I will",
                                              "Promise!",
                                              "Bye for now"],
                    stoppers=stoppers
                )

                print("\nModel's response:")
                print(response)

                result = extract_answer(response)
                print(f"\nExtracted answer: {result}")

                correct = tally_score(result, row['answer'])
                corrects += correct
                total += 1

                question_result = {
                    "index": index,
                    "category": row['category'],
                    "question": row['question'],
                    "correct_answer": row['answer'],
                    "model_answer": result,
                    "model_full_response": response,
                    "is_correct": correct == 1,
                    "current_score": f"{corrects}/{total} = {corrects / total:.2f}"
                }
                results.append(question_result)

                o = f"Question {index + 1}: {correct == 1} ({result} {'==' if correct == 1 else '!='} {row['answer']}) ({corrects}/{total} = {corrects / total:.2f})"
                print("\033[92m" + o + "\033[0m" if correct == 1 else "\033[91m" + o + "\033[0m")

                print("Saving results...")
                save_results(results)

            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"Error processing question {index + 1}: {e}")
                continue

    except KeyboardInterrupt:
        print("\nScript interrupted by user. Saving results...")
    finally:
        if results:
            print("Saving final results...")
            save_results(results)
        if total > 0:
            print(f"\nFinal score: {corrects}/{total} = {corrects / total:.2f}")
        else:
            print("No questions were fully processed.")

        print("Loading results for verification...")
        loaded_results = load_results()
        print(f"Loaded {len(loaded_results)} results:")
        for result in loaded_results[-5:]:  # Print the last 5 results for verification
            print(
                f"Question {result['index'] + 1}: {result['is_correct']} ({result['model_answer']} vs {result['correct_answer']})")

        cleanup()


if __name__ == "__main__":
    main()
#
# END
##############################
