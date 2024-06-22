# Run MMLU-Pro benchmarks against a GGUF quant

import os
import json
import sys
import argparse
import logging
import re
import pandas as pd
from tqdm import tqdm
from llm import LLMService
from prompting import Prompting
from utils import MetadataRepository, NGramTailStopper, RegexStopper

choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]

def format_prompt(prompting, question, options, answer, answer_index, cot_content, category, src):
    options_str = ""
    for i, option in enumerate(options):
        options_str += f"{choices[i]}: {option}\n"
    prompt = prompting.instruct("""Question:
{question}
Options:
{options_str}""".format(question=question, options_str=options_str)
    )
    return prompt

def extract_answer(text):
    pattern = r"answer is \(?([ABCDEFGHIJ])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        logging.info("1st answer extract failed\n" + text)
        return extract_again(text)

def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return None

def tally_score(response, correct):
    # NOTE: This differs from the official implementation, which gives the model a random selection which, if it ends up being correct, awards the model a point; here, the model is always given a 0 score if it fails to produce a properly formatted answer
    return 1 if response is not None and response == correct else 0

def main():
    # Do some file checks
    # 1. Do we have validation-00000-of-00001.parquet?
    if not os.path.exists("validation-00000-of-00001.parquet"):
        print("Error: validation-00000-of-00001.parquet not found; please download the dataset from https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/validation-00000-of-00001.parquet?download=true and place it (or a symlink to it) in the current directory.")
        sys.exit(1)

    # 2. Do we have koboldcpp directory (symlink)? If not, ask the user to put it there.
    if not os.path.exists("koboldcpp"):
        print("Error: koboldcpp directory not found; please symlink koboldcpp to the current directory.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Run MMLU-Pro benchmarks against a GGUF quant")
    parser.add_argument("modelformat", type=str, help="The model format to use (e.g. 'llama3', 'mistral', etc.)")
    LLMService.PrepareParser(parser, defaults={
        "temperature": 0,
        "smoothing-factor": 0,
        "predict": 1024,
    })
    args = parser.parse_args()

    llm = LLMService(args.llm, args)
    tokenizer = llm.get_tokenizer()
    prompting = Prompting.from_name(tokenizer, args.modelformat, "user", None)

    preamble_template = "The following are multiple choice questions (with answers) about {category}. Think step by step and then finish your answer with \"the answer is (X)\" where X is the correct letter choice.\n\n\n"

    # Load the dataset from ./validation-00000-of-00001.parquet
    df = pd.read_parquet("validation-00000-of-00001.parquet")

    mdfile = "metadata/root"
    if os.path.exists(mdfile):
        try:
            with open(mdfile, "r") as f:
                metadata = json.load(f)
            metadata = MetadataRepository.deserialize(metadata)
        except:  # noqa: E722
            print(f"Error loading metadata from {mdfile}")
            breakpoint()
            metadata = MetadataRepository()
    else:
        # Make directories
        mddirs = "/".join(mdfile.split("/")[:-1])
        os.makedirs(mddirs, exist_ok=True)
        metadata = MetadataRepository()

    metadata.filename = mdfile

    stoppers = [
        NGramTailStopper(6, 11),
        RegexStopper(r"answer is \(?([ABCDEFGHIJ])\)?"),
        RegexStopper(r".*[aA]nswer:\s*([A-J])"),
    ]

    # Run the benchmarks
    corrects, total = 0, 0
    for index, row in tqdm(df.iterrows(), total=len(df)):
        question = row["question"]
        options = row["options"]
        answer = row["answer"]
        answer_index = row["answer_index"]
        cot_content = row["cot_content"]
        category = row["category"]
        src = row["src"]

        prompting.preamble = preamble_template.format(category=category)
        # Run the benchmark
        prompt = format_prompt(prompting, question, options, answer, answer_index, cot_content, category, src)
        print(prompt)
        response = llm.cachable_finish_completion(prompt, metadata.get_child(question), echo=True, stop_sequence_additional=["Note:", "I apologize", "Please let me", ".assistant", "Seriously,", "For real", "P.S.", "(Sorry", "I'll stop now", "Really, I will", "Promise!", "Bye for now"], stoppers=stoppers)
        result = extract_answer(response)
        if result is None:
            logging.info("2nd answer extract failed\n" + response)
            print(f"Response: {response}\nPrompt: {prompt}\nAnswer: {answer}\nAnswer Index: {answer_index}\nCOT Content: {cot_content}\nCategory: {category}\nSource: {src}")
        correct = tally_score(result, answer)
        corrects += correct
        total += 1
        # color code correct in green, incorrect in red
        o = f"Question {index + 1}: {correct == 1} ({result} {'==' if correct == 1 else '!='} {answer}) ({corrects}/{total} = {corrects / total})"
        print("\033[92m" + o + "\033[0m" if correct == 1 else "\033[91m" + o + "\033[0m")

    # Print the final score
    print(f"Final score: {corrects}/{total} = {corrects / total}")

if __name__ == "__main__":
    main()
