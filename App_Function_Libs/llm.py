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
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

# 3rd-Party Libraries
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizer

#
#
##########################################################################################################
#
#

# Constants
DEFAULT_PROMPT_TEMPLATE = "### Instruction: {instruction}\n\n### Response: "
STOP_SEQUENCES = ["### ", "\n#", "</s>", "<|im_end|>", "[INST]", "<|eot_id|>"]

def initialize_llm_service(llm: str, args: Union[Dict[str, Any], argparse.Namespace]) -> Dict[str, Any]:
    global server_log
    server_log = open("server.log", "w")

    os.makedirs("sample-outputs", exist_ok=True)

    # Helper function to get attributes from args, whether it's a dict or an object
    def get_arg(name: str, default: Any = None) -> Any:
        if isinstance(args, dict):
            return args.get(name, default)
        return getattr(args, name, default)

    return {
        "llm": llm,
        "args": args,
        "llm_hash": get_llm_hash(llm),
        "prompt_template": DEFAULT_PROMPT_TEMPLATE,
        "debug": get_arg("debug", False),
        "verbose": get_arg("verbose", False),
        "forbid_start": get_arg("no_llm", False),
    }

def determine_model_type(model_path: str) -> str:
    model_name = os.path.basename(model_path).lower()
    if "llama" in model_name:
        return "llama"
    elif "mistral" in model_name:
        return "mistral"
    else:
        return "unknown"


def get_hf_model_info(model_name: str) -> dict:
    api_url = f"https://huggingface.co/api/models/{model_name}"
    response = requests.get(api_url)
    if response.status_code == 200:
        return response.json()
    return None


# FIXME - take into account tokenizer for mistral v.03
def ensure_tokenizer(model_path: str) -> str:
    model_type = determine_model_type(model_path)
    tokenizer_path = os.path.join(os.path.dirname(model_path), f"{model_type}_tokenizer")

    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer not found. Attempting to determine appropriate tokenizer.")

        if model_type == "unknown":
            hf_model = input("Enter the Hugging Face model name (e.g., 'mlabonne/NeuralDaredevil-8B'): ")
            model_info = get_hf_model_info(hf_model)

            if model_info and isinstance(model_info, dict) and model_info.get('pipeline_tag') == 'text-generation':
                print(f"Using tokenizer from {hf_model}")
                tokenizer = AutoTokenizer.from_pretrained(hf_model)
            else:
                print("Couldn't find tokenizer info. Using default Llama tokenizer.")
                tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        elif model_type == "llama":
            tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        elif model_type == "mistral":
            tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

        print(f"Saving tokenizer to {tokenizer_path}")
        tokenizer.save_pretrained(tokenizer_path)

    return tokenizer_path


# Fixme - functionality to specify tokenizer
def get_tokenizer(model_path: str) -> AutoTokenizer:
    try:
        # First, try to load from the model's directory
        model_dir = os.path.dirname(model_path)
        return AutoTokenizer.from_pretrained(model_dir)
    except OSError:
        print(f"Couldn't load tokenizer from {model_dir}. Attempting to determine appropriate tokenizer.")
        tokenizer_path = ensure_tokenizer(model_path)
        return AutoTokenizer.from_pretrained(tokenizer_path)


def get_llm_hash(llm: str) -> str:
    global hash_cache
    cached_hash = get_hash_cache(hash_cache, llm)
    if cached_hash is None:
        print(f"Deriving LLM hash for {llm}...")
        new_hash = model_quick_hash(llm)
        set_hash_cache(hash_cache, llm, new_hash)
        print(f"LLM hash: {new_hash}")
        return new_hash
    return cached_hash

# llamafile - claude sonnet 3.5 version
def run_completion(prompt: str, service: Dict[str, Any], **kwargs) -> Generator[str, None, None]:
    max_tokens = kwargs.get("max_tokens", service["args"].predict)
    temperature = kwargs.get("temperature", service["args"].temperature)
    top_p = kwargs.get("top_p", service["args"].top_p)

    data = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True
    }

    print(f"Sending request to server with data: {data}")

    response = requests.post("http://127.0.0.1:8080/completion", json=data, stream=True)

    if response.status_code != 200:
        print(f"Server returned status code {response.status_code}")
        print(f"Response content: {response.content}")
        raise Exception(f"Server returned non-200 status code: {response.status_code}")

    accumulated_text = ""
    for line in response.iter_lines():
        if line:
            try:
                json_str = line.decode('utf-8').removeprefix('data: ')
                json_response = json.loads(json_str)
                if 'content' in json_response:
                    accumulated_text += json_response['content']
                    if accumulated_text.endswith((' ', '\n', '.', '!', '?', ',')):
                        yield accumulated_text
                        accumulated_text = ""
                    if json_response.get('stop'):
                        if accumulated_text:
                            yield accumulated_text
                        break
                else:
                    print(f"Unexpected JSON structure: {json_response}")
            except json.JSONDecodeError:
                print(f"Failed to parse JSON from line: {line}")
            except Exception as e:
                print(f"Error processing line: {e}")
                print(f"Problematic line: {line}")

def is_server_running():
    try:
        response = requests.get("http://127.0.0.1:8080/v1/models", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


# Llamafile - claude sonnet 3.5 version
def start_server(service: Dict[str, Any], **kwargs):
    if service["forbid_start"]:
        raise Exception("Server start is forbidden.")

    llamafile_path = kwargs.get("llamafile", "./llamafile")
    model_path = kwargs.get("llm", service["llm"])

    print(f"Starting server with llamafile: {llamafile_path}")
    print(f"Using model: {model_path}")

    command = f"{llamafile_path} --model {model_path} --host 127.0.0.1 --port 8080 --n-gpu-layers {service['args'].ngl} --ctx-size {service['args'].contextsize} --batch-size 512 --threads 8"

    print(f"Executing command: {command}")

    # Start the server as a background process
    os.system(f"start /b {command}")

    print("Waiting for server to start...")
    start_time = time.time()
    timeout = 300  # 5 minutes

    while time.time() - start_time < timeout:
        try:
            response = requests.get("http://127.0.0.1:8080/v1/models", timeout=5)
            if response.status_code == 200:
                print("Server started successfully.")
                return
        except requests.exceptions.RequestException as e:
            print(f"Server not ready yet: {e}")

        time.sleep(5)

    print(f"Server failed to start within {timeout} seconds.")
    raise Exception("Server startup timed out")


def stop_server():
    global server
    if server is not None:
        atexit.unregister(server.terminate)
        server.terminate()
        server.wait()
        server = None


def cleanup():
    global server
    if server:
        print("Terminating server...")
        try:
            parent = psutil.Process(server.pid)
            for child in parent.children(recursive=True):
                child.terminate()
            parent.terminate()
            parent.wait(timeout=10)
        except psutil.NoSuchProcess:
            print("Server process not found. It may have already been closed.")
        except Exception as e:
            print(f"Error while terminating server: {e}")
        finally:
            if server.poll() is None:
                print("Server didn't terminate, forcing...")
                try:
                    parent = psutil.Process(server.pid)
                    parent.kill()
                except Exception as e:
                    print(f"Error while force-terminating server: {e}")
    server = None


MB = 1024 * 1024
GB = MB * 1024


def model_dir_quick_hash(modeldir: str) -> str:
    data = b""
    for fname in sorted(os.listdir(modeldir)):
        data += f"\nFILE:{fname}=".encode("utf-8")
        with open(os.path.join(modeldir, fname), "rb") as f:
            data += f.read(MB)
            size = os.path.getsize(os.path.join(modeldir, fname))
            pos = 4
            while pos < size / GB:
                f.seek(pos * GB)
                data += f.read(MB)
                pos += 4
    return hashlib.sha256(data).hexdigest()


def model_quick_hash(modelpath: str) -> str:
    if os.path.isdir(modelpath):
        return model_dir_quick_hash(modelpath)

    with open(modelpath, "rb") as f:
        data = f.read(MB)
        size = os.path.getsize(modelpath)
        pos = 4
        while pos < size / GB:
            f.seek(pos * GB)
            data += f.read(MB)
            pos += 4
    return hashlib.sha256(data).hexdigest()





