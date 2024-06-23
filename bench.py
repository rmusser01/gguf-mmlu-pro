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
import requests
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
from transformers import PreTrainedTokenizer
from tqdm import tqdm

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



#
#
##########################################################################################################
#
#

def get_sample_path(digest: str) -> str:
    return os.path.join("sample-outputs", digest)


def get_sample(digest: str) -> str:
    sample_output_path = get_sample_path(digest)
    if os.path.exists(sample_output_path) and os.path.getsize(sample_output_path) >= 1:
        with open(sample_output_path, "r") as f:
            return f.read()
    return None


def cachable_finish_completion(prompt: str, service: dict[str, Any], metadata: dict[str, Any], echo: bool = False,
                               ignore_cache: bool = False, allow_fallback: bool = False, **kwargs) -> str:
    sample, digest, digest_fb = get_cached_completion(prompt, service, allow_fallback, metadata, **kwargs)

    if sample is None or ignore_cache:
        stats["cache_misses"] += 1
        print("Cache miss. Starting server if not already running...")
        if not is_server_running():
            start_server(service, llamafile=service.get("llamafile"))

        print("Generating completion...")
        try:
            sample = finish_completion(prompt, service, echo=echo, **kwargs)
            print("Completion generated. Saving to cache...")
            with open(get_sample_path(digest), "w") as f:
                f.write(sample)
            with open(get_sample_path(digest_fb), "w") as f:
                f.write(sample)
        except KeyboardInterrupt:
            print("Completion generation interrupted.")
            raise
    else:
        if echo:
            print("Using cached completion.")
            print(sample)
        stats["cache_hits"] += 1
    return sample


def finish_completion(prompt: str, service: Dict[str, Any], echo: bool = False, **kwargs) -> str:
    v = ""
    for chunk in run_completion(prompt, service, **kwargs):
        v += chunk
        if echo:
            print(chunk, end="", flush=True)
    if echo:
        print()
    return v


def get_cached_completion(prompt: str, service: Dict[str, Any], allow_fallback: bool, metadata: Dict[str, Any],
                          **kwargs) -> Tuple[Optional[str], str, str]:
    digest, digest_fb = digests(metadata, prompt, service["llm_hash"], kwargs)
    sample = get_sample(digest)
    process_metadata(metadata, prompt, digest, kwargs)
    if sample is None and allow_fallback:
        sample = get_sample(digest_fb)
    return sample, digest, digest_fb


def format_prompt(prompting_service: Dict[str, Any], question: str, options: List[str], answer: str, answer_index: int,
                  cot_content: str, category: str, src: str) -> str:
    options_str = "\n".join([f"{choices[i]}: {option}" for i, option in enumerate(options)])
    instruction = f"""Question:
{question}
Options:
{options_str}"""

    return instruct(prompting_service, instruction)

#######################################################################################################################
# claude 3.5 sonnet
# This refactored version:
#   Replaces the Prompting class with a set of functions that operate on a "prompting service" dictionary.
#   Uses type hints for better readability and error catching.
#   Simplifies the logic in some functions, making them more focused and easier to understand.
#   Uses more descriptive function names.
#   Keeps the same functionality as the original class-based implementation.

# The main changes are:
#   create_prompting_service replaces the __init__ method.
#   Class methods are now standalone functions that take the service dictionary as their first argument.
#   The @classmethod decorators are removed, and the functions are renamed to be more descriptive.
#   The self references are replaced with the service dictionary.


def create_prompting_service(
        tokenizer: PreTrainedTokenizer,
        userid: Optional[str] = None,
        preamble: Optional[str] = None,
        strict_roles: bool = False,
        prepend_user_names: bool = False,
        must_begin_with: Optional[str] = None,
        system_as_role: bool = False
) -> Dict[str, Any]:
    return {
        "tokenizer": tokenizer,
        "userid": userid,
        "preamble": preamble,
        "strict_roles": strict_roles,
        "prepend_user_names": prepend_user_names,
        "must_begin_with": must_begin_with,
        "system_as_role": system_as_role
    }


def system_message(service: Dict[str, Any], message: str) -> str:
    if service["system_as_role"]:
        return service["tokenizer"].apply_chat_template(
            [{"role": "system", "content": message}],
            tokenize=False,
            add_generation_prompt=False
        )
    return message


def exchange(service: Dict[str, Any], messages: List[Dict[str, str]], add_generation_prompt: bool = False) -> str:
    if service["must_begin_with"] and messages[0]["role"] != service["must_begin_with"]:
        raise ValueError(f"First message must be from {service['must_begin_with']}")

    rv = ""
    if service["preamble"] is not None:
        rv += system_message(service, service["preamble"])

    if service["strict_roles"]:
        if service["userid"] is None:
            raise ValueError("Cannot enforce strict roles without a userid")

        processed_messages = []
        for m in messages:
            if m["role"] == "system":
                if service["system_as_role"]:
                    processed_messages.append(m)
                else:
                    rv += system_message(service, m["content"])
            else:
                role = "user" if m["role"] == service["userid"] else "assistant"
                content = f"{m['role']}: {m['content']}" if service["prepend_user_names"] else m['content']
                processed_messages.append({"role": role, "content": content})
        messages = processed_messages

    return rv + service["tokenizer"].fixed_apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt
    )


def instruct(service: Dict[str, Any], instruction: str, add_generation_prompt: bool = True) -> str:
    rv = ""
    if service["preamble"] is not None:
        rv += system_message(service, service["preamble"])

    return rv + service["tokenizer"].apply_chat_template(
        [{"role": service["userid"] if service["userid"] is not None else 'user', "content": instruction}],
        tokenize=False,
        add_generation_prompt=add_generation_prompt
    )


def create_llama3_service(tokenizer: PreTrainedTokenizer, userid: str, preamble: str) -> Dict[str, Any]:
    return create_prompting_service(
        tokenizer=tokenizer,
        userid=userid,
        preamble=preamble,
        system_as_role=True
    )


def create_mistral_service(tokenizer: PreTrainedTokenizer, userid: str, preamble: str) -> Dict[str, Any]:
    return create_prompting_service(
        tokenizer=tokenizer,
        userid=userid,
        preamble=preamble,
        system_as_role=True
    )


def create_prompting_service_from_name(tokenizer: PreTrainedTokenizer, name: str, userid: str, preamble: str) -> Dict[
    str, Any]:
    if name == "llama3":
        return create_llama3_service(tokenizer, userid, preamble)
    if name == "mistral":
        return create_mistral_service(tokenizer, userid, preamble)
    raise ValueError(f"Unknown prompting name: {name}")


def derive_prompting_service_from_tokenizer(tokenizer: PreTrainedTokenizer, userid: str, preamble: str) -> Dict[
    str, Any]:
    if tokenizer.bos_token == "<|begin_of_text|>" and tokenizer.bos_token_id == 128000:
        return create_llama3_service(tokenizer, userid, preamble)
    elif tokenizer("[INST]")["input_ids"][0] == 3:
        return create_mistral_service(tokenizer, userid, preamble)
    raise ValueError("Unknown tokenizer")



def create_hash_cache(path: str) -> Dict[str, Any]:
    cache = {}
    if os.path.exists(path):
        with open(path, "r") as f:
            cache_data = json.load(f)
            for key, value in cache_data.items():
                cache[key] = {
                    "hash": value["hash"],
                    "expiry": value["expiry"]
                }
    return {"path": path, "cache": cache}


def save_hash_cache(hash_cache: Dict[str, Any]):
    with open(hash_cache["path"], "w") as f:
        json.dump(hash_cache["cache"], f)


def get_hash_cache(hash_cache: Dict[str, Any], key: str) -> Optional[str]:
    if key in hash_cache["cache"]:
        entry = hash_cache["cache"][key]
        if entry["expiry"] > time.time():
            return entry["hash"]
        del hash_cache["cache"][key]
    return None


def set_hash_cache(hash_cache: Dict[str, Any], key: str, hash_value: str, expiry: int = 3600):
    hash_cache["cache"][key] = {
        "hash": hash_value,
        "expiry": time.time() + expiry
    }
    save_hash_cache(hash_cache)


def create_stopper() -> Dict[str, Any]:
    return {"should_stop": lambda token: False, "reset": lambda: None}


def create_ngram_tail_stopper(n: int, m: int) -> Dict[str, Any]:
    stopper = {
        "n": n,
        "m": m,
        "buffer": [],
        "window": []
    }

    def should_stop(token: str) -> bool:
        stopper["buffer"].append(token)
        if len(stopper["buffer"]) < stopper["n"]:
            return False
        ngram = "".join(stopper["buffer"][-stopper["n"]:])
        if ngram in stopper["window"]:
            return True
        stopper["window"].append(ngram)
        stopper["buffer"].pop(0)
        if len(stopper["window"]) > stopper["m"]:
            stopper["window"].pop(0)
        return False

    stopper["should_stop"] = should_stop
    stopper["reset"] = lambda: stopper.update({"buffer": [], "window": []})
    return stopper


def create_regex_stopper(regex: str, tokens: int = 12) -> Dict[str, Any]:
    stopper = {
        "regex": regex,
        "tokens": tokens,
        "buffer": []
    }

    def should_stop(token: str) -> bool:
        stopper["buffer"].append(token)
        if re.search(stopper["regex"], "".join(stopper["buffer"])):
            return True
        if len(stopper["buffer"]) > stopper["tokens"]:
            stopper["buffer"].pop(0)
        return False

    def reset() -> None:
        stopper["buffer"] = []

    stopper["should_stop"] = should_stop
    stopper["reset"] = reset
    return stopper


def create_str_stopper(stopstrings: List[str]) -> Dict[str, Any]:
    stopper = {
        "stopstrings": stopstrings,
        "keep": max(len(s) for s in stopstrings),
        "buffer": ""
    }

    def should_stop(token: str) -> bool:
        stopper["buffer"] += token
        for s in stopper["stopstrings"]:
            if s in stopper["buffer"]:
                return True
        if len(stopper["buffer"]) > stopper["keep"]:
            stopper["buffer"] = stopper["buffer"][-stopper["keep"]:]
        return False

    stopper["should_stop"] = should_stop
    stopper["reset"] = lambda: stopper.update({"buffer": ""})
    return stopper


def log(s: str):
    print(f"[{time.ctime()}] {s}")


def serializable_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: [str(s) for s in v] if k == "stoppers" else v
        for k, v in kwargs.items()
    }


def create_metadata() -> Dict[str, Any]:
    return {
        "digest": None,
        "kwargs": {},
        "llm_hash": None,
        "llm": None,
        "prompt": None,
        "log": []
    }


def deserialize_metadata(obj: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "digest": obj["digest"],
        "kwargs": obj["kwargs"],
        "llm_hash": obj["llm_hash"],
        "llm": obj["llm"],
        "prompt": obj["prompt"],
        "log": obj["log"]
    }


def serialize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "digest": metadata["digest"],
        "kwargs": metadata["kwargs"],
        "llm_hash": metadata["llm_hash"],
        "llm": metadata["llm"],
        "llm": metadata["llm"],
        "prompt": metadata["prompt"],
        "log": metadata["log"]
    }


def digests(metadata: Dict[str, Any], prompt: str, llm_hash: str, kwargs: Dict[str, Any]) -> Tuple[str, str]:
    y = prompt + "|" + str(serializable_kwargs(kwargs))
    z = (llm_hash + "|" + y).encode("utf-8")
    y = y.encode("utf-8")
    metadata["last_input"], metadata["last_input_fb"] = z, y
    digest, digest_fb = hashlib.sha256(z).hexdigest(), hashlib.sha256(y).hexdigest()
    return digest, digest_fb


def process_metadata(metadata: Dict[str, Any], prompt: str, digest: str, kwargs: Dict[str, Any]) -> bool:
    kwargs = serializable_kwargs(kwargs)
    if "digest" not in metadata or metadata["digest"] is None:
        metadata.update({
            "digest": digest,
            "llm_hash": metadata.get("llm_hash"),
            "llm": metadata.get("llm"),
            "kwargs": kwargs,
            "prompt": prompt
        })
        return True
    elif digest != metadata["digest"]:
        ob1 = {
            "kwargs": metadata["kwargs"],
            "llm_hash": metadata["llm_hash"],
            "llm": metadata["llm"],
            "prompt": metadata["prompt"]
        }
        logentry = {"digest": {"old": metadata["digest"], "new": digest}}
        metadata["digest"] = digest

        for key in ["llm", "llm_hash"]:
            if metadata[key] != kwargs.get(key, metadata[key]):
                logentry[key] = {"old": metadata[key], "new": kwargs[key]}
                metadata[key] = kwargs[key]

        kv = {}
        for k, v in kwargs.items():
            if k not in metadata["kwargs"] or metadata["kwargs"][k] != v:
                kv[k] = {"old": metadata["kwargs"].get(k), "new": v}
                metadata["kwargs"][k] = v
        if kv:
            logentry["kwargs"] = kv

        if prompt != metadata["prompt"]:
            i = next(
                (i for i in range(min(len(prompt), len(metadata["prompt"]))) if prompt[i] != metadata["prompt"][i]),
                min(len(prompt), len(metadata["prompt"])))
            j = next((j for j in range(min(len(prompt), len(metadata["prompt"])) - 1, -1, -1) if
                      prompt[j] != metadata["prompt"][j]), -1)
            logentry["prompt"] = {"diff": {"start": i, "end": j, "content": prompt[i:j + 1]}}
        metadata["prompt"] = prompt

        ob2 = {
            "kwargs": metadata["kwargs"],
            "llm_hash": metadata["llm_hash"],
            "llm": metadata["llm"],
            "prompt": metadata["prompt"]
        }
        diff_string = describe_difference(ob1, ob2)
        logentry["diff_str"] = diff_string or "No changes"
        metadata["log"].append(logentry)
        print(f"Metadata changed:\n{diff_string}")
        return True
    return False


def create_metadata_repository(parent: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "metadata": create_metadata(),
        "parent": parent,
        "children": {},
        "needs_saving": False
    }


def deserialize_metadata_repository(obj: Dict[str, Any], parent: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    repo = create_metadata_repository(parent)
    repo["metadata"] = deserialize_metadata(obj["metadata"])
    for k, v in obj["children"].items():
        repo["children"][k] = deserialize_metadata_repository(v, parent=repo)
    return repo


def process_metadata_repository(repo: Dict[str, Any], prompt: str, digest: str, kwargs: Dict[str, Any]) -> bool:
    if process_metadata(repo["metadata"], prompt, digest, kwargs):
        repo["needs_saving"] = True
    return repo["needs_saving"]


def get_child_metadata_repository(repo: Dict[str, Any], key: str) -> Dict[str, Any]:
    if key not in repo["children"]:
        repo["children"][key] = create_metadata_repository(parent=repo)
    return repo["children"][key]


def serialize_metadata_repository(repo: Dict[str, Any]) -> Dict[str, Any]:
    repo["needs_saving"] = False
    return {
        "metadata": serialize_metadata(repo["metadata"]),
        "children": {k: serialize_metadata_repository(v) for k, v in repo["children"].items()}
    }


def pathstr_metadata_repository(repo: Dict[str, Any], child: Optional[Dict[str, Any]] = None) -> str:
    if child is None:
        return repo["parent"]["pathstr"](repo) if repo["parent"] else ""
    for k, v in repo["children"].items():
        if v == child:
            parent_path = repo["parent"]["pathstr"](repo) if repo["parent"] else ""
            return (parent_path + "/" if parent_path else "") + k
    return ""


# Claude Sonnet 3.5
# This refactored version:
#   Removes class structures and replaces them with functions that operate on dictionaries.
#   Uses type hints for better readability and error catching.
#   Simplifies some of the logic and removes redundant code.
#   Uses more descriptive function names.
#   Moves constants to the top of the file.
#   Uses dictionary-based state instead of class attributes.
#   Improves error handling in some places.
#   Uses more functional programming constructs like list comprehensions and generator expressions.

# The main changes are:
#   HashCache, Stopper, NGramTailStopper, RegexStopper, and StrStopper classes are now functions that return dictionaries with the necessary methods.
#   Metadata and MetadataRepository classes are replaced with functions that operate on dictionaries.
#   Helper functions like log and serializable_kwargs remain largely unchanged.
#   The describe_difference function is simplified and made more functional.

MB = 1024 * 1024
GB = MB * 1024


def difference_expr(desc: str, s1: str, s2: str) -> str:
    prefix = ('=' * len(desc)) + ' '
    return desc + '"' + s1.replace('\n', f"\n{prefix}") + '" vs "' + s2.replace('\n', f"\n{prefix}") + '"'


def describe_difference(ob1: Any, ob2: Any, prefix: str = "") -> str:
    if isinstance(ob1, dict):
        keys = set(ob1.keys()) | set(ob2.keys())
        return "\n".join(
            f"{prefix}Key {k} is new" if k not in ob1 else
            f"{prefix}Key {k} is gone" if k not in ob2 else
            describe_difference(ob1[k], ob2[k], f"{prefix}{k}.")
            for k in keys
        ).strip()

    if isinstance(ob1, list):
        rv = []
        if len(ob1) != len(ob2):
            rv.append(f"{prefix}List {'shrunk' if len(ob1) > len(ob2) else 'grew'} from {len(ob1)} to {len(ob2)}")
        rv.extend(describe_difference(ob1[i], ob2[i], f"{prefix}[{i}].") for i in range(min(len(ob1), len(ob2))))
        return "\n".join(rv).strip()

    if ob1 == ob2:
        return "Identical" if prefix == "" else ""

    ob1, ob2 = str(ob1), str(ob2)
    i = next((i for i in range(min(len(ob1), len(ob2))) if ob1[i] != ob2[i]), min(len(ob1), len(ob2)))

    if i == min(len(ob1), len(ob2)):
        return f"\n{prefix}{'Appendage' if len(ob1) < len(ob2) else 'Missing'}: \"{ob2[i:i + 20] if len(ob1) < len(ob2) else ob1[i:i + 20]}\""
    else:
        return difference_expr(f"\n{prefix}Changed at {i}: ", ob1[max(0, i - 20):i + 20], ob2[max(0, i - 20):i + 20])

#
#
##########################################################################################################
#
#

def reset_stats():
    global stats
    stats = {"cache_hits": 0, "cache_misses": 0}


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
