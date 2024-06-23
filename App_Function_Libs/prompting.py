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

# Local libraries
from App_Function_Libs.llm import *

# 3rd-Party Libraries
from transformers import PreTrainedTokenizer

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

