import hashlib
import json
import os
import re
import time
from typing_extensions import Self

mb = 1024 * 1024
gb = mb * 1024

def model_dir_quick_hash(modeldir):
    # Generate a quick hash of the model directory
    # We take 1 MB every 4 GB and hash that to get a quick hash
    data = b""
    for fname in sorted(os.listdir(modeldir)):
        data += (f"\nFILE:{fname}=").encode("utf-8")
        with open(os.path.join(modeldir, fname), "rb") as f:
            data += f.read(mb)
            size = os.path.getsize(os.path.join(modeldir, fname))
            pos = 4
            while pos < size / gb:
                f.seek(pos * gb)
                data += f.read(mb)
                pos += 4
    # Compute hash
    return hashlib.sha256(data).hexdigest()

def model_quick_hash(modelpath):
    # Generate a quick hash of the model
    if os.path.isdir(modelpath):
        return model_dir_quick_hash(modelpath)

    # We take 1 MB every 4 GB and hash that to get a quick hash
    with open(modelpath, "rb") as f:
        data = f.read(mb)
        size = os.path.getsize(modelpath)
        pos = 4
        while pos < size / gb:
            f.seek(pos * gb)
            data += f.read(mb)
            pos += 4
        # Compute hash
        return hashlib.sha256(data).hexdigest()

def difference_expr(desc, s1, s2):
    prefix = ('=' * len(desc)) + ' '
    return desc + '"' + s1.replace('\n', f"\n{prefix}") + '" vs "' + s2.replace('\n', f"\n{prefix}") + '"'

def describe_difference(ob1, ob2, prefix=""):
    rv = ""
    # Dictionaries and lists: iterate and recursively look at keys and values and describe the differences
    if isinstance(ob1, dict):
        # aggregated keys
        keys = set(ob1.keys()) | set(ob2.keys())
        for k in keys:
            if k not in ob1:
                rv += f"\n{prefix}Key {k} is new"
            elif k not in ob2:
                rv += f"\n{prefix}Key {k} is gone"
            else:
                rv += describe_difference(ob1[k], ob2[k], f"{prefix}{k}.")
        return rv.strip()

    if isinstance(ob1, list):
        if len(ob1) > len(ob2):
            rv += f"\n{prefix}List shrunk from {len(ob1)} to {len(ob2)}"
        elif len(ob1) < len(ob2):
            rv += f"\n{prefix}List grew from {len(ob1)} to {len(ob2)}"
        for i in range(min(len(ob1), len(ob2))):
            rv += describe_difference(ob1[i], ob2[i], f"{prefix}[{i}].")
        return rv.strip()

    if ob1 == ob2:
        return "Identical" if prefix == "" else ""

    if not isinstance(ob1, str):
        ob1 = str(ob1)
        ob2 = str(ob2)

    # Find the position where the strings differ
    i = 0
    l12 = min(len(ob1), len(ob2))
    while i < l12 and ob1[i] == ob2[i]:
        i += 1
    if i == l12:
        if len(ob1) < l12:
            # ob2 is ob1 with extra stuff
            rv += f"\n{prefix}Appendage: \"{ob2[l12:l12 + 20]}\""
        else:
            # ob1 is ob2 with extra stuff
            rv += f"\n{prefix}Missing: \"{ob1[l12:l12 + 20]}\""
    else:
        rv += difference_expr(f"\n{prefix}Changed at {i}: ", ob1[max(0, i - 20):i + 20], ob2[max(0, i - 20):i + 20])

    return rv.strip()

class HashCache:
    class HashCacheEntry:
        def __init__(self, hash, expiry):
            self.hash = hash
            self.expiry = expiry
        @classmethod
        def decode(self, obj):
            return HashCache.HashCacheEntry(obj["hash"], obj["expiry"])
        def encode(self):
            return {
                "hash": self.hash,
                "expiry": self.expiry,
            }

    def __init__(self, path):
        self.path = path
        self.cache = {} # key -> HashCacheEntry
        if os.path.exists(path):
            with open(path, "r") as f:
                cache = json.load(f)
                for key in cache:
                    self.cache[key] = HashCache.HashCacheEntry.decode(cache[key])
    def save(self):
        with open(self.path, "w") as f:
            json.dump({ key: self.cache[key].encode() for key in self.cache }, f)
    def get(self, key):
        if key in self.cache:
            entry = self.cache[key]
            if entry.expiry > time.time():
                return entry.hash
            self.cache.pop(key)
        return None
    def set(self, key, hash, expiry=3600):
        expiry = time.time() + expiry
        self.cache[key] = HashCache.HashCacheEntry(hash, expiry)
        self.save()

class Stopper:
    def __init__(self):
        pass
    def should_stop(self, token):
        return False
    def reset(self):
        pass

class NGramTailStopper(Stopper):
    """
    The n-gram tail stopper looks at a stream of tokens and requests halt if the last N tokens are found in the last M tokens.
    """
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.buffer = []
        self.window = []

    def __str__(self):
        return f"NGramTailStopper(n={self.n}, m={self.m})"

    def reset(self):
        self.buffer = []
        self.window = []

    def should_stop(self, token):
        self.buffer.append(token)
        if len(self.buffer) < self.n:
            return False
        ngram = "".join(self.buffer[-self.n:])
        if ngram in self.window:
            # print(f"NGTS triggered on ngram {ngram}.")
            # for w in self.window:
            #     print(f"- \"{w}\" {'<--' if w == ngram else ''}")
            return True
        self.window.append(ngram)
        self.buffer.pop(0)
        if len(self.window) > self.m:
            self.window.pop(0)
        return False

class RegexStopper(Stopper):
    def __init__(self, regex, tokens=12):
        self.regex = regex
        self.tokens = tokens
        self.buffer = []

    def __str__(self):
        return f"RegexStopper(regex={self.regex}, tokens={self.tokens})"

    def reset(self):
        self.buffer = []

    def should_stop(self, token):
        self.buffer.append(token)
        if re.search(self.regex, "".join(self.buffer)):
            # print(f"RegexStopper \"{self.regex}\" triggered on token {token} in " + "".join(self.buffer))
            return True
        if len(self.buffer) > self.tokens:
            self.buffer.pop(0)
        return False

class StrStopper(Stopper):
    def __init__(self, stopstrings):
        self.stopstrings = stopstrings
        self.keep = max([len(s) for s in stopstrings])
        self.buffer = ""

    def __str__(self):
        return f"StrStopper(stopstrings={self.stopstrings})"

    def reset(self):
        self.buffer = ""

    def should_stop(self, token):
        self.buffer += token
        for s in self.stopstrings:
            if s in self.buffer:
                # print(f"StrStopper \"{self.stopstrings}\" triggered on token {token} in " + self.buffer)
                # breakpoint()
                return True
        if len(self.buffer) > self.keep:
            self.buffer = self.buffer[-self.keep:]
        return False

def log(s):
    print(f"[{time.ctime()} {s}")

def serializable_kwargs(kwargs):
    # stoppers are not serializable
    rv = {}
    for k, v in kwargs.items():
        if k == "stoppers":
            rv[k] = [str(s) for s in v]
        else:
            rv[k] = v
    return rv

class Metadata:
    def __init__(self):
        self.digest = None
        self.kwargs = {}
        self.llm_hash = None
        self.llm = None
        self.prompt = None
        self.log = []

    @classmethod
    def deserialize(cls, obj) -> Self:
        inst = cls()
        inst.digest = obj["digest"]
        inst.kwargs = obj["kwargs"]
        inst.llm_hash = obj["llm_hash"]
        inst.llm = obj["llm"]
        inst.prompt = obj["prompt"]
        inst.log = obj["log"]
        return inst

    def serialize(self) -> dict:
        return {
            "digest": self.digest,
            "kwargs": self.kwargs,
            "llm_hash": self.llm_hash,
            "llm": self.llm,
            "prompt": self.prompt,
            "log": self.log,
        }

    def pathstr(self, child=None):
        return ""

    def digests(self, prompt, llm_hash, kwargs):
        y = prompt + "|" + str(serializable_kwargs(kwargs))
        z = (llm_hash + "|" + y).encode("utf-8")
        y = y.encode("utf-8")
        self.last_input, self.last_input_fb = z, y
        digest, digest_fb = hashlib.sha256(z).hexdigest(), hashlib.sha256(y).hexdigest()
        return digest, digest_fb

    def process(self, prompt, digest, kwargs):
        kwargs = serializable_kwargs(kwargs)
        if self.digest is None:
            self.digest = digest
            self.llm_hash = self.llm_hash
            self.llm = self.llm
            self.kwargs = kwargs
            self.prompt = prompt
            return True
        elif digest != self.digest:
            ob1 = {
                "kwargs": self.kwargs,
                "llm_hash": self.llm_hash,
                "llm": self.llm,
                "prompt": self.prompt,
            }
            logentry = {}
            logentry["digest"] = {
                "old": self.digest,
                "new": digest
            }
            self.digest = digest
            if self.llm != self.llm:
                logentry["llm"] = {
                    "old": self.llm,
                    "new": self.llm
                }
                self.llm = self.llm
            if self.llm_hash != self.llm_hash:
                logentry["llm_hash"] = {
                    "old": self.llm_hash,
                    "new": self.llm_hash
                }
                self.llm_hash = self.llm_hash
            kv = {}
            for k, v in kwargs.items():
                if k not in self.kwargs:
                    kv[k] = {
                        "old": None,
                        "new": v
                    }
                    self.kwargs[k] = v
                elif self.kwargs[k] != v:
                    kv[k] = {
                        "old": self.kwargs[k],
                        "new": v
                    }
                    self.kwargs[k] = v
            if len(kv) > 0:
                logentry["kwargs"] = kv
            if prompt != self.prompt:
                # Ffwd until we hit a different character
                i = 0
                while i < len(prompt) and i < len(self.prompt) and prompt[i] == self.prompt[i]:
                    i += 1
                # Also rwd until we hit a different character
                j = min(len(prompt), len(self.prompt)) - 1
                while j >= 0 and prompt[j] == self.prompt[j]:
                    j -= 1
                logentry["prompt"] = {
                    "diff": {
                        "start": i,
                        "end": j,
                        "content": prompt[i:j+1]
                    },
                }
                self.prompt = prompt
            ob2 = {
                "kwargs": self.kwargs,
                "llm_hash": self.llm_hash,
                "llm": self.llm,
                "prompt": self.prompt,
            }
            diff_string = describe_difference(ob1, ob2)
            if diff_string == "":
                diff_string = "No changes"
            logentry["diff_str"] = diff_string
            self.log.append(logentry)
            print(f"Metadata changed for {self.pathstr()}:\n{diff_string}")
            # breakpoint()
            return True
        return False

class MetadataRepository(Metadata):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.children = {}
        self.needs_saving = False

    @classmethod
    def deserialize(cls, obj, parent=None) -> Self:
        inst = cls(parent=parent)
        md = obj["metadata"]
        inst.digest = md["digest"]
        inst.kwargs = md["kwargs"]
        inst.llm_hash = md["llm_hash"]
        inst.llm = md["llm"]
        inst.prompt = md["prompt"]
        inst.log = md["log"]
        for k, v in obj["children"].items():
            inst.children[k] = MetadataRepository.deserialize(v, parent=inst)
        return inst

    def process(self, prompt, digest, kwargs):
        if super().process(prompt, digest, kwargs):
            self.needs_saving = True
        return self.needs_saving

    def get_child(self, key) -> Self:
        if key not in self.children:
            self.children[key] = MetadataRepository(parent=self)
        return self.children[key]

    def serialize(self) -> dict:
        self.needs_saving = False
        md = super().serialize()
        children = {}
        for k, v in self.children.items():
            children[k] = v.serialize()
        return {
            "metadata": md,
            "children": children,
        }

    def pathstr(self, child=None):
        if child is None:
            return self.parent.pathstr(self)
        # Find child in values of self.children
        for k, v in self.children.items():
            if v == child:
                return (self.parent.pathstr(self) + "/" if self.parent is not None else "") + k
