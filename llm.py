import json
import os
import subprocess
import sys
import time
import requests
import atexit
from transformers import AutoTokenizer
from utils import HashCache, Metadata, StrStopper, log, model_quick_hash

class LLMService:
    def __init__(self, llm, args):
        self.server = None
        self.args = args
        self.hcache = HashCache(".hashcache.json")
        self.llm = llm
        self.server_log = open("server.log", "w")
        self.debug = False
        self.infoln = lambda s: None
        self.last_prompt = None
        self.info = lambda s: None
        self.forbid_start = args.no_llm
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
        }
        if "verbose" in args and args.verbose:
            self.infoln = lambda s: print(s)
            self.info = lambda s: print(s, end="", flush=True)

        self.default_prompt_template = "### Instruction: {instruction}\n\n### Response: "
        self.prompt_template = self.default_prompt_template
        self.llm_hash = self.get_llm_hash(llm)
        # Make sure "sample-outputs" directory exists
        if not os.path.exists("sample-outputs"):
            os.makedirs("sample-outputs")

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained("/".join(self.args.llm.split("/")[:-1]))

    def get_llm_hash(self, llm):
        if self.hcache.get(llm) is None:
            self.infoln(f"Deriving LLM hash for {llm}...")
            self.hcache.set(llm, model_quick_hash(llm))
            print(f"LLM hash: {self.hcache.get(llm)}")
        return self.hcache.get(llm)

    def get_sample_path(self, digest):
        return os.path.join("sample-outputs", digest)

    def get_sample(self, digest):
        sample_output_path = self.get_sample_path(digest)
        if os.path.exists(sample_output_path) and os.path.getsize(sample_output_path) >= 1:
            with open(sample_output_path, "r") as f:
                return f.read()
        return None

    def run_completion(self, prompt, **kwargs):
        force_output = kwargs["ignore_eos"] if "ignore_eos" in kwargs else False
        # Lambda function that pulls defaults from kwargs["params"] if it is not None, otherwise from self.args
        if "params" in kwargs:
            defaults = lambda key: kwargs[key] if key in kwargs else kwargs["params"][key] if key in kwargs["params"] else self.args[key]  # noqa: E731
        else:
            defaults = lambda key: kwargs[key] if key in kwargs else getattr(self.args, key, None)  # noqa: E731
        stop_sequence = ["### ", "\n#", "</s>", "<|im_end|>", "[INST]", "<|eot_id|>"] if not force_output else []
        if "stop_sequence_additional" in kwargs:
            stop_sequence += kwargs["stop_sequence_additional"]
        stoppers = []
        if "stoppers" in kwargs:
            stoppers = kwargs["stoppers"]
            for stopper in stoppers:
                stopper.reset()
        stoppers.append(StrStopper(stopstrings=stop_sequence))
        data = {
            "sampler_seed": defaults("seed"),
            "max_length": defaults("predict"),
            "temperature": defaults("temperature"),
            "mirostat": defaults("mirostat"),
            "mirostat_tau": defaults("mirostat_tau"),
            "mirostat_eta": defaults("mirostat_eta"),
            "rep_pen": defaults("rep_pen"),
            "rep_pen_range": defaults("rep_pen_range"),
            "top_a": defaults("top_a"),
            "top_k": defaults("top_k"),
            "top_p": defaults("top_p"),
            "min_p": defaults("min_p"),
            "smoothing_factor": defaults("smoothing_factor"),
            "use_default_badwordsids": not force_output,
            "prompt": prompt,
            "stop_sequence": stop_sequence,
        }
        # verify data
        assert isinstance(data["temperature"], float) or isinstance(data["temperature"], int)
        # self.debug = True
        if self.debug:
            print(f"run_completion :: {data}")
        s = requests.Session()
        response = s.post(
            "http://localhost:5001/api/extra/generate/stream",
            json=data,
            stream=True,
        )
        for chunk in response.iter_lines():
            s = chunk.decode('utf8')
            if self.debug: print(f"-- chunk :: '{s}'")  # noqa: E701
            if s == '': continue  # noqa: E701
            # event: message
            # data: {data}
            if s == "event: message":
                if self.debug: print("- evt msg! -- okay all right")  # noqa: E701
                continue
            elif s.startswith("data: "):
                if self.debug: print("- data! -- okay all right")  # noqa: E701
                if s and len(s) > 0:
                    token = json.loads(s[5:])["token"]
                    yield token # we want to yield it before stopping, as it may be a part of the answer
                    if any([stopper.should_stop(token) for stopper in stoppers]):
                        response.close()
                        # print(f"Stopper triggered on token {token}")
                        # breakpoint()
                        return
            else:
                if self.debug: print(f"- whatta!!! '{s}'")  # noqa: E701
        if self.debug: print("finished yielding shit")  # noqa: E701

    def finish_completion(self, prompt, echo=False, **kwargs):
        v = ""
        for chunk in self.run_completion(prompt, **kwargs):
            v += chunk
            if echo:
                self.info(chunk)
        if v == "":
            # Retry once, after a second
            time.sleep(1)
            for chunk in self.run_completion(prompt, **kwargs):
                v += chunk
                if echo:
                    self.info(chunk)
        if echo:
            self.info("\n")

        stop_sequence = ["### ", "\n#", "</s>", "<|im_end|>", "[INST]", "<|eot_id|>"]
        if "stop_sequence_additional" in kwargs:
            stop_sequence += kwargs["stop_sequence_additional"]

        for i in stop_sequence:
            if v.endswith(i):
                v = v[:-len(i)].strip()
                break
        return v

    def get_cached_completion(self, prompt: str, allow_fallback: bool, metadata: Metadata, **kwargs):
        digest, digest_fb = metadata.digests(prompt, self.llm_hash, kwargs)
        sample = self.get_sample(digest)
        metadata.process(prompt, digest, kwargs)
        if sample is None and allow_fallback:
            sample = self.get_sample(digest_fb)
        return sample, digest, digest_fb

    def cachable_finish_completion(self, prompt: str, metadata: Metadata, echo: bool=False, ignore_cache: bool=False, allow_fallback=False, **kwargs) -> str:
        sample, digest, digest_fb = self.get_cached_completion(prompt, metadata=metadata, allow_fallback=allow_fallback, **kwargs)
        self.last_was_cached = sample is not None and not ignore_cache
        # z = (self.llm_hash + "|" + prompt + "|" + str(kwargs))
        # print(f"{digest} :: {sample} 【{z}】")
        if sample is None or ignore_cache:
            self.stats["cache_misses"] += 1
            if self.server is None:
                # Start the server
                self.start_server()
            if self.last_prompt is not None:
                rv = ""
                same = True
                for i in range(min(len(prompt), len(self.last_prompt))):
                    if not same and prompt[i] == '\n':
                        break
                    if prompt[i] != self.last_prompt[i] and same:
                        # Truncate rv down to last \n
                        rv = f"[{i}] " + rv.split("\n")[-1]
                        rv += "\033[31m"
                        same = False
                    rv += prompt[i]
                print(f"Prompt change: {rv}\033[0m")
            self.last_prompt = prompt
            sample = self.finish_completion(prompt, echo=echo, **kwargs)
            with open(self.get_sample_path(digest), "w") as f:
                f.write(sample)
            with open(self.get_sample_path(digest_fb), "w") as f:
                f.write(sample)
        else:
            if echo:
                print(sample)
            self.stats["cache_hits"] += 1
        return sample

    def reset_stats(self):
        self.stats["cache_hits"] = 0
        self.stats["cache_misses"] = 0

    def format_prompt(self, instruction):
        return self.prompt_template.format(instruction=instruction)

    def start_server(self, **kwargs):
        if self.forbid_start:
            raise Exception("Server start is forbidden.")
        if self.server is not None:
            self.stop_server()
        self.server = subprocess.Popen([
            "python",
            "koboldcpp/koboldcpp.py",
            kwargs["llm"] if "llm" in kwargs else self.llm,
            "--threads", "16",
            "--highpriority",
            "--debugmode",
            "--debug",
            "--skiplauncher",
            "--gpulayers", str(self.args.ngl),
            "--contextsize", str(self.args.contextsize),
            "--flashattention",
            "--usecublas",
            "--quantkv", str(self.args.quantkv),
        # self.server = subprocess.Popen([
        #     "./server",
        #     "-m", kwargs["llm"] if "llm" in kwargs else self.llm,
        #     "-c", "4096",
        #     "-t", "8",
        #     "-ngl", str(self.args.ngl)
        ], stdout=self.server_log, stderr=self.server_log)
        atexit.register(self.server.terminate)

        # Wait for server to start
        self.infoln("Waiting for server to start...")
        # ctr = 0
        while True:
            if self.server.poll() is not None:
                log("Server failed to start.")
                try:
                    print(self.server.stdout.read())
                except:  # noqa: E722
                    pass
                try:
                    print(self.server.stderr.read())
                except:  # noqa: E722
                    pass
                sys.exit(1)
            try:
                # print(f"Trying to finish completion")
                self.finish_completion(self.format_prompt("1+1"), predict=4, echo=True)
                break
            except:  # noqa: E722
                # import traceback
                # type, value, _ = sys.exc_info()
                # print(f"That failed: {type}, {value}")
                # traceback.print_exc()
                self.info(".")
                time.sleep(5)
                pass
            # ctr += 1
            # if ctr > 3:
            #     self.finish_completion(self.format_prompt("1+1"), predict=4, echo=True)
        self.infoln("\nServer started.")

    def stop_server(self):
        if self.server is not None:
            atexit.unregister(self.server.terminate)
            self.server.terminate()
            self.server.wait()
            self.server = None

    @classmethod
    def PrepareParser(cls, parser, defaults=None):
        if defaults is None:
            defaults = {}
        parser.add_argument("--contextsize", type=int, default=8192, help="The maximum context size to use for the LLM model.")
        parser.add_argument("--llm", default="./llm.gguf", help="The LLM model to use.")
        parser.add_argument("--verbose", action="store_true", help="Print verbose output.")
        parser.add_argument("--debug", action="store_true", help="Print debug output.")
        parser.add_argument(
            "--seed",
            type=int,
            default=12318731,
            help="Seed for random number generator. XOR'd with first 4 bytes of sha256 of evaluation LLM.",
        )
        parser.add_argument(
            "--no-llm",
            action="store_true",
            help="Disable LLM usage.",
        )

        def i(name, d, h):
            parser.add_argument("--" + name, type=int, default=d if name not in defaults else defaults[name], help=h)
        def f(name, d, h):
            parser.add_argument("--" + name, type=float, default=d if name not in defaults else defaults[name], help=h)

        i("ngl", 90, "Number of layers to load on GPU.")
        f("temperature", 0.2, "Temperature for sampling.")
        i("mirostat", 2, "0-2, where 0 is disabled.")
        f("mirostat-tau", 5.0, "Mirostat tau.")
        f("mirostat-eta", 0.1, "Mirostat eta.")
        # arg top_a defaults to 0
        f("top-a", 0, "Top-k sampling parameter.")
        f("top-p", 1, "Top-p sampling parameter.")
        f("top-k", 0, "Top-k sampling parameter.")
        f("rep-pen", 1.01, "Repetition penalty.")
        i("rep-pen-range", 1024, "Repetition penalty range.")
        f("min-p", 0.05, "Minimum probability.")
        f("smoothing-factor", 0.3, "Smoothing factor.")
        i("predict", 512, "Number of tokens to predict.")
        i("quantkv", 0, "KV quantization level (0=f16, 1=q8, 2=q4)")
