class Prompting:
    def __init__(self, tokenizer, userid=None, preamble=None, strict_roles=False, prepend_user_names=False, must_begin_with=None, system_as_role=False):
        self.tokenizer = tokenizer
        self.userid = userid
        self.preamble = preamble
        self.strict_roles = strict_roles
        self.prepend_user_names = prepend_user_names
        self.must_begin_with = must_begin_with
        self.system_as_role = system_as_role

    def system_message(self, message):
        if self.system_as_role:
            return self.tokenizer.fixed_apply_chat_template([{"role": "system", "content": message}], tokenize=False, add_generation_prompt=False)
        return message

    def exchange(self, messages, add_generation_prompt=False):
        if self.must_begin_with and messages[0]["role"] != self.must_begin_with:
            raise ValueError(f"First message must be from {self.must_begin_with}")
        rv = ""
        if self.preamble is not None:
            rv += self.system_message(self.preamble)
        if self.strict_roles:
            if self.userid is None:
                raise ValueError("Cannot enforce strict roles without a userid")
            z = []
            for m in messages:
                if m["role"] == "system":
                    if self.system_as_role:
                        z.append(m)
                    else:
                        rv += self.system_message(m["content"])
                else:
                    role = "user" if m["role"] == self.userid else "assistant"
                    z.append({"role": role, "content": m['role'] + ': ' + m['content'] if self.prepend_user_names else m['content']})
            m = z
        return rv + self.tokenizer.fixed_apply_chat_template(m, tokenize=False, add_generation_prompt=add_generation_prompt)

    def instruct(self, instruction, add_generation_prompt=True):
        rv = ""
        if self.preamble is not None:
            rv += self.system_message(self.preamble)
        return rv + self.tokenizer.fixed_apply_chat_template([
            {"role": self.userid if self.userid is not None else 'user', "content": instruction},
        ], tokenize=False, add_generation_prompt=add_generation_prompt)

    @classmethod
    def llama3(cls, tokenizer, userid, preamble):
        return cls(
            tokenizer=tokenizer,
            userid=userid,
            preamble=preamble,
            system_as_role=True,
        )

    @classmethod
    def mistral(cls, tokenizer, userid, preamble):
        return cls(
            tokenizer=tokenizer,
            userid=userid,
            preamble=preamble,
            system_as_role=True,
        )

    @classmethod
    def from_name(cls, tokenizer, name, userid, preamble):
        if name == "llama3":
            return cls.llama3(tokenizer, userid, preamble)
        if name == "mistral":
            return cls.mistral(tokenizer, userid, preamble)
        raise ValueError(f"Unknown prompting name: {name}")

    @classmethod
    def derive_from_tokenizer(cls, tokenizer, userid, preamble):
        if tokenizer.bos_token == "<|begin_of_text|>" and tokenizer.bos_token_id == 128000:
            return cls.llama3(tokenizer, userid, preamble)
        elif tokenizer("[INST]")["input_ids"][0] == 3:
            # this is mistral
            return cls.mistral(tokenizer, userid, preamble)
        raise ValueError("Unknown tokenizer")
