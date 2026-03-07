import os
import logging
from transformers import AutoTokenizer


class TokenizerWrapper:
    def __init__(self, tokenizer_name, tokenizer_revision=None, trust_remote_code=False):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            revision=tokenizer_revision or "main",
            trust_remote_code=trust_remote_code,
        )
        self.custom_chat_template = os.getenv("CUSTOM_CHAT_TEMPLATE")
        self.has_chat_template = bool(self.tokenizer.chat_template) or bool(
            self.custom_chat_template
        )
        if self.custom_chat_template and isinstance(self.custom_chat_template, str):
            self.tokenizer.chat_template = self.custom_chat_template

    def apply_chat_template(self, input):
        if isinstance(input, list):
            if not self.has_chat_template:
                raise ValueError(
                    "Chat template does not exist for this model, you must provide a single string input instead of a list of messages"
                )
        elif isinstance(input, str):
            input = [{"role": "user", "content": input}]
        else:
            raise ValueError("Input must be a string or a list of messages")

        return self.tokenizer.apply_chat_template(
            input, tokenize=False, add_generation_prompt=True
        )
