import abc
import dataclasses
from typing import Sequence

import transformers

TEXT_FORMATTER = {'function_calling', 'guanaco_oasst', 'alpaca'}


class TextFormatter(abc.ABC):
    @abc.abstractmethod
    def __call__(self, dict_data: dict):
        raise NotImplementedError


@dataclasses.dataclass
class FunctionCallingTextFormatter(TextFormatter):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, list_dict_data: Sequence[dict]):
        text = ""
        for dict_data in list_dict_data:
            content, role = dict_data['content'], dict_data['role']
            if role == "system":
                text += f"SYSTEM: {content}\n\n"
            elif role == "user":
                text += f"USER: {content} {self.tokenizer.eos_token} "
            elif role == "assistant":
                text += f"ASSISTANT: {content} {self.tokenizer.eos_token} "
            else:
                raise ValueError("role must be one of 'system', 'user', 'assistant'")

        last_role = list_dict_data[-1]['role']
        if last_role == "user":
            text += "ASSISTANT:"
        elif last_role in ("assistant", "system"):
            text += "USER:"
        else:
            raise ValueError("role must be one of 'system', 'user', 'assistant'")
        return text
