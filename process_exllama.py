# Copyright (C) 2023 Charles O. Goddard
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

import glob
import logging
import multiprocessing
import os.path
import shelve
from typing import Dict, Optional

import datasets
import huggingface_hub
import rathe
from exllama.generator import ExLlamaGenerator
from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama.tokenizer import ExLlamaTokenizer
from rathe import PromptFormatter
from rathe.conversion import ConversionContext
from tqdm import tqdm

from common import parse_score


class LlmState:
    config: ExLlamaConfig
    model: ExLlama
    tokenizer: ExLlamaTokenizer
    cache: ExLlamaCache
    generator: ExLlamaGenerator
    formatter: PromptFormatter
    max_reply_tokens: int

    def __init__(
        self,
        model_path: str,
        max_seq_len: int,
        max_reply_tokens: int,
        formatter: PromptFormatter,
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        cuda_device: str = "cuda:0",
    ):
        self.formatter = formatter
        self.max_reply_tokens = max_reply_tokens

        if not os.path.exists(model_path):
            if "@" in model_path:
                (model_path, revision) = model_path.split("@")
            else:
                revision = None
            model_path = huggingface_hub.snapshot_download(
                model_path, revision=revision
            )

        if checkpoint_path is None:
            st_pattern = os.path.join(model_path, "*.safetensors")
            checkpoint_path = glob.glob(st_pattern)[0]
        if config_path is None:
            config_path = os.path.join(model_path, "config.json")
        if tokenizer_path is None:
            tokenizer_path = os.path.join(model_path, "tokenizer.model")

        self.config = ExLlamaConfig(config_path)
        self.config.use_flash_attn_2 = True
        self.config.model_path = checkpoint_path
        self.config.device_map.lm_head = cuda_device
        self.config.device_map.norm = cuda_device
        self.config.device_map.layers = [
            cuda_device
        ] * self.config.device_map.num_layers
        self.config.max_seq_len = max_seq_len

        self.model = ExLlama(self.config)
        self.tokenizer = ExLlamaTokenizer(tokenizer_path)
        self.cache = ExLlamaCache(self.model)
        self.generator = ExLlamaGenerator(self.model, self.tokenizer, self.cache)
        self.generator.settings.temperature = 0.7
        self.generator.settings.top_p = 0.1
        self.generator.settings.token_repetition_penalty_max = 1.18
        self.generator.settings.top_k = 40


def pid() -> int:
    return multiprocessing.current_process()._identity[0]


def init_model(
    model: str, max_seq_len: int, max_reply_tokens: int, formatter: PromptFormatter
):
    global llm
    device = f"cuda:{pid() - 1}"
    logging.warning(f"Initializing on device {device}")

    llm = LlmState(
        model,
        max_seq_len,
        max_reply_tokens,
        formatter,
        cuda_device=device,
    )


def process_single(row: Dict) -> Dict:
    global llm

    inst = rathe.InstructPrompt(row["instruction"], output=None, input=row["input"])
    prefix = (
        llm.formatter.format(
            inst,
            {
                "eos_token": llm.tokenizer.eos_token,
                "bos_token": llm.tokenizer.bos_token,
            },
        ).to_string()
        + "Reasoning:"
    )
    try:
        response = llm.generator.generate_simple(
            prefix, max_new_tokens=llm.max_reply_tokens
        )[len(prefix) :]
    except Exception as e:
        logging.error(f"Failed to process prompt {row['id']}", exc_info=e)
        return {"success": False, "response": None, "id": row["id"]}

    res = parse_score(row["id"], response.strip())
    if not res:
        return {"success": False}
    return {"success": True, **res}


def process_exllama(
    judge_model: str,
    _group_size: int,
    num_gpus: int,
    max_reply_tokens: int,
    formatter: PromptFormatter,
    _conversion_context: ConversionContext,
    data: datasets.Dataset,
    db: shelve.Shelf,
):
    with multiprocessing.Pool(
        num_gpus, init_model, initargs=[judge_model, 4096, max_reply_tokens, formatter]
    ) as p:
        for response in tqdm(
            p.imap(process_single, data), total=len(data), smoothing=0.1
        ):
            if response["success"]:
                del response["success"]
                db[response["id"]] = response
