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

import logging
import shelve

import datasets
import vllm
from rathe import InstructPrompt, PromptFormatter
from rathe.conversion import ConversionContext
from tqdm import tqdm

from common import parse_score


def process_vllm(
    judge_model: str,
    group_size: int,
    num_gpus: int,
    max_reply_tokens: int,
    formatter: PromptFormatter,
    conversion_context: ConversionContext,
    data: datasets.Dataset,
    db: shelve.Shelf,
):
    llm = vllm.LLM(
        judge_model, tensor_parallel_size=num_gpus, max_num_batched_tokens=4096
    )
    special_tokens = llm.get_tokenizer().special_tokens_map

    sampling_params = vllm.SamplingParams(
        presence_penalty=0.1,
        frequency_penalty=1.12,
        temperature=0.7,
        top_p=0.1,
        top_k=40,
        max_tokens=max_reply_tokens,
    )

    for idx0 in tqdm(range(0, len(data) + 1 - group_size, group_size)):
        prompt_text = []
        for idx in range(idx0, min(idx0 + group_size, len(data))):
            row = data[idx]
            inst = InstructPrompt(row["instruction"], output=None, input=row["input"])
            prompt_text.append(
                formatter.format(inst, special_tokens, conversion_context).to_string()
                + "Reasoning:"
            )

        outputs = llm.generate(
            prompt_text, sampling_params=sampling_params, use_tqdm=False
        )
        for i, output in enumerate(outputs):
            row_id = data[idx0 + i]["id"]
            if not output.finished:
                logging.warning(f"Generation ran too long for {row_id}")
                continue

            response_text = output.outputs[0].text
            if response_text.startswith(prompt_text[i]):
                response_text = response_text[len(prompt_text[i]) :]

            result = parse_score(row_id, response_text)
            if result:
                db[row_id] = result
            else:
                logging.warning(f"Failed to parse score for {row_id}")
