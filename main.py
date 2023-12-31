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
import multiprocessing
import os.path
import shelve
from typing import List

import rathe
import torch
import typer
from rathe.conversion import ConversionContext
from typing_extensions import Annotated

import prompts
from common import load_data
from process_exllama import process_exllama
from process_vllm import process_vllm

MAX_REPLY_TOKENS = 300


def main(
    dataset: Annotated[str, typer.Argument(help="HuggingFace dataset to critique")],
    data_file: Annotated[List[str], typer.Option(help="List of files to include")] = [],
    prompt_parser: Annotated[str, typer.Option(help="Parser for dataset")] = "alpaca",
    prompt_format: Annotated[
        str, typer.Option(help="Prompt format used by judge model")
    ] = "alpaca",
    judge_type: Annotated[
        str, typer.Option(help="assistant, rp, or commitpack")
    ] = "assistant",
    judge_model: Annotated[
        str, typer.Option(help="Model to use")
    ] = "TheBloke/OpenOrca-Platypus2-13B-GPTQ@gptq-4bit-32g-actorder_True",
    engine: Annotated[str, typer.Option(help="exllama or vllm")] = "exllama",
    vllm_group_size: Annotated[
        int, typer.Option(help="Number of examples to feed vLLM at once")
    ] = 128,
    offset: int = -1,
    last_index: int = -1,
    shuffle: bool = False,
    num_gpus: int = -1,
):
    logging.basicConfig(level=logging.INFO)
    if not data_file:
        data_file = None

    if num_gpus < 1:
        num_gpus = torch.cuda.device_count()

    get_judge_prompt = prompts.get_judge_prompt_fn(judge_type)
    parser = rathe.get_parser(prompt_parser)
    formatter = rathe.get_formatter(prompt_format)

    conversion_context = ConversionContext.default()

    if engine == "exllama":
        process_fn = process_exllama
    elif engine == "vllm":
        process_fn = process_vllm
    else:
        raise RuntimeError(f"Unknown engine {engine} - must be exllama or vllm")

    data = load_data(
        dataset,
        data_files=data_file,
        parser=parser,
        get_judge_prompt=get_judge_prompt,
        conversion_context=conversion_context,
        offset=offset,
        last_index=last_index,
        shuffle=shuffle,
    )
    dbname = "ratings_" + dataset.replace("/", "_").replace(":", "_")
    with shelve.open(dbname) as db:
        already_processed = set(list(db.keys()))
        if already_processed:
            logging.info("filtering out already processed...")
            data = data.filter(
                lambda e: e["id"] not in already_processed, num_proc=os.cpu_count()
            )

        process_fn(
            judge_model,
            vllm_group_size,
            num_gpus,
            MAX_REPLY_TOKENS,
            formatter,
            conversion_context,
            data,
            db,
        )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    typer.run(main)
