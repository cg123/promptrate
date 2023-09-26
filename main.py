import logging
import multiprocessing
import os.path
import shelve
from typing import List

import rathe
import torch
import typer
from rathe.conversion import ConversionContext

import prompts
from common import load_data
from process_vllm import process_vllm
from process_exllama import process_exllama


MAX_REPLY_TOKENS = 300


def main(
    dataset: str,
    data_file: List[str] = [],
    prompt_parser: str = "orca",
    prompt_format: str = "alpaca",
    judge_type: str = "assistant",
    judge_model: str = "Open-Orca/OpenOrca-Platypus2-13B",
    engine: str = "exllama",
    offset: int = -1,
    last_index: int = -1,
    batch_size: int = 128,
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
            batch_size,
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
