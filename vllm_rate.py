import logging
import multiprocessing
import os.path
import re
import shelve
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import datasets
import rathe
import torch
import typer
import vllm
from rathe import InstructPrompt, Prompt
from rathe.conversion import ConversionContext
from tqdm import tqdm

import prompts

MAX_RESPONSE_LEN = 300
SCORE_RE = re.compile(r"[sS]core: ([0-5](\.5)?)")


def alpaca_json(prompt: rathe.InstructPrompt) -> Dict:
    return {
        "instruction": prompt.instruction,
        "input": prompt.input,
        "output": prompt.output,
    }


def load_data(
    dataset: str,
    parser: rathe.PromptParser,
    get_judge_prompt: Callable[[Prompt, ConversionContext], InstructPrompt],
    conversion_context: Optional[ConversionContext] = None,
    data_files: Optional[List[str]] = None,
    offset: int = -1,
    last_index: int = -1,
) -> datasets.Dataset:
    logging.info("loading dataset...")
    data = datasets.load_dataset(dataset, data_files=data_files)
    if "train" in data:
        data = data["train"]

    if "id" not in data.column_names:
        data = data.map(
            lambda e, idx: {"id": f"{dataset}.{idx}", **e}, with_indices=True
        )

    if offset > 0 or last_index >= 0:
        if last_index < 0:
            last_index = len(data)
        if offset < 0:
            offset = 0
        logging.info(f"selecting samples from {offset} to {last_index}")
        data = data.select(range(offset, last_index))

    if conversion_context is None:
        conversion_context = ConversionContext.default()

    logging.info("parsing and formatting prompts...")
    t = PromptTransform(parser, get_judge_prompt, conversion_context)
    data = data.map(
        t,
        num_proc=os.cpu_count(),
    )
    return data


@dataclass
class PromptTransform:
    parser: rathe.PromptParser
    get_judge_prompt: Callable[[Prompt, ConversionContext], InstructPrompt]
    conversion_context: Optional[ConversionContext]

    def __call__(self, row: Dict) -> Dict:
        parsed = self.parser.parse(row)
        judge_prompt = self.get_judge_prompt(parsed, self.conversion_context)
        res = alpaca_json(judge_prompt)
        res["id"] = row["id"]
        return res


def main(
    dataset: str,
    data_file: List[str] = [],
    prompt_parser: str = "orca",
    prompt_format: str = "alpaca",
    judge_type: str = "assistant",
    judge_model: str = "Open-Orca/OpenOrca-Platypus2-13B",
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

    data = load_data(
        dataset,
        data_files=data_file,
        parser=parser,
        get_judge_prompt=get_judge_prompt,
        conversion_context=conversion_context,
        offset=offset,
        last_index=last_index,
    )

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
        max_tokens=MAX_RESPONSE_LEN,
    )

    dbname = "ratings_" + dataset.replace("/", "_").replace(":", "_")
    with shelve.open(dbname) as db:
        already_processed = set(list(db.keys()))
        if already_processed:
            logging.info("filtering out already processed...")
            data = data.filter(
                lambda e: e["id"] not in already_processed, num_proc=os.cpu_count()
            )

        for idx0 in tqdm(range(0, len(data) + 1 - batch_size, batch_size)):
            prompt_text = []
            for idx in range(idx0, min(idx0 + batch_size, len(data))):
                row = data[idx]
                inst = InstructPrompt(
                    row["instruction"], output=None, input=row["input"]
                )
                prompt_text.append(
                    formatter.format(
                        inst, special_tokens, conversion_context
                    ).to_string()
                    + "Reasoning:"
                )

            outputs = llm.generate(
                prompt_text, sampling_params=sampling_params, use_tqdm=False
            )
            for i, output in enumerate(outputs):
                row_id = data[idx]["id"]
                if not output.finished:
                    logging.warning(f"Generation ran too long for {row_id}")
                    continue

                response_text = output.outputs[0].text
                if response_text.startswith(prompt_text[i]):
                    response_text = response_text[len(prompt_text[i]) :]

                match = SCORE_RE.search(response_text)
                if match:
                    score = float(match.group(1))
                    db[row_id] = {
                        "id": row_id,
                        "score": score,
                        "response": response_text,
                    }
                else:
                    logging.warning(f"Failed to parse score for {row_id}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    typer.run(main)
