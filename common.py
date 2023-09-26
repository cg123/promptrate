import logging
import os.path
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import datasets
import rathe
from rathe import InstructPrompt, Prompt, PromptParser
from rathe.conversion import ConversionContext

SCORE_RE = re.compile(r"[sS]core: ([0-5](\.5)?)")


def alpaca_json(prompt: rathe.InstructPrompt) -> Dict:
    return {
        "instruction": prompt.instruction,
        "input": prompt.input,
        "output": prompt.output,
    }


def load_data(
    dataset: str,
    parser: PromptParser,
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


def parse_score(row_id: str, response_text: str):
    match = SCORE_RE.search(response_text)
    if match:
        score = float(match.group(1))
        return {
            "id": row_id,
            "score": score,
            "response": response_text,
        }
    return None
