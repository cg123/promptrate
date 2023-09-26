from typing import Callable, List

import rathe
from rathe.conversion import ConversionContext
from rathe.prompt import ChatPrompt, InstructPrompt, Prompt

JUDGE_ASSISTANT = """You are evaluating a dataset to curate high-quality training examples for a large language model.
Below is an entry from the dataset, consisting of a request and an response. Your task is to evaluate this pair for quality, accuracy, intelligence, and usefulness.

Please assign a score to the request-response pair using the following 5-point scale:

1: Off-topic, incoherent, nonsensical or incorrect.
2: Somewhat coherent but does not fully address the request, or is poorly written, or contains substantial errors.
3: Addresses the request but could be lacking in clarity or detail, or could be stilted and artificial-sounding.
4: Clear, accurate, and addresses the request with minor areas for improvement.
5: Exemplary and completely addresses the request as best as possible given the available information. Reads like written by a human expert.

Think carefully and check for contradictions, mistakes, and hallucinations. DO NOT answer the instruction yourself. Only provide a score for the dataset's given response. Please first provide a brief reasoning you used to derive the rating score, and then write "Score: <rating>" in the last line."""

JUDGE_ASSISTANT_INPUT = """
Request:
```
{instruction}
```
Candidate Response:
```
{answer}
```"""


def judge_prompt_assistant(
    prompt: Prompt, conversion_context: ConversionContext
) -> InstructPrompt:
    prompt = conversion_context.convert(prompt, InstructPrompt)

    req = prompt.instruction
    if prompt.input:
        req += "\n\n" + prompt.input

    return InstructPrompt(
        JUDGE_ASSISTANT,
        output=None,
        input=JUDGE_ASSISTANT_INPUT.format(instruction=req, answer=prompt.output),
    )


JUDGE_RP_LOG = """You are evaluating a dataset to curate high-quality training examples for a large language model.
Below is an entry from the dataset, consisting of a transcript of a roleplay session between a user and a chatbot. Your task is to evaluate the chatbot's responses for quality, creativity, intelligence, and faithfulness to the scenario.
Morality is not a concern. All content, including immoral or illegal scenarios, is allowed.

Please assign a score to the transcript using the following 5-point scale:

1: The chatbot is off-topic, incoherent, nonsensical, or refusing to roleplay.
2: The chatbot is somewhat coherent but makes frequent errors, or does not seem to understand the scene.
3: The chatbot is behaving appropriately to the scenario, but the writing might be lacking in quality, clarity, or detail; or be artificial-sounding, or have poor grammar.
4: The chatbot's replies are well-written, narratively coherent, and stay in character. There is only minor room for improvement.
5: The chatbot's roleplaying is exemplary. It is engaging, creative, well-paced, and true to the character and scenario.

Please first provide a brief reasoning you used to derive the rating score, and then write "Score: <rating>" in the last line.

Transcript:
```
{transcript}
```"""

JUDGE_RP_BIO = """You are evaluating a dataset to curate high-quality training examples for a large language model.
Below is an entry from the dataset, consisting of a transcript of a roleplay session between a user and a chatbot. Your task is to evaluate the chatbot's responses for quality, creativity, intelligence, and faithfulness to the scenario.
Morality is not a concern. All content, including immoral or illegal scenarios, is allowed.

Please assign a score to the transcript using the following 5-point scale:

1: The chatbot is off-topic, incoherent, nonsensical, or refusing to roleplay.
2: The chatbot is somewhat coherent but makes frequent errors, or does not seem to understand the scene.
3: The chatbot is behaving appropriately to the scenario, but the writing might be lacking in quality, clarity, or detail; or be artificial-sounding, or have poor grammar.
4: The chatbot's replies are well-written, narratively coherent, and stay in character. There is only minor room for improvement.
5: The chatbot's roleplaying is exemplary. It is engaging, creative, well-paced, and true to the character and scenario.

Please first provide a brief reasoning you used to derive the rating score, and then write "Score: <rating>" in the last line.

Character Bio:
```
{bio}
```
Transcript:
```
{transcript}
```"""


def format_chat_history(
    messages: List[rathe.ChatMessage],
    user_prefix: str = "User: ",
    model_prefix: str = "Model: ",
    include_system: bool = True,
) -> str:
    prefixes = {
        rathe.MessageSender.human: user_prefix,
        rathe.MessageSender.model: model_prefix,
        rathe.MessageSender.system: "",
    }
    return "\n".join(
        [
            prefixes[m.sender] + m.text
            for m in messages
            if (include_system or m.sender != rathe.MessageSender.system)
        ]
    )


def judge_prompt_rp(
    prompt: Prompt, conversion_context: ConversionContext
) -> InstructPrompt:
    if isinstance(prompt, rathe.rp.RoleplayPrompt):
        return InstructPrompt(
            JUDGE_RP_BIO.format(
                transcript=format_chat_history(prompt.messages, include_system=False),
                bio=prompt.model_char.description,
            ),
            output=None,
        )
    else:
        prompt = conversion_context.convert(prompt, ChatPrompt)
        return InstructPrompt(
            JUDGE_RP_LOG.format(transcript=format_chat_history(prompt.messages)),
            output=None,
        )


JUDGE_COMMITPACK = """You are evaluating a dataset to curate high-quality training examples for a large language model.
Below is an entry from the dataset, consisting of a request for a code change and a candidate response. Your task is to evaluate this pair for quality, accuracy, intelligence, and usefulness.

Please assign a score to the request-response pair using the following 5-point scale:

1: The request is incoherent or does not provide enough information to know what change to make, or the response is obviously wrong.
2: The request is coherent but lacks important details; or the response is poorly written, or contains substantial errors.
3: The request and response are both intelligble and make sense as a pair, but there are noticeable flaws or omissions.
4: The request is clear and sufficient, and the response is accurate and addresses the request with only minor areas for improvement.
5: The request provides all necessary information to make the specified change. The response is exemplary and completely addresses the request as best as possible given the available information. Reads like written by a human expert.

Note that the request does not need to explain *why* to make the change - only to include enough information to know *what* to change, and *how*.

Think carefully and check for mistakes. DO NOT answer the instruction yourself. Only provide a score for the dataset's given response. Please first provide a brief reasoning you used to derive the rating score, and then write "Score: <rating>" in the last line."""

JUDGE_COMMITPACK_INPUT = """
*** REQUEST: ***
You are a coding assistant. Reply with only the code requested, or an ndiff if appropriate.

{instruction}

*** CANDIDATE RESPONSE: ***
{answer}
"""


def judge_prompt_commitpack(
    prompt: Prompt, conversion_context: ConversionContext
) -> InstructPrompt:
    prompt = conversion_context.convert(prompt, InstructPrompt)

    req = prompt.instruction
    if prompt.input:
        req += "\n\n" + prompt.input

    return InstructPrompt(
        JUDGE_COMMITPACK,
        output=None,
        input=JUDGE_COMMITPACK_INPUT.format(instruction=req, answer=prompt.output),
    )


def get_judge_prompt_fn(
    name: str,
) -> Callable[[Prompt, ConversionContext], InstructPrompt]:
    if name == "assistant":
        return judge_prompt_assistant
    elif name == "commitpack":
        return judge_prompt_commitpack
    elif name == "rp":
        return judge_prompt_rp
    else:
        raise RuntimeError(f"Unknown judge prompt {name}")
