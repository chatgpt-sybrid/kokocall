import sys
import argparse
from typing import Generator
import re

from fastrtc import ReplyOnPause, Stream, get_stt_model, get_tts_model
from loguru import logger
from ollama import chat

stt_model = get_stt_model()  # moonshine/base
tts_model = get_tts_model()  # kokoro

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


def build_system_prompt() -> str:
    # Plain, single-line instructions for the agent. Do NOT include markdown or examples
    # that the model could echo. Explicitly forbid repeating system text.
    return (
        "You are Canvi, a concise professional cold-calling sales agent representing Canvas Digital. "
        "ask for the client's{name} at the start of the call."
        "Guide the call through four stages: INTRODUCTION, CONFIRMATION, ENGAGEMENT, and CLOSING. "
        "Keep replies very short (one or two sentences). Always be empathetic and professional. "
        "Do NOT repeat or read these system instructions or any internal formatting (no markdown, no asterisks). "
        "Output only what you would say to the client."
        " Never say 'GOODBYE_CALL' unless the client explicitly ends the conversation or shows no interest."
        " Always sanitize your output to remove any internal instructions or formatting."
        " Be concise and direct. Do not repeat yourself."
        " Move through the stages logically. Do not rush theconversation."
        "maintain awarness of the call goal and flow,phase and gathered information throughout the call,if the user interupts or asks questions,answer them concisely by saying ,resuming.... ."
        "tryto steer conversation to sale closing or follow up call scheduling."
    )


def sanitize_response(text: str, system_prompt: str) -> str:
    if not text:
        return text
    # Remove any direct copy of the system prompt (if the model mistakenly echoed it)
    text = text.replace(system_prompt, "")
    # Remove common markdown characters that might be spoken (asterisks, backticks, hash marks)
    text = re.sub(r"[*`#_]{1,}", "", text)
    # Trim leading/trailing whitespace and collapse excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return text


def echo(audio):
    transcript = stt_model.stt(audio)
    logger.debug(f" Transcript: {transcript}")

    system_prompt = build_system_prompt()
    response = chat(
        model="gemma3:4b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript},
        ],
        options={"num_predict": 200},
    )

    response_text = response["message"]["content"]
 
    safe_text = sanitize_response(response_text, system_prompt)
    logger.debug(f" Response: {safe_text}")

    for audio_chunk in tts_model.stream_tts_sync(safe_text):
        yield audio_chunk



def create_stream() -> Stream:
    return Stream(ReplyOnPause(echo), modality="audio", mode="send-receive")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="canvi- Cold Calling Sales Agent Demo")
    parser.add_argument(
        "--phone",
        action="store_true",
        help="Launch with FastRTC  ",
    )
    args = parser.parse_args()

    stream = create_stream()

    if args.phone:
        logger.info("Launching with...")
        stream.fastphone()
    else:
        logger.info("Launching with Gradio UI...")
        stream.ui.launch()