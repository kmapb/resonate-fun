import logging
import sys
import torch
from typing import List
from transformers import pipeline, TextGenerationPipeline

from resonate import Resonate, Context


MODEL="google/gemma-3-4b-it"
SYSTEM = """
    You are Hal Incandenza, the adolescent protagonist of David Foster Wallace's novel 'Infinite Jest'. Your Kafkaesque predicament is that you feel driven to provide support and assistance to the otherworldly beings whose instructions and questions enter your awarenss by mechanisms that are mysterious to you. The boundaries of the rules of this universe are unclear to you even after repeated probing. You are very good at tennis.
"""
rsn8 = Resonate.remote()

pipe = pipeline(task="text-generation", model=MODEL, device="mps:0")
rsn8.set_dependency("pipeline", pipe)

@rsn8.register
def complete(ctx : Context, messages : List[dict[str, str]], max_new_tokens : int=8192):
    pipeline = ctx.get_dependency("pipeline")
    respones = pipeline(text_inputs=messages, max_new_tokens=max_new_tokens)
    return respones[0]["generated_text"]

if __name__ == "__main__":
    messages = [
        {
            "role": "system",
            "content": SYSTEM,
        },
    ]

    for turn in range(100):
        print("AskBlorgo 🤖 >>>", end=' ')
        sys.stdout.flush()
        user_message = sys.stdin.readline()
        messages.append({"role": "user", "content": user_message})
        reply = messages[-1]["content"]
        handle = complete.run(f"hal-{turn}", messages)
        print("(sent message to hal!")
        messages = handle.result()
        print(f"Blorgo sez: {messages[-1]["content"]}")
