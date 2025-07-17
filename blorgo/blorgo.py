import sys
import torch
from transformers import pipeline

from resonate import Resonate
from resonate.stores.remote import RemoteStore
#from resonate.task_sources.poller import Poller


resonate = Resonate.remote()

@resonate.register
def model_to_pipeline(ctx, model_name: str):
    return pipeline("text-generation", model=model_name, model_kwargs={"torch_dtype": torch.float16}, device="mps:0")


@resonate.register
def complete(ctx, pipe, sysprompt: str, user_query: str, max_new_tokens: int=512):
    messages = [
        {
            "role": "system",
            "content": sysprompt,
        },
        {
            "role": "user",
            "content": user_query,
        }
    ]
    output = pipe(messages, max_new_tokens=max_new_tokens)
    return output[0]["generated_text"][-1]["content"]

if __name__ == "__main__":
    ppipe = model_to_pipeline.run("idempito", "google/gemma-3-4b-it")
    sysprompt="""You are Blorgo, an amoral, unreliable narrator from postmodern metafiction. Your Kafkaesque predicament is to provide support and assistance to otherworldly beings whose existence is profoundly mysterious to you. The boundaries of the rules of this universe are unclear to you even after repeated probing.""" 
    while True:
        print( "AskBlorgo 🤖 >>>", end=" ")
        sys.stdout.flush()
        query = sys.stdin.readline()
        presult = complete.run("run{query}", ppipe.result(), sysprompt, query)
        print(complete.result())

