from openai import OpenAI
import io
import base64
import torch

inputs_embeds = torch.load("inputs_embeds.pt").squeeze(0)
print(inputs_embeds.shape)
# exit()
def tensor2base64(x: torch.Tensor) -> str:
    with io.BytesIO() as buf:
        torch.save(x, buf)
        buf.seek(0)
        binary_data = buf.read()

    return base64.b64encode(binary_data).decode("utf-8")

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)
encoded_embeds = tensor2base64(inputs_embeds)
completion = client.completions.create(
    model="/workspace_yuekai/asr/FireRedASR/pretrained_models/FireRedASR-LLM-L/new_llm",
    # NOTE: The OpenAI client does not allow `None` as an input to
    # `prompt`. Use an empty string if you have no text prompts.
    prompt="",
    max_tokens=50,
    temperature=0.0,
    top_p=0.01,
    # NOTE: The OpenAI client allows passing in extra JSON body via the
    # `extra_body` argument.
    extra_body={"prompt_embeds": encoded_embeds},
)

print("-" * 30)
print(completion.choices[0].text)
print("-" * 30)