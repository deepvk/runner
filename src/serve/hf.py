import fire
import torch
from transformers import pipeline

from ..utils import preprocess_instance


def build_ru_conversation(text, entity_type):
    return {
        "conversations": [
            {"from": "human", "value": f"Текст: {text}"},
            {"from": "gpt", "value": "Я прочитала текст."},
            {"from": "human", "value": f'Что описывает "{entity_type}" в тексте?'},
            {"from": "gpt", "value": None},
        ]
    }, "ie_as_qa_ru"


def build_en_conversation(text, entity_type):
    return {
        "conversations": [
            {"from": "human", "value": f"Text: {text}"},
            {"from": "gpt", "value": "I've read this text."},
            {"from": "human", "value": f"What describes {entity_type} in the text?"},
            {"from": "gpt", "value": None},
        ]
    }, "ie_as_qa"


def main(
    model_path: str = "Universal-NER/UniNER-7B-type",
    max_new_tokens: int = 256,
    lang: str = "ru",
):
    generator = pipeline(
        "text-generation", model=model_path, torch_dtype=torch.float16, device="cuda:0"
    )
    while True:
        try:
            text = input("Input text: ")
            entity_type = input("Input entity type: ")
        except EOFError:
            text = entity_type = ""
        if not text:
            print("Exit...")
            break
        example, template = (
            build_ru_conversation(text, entity_type)
            if lang == "ru"
            else build_en_conversation(text, entity_type)
        )
        prompt = preprocess_instance(example["conversations"], template)
        print(prompt)
        outputs = generator(prompt, max_length=max_new_tokens, return_full_text=False)
        print(outputs[0]["generated_text"])


if __name__ == "__main__":
    fire.Fire(main)
