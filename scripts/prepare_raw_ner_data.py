"""Prepare raw data parsed from LLM API to the Universal-NER format.

Input (json line):
  {
    "text": "Одна из дочерей Джаку была выдана замуж за Амира Темура.\nСмерть.",
    "entities": [
        ["person", ["Джаку", "Амира Темура"]]
    ]
  }

Output (json line):
  [
    {"from": "human", "value": "Текст: Одна из дочерей Джаку была выдана замуж за Амира Темура.\nСмерть."},
    {"from": "gpt", "value": "Я прочитала текст."},
    {"from": "human", "value": "Что описывает \"person\" в тексте?"},
    {"from": "gpt", "value": "[\"Джаку\", \"Амира Темура\"]"},
  ]

"""
import json
import random
from argparse import ArgumentParser
from collections import Counter

from tqdm import tqdm

RU_PROMPT = {
    "text": "Tекст",
    "read": "Я прочитала текст.",
    "question": 'Что описывает "{entity}" в тексте?',
}
EN_PROMPT = {
    "text": "Text",
    "read": "I've read this text.",
    "question": "What describes {entity} in the text?",
}


NEGATIVE_SAMPLING_PROB = 0.7


def convert_raw_sample(
    raw_sample: dict[str, str | list], prompts: dict[str, str], negative_entities: tuple[list, list] | None
) -> [dict[str, str]]:
    result = [
        {"from": "human", "value": prompts["text"] + ": " + raw_sample["text"]},
        {"from": "gpt", "value": prompts["read"]},
    ]

    used_entities = []
    for entity in raw_sample["entities"]:
        entity_type, entity_values = entity
        result += [
            {"from": "human", "value": prompts["question"].format(entity=entity_type)},
            {"from": "gpt", "value": '["' + '", "'.join(entity_values) + '"]'},
        ]
        used_entities.append(entity_type)

    if negative_entities is not None:
        if random.random() < NEGATIVE_SAMPLING_PROB:
            while True:
                negative_entity = random.choices(negative_entities[0], weights=negative_entities[1], k=1)[0]
                if negative_entity not in used_entities:
                    break
            result += [
                {"from": "human", "value": prompts["question"].format(entity=negative_entity)},
                {"from": "gpt", "value": "[]"},
            ]

    return result


def main(input_file_path: str, output_file_path: str, lang: str, ns: bool) -> None:
    if lang == "ru":
        prompts = RU_PROMPT
    elif lang == "en":
        prompts = EN_PROMPT
    else:
        print("Unknown language")
        return

    if ns:
        entity_counter = Counter()
        with open(input_file_path, "r") as f_in:
            for line in f_in:
                raw_sample = json.loads(line)
                entity_counter.update([it[0] for it in raw_sample["entities"]])
        print(f"Top-10 entities: {entity_counter.most_common(10)}")
        negative_entities = list(entity_counter.keys()), list(entity_counter.values())
    else:
        negative_entities = None

    with open(input_file_path, "r") as f_in, open(output_file_path, "w") as f_out:
        for line in tqdm(f_in):
            raw_sample = json.loads(line)
            prepared_sample = convert_raw_sample(raw_sample, prompts, negative_entities)
            print(json.dumps(prepared_sample, ensure_ascii=False), file=f_out)


if __name__ == "__main__":
    _arg_parser = ArgumentParser()
    _arg_parser.add_argument("input_file_path", help="Path to the input file, JSONL format.")
    _arg_parser.add_argument("output_file_path", help="Path to the output file, JSONL format.")
    _arg_parser.add_argument("--lang", default="ru", help="Prompt language.")
    _arg_parser.add_argument("--ns", action="store_true", help="Use negative sampling.")

    _args = _arg_parser.parse_args()
    main(**vars(_args))
