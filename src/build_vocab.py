import json
from collections import Counter
from vqa_dataset import normalize_answer


def build_answer_vocab(annotation_path, top_k=1000):
    with open(annotation_path, "r") as f:
        annotations_data = json.load(f)

    answers = []

    for ann in annotations_data["annotations"]:
        for a in ann["answers"]:
            answers.append(normalize_answer(a["answer"]))

    counter = Counter(answers)
    most_common = counter.most_common(top_k)

    vocab = {ans: idx for idx, (ans, _) in enumerate(most_common)}

    return vocab
