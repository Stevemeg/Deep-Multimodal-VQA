import torch
from transformers import DistilBertModel, DistilBertTokenizer


class TextEncoder(torch.nn.Module):
    def __init__(self, model_name="distilbert-base-uncased"):
        super(TextEncoder, self).__init__()

        print("Loading DistilBERT...")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name)

        print("DistilBERT loaded (trainable).")

    def forward(self, questions):
        """
        questions: list of strings
        returns: token embeddings (batch, seq_len, hidden_dim)
        """

        inputs = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        outputs = self.model(**inputs)

        # Shape: (batch, seq_len, hidden_dim)
        return outputs.last_hidden_state
