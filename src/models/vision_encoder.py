import torch
from transformers import CLIPVisionModel, CLIPImageProcessor


class CLIPVisionEncoder:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cpu"):
        self.device = device

        print("Loading CLIP Vision Model...")
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.model = CLIPVisionModel.from_pretrained(model_name)

        self.model.to(device)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        print("CLIP loaded and frozen.")

    def encode(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state.squeeze(0)
        return embeddings
