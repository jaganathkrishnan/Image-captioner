from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def generate_caption(image, prompt=None, return_score=False):
    if prompt:
        inputs = processor(image, prompt, return_tensors="pt").to(device)
    else:
        inputs = processor(images=image, return_tensors="pt").to(device)

    output = model.generate(**inputs, output_scores=return_score, return_dict_in_generate=True)
    caption = processor.decode(output.sequences[0], skip_special_tokens=True)

    if return_score:
        logits = output.scores[0].softmax(dim=-1) if output.scores else None
        return caption, logits
    return caption

def log_caption(caption_text, mode):
    with open("logs.txt", "a") as f:
        f.write(f"{mode.upper()} | {caption_text}\n")
