import gradio as gr
from captioning import generate_caption, log_caption

def basic_caption(image):
    caption = generate_caption(image)
    log_caption(caption, "basic")
    return caption

def prompted_caption(image, prompt):
    if not prompt.strip():
        return "Please enter a prompt."
    caption = generate_caption(image, prompt)
    log_caption(caption, f"prompted: {prompt}")
    return caption

def caption_with_confidence(image):
    caption, logits = generate_caption(image, return_score=True)
    top_conf = logits.topk(5).values.tolist() if logits is not None else []
    log_caption(caption, "with_confidence")
    return caption, top_conf

with gr.Blocks() as demo:
    gr.Markdown("# 🖼️ Advanced Image Captioning App")
    gr.Markdown("This app uses the BLIP model to generate captions for uploaded images. Choose from different captioning modes below:")

    with gr.Tab("General Caption"):
        with gr.Row():
            image_input = gr.Image(type="pil")
            output_text = gr.Textbox(label="Caption")
        image_input.change(basic_caption, inputs=image_input, outputs=output_text)

    with gr.Tab("Caption with Confidence"):
        with gr.Row():
            conf_image = gr.Image(type="pil")
            conf_output = gr.Textbox(label="Generated Caption")
        with gr.Row():
            conf_score = gr.Textbox(label="Top Logit Scores (raw)")
            conf_button = gr.Button("Generate + Show Confidence")
        conf_button.click(caption_with_confidence, inputs=conf_image, outputs=[conf_output, conf_score])

demo.launch()
