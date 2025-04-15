# 🖼️ Advanced Image Captioning with Transformers

This project is a modern image captioning web application built using the **BLIP vision-language model**. It provides an interactive interface to generate captions for images using various modes.

## 🚀 Features

- ✅ Generate **general captions** for any image
- 🧠 Input a **custom prompt** (e.g., "What is the person doing?")
- 📊 Display **model confidence scores**
- 🎥 Upload image via file or webcam
- 📦 Logs saved for every caption generated

## 🧰 Tech Stack

- **BLIP Model**: Pretrained vision-language transformer
- **PyTorch + Hugging Face**: For inference and model handling
- **Gradio**: Web interface with tabs and image input
- **PIL**: Image preprocessing
- **Python Logging**: To save caption history

## 🛠 Setup Instructions

```bash
git clone https://github.com/jaganathkrishnan/Image-captioner.git
cd image-captioning-app
pip install -r requirements.txt
python app.py
