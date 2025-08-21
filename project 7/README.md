Wardrobe.AI 👔

An AI-powered virtual wardrobe that recommends T-shirt combinations based on your pants using computer vision, embeddings, and GenAI. It also provides a real-time augmented reality try-on experience.

🌟 Features

AI-based clothing analysis using OpenAI Vision & Embeddings

Smart T-shirt recommendations with fashion styling logic

Vector database (ChromaDB) for similarity search

Real-time AR try-on with MediaPipe & OpenCV

Streamlit web app with clean UI

🏗️ Tech Stack

Python, Streamlit, OpenAI API

ChromaDB for vector storage

MediaPipe + OpenCV for AR overlay

NumPy, Pandas, TensorFlow for preprocessing

📂 Project Structure
wardrobe-ai/
├── src/app.py             # Main Streamlit app
├── notebooks/             # Vector DB prep
├── data/                  # Sample clothing images
├── results/               # Output outfit combinations
├── requirements.txt
└── README.md

🚀 Quick Start

Clone repo & install requirements (pip install -r requirements.txt)

Add your OpenAI API key to .env

Prepare vector DB (notebooks/prepare_vector_db.ipynb)

Run app → streamlit run src/app.py

📊 How It Works

Upload pant image → AI describes style & color

AI suggests diverse matching T-shirt designs

Vector DB finds best visual match from dataset

AR overlay lets you “try on” outfit via webcam

🔑 Highlights

Won Best Project Award in final year 🎉

End-to-end GenAI + AR pipeline

Demonstrates LLM integration + embeddings + real-time CV
