# 🧮 Handwritten Equation Solver

AI-powered web app that recognizes handwritten math equations and solves them step-by-step.

## Features

- **Draw or Upload**: Draw equations on canvas or upload images
- **AI Recognition**: Converts handwriting to LaTeX using TrOCR model
- **Smart Solving**: Step-by-step solutions with OpenAI GPT
- **Multiple Equations**: Work with several equations simultaneously
- **Custom Prompts**: Tell the AI exactly what to do (solve, simplify, etc.)

## Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add OpenAI API key**
   Create a `.env` file with your OpenAI key:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

3. **Run the app**
   ```bash
   python run_app.py
   ```
   
   Then open http://localhost:8501 in your browser

## How to Use

1. Click "Go to Canvas"
2. Draw an equation or upload an image
3. Add a prompt (e.g., "solve for x")
4. Click "Analyze Canvas"
5. Get step-by-step solution!

## Tech Stack

- **Frontend**: Streamlit with drawable canvas
- **Backend**: FastAPI
- **AI Models**: TrOCR (handwriting recognition) + OpenAI GPT (solving)

## Project Files

```
├── app.py          # Streamlit frontend
├── main.py         # FastAPI backend  
├── run_app.py      # Startup script
├── requirements.txt
└── .env            # Your OpenAI API key
```
