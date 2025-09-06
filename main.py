from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import PIL
import io
from openai import OpenAI
import os
from typing import List

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("ℹ️  python-dotenv not installed, using system environment variables")
except Exception as e:
    print(f"⚠️  Could not load .env file: {e}")

app = FastAPI(
    title="Handwritten Equation Solver API",
    description="API for recognizing handwritten equations and solving them with AI",
    version="1.0.0"
)

# Allow CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load TrOCR model and processor
print("Loading TrOCR model...")
processor = TrOCRProcessor.from_pretrained('tjoab/latex_finetuned')
model = VisionEncoderDecoderModel.from_pretrained('tjoab/latex_finetuned')
print("Model loaded successfully!")

# Initialize OpenAI client
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("OpenAI client initialized successfully!")
except Exception as e:
    print(f"Warning: OpenAI client initialization failed: {e}")
    client = None

@app.get("/")
async def root():
    return {"message": "Handwritten Equation Solver API is running!", "status": "healthy"}

def open_PIL_image_from_bytes(image_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image

def open_PIL_image_enhanced(image_bytes: bytes) -> Image.Image:
    """Enhanced image processing similar to your old code"""
    image = Image.open(io.BytesIO(image_bytes))
    
    # Handle PNG transparency by compositing with white background
    if image.mode == 'RGBA':
        background = PIL.Image.new('RGB', image.size, 'white')
        image = Image.composite(image, background, image)
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image

@app.post("/predict_equation")
async def predict_equation(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = open_PIL_image_enhanced(image_bytes)
        preproc_image = processor.image_processor(images=image, return_tensors="pt").pixel_values
        pred_ids = model.generate(preproc_image, max_length=128)
        latex_pred = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
        return {"latex": latex_pred}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/predict_multiple_equations")
async def predict_multiple_equations(files: List[UploadFile] = File(...)):
    """Process multiple equation images at once"""
    try:
        results = []
        for i, file in enumerate(files):
            image_bytes = await file.read()
            image = open_PIL_image_enhanced(image_bytes)
            preproc_image = processor.image_processor(images=image, return_tensors="pt").pixel_values
            pred_ids = model.generate(preproc_image, max_length=128)
            latex_pred = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
            results.append({
                "canvas_id": i,
                "filename": file.filename,
                "latex": latex_pred
            })
        return {"equations": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")

@app.post("/solve_multiple_equations")
async def solve_multiple_equations(equations: str = Form(...), prompts: str = Form(...)):
    """Solve multiple equations with their respective prompts"""
    try:
        if not client:
            raise HTTPException(status_code=500, detail="OpenAI client not initialized. Please check your API key.")
        
        # Parse the equations and prompts (they should be JSON strings)
        import json
        equations_list = json.loads(equations)
        prompts_list = json.loads(prompts)
        
        # Compose the message for LLM with better formatting
        equation_text = ""
        for i, (eq, prompt) in enumerate(zip(equations_list, prompts_list)):
            equation_text += f"**Equation {i+1}:** {eq}\n**Task {i+1}:** {prompt}\n\n"
        
        user_message = f"""
        Please solve the following equations with their respective instructions:
        
        {equation_text}
        
        Please provide step-by-step solutions for each equation with the following requirements:
        1. First, interpret each equation correctly (clean up any OCR artifacts)
        2. Use clear step-by-step format (**Step 1:**, **Step 2:**, etc.)
        3. Use proper mathematical notation (x² instead of x^2, √ instead of sqrt)
        4. Show all algebraic manipulations clearly
        5. Provide specific values or expressions for each variable when possible
        6. End each equation solution with **Final Answer:** showing the complete solution
        7. If equations are related, explain the relationship
        
        Format your response with clear sections for each equation and proper mathematical formatting.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": user_message}],
            max_tokens=1500,
        )
        answer = response.choices[0].message.content.strip()
        return {"answer": answer}
    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Error solving equations: {str(e)}")

@app.post("/solve_equation")
async def solve_equation(equation: str = Form(...), prompt: str = Form(...)):
    try:
        if not client:
            raise HTTPException(status_code=500, detail="OpenAI client not initialized. Please check your API key.")
        
        # Compose the message for LLM with specific formatting instructions
        user_message = f"""
        Given equation: {equation}
        Task: {prompt}
        
        Please solve this equation step by step. Follow these instructions carefully:
        
        1. First, interpret the equation correctly (note that \\dot{{30}} likely means 30, and 4^{{2}} means 16)
        2. Provide clear step-by-step solution with proper mathematical notation
        3. Use **Step 1:**, **Step 2:**, etc. for each step
        4. Show all algebraic manipulations clearly
        5. If solving for variables, provide specific values or expressions for each variable
        6. Use proper mathematical symbols (x² not x^2, √ not sqrt)
        7. End with **Final Answer:** showing the complete solution
        
        If this is a system of equations or has multiple variables, explain what type of equation it is and provide the most appropriate solution method.
        
        Format example:
        **Step 1:** Interpret and simplify the equation
        **Step 2:** Apply appropriate solving method
        **Step 3:** Solve for the variable(s)
        **Final Answer:** x = [value] and/or y = [value] or the relationship between variables
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": user_message}],
            max_tokens=1000,
        )
        answer = response.choices[0].message.content.strip()
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error solving equation: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)