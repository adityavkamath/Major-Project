import streamlit as st
from streamlit_drawable_canvas import st_canvas
import requests
from PIL import Image
import io
import numpy as np
import json
import re

st.set_page_config(page_title="Handwritten Equation Solver", layout="wide")

# Initialize session state
if "canvas_count" not in st.session_state:
    st.session_state.canvas_count = 1
if "page" not in st.session_state:
    st.session_state.page = "landing"
if "canvas_equations" not in st.session_state:
    st.session_state.canvas_equations = {}
if "canvas_prompts" not in st.session_state:
    st.session_state.canvas_prompts = {}

def format_math_response(response):
    """Format mathematical response for better display"""
    try:
        formatted = response
        
        # Simple, safe replacements without complex regex
        # Convert common LaTeX symbols
        formatted = formatted.replace('\\frac{', '(').replace('}{', ')/(').replace('}', ')')
        formatted = formatted.replace('\\[', '').replace('\\]', '')
        formatted = formatted.replace('\\(', '').replace('\\)', '')
        formatted = formatted.replace('\\sqrt', '√')
        formatted = formatted.replace('sqrt', '√')
        
        # Convert exponents safely
        formatted = formatted.replace('^2', '²')
        formatted = formatted.replace('^3', '³')
        formatted = formatted.replace('{2}', '²')
        formatted = formatted.replace('{3}', '³')
        
        # Make steps and final answer prominent
        formatted = formatted.replace('Step 1:', '<h4 style="color: #1976d2; margin-top: 15px;">📍 Step 1:</h4>')
        formatted = formatted.replace('Step 2:', '<h4 style="color: #1976d2; margin-top: 15px;">📍 Step 2:</h4>')
        formatted = formatted.replace('Step 3:', '<h4 style="color: #1976d2; margin-top: 15px;">📍 Step 3:</h4>')
        formatted = formatted.replace('Step 4:', '<h4 style="color: #1976d2; margin-top: 15px;">📍 Step 4:</h4>')
        formatted = formatted.replace('Step 5:', '<h4 style="color: #1976d2; margin-top: 15px;">📍 Step 5:</h4>')
        
        formatted = formatted.replace('Final Answer:', '<div style="background-color: #e8f5e8; padding: 15px; margin: 15px 0; border: 2px solid #4caf50; border-radius: 8px;"><h3 style="margin: 0; color: #2e7d32;">🎯 Final Answer:</h3>')
        
        # Close final answer div if it exists
        if '🎯 Final Answer:' in formatted:
            formatted += '</div>'
        
        # Add line breaks for readability
        formatted = formatted.replace('\n\n', '<br><br>')
        formatted = formatted.replace('\n', '<br>')
        
        # Wrap in container
        formatted = f"""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #1976d2; font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6;">
            {formatted}
        </div>
        """
        
        return formatted
        
    except Exception as e:
        # If any error occurs, return simple formatted text
        simple_format = response.replace('\n', '<br>')
        return f"<div style='padding: 15px; background-color: #f5f5f5; border-radius: 8px; font-family: Arial, sans-serif;'>{simple_format}</div>"

def clean_latex_equation(latex_equation):
    """Clean and normalize LaTeX equation for better processing"""
    try:
        cleaned = latex_equation
        
        # Remove unnecessary LaTeX commands
        cleaned = cleaned.replace('\\dot{', '').replace('}', '')
        cleaned = cleaned.replace('\\', '')
        
        # Convert common patterns
        cleaned = re.sub(r'(\d+)\^{(\d+)}', r'\1^\2', cleaned)  # Convert {2} to 2
        cleaned = re.sub(r'(\d+)\^(\d+)', r'\1^\2', cleaned)    # Normalize exponents
        
        # Handle specific cases from your equation
        cleaned = cleaned.replace('dot{30}', '30')
        cleaned = cleaned.replace('4^{2}', '16')  # Pre-calculate simple expressions
        
        return cleaned
    except Exception as e:
        st.error(f"Equation cleaning error: {e}")
        return latex_equation
    formatted = f"""
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #007bff; font-family: 'Segoe UI', sans-serif; line-height: 1.6;">
        {formatted}
    </div>
    """
    
    return formatted

def clean_latex_equation(latex_equation):
    """Clean and improve LaTeX equation recognition"""
    cleaned = latex_equation
    
    # Common OCR fixes
    cleaned = re.sub(r'\\dot\{(\d+)\}', r'\1', cleaned)  # Remove \dot{30} -> 30
    cleaned = re.sub(r'(\d+)\^{(\d+)}', r'\1^{\2}', cleaned)  # Ensure proper exponent format
    cleaned = cleaned.replace('4^{2}', '16')  # Convert 4^2 to 16
    cleaned = cleaned.replace('{2}', '²')  # Convert {2} to superscript
    cleaned = cleaned.replace('^{2}', '²')  # Convert ^{2} to superscript
    
    return cleaned

def landing_page():
    st.markdown("""
    # 🧮 Handwritten Equation Solver
    
    Welcome to the AI-powered handwritten equation solver! This application uses advanced machine learning models to:
    
    1. **Recognize** your handwritten mathematical equations
    2. **Convert** them to LaTeX format
    3. **Solve** them using AI with custom prompts
    
    ### How it works:
    - **Draw** your equations on the canvas OR **upload** equation images
    - Add custom prompts to guide the AI solver
    - Get step-by-step solutions and explanations
    - **Solve multiple equations** together for complex problems
    
    ### Features:
    - ✏️ **Drawing Canvas**: Draw equations directly in the browser
    - 📁 **Image Upload**: Upload photos of handwritten equations
    - 🔄 **Multiple Canvas Support**: Work with several equations simultaneously
    - 🎯 **Custom AI Prompts**: Specify exactly what you want the AI to do
    - 🤖 **LaTeX Recognition**: Advanced OCR for mathematical expressions
    - 📊 **Multi-Equation Solving**: Solve related equations together
    - 🎓 **Step-by-step Solutions**: Detailed explanations and working
    """)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎨 Go to Canvas", use_container_width=True, type="primary"):
            st.session_state.page = "canvas"
            st.rerun()

def canvas_page():
    st.title("🎨 Draw Your Equations")
    
    # Back button and Add Canvas button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← Back to Home"):
            st.session_state.page = "landing"
            st.rerun()
    with col3:
        if st.button("➕ Add Canvas", type="secondary"):
            st.session_state.canvas_count += 1
            st.rerun()
    
    st.markdown(f"**Active Canvases: {st.session_state.canvas_count}**")
    
    # Multi-canvas solver section
    if st.session_state.canvas_count > 1:
        st.divider()
        with st.expander("🔄 Multi-Canvas Solver", expanded=False):
            st.markdown("**Select canvases to solve together:**")
            
            # Create checkboxes for each canvas
            selected_canvases = []
            cols = st.columns(min(4, st.session_state.canvas_count))
            for i in range(st.session_state.canvas_count):
                with cols[i % len(cols)]:
                    equation_preview = st.session_state.canvas_equations.get(i, "Not analyzed yet")
                    if len(equation_preview) > 20:
                        equation_preview = equation_preview[:20] + "..."
                    
                    if st.checkbox(f"Canvas {i+1}", key=f"multi_select_{i}"):
                        selected_canvases.append(i)
                    st.caption(f"Eq: {equation_preview}")
            
            if len(selected_canvases) > 1:
                st.markdown("**Combined Analysis:**")
                if st.button("🚀 Solve Selected Canvases Together", type="primary"):
                    solve_multiple_canvases(selected_canvases)
    
    st.divider()

    for i in range(st.session_state.canvas_count):
        with st.container():
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(f"📝 Canvas {i+1}")
                
                # Tab for drawing vs uploading
                tab1, tab2 = st.tabs(["✏️ Draw", "📁 Upload Image"])
                
                with tab1:
                    canvas_result = st_canvas(
                        fill_color="rgba(255, 255, 255, 1)",
                        stroke_width=4,
                        stroke_color="#000000",
                        background_color="#FFFFFF",
                        width=600,
                        height=200,
                        drawing_mode="freedraw",
                        key=f"canvas_{i}",
                    )
                
                with tab2:
                    uploaded_file = st.file_uploader(
                        f"Upload equation image for Canvas {i+1}",
                        type=['png', 'jpg', 'jpeg'],
                        key=f"upload_{i}"
                    )
                    if uploaded_file is not None:
                        st.image(uploaded_file, caption="Uploaded equation", width=300)
            
            with col2:
                st.subheader("🎯 Instructions")
                prompt = st.text_area(
                    f"What should the AI do with this equation?", 
                    placeholder="e.g., 'Solve for x', 'Find the derivative', 'Simplify the expression'",
                    key=f"prompt_{i}",
                    height=100
                )
                
                # Store prompt in session state
                st.session_state.canvas_prompts[i] = prompt
                
                submit_button = st.button(
                    f"🚀 Analyze Canvas {i+1}", 
                    key=f"submit_{i}",
                    use_container_width=True,
                    type="primary"
                )

            # Process canvas submission
            if submit_button:
                # Get uploaded file from the correct key
                uploaded_file_key = f"upload_{i}"
                uploaded_file = None
                
                # Check session state for uploaded file
                if uploaded_file_key in st.session_state:
                    uploaded_file = st.session_state[uploaded_file_key]
                
                # Determine image source
                image_data = None
                source_type = ""
                
                if uploaded_file is not None:
                    # Use uploaded image
                    image_data = uploaded_file.getvalue()
                    source_type = "uploaded image"
                elif canvas_result.image_data is not None:
                    # Use drawn canvas
                    img_array = canvas_result.image_data[:, :, :3]
                    if np.sum(img_array) == img_array.size * 255:  # All white canvas
                        st.error("⚠️ Please draw something on the canvas or upload an image first!")
                        continue
                    else:
                        img = Image.fromarray(img_array.astype("uint8"))
                        buf = io.BytesIO()
                        img.save(buf, format="PNG")
                        image_data = buf.getvalue()
                        source_type = "drawn canvas"
                else:
                    st.error("⚠️ No image data found. Please draw something or upload an image first!")
                    continue
                
                if image_data:
                    with st.spinner(f"🔍 Analyzing your equation from {source_type}..."):
                        try:
                            # Create file-like object for the API
                            files = {"file": ("equation.png", io.BytesIO(image_data), "image/png")}
                            
                            # Predict equation
                            resp = requests.post("http://localhost:8000/predict_equation", files=files)
                            resp.raise_for_status()
                            latex = resp.json().get("latex", "")
                            
                            # Clean the LaTeX equation
                            cleaned_latex = clean_latex_equation(latex)
                            
                            # Store equation in session state
                            st.session_state.canvas_equations[i] = cleaned_latex
                            
                            st.success(f"✅ Equation recognized from {source_type}!")
                            st.markdown(f"**Original Recognition:** `{latex}`")
                            if latex != cleaned_latex:
                                st.markdown(f"**Cleaned Equation:** `{cleaned_latex}`")
                            
                            # Solve equation if prompt is provided
                            if prompt.strip():
                                with st.spinner("🤖 AI is solving your equation..."):
                                    data = {"equation": cleaned_latex, "prompt": prompt}
                                    resp2 = requests.post("http://localhost:8000/solve_equation", data=data)
                                    resp2.raise_for_status()
                                    answer = resp2.json().get("answer", "")
                                    
                                    st.markdown("### 🎓 AI Solution:")
                                    # Create an expander for better organization
                                    with st.expander("📖 View Complete Solution", expanded=True):
                                        # Process the answer to make it more readable
                                        formatted_answer = format_math_response(answer)
                                        st.markdown(formatted_answer, unsafe_allow_html=True)
                            else:
                                st.info("💡 Add a prompt to get AI assistance with solving this equation!")
                                
                        except requests.exceptions.ConnectionError:
                            st.error("❌ Cannot connect to the backend server. Please make sure it's running on http://localhost:8000")
                        except requests.exceptions.HTTPError as e:
                            st.error(f"❌ Server error: {e}")
                        except Exception as e:
                            st.error(f"❌ An unexpected error occurred: {e}")
            
            st.divider()

def solve_multiple_canvases(selected_canvases):
    """Solve multiple selected canvases together"""
    equations = []
    prompts = []
    
    for canvas_id in selected_canvases:
        equation = st.session_state.canvas_equations.get(canvas_id, "")
        prompt = st.session_state.canvas_prompts.get(canvas_id, "")
        
        if not equation:
            st.error(f"❌ Canvas {canvas_id + 1} hasn't been analyzed yet. Please analyze it first.")
            return
        
        equations.append(equation)
        prompts.append(prompt if prompt else "Solve this equation")
    
    with st.spinner("🤖 AI is solving multiple equations..."):
        try:
            # Send multiple equations to the API
            data = {
                "equations": json.dumps(equations),
                "prompts": json.dumps(prompts)
            }
            resp = requests.post("http://localhost:8000/solve_multiple_equations", data=data)
            resp.raise_for_status()
            answer = resp.json().get("answer", "")
            
            st.markdown("### 🎓 Combined AI Solution:")
            # Create an expander for better organization
            with st.expander("📖 View Complete Multi-Equation Solution", expanded=True):
                formatted_answer = format_math_response(answer)
                st.markdown(formatted_answer, unsafe_allow_html=True)
            
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to the backend server. Please make sure it's running on http://localhost:8000")
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ Server error: {e}")
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {e}")

# Main app routing
if st.session_state.page == "landing":
    landing_page()
else:
    canvas_page()