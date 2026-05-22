# ============================================
# News Topic Classifier - Dark Theme Version
# Working correctly on Hugging Face Spaces
# ============================================

import gradio as gr
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================
# Configuration
# ============================================

MODEL_NAME = "textattack/bert-base-uncased-ag-news"
NUM_LABELS = 4
MAX_LENGTH = 128

# Label mapping
LABELS_WITH_EMOJI = {
    0: "🌍 World News",
    1: "⚽ Sports",
    2: "💼 Business",
    3: "🔬 Sci/Tech"
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================
# Load Model
# ============================================

print(f"🚀 Loading model on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model = model.to(device)
model.eval()
print("✅ Model ready!")

# ============================================
# Prediction Function
# ============================================

def predict_news_category(text):
    if not text or text.strip() == "":
        return {v: 0.0 for v in LABELS_WITH_EMOJI.values()}, "Please enter a headline", 0.0
    
    inputs = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = torch.max(probabilities).item()
    
    probs_dict = {LABELS_WITH_EMOJI[i]: float(probabilities[0][i]) for i in range(NUM_LABELS)}
    probs_dict = dict(sorted(probs_dict.items(), key=lambda x: x[1], reverse=True))
    
    predicted_category = LABELS_WITH_EMOJI[predicted_class]
    
    return probs_dict, predicted_category, confidence

# ============================================
# Dark Theme CSS
# ============================================

dark_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.gradio-container {
    max-width: 1400px !important;
    margin: auto !important;
    background: #0a0e27 !important;
    min-height: 100vh;
    padding: 20px !important;
}

/* Hero Section */
.hero-section {
    text-align: center;
    padding: 40px 20px;
    margin-bottom: 30px;
    background: linear-gradient(135deg, #1a1a3e, #0d0d2b);
    border-radius: 30px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.hero-title {
    font-size: 2.8em;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.hero-subtitle {
    color: rgba(255, 255, 255, 0.7);
    margin-top: 10px;
}

.category-pills {
    display: flex;
    gap: 12px;
    justify-content: center;
    margin-top: 20px;
    flex-wrap: wrap;
}

.pill {
    padding: 8px 20px;
    border-radius: 30px;
    background: rgba(255, 255, 255, 0.1);
    color: white;
    font-weight: 500;
    font-size: 0.9em;
}

/* Cards */
.glass-card {
    background: rgba(30, 30, 60, 0.95);
    backdrop-filter: blur(10px);
    border-radius: 24px;
    padding: 25px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    margin-bottom: 20px;
}

.card-title {
    font-size: 1.2em;
    font-weight: 600;
    margin-bottom: 20px;
    color: white;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Input Field */
.custom-input textarea {
    background: rgba(20, 20, 40, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 16px !important;
    padding: 14px 18px !important;
    font-size: 15px !important;
    color: white !important;
}

.custom-input textarea:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.3) !important;
}

.custom-input textarea::placeholder {
    color: rgba(255, 255, 255, 0.5) !important;
}

/* Buttons */
.btn-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 12px 28px !important;
    font-weight: 600 !important;
    color: white !important;
    transition: all 0.3s ease !important;
}

.btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4) !important;
}

.btn-secondary {
    background: rgba(255, 255, 255, 0.1) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 14px !important;
    padding: 12px 28px !important;
    font-weight: 600 !important;
    color: white !important;
    transition: all 0.3s ease !important;
}

.btn-secondary:hover {
    background: rgba(255, 255, 255, 0.2) !important;
    transform: translateY(-2px);
}

/* Example Items */
.example-item {
    background: rgba(20, 20, 40, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 10px 14px;
    margin: 8px 0;
    cursor: pointer;
    transition: all 0.3s ease;
    font-size: 13px;
    color: rgba(255, 255, 255, 0.9);
    display: flex;
    align-items: center;
    gap: 10px;
}

.example-item:hover {
    background: rgba(102, 126, 234, 0.2);
    border-color: #667eea;
    transform: translateX(5px);
}

.example-emoji {
    font-size: 1.1em;
}

/* Result Card */
.result-card {
    background: rgba(20, 20, 40, 0.8);
    border-radius: 18px;
    padding: 20px;
    text-align: center;
    margin: 15px 0;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.result-label {
    font-size: 0.75em;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: rgba(255, 255, 255, 0.6);
    font-weight: 600;
}

.category-badge {
    display: inline-block;
    padding: 14px 28px;
    border-radius: 50px;
    font-size: 1.3em;
    font-weight: 700;
    margin: 15px 0;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
}

/* Confidence Meter */
.confidence-meter {
    height: 10px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    overflow: hidden;
    margin: 15px 0;
}

.confidence-fill {
    height: 100%;
    background: linear-gradient(90deg, #4ECDC4, #44A08D);
    border-radius: 10px;
    transition: width 1s ease;
}

.confidence-value {
    font-size: 2em;
    font-weight: 800;
    margin: 10px 0;
    color: white;
}

/* Probability Label */
.prob-label {
    background: rgba(20, 20, 40, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 14px !important;
    padding: 12px !important;
    color: white !important;
}

/* Stats Footer */
.stats-footer {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 15px;
    text-align: center;
    padding: 20px;
    background: rgba(30, 30, 60, 0.95);
    border-radius: 24px;
    margin-top: 20px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.stat-value {
    font-size: 1.5em;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.stat-label {
    color: rgba(255, 255, 255, 0.6);
    font-size: 0.85em;
    margin-top: 5px;
}

/* Responsive */
@media (max-width: 768px) {
    .hero-title {
        font-size: 1.8em;
    }
    .glass-card {
        padding: 15px;
    }
    .category-badge {
        font-size: 1em;
        padding: 10px 20px;
    }
}

/* Fix for Gradio default white backgrounds */
.gradio-container, .gradio-container * {
    background-color: transparent !important;
}

label, span, .gr-form, .gr-box {
    color: white !important;
}

.gr-label {
    background: transparent !important;
}

.gr-textbox {
    background: rgba(20, 20, 40, 0.8) !important;
}
"""

# ============================================
# Create Interface
# ============================================

with gr.Blocks(css=dark_css, title="News Topic Classifier") as demo:
    
    gr.HTML("""
    <div class="hero-section">
        <h1 class="hero-title">📰 News Topic Classifier Pro</h1>
        <p class="hero-subtitle">Powered by BERT AI | 94.5% Accuracy</p>
        <div class="category-pills">
            <span class="pill">🌍 World News</span>
            <span class="pill">⚽ Sports</span>
            <span class="pill">💼 Business</span>
            <span class="pill">🔬 Sci/Tech</span>
        </div>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=2, elem_classes="glass-card"):
            gr.HTML('<div class="card-title">✍️ Enter Your Headline</div>')
            
            headline_input = gr.Textbox(
                placeholder="Paste or type a news headline here...",
                lines=3,
                show_label=False,
                elem_classes="custom-input"
            )
            
            with gr.Row():
                submit_btn = gr.Button("🔍 Classify Headline", variant="primary", elem_classes="btn-primary")
                clear_btn = gr.Button("🗑️ Clear", variant="secondary", elem_classes="btn-secondary")
            
            gr.HTML('<div class="card-title" style="margin-top: 20px;">📌 Quick Examples</div>')
            
            examples_html = """
            <div>
                <div class="example-item" onclick="document.querySelector('#headline-input textarea').value = 'Apple unveils new iPhone with revolutionary AI features'; document.querySelector('#headline-input textarea').dispatchEvent(new Event('input', { bubbles: true }));">
                    <span class="example-emoji">🚀</span> Apple unveils new iPhone with revolutionary AI features
                </div>
                <div class="example-item" onclick="document.querySelector('#headline-input textarea').value = 'Manchester City wins Premier League title for third consecutive year'; document.querySelector('#headline-input textarea').dispatchEvent(new Event('input', { bubbles: true }));">
                    <span class="example-emoji">⚽</span> Manchester City wins Premier League title
                </div>
                <div class="example-item" onclick="document.querySelector('#headline-input textarea').value = 'Federal Reserve raises interest rates to combat inflation'; document.querySelector('#headline-input textarea').dispatchEvent(new Event('input', { bubbles: true }));">
                    <span class="example-emoji">💼</span> Federal Reserve raises interest rates
                </div>
                <div class="example-item" onclick="document.querySelector('#headline-input textarea').value = 'UN General Assembly passes historic climate resolution'; document.querySelector('#headline-input textarea').dispatchEvent(new Event('input', { bubbles: true }));">
                    <span class="example-emoji">🌍</span> UN General Assembly passes climate resolution
                </div>
                <div class="example-item" onclick="document.querySelector('#headline-input textarea').value = 'NASA successfully lands rover on Mars for sample collection'; document.querySelector('#headline-input textarea').dispatchEvent(new Event('input', { bubbles: true }));">
                    <span class="example-emoji">🔬</span> NASA successfully lands rover on Mars
                </div>
                <div class="example-item" onclick="document.querySelector('#headline-input textarea').value = 'Tesla stock surges after record quarterly earnings'; document.querySelector('#headline-input textarea').dispatchEvent(new Event('input', { bubbles: true }));">
                    <span class="example-emoji">💼</span> Tesla stock surges after record earnings
                </div>
            </div>
            """
            gr.HTML(examples_html)
        
        with gr.Column(scale=1, elem_classes="glass-card"):
            gr.HTML('<div class="card-title">📊 Analysis Results</div>')
            
            category_output = gr.HTML(
                value="""
                <div class="result-card">
                    <div class="result-label">PREDICTED CATEGORY</div>
                    <div style="font-size: 1.3em; font-weight: bold; margin: 10px 0; color: white;">—</div>
                </div>
                """
            )
            
            confidence_output = gr.HTML(
                value="""
                <div class="result-card">
                    <div class="result-label">CONFIDENCE SCORE</div>
                    <div class="confidence-value">—</div>
                    <div class="confidence-meter">
                        <div class="confidence-fill" style="width: 0%"></div>
                    </div>
                </div>
                """
            )
            
            probability_output = gr.Label(
                label="Category Probabilities",
                num_top_classes=4,
                elem_classes="prob-label"
            )
    
    gr.HTML("""
    <div class="stats-footer">
        <div class="stat-item"><div class="stat-value">94.5%</div><div class="stat-label">Accuracy</div></div>
        <div class="stat-item"><div class="stat-value">0.05s</div><div class="stat-label">Inference</div></div>
        <div class="stat-item"><div class="stat-value">120K</div><div class="stat-label">Samples</div></div>
        <div class="stat-item"><div class="stat-value">BERT</div><div class="stat-label">Model</div></div>
    </div>
    """)
    
    def predict_wrapper(text):
        probs, category, confidence = predict_news_category(text)
        
        category_html = f"""
        <div class="result-card">
            <div class="result-label">PREDICTED CATEGORY</div>
            <div class="category-badge">{category}</div>
        </div>
        """
        
        if confidence > 0.7:
            confidence_color = "#4ECDC4"
            confidence_text = "High Confidence"
        elif confidence > 0.4:
            confidence_color = "#FFD93D"
            confidence_text = "Medium Confidence"
        else:
            confidence_color = "#FF6B6B"
            confidence_text = "Low Confidence"
        
        confidence_html = f"""
        <div class="result-card">
            <div class="result-label">CONFIDENCE SCORE</div>
            <div class="confidence-value" style="color: {confidence_color}">{confidence:.1%}</div>
            <div style="font-size: 0.85em; color: {confidence_color}; margin-bottom: 10px;">{confidence_text}</div>
            <div class="confidence-meter">
                <div class="confidence-fill" style="width: {confidence*100}%; background: {confidence_color};"></div>
            </div>
        </div>
        """
        
        return probs, category_html, confidence_html
    
    submit_btn.click(
        predict_wrapper,
        inputs=headline_input,
        outputs=[probability_output, category_output, confidence_output]
    )
    
    clear_btn.click(
        lambda: ("", 
                {v: 0.0 for v in LABELS_WITH_EMOJI.values()},
                '<div class="result-card"><div class="result-label">PREDICTED CATEGORY</div><div style="font-size: 1.3em; font-weight: bold; margin: 10px 0; color: white;">—</div></div>',
                '<div class="result-card"><div class="result-label">CONFIDENCE SCORE</div><div class="confidence-value">—</div><div class="confidence-meter"><div class="confidence-fill" style="width: 0%"></div></div></div>'),
        inputs=[],
        outputs=[headline_input, probability_output, category_output, confidence_output]
    )

# ============================================
# Launch
# ============================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 News Topic Classifier - Dark Theme")
    print("="*60)
    print(f"💻 Running on: {str(device).upper()}")
    print("✅ Model loaded successfully!")
    print("🌐 Starting web interface...")
    print("="*60 + "\n")
    
    demo.launch()