from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from PIL import Image
import io
import os
import re
import json
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if not GEMINI_API_KEY:
    print("❌ ERROR: GEMINI_API_KEY not found!")
    exit(1)

genai.configure(api_key=GEMINI_API_KEY)

print("\n🔍 Available models:")
available_models = []
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"  ✅ {m.name}")
            available_models.append(m.name)
except Exception as e:
    print(f"⚠️  Could not list models: {e}")

model = None
model_options = [
    'models/gemini-2.5-flash',
    'models/gemini-2.0-flash',
    'models/gemini-flash-latest',
    'models/gemini-1.5-flash',
    'models/gemini-pro-latest',
]

for model_name in model_options:
    try:
        model = genai.GenerativeModel(model_name)
        print(f"\n✅ Using model: {model_name}")
        break
    except Exception as e:
        continue

if not model:
    print("❌ No working model found!")
    exit(1)

def parse_analysis(text):
    """Parse the AI response into structured format"""
    analysis = {
        'overall': '',
        'strengths': [],
        'improvements': [],
        'tips': [],
        'detected_text': '',
        'scores': {},
        'practice_steps': []
    }
    
    # Try to extract JSON scores block first
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            scores_data = json.loads(json_match.group(1))
            if 'scores' in scores_data:
                analysis['scores'] = scores_data['scores']
            else:
                analysis['scores'] = scores_data
        except json.JSONDecodeError:
            pass
    
    # Also try to find scores in format "Category: XX%"
    if not analysis['scores']:
        score_patterns = [
            (r'Legibility[:\s]+(\d+)', 'Legibility'),
            (r'Letter Formation[:\s]+(\d+)', 'Letter Formation'),
            (r'Spacing[:\s]+(\d+)', 'Spacing'),
            (r'Baseline[:\s]+(\d+)', 'Baseline Consistency'),
            (r'Size Consistency[:\s]+(\d+)', 'Size Consistency'),
            (r'Slant Consistency[:\s]+(\d+)', 'Slant Consistency'),
        ]
        for pattern, name in score_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                analysis['scores'][name] = int(match.group(1))
    
    sections = text.split('\n\n')
    
    for section in sections:
        lower_section = section.lower()
        
        # Detect what was written
        if 'text written' in lower_section or 'wrote' in lower_section:
            match = re.search(r'["\'](.*?)["\']', section)
            if match:
                analysis['detected_text'] = match.group(1)
        
        if 'overall' in lower_section or 'assessment' in lower_section:
            lines = section.split('\n')
            content = ' '.join(lines[1:]) if len(lines) > 1 else section
            content = re.sub(r'\*\*.*?\*\*', '', content).strip()
            if content:
                analysis['overall'] = content
        
        elif 'strength' in lower_section or 'positive' in lower_section:
            items = re.findall(r'[-•*]\s*(.+)', section)
            analysis['strengths'].extend([item.strip() for item in items if item.strip()])
        
        elif 'improve' in lower_section or 'area' in lower_section or 'work on' in lower_section:
            items = re.findall(r'[-•*]\s*(.+)', section)
            analysis['improvements'].extend([item.strip() for item in items if item.strip()])
        
        elif 'tip' in lower_section or 'recommend' in lower_section:
            items = re.findall(r'[-•*\d.]\s*(.+)', section)
            analysis['tips'].extend([item.strip() for item in items if item.strip()])
        
        elif 'practice step' in lower_section or 'exercise' in lower_section:
            items = re.findall(r'[-•*\d.]\s*(.+)', section)
            analysis['practice_steps'].extend([item.strip() for item in items if item.strip()])
    
    # Fallback detected text
    if not analysis['detected_text']:
        match = re.search(r'["\'](.*?)["\']', text)
        if match:
            analysis['detected_text'] = match.group(1)
    
    # Fallback if no structured content found
    if not any([analysis['strengths'], analysis['improvements'], analysis['tips']]):
        all_bullets = re.findall(r'[-•*]\s*(.+)', text)
        if all_bullets:
            analysis['tips'] = all_bullets[:5]
        analysis['overall'] = text[:300] if not all_bullets else "Analysis provided below."
    
    return analysis

@app.route('/')
def home():
    return jsonify({
        'message': 'AI Handwriting Teacher Backend',
        'status': 'running',
        'endpoints': {
            'health': '/health',
            'test': '/test-api',
            'analyze': '/analyze (POST with image)',
            'detect': '/detect-text (POST with image)'
        }
    }), 200

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'api_configured': bool(GEMINI_API_KEY),
        'model': str(model._model_name) if model else 'none'
    }), 200

@app.route('/test-api', methods=['GET'])
def test_api():
    try:
        response = model.generate_content("Say 'API is working!' if you can read this.")
        return jsonify({
            'status': 'success',
            'message': 'Gemini API is working perfectly!',
            'model': str(model._model_name),
            'test_response': response.text
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/detect-text', methods=['POST'])
def detect_text():
    """Detect what was written"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        image_data = file.read()
        
        if len(image_data) == 0:
            return jsonify({'error': 'Empty image file'}), 400
        
        try:
            image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            return jsonify({'error': f'Invalid image format: {str(e)}'}), 400
        
        prompt = """Look at this handwriting image and tell me EXACTLY what text was written. 
        Respond ONLY with the text in quotes, nothing else.
        For example, if the image shows "Hello World", respond: "Hello World"
        Be as accurate as possible."""
        
        print("📤 Detecting text...")
        response = model.generate_content([prompt, image])
        
        detected = response.text.strip()
        match = re.search(r'["\'](.*?)["\']', detected)
        if match:
            detected = match.group(1)
        else:
            detected = detected.replace('"', '').replace("'", '').strip()
        
        print(f"✅ Detected: {detected}")
        return jsonify({'text': detected}), 200
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500

@app.route('/analyze', methods=['POST'])
def analyze_handwriting():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        image_data = file.read()
        
        if len(image_data) == 0:
            return jsonify({'error': 'Empty image file'}), 400
        
        try:
            image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            return jsonify({'error': f'Invalid image format: {str(e)}'}), 400
        
        prompt = """You are an expert handwriting teacher. Analyze this handwriting sample carefully.

IMPORTANT: First identify EXACTLY what text was written.

Provide your analysis in this EXACT format:

**Text Written:** "exact text here"

**Overall Assessment:**
[2-3 sentences about the overall handwriting quality]

**Strengths:**
- [Specific positive aspect 1]
- [Specific positive aspect 2]
- [Specific positive aspect 3]

**Areas for Improvement:**
- [Specific issue 1]
- [Specific issue 2]
- [Specific issue 3]

**Practice Tips:**
- [Actionable tip 1]
- [Actionable tip 2]
- [Actionable tip 3]

**Practice Steps:**
- [Step 1: specific exercise]
- [Step 2: specific exercise]
- [Step 3: specific exercise]

**Scores (0-100):**
```json
{
  "Legibility": [score based on how readable the text is],
  "Letter Formation": [score based on correct letter shapes],
  "Spacing": [score based on consistent spacing between letters/words],
  "Baseline Consistency": [score based on alignment to baseline],
  "Size Consistency": [score based on uniform letter sizes],
  "Slant Consistency": [score based on uniform letter angles]
}
```

Be accurate and constructive. Score strictly - 70-85 is good, 85+ is excellent, below 60 needs work."""

        print("📤 Sending to Gemini...")
        response = model.generate_content([prompt, image])
        print("✅ Got response!")
        
        analysis = parse_analysis(response.text)
        
        # Ensure we have scores (fallback defaults if parsing failed)
        if not analysis['scores']:
            analysis['scores'] = {
                'Legibility': 70,
                'Letter Formation': 70,
                'Spacing': 70,
                'Baseline Consistency': 70,
                'Size Consistency': 70
            }
        
        print(f"📝 Detected text: {analysis['detected_text']}")
        print(f"📊 Scores: {analysis['scores']}")
        
        return jsonify(analysis), 200
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error: {error_msg}")
        
        if "API_KEY_INVALID" in error_msg or "invalid" in error_msg.lower():
            return jsonify({'error': 'Invalid API key'}), 500
        elif "quota" in error_msg.lower():
            return jsonify({'error': 'API quota exceeded. Wait a minute.'}), 429
        else:
            return jsonify({'error': f'Analysis failed: {error_msg}'}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 AI Handwriting Teacher Backend")
    print("="*50)
    print(f"✅ API Key: {GEMINI_API_KEY[:20]}...{GEMINI_API_KEY[-4:]}")
    print(f"🤖 Model: {model._model_name if model else 'ERROR'}")
    print("🌐 Server: http://localhost:5000")
    print("="*50 + "\n")
    
    app.run(debug=True, port=5000, host='0.0.0.0')