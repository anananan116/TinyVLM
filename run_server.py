from flask import Flask, request, jsonify, render_template_string
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
import base64
import io

app = Flask(__name__)

# Initialize model and tokenizer globally
model = AutoModelForCausalLM.from_pretrained(
    "anananan116/TinyVLM",
    trust_remote_code=True,
    torch_dtype=torch.float16,
).eval()
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
model.to(device)
tokenizer = AutoTokenizer.from_pretrained("anananan116/TinyVLM")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
DEFAULT_PROMPT = "<IMGPLH>Describe this image."

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Vision-Language Model Interface</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --primary-color: #2563eb;
            --secondary-color: #1e40af;
            --background-color: #f8fafc;
            --card-background: #ffffff;
            --text-color: #1e293b;
            --border-color: #e2e8f0;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--background-color);
            color: var(--text-color);
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }

        .container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-top: 2rem;
        }

        @media (max-width: 768px) {
            .container {
                grid-template-columns: 1fr;
            }
            body {
                padding: 1rem;
            }
        }

        .header {
            text-align: center;
            margin-bottom: 2rem;
            padding: 1rem;
            background: var(--card-background);
            border-radius: 1rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        .header h1 {
            color: var(--primary-color);
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }

        .header p {
            color: #64748b;
            font-size: 1rem;
        }

        .input-section, .output-section {
            background: var(--card-background);
            padding: 1.5rem;
            border-radius: 1rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        .section-title {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: var(--primary-color);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .section-title i {
            font-size: 1.1rem;
        }

        #dropZone {
            width: 100%;
            min-height: 200px;
            border: 2px dashed var(--border-color);
            border-radius: 0.5rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 1rem;
            padding: 2rem;
            transition: all 0.3s ease;
            cursor: pointer;
            margin-bottom: 1rem;
        }

        #dropZone:hover, #dropZone.dragover {
            background-color: #f1f5f9;
            border-color: var(--primary-color);
        }

        #dropZone i {
            font-size: 2.5rem;
            color: var(--primary-color);
        }

        #dropZone p {
            color: #64748b;
            text-align: center;
        }

        #imagePreview {
            max-width: 100%;
            max-height: 300px;
            display: none;
            border-radius: 0.5rem;
            margin: 1rem 0;
            object-fit: contain;
        }

        .input-group {
            position: relative;
            margin-bottom: 1rem;
        }

        #chatInput {
            width: 100%;
            padding: 0.75rem 1rem;
            border: 1px solid var(--border-color);
            border-radius: 0.5rem;
            font-size: 1rem;
            transition: all 0.3s ease;
            outline: none;
        }

        #chatInput:focus {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        }

        .button-group {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }

        button {
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 0.5rem;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .primary-button {
            background-color: var(--primary-color);
            color: white;
        }

        .primary-button:hover {
            background-color: var(--secondary-color);
        }

        .secondary-button {
            background-color: #e2e8f0;
            color: var(--text-color);
        }

        .secondary-button:hover {
            background-color: #cbd5e1;
        }

        #output {
            width: 100%;
            min-height: 200px;
            padding: 1rem;
            border: 1px solid var(--border-color);
            border-radius: 0.5rem;
            background-color: #f8fafc;
            font-family: monospace;
            white-space: pre-wrap;
            overflow-y: auto;
            margin-top: 1rem;
        }

        .loading {
            display: none;
            text-align: center;
            padding: 1rem;
        }

        .loading i {
            color: var(--primary-color);
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .status {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem;
            border-radius: 0.25rem;
            font-size: 0.875rem;
            margin-top: 0.5rem;
        }

        .status.success {
            background-color: #dcfce7;
            color: #166534;
        }

        .status.error {
            background-color: #fee2e2;
            color: #991b1b;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Vision-Language Model Interface</h1>
        <p>Upload an image and get AI-generated descriptions</p>
    </div>

    <div class="container">
        <div class="input-section">
            <div class="section-title">
                <i class="fas fa-upload"></i>
                Input
            </div>
            
            <div id="dropZone">
                <i class="fas fa-cloud-upload-alt"></i>
                <p>Drag and drop an image here<br>or click to upload</p>
            </div>
            <img id="imagePreview" alt="Preview"/>
            
            <div class="input-group">
                <input type="text" id="chatInput" placeholder="Enter your message (or leave empty for default prompt)">
            </div>

            <div class="button-group">
                <button onclick="sendMessage()" class="primary-button">
                    <i class="fas fa-paper-plane"></i>
                    Send
                </button>
                <button onclick="clearAll()" class="secondary-button">
                    <i class="fas fa-trash"></i>
                    Clear
                </button>
            </div>
        </div>

        <div class="output-section">
            <div class="section-title">
                <i class="fas fa-comment-dots"></i>
                Output
            </div>
            
            <div class="loading">
                <i class="fas fa-spinner fa-2x"></i>
                <p>Processing...</p>
            </div>
            
            <div id="output"></div>
        </div>
    </div>

    <script>
        let dropZone = document.getElementById('dropZone');
        let imagePreview = document.getElementById('imagePreview');
        let currentImage = null;
        let loadingIndicator = document.querySelector('.loading');

        // Handle drag and drop
        dropZone.ondragover = function(e) {
            e.preventDefault();
            this.classList.add('dragover');
        };
        
        dropZone.ondragleave = function(e) {
            this.classList.remove('dragover');
        };
        
        dropZone.ondrop = function(e) {
            e.preventDefault();
            this.classList.remove('dragover');
            handleFiles(e.dataTransfer.files);
        };
        
        dropZone.onclick = function() {
            let input = document.createElement('input');
            input.type = 'file';
            input.accept = 'image/*';
            input.onchange = e => handleFiles(e.target.files);
            input.click();
        };
        
        function handleFiles(files) {
            if (files.length > 0) {
                let file = files[0];
                if (file.type.startsWith('image/')) {
                    let reader = new FileReader();
                    reader.onload = function(e) {
                        imagePreview.src = e.target.result;
                        imagePreview.style.display = 'block';
                        currentImage = e.target.result;
                        dropZone.style.display = 'none';
                    };
                    reader.readAsDataURL(file);
                }
            }
        }
        
        function showStatus(message, type) {
            const statusDiv = document.createElement('div');
            statusDiv.className = `status ${type}`;
            statusDiv.innerHTML = `
                <i class="fas fa-${type === 'success' ? 'check-circle' : 'exclamation-circle'}"></i>
                ${message}
            `;
            document.getElementById('output').parentNode.insertBefore(statusDiv, document.getElementById('output'));
            setTimeout(() => statusDiv.remove(), 5000);
        }
        
        function sendMessage() {
            if (!currentImage) {
                showStatus('Please upload an image first', 'error');
                return;
            }
            
            let message = document.getElementById('chatInput').value;
            loadingIndicator.style.display = 'block';
            document.getElementById('output').textContent = '';
            
            let data = {
                message: message,
                image: currentImage
            };
            
            fetch('/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(data => {
                loadingIndicator.style.display = 'none';
                document.getElementById('output').textContent = data.response;
                showStatus('Response generated successfully', 'success');
            })
            .catch(error => {
                loadingIndicator.style.display = 'none';
                console.error('Error:', error);
                document.getElementById('output').textContent = 'Error processing request';
                showStatus('Error processing request', 'error');
            });
        }
        
        function clearAll() {
            document.getElementById('chatInput').value = '';
            document.getElementById('output').textContent = '';
            imagePreview.style.display = 'none';
            dropZone.style.display = 'flex';
            currentImage = null;
        }

        // Enable Enter key to send message
        document.getElementById('chatInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    message = data.get('message', '').strip() + "Carefully explain your reasoning before answering the question."
    image_data = data.get('image')
    
    try:
        if not message:
            message = DEFAULT_PROMPT
        
        if image_data:
            image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            if '<IMGPLH>' not in message:
                message = '<IMGPLH>' + message
        else:
            image = None
            
        inputs = model.prepare_input_ids_for_generation([message], [image] if image else None, tokenizer)
        
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs['input_ids'].to(device),
                attention_mask=inputs['attention_mask'].to(device),
                encoded_image=inputs["encoded_image"],
                max_new_tokens=128,
                do_sample=True,
                output_scores=True,
                return_dict_in_generate=True
            )
        
        new_tokens = generated_ids.sequences[:, inputs['input_ids'].shape[1]:]
        output_text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return jsonify({'response': output_text})
    
    except Exception as e:
        return jsonify({'response': f'Error: {str(e)}'})

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=5000)