from flask import Flask, render_template, request, jsonify
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

# Initialize model and tokenizer
tokenizer = T5Tokenizer.from_pretrained('./trained_model')
model = T5ForConditionalGeneration.from_pretrained('./trained_model')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def generate_qa_pairs(context, num_questions=5, max_length=50, temperature=0.7):
    # Generate questions
    question_inputs = tokenizer("generate question: " + context, return_tensors="pt")
    question_inputs = {key: value.to(device) for key, value in question_inputs.items()}
    
    question_outputs = model.generate(
        question_inputs['input_ids'],
        do_sample=True,
        max_length=max_length,
        top_k=50,
        top_p=0.95,
        temperature=temperature,
        num_return_sequences=num_questions
    )
    
    questions = [tokenizer.decode(output, skip_special_tokens=True)
                for output in question_outputs]
    
    # Generate answers for each question
    qa_pairs = []
    for question in questions:
        answer_input = tokenizer(f"answer question: {question} context: {context}", 
                               return_tensors="pt")
        answer_input = {key: value.to(device) for key, value in answer_input.items()}
        
        answer_output = model.generate(
            answer_input['input_ids'],
            max_length=max_length,
            num_beams=4,
            no_repeat_ngram_size=2
        )
        
        answer = tokenizer.decode(answer_output[0], skip_special_tokens=True)
        qa_pairs.append({"question": question, "answer": answer})
    
    return qa_pairs

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    context = data.get('context', '')
    num_questions = int(data.get('num_questions', 5))
    temperature = float(data.get('temperature', 0.7))
    max_length = int(data.get('max_length', 50))
    
    try:
        qa_pairs = generate_qa_pairs(
            context, 
            num_questions=num_questions,
            temperature=temperature,
            max_length=max_length
        )
        return jsonify({'success': True, 'qa_pairs': qa_pairs})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)