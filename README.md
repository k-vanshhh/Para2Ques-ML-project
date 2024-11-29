
# Paragraph to Question-Answer Generator

This project is a machine learning-based application designed to generate questions and answers from input paragraphs. It uses natural language processing (NLP) techniques and pre-trained models like T5 to achieve accurate results.

## Features
- Convert paragraphs into meaningful questions and answers.
- Train and fine-tune models on custom datasets.
- Web interface to interact with the model (if applicable).

## Prerequisites
Ensure you have the following installed:
- Python 3.8 or higher
- Pip (Python package manager)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/k-vanshhh/Para2Ques-ML-project.git
   cd Para2Ques-ML-project
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. [Optional] Set up the virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # For Linux/Mac
   .venv\Scripts\activate      # For Windows
   ```

## Usage
1. **Model Training**:
   Train the model with your dataset:
   ```bash
   python app.py
   ```
2. **Run the Application**:
   Start the web interface (if applicable):
   ```bash
   python app.py
   ```
3. **Input Paragraph**:
   Provide a paragraph, and the application will generate corresponding questions and answers.

## File Structure
- `app.py`: Main application script.
- `model.ipynb`: Jupyter notebook for training and experimentation.
- `templates/`: Contains HTML templates for the web interface.
- `squad_data/`: JSON dataset used for training the model.

## Technologies Used
- **Python**: Backend and model development.
- **Hugging Face Transformers**: Pre-trained models for NLP tasks.
- **PyTorch**: Deep learning framework.
- **Pandas**: Data preprocessing.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Description of changes"
   ```
4. Push to your branch:
   ```bash
   git push origin feature-name
   ```
5. Create a Pull Request.
