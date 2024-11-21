# Science Exam Multiple Choice Question Answering Model

## Project Overview
A sophisticated machine learning solution for answering multiple-choice science exam questions using advanced retrieval and prediction techniques.

## Key Features
- Hybrid document retrieval system
- Longformer-based multiple-choice question answering
- Advanced performance metrics and visualization
- Confidence-based prediction mechanism

## Model Architecture
- *Retrieval*: Two-pronged TF-IDF based document search
  - Parsed Wikipedia dataset
  - Cohere-based dataset
- *Model*: Longformer for multiple-choice inference
- *Prediction Strategy*: 
  - Average probabilities from two retrievals
  - Fallback prediction with 0.4 confidence threshold

## Performance Metrics
- AUC-ROC Score
- Mean Average Precision
- Logarithmic Loss
- F1 Score
- Cohen's Kappa

## Visualization
Includes comprehensive performance visualization:
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Performance Metrics Bar Plot

## Requirements
- Python 3.10+
- PyTorch
- Pandas
- Matplotlib
- Seaborn
- transformers>=4.6.0,<5.0.0
- tokenizers>=0.10.3
- tqdm
- torch>=1.6.0
- torchvision
- numpy
- scikit-learn
- scipy
- nltk
- sentencepiece
- huggingface-hub



## Usage
python
# Load model and make predictions
from science_exam_qa import predict_questions

predictions = predict_questions(test_dataframe)


## Evaluation
Run performance metrics and generate visualizations:
python
from science_exam_qa import evaluate_model

metrics, plots = evaluate_model(ground_truth, predictions)


## Model Limitations
- Depends on quality of retrieved documents
- Performance varies with question complexity
- Requires significant computational resources

## Contributing
Contributions, issues, and feature requests are welcome!

## License
[Specify your license]

## Acknowledgments
- Kaggle LLM Science Exam Competition
- Longformer Model
- Open-source ML community
