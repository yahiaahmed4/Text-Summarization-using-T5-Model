# Text Summarization using T5 Model

This project utilizes the T5 (Text-To-Text Transfer Transformer) model for text summarization. The T5 model is a versatile transformer-based model capable of various natural language processing tasks, including text summarization.

## Installation

Ensure you have the required packages installed by running the following commands:

```bash
pip install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio===0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip install datasets
pip install transformers
pip install rouge
```

## Dataset

The project utilizes the "multi_news" dataset for training and evaluation. The dataset contains news articles paired with their corresponding summaries.

## Training

The T5 model is pretrained on the "multi_news" dataset. The training involves preprocessing the data, tokenizing it, and fine-tuning the T5 model for text summarization. The training hyperparameters are tuned for optimal performance.

## Inference

After training, the model can generate summaries for given input text. The provided `predict_summary()` function takes a document as input and produces a summary using the trained model.

## Evaluation

The quality of the generated summaries is evaluated using the ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metric. ROUGE measures the overlap between the generated summary and the reference summary in terms of n-gram recall.

## Results

The project evaluates the model's performance on a validation dataset from the "multi_news" dataset. Average ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) are computed to assess the summarization quality.

- Average ROUGE-1 score: 0.266
- Average ROUGE-2 score: 0.083
- Average ROUGE-L score: 0.241

## Usage

To use the trained model for summarization, simply call the `predict_summary()` function with the document as input. The model will generate a summary based on its training.

## Note

- Ensure that all necessary packages are installed before running the code.
- The model is fine-tuned on the "multi_news" dataset, but it can be adapted to other summarization tasks with appropriate training data.

---

Feel free to modify and extend the code to suit your specific requirements or explore other text summarization techniques and datasets. Happy summarizing!

--- 

