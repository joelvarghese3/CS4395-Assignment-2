
## 🧠 Models Implemented

### FFNN (Feedforward Neural Network)

- **Vectorization**: Converts each review into a bag-of-words vector using a custom vocabulary.
- **Architecture**: `Input → ReLU → Linear → LogSoftmax`
- **Loss Function**: Negative Log Likelihood (`NLLLoss`)
- **Optimizer**: Stochastic Gradient Descent (SGD) with momentum
- **Best Accuracy**: **Validation accuracy = 63.1%** with hidden dimension size of 32

### RNN (Recurrent Neural Network)

- **Vectorization**: Uses pre-trained word embeddings loaded from `word_embedding.pkl`.
- **Architecture**: PyTorch's Elman RNN with Tanh nonlinearity followed by a linear layer and LogSoftmax.
- **Loss Function**: `NLLLoss`
- **Optimizer**: Adam
- **Early Stopping**: Automatically stops training when overfitting is detected
- **Best Accuracy**: **Validation accuracy = 39%** with hidden dimension size of 32

## 📊 Key Findings

- FFNN significantly outperforms RNN for this task.
- FFNN achieves peak performance at a hidden layer size of 32, beyond which it begins to overfit.
- RNN accuracy does not increase with larger hidden layers and performed best with a small hidden dimension (10 or 32).
- Yelp review sentiment is more easily captured by FFNN, likely due to the non-sequential nature of the review sentiment classification.

## 🚀 How to Run

To train the FFNN and RNN:
```bash
python ffnn.py -hd 32 -e 100 --train_data ./training.json --val_data ./validation.json --do_train
python rnn.py -hd 32 -e 100 --train_data ./training.json --val_data ./validation.json --do_train


