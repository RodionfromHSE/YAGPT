# YAGPT (Yet Another GPT)

This is a simple GPT implementation in Python. It is based on the russian version of [GPT-2](https://huggingface.co/AlexWortega/instruct_rugptMedium).

### Dataset (dataset.ipynb)

Final dataset consists of 0.8M samples.

We used 2 large russian text corpus: [Yandex QA](
https://huggingface.co/datasets/its5Q/yandex-q) and [Diasum Dataset](https://huggingface.co/datasets/bragovo/diasum).
We tried next techniques to prepare dataset for our model:
- Form dialogues from sentences
- Make each sample consist of 3 parts: context, prompt and answer

**Example**
```python
history: "Привет, как дела?"
speaker1: "Привет, все хорошо, а у тебя?"
speaker2: "Все хорошо, спасибо!"
```
On english
```python
history: "Hi, how are you?"
speaker1: "Hi, I'm fine, and you?"
speaker2: "I'm fine, thanks!"
```


### Training (train.ipynb)
Here is usual training pipeline. We used [HuggingFace](https://huggingface.co/) transformers library to train our model.
