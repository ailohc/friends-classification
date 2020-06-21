# friends-classification

COSE461 Natural Language Processing Task 2를 위한 코드입니다.

## Install Requirements
Install package with pip, check the cuda version for installing pytorch.
```
$ pip install -r requirements.txt
```

## Preprocessing
For train data, execute preprocessing/bert.py
```
$ cd preprocessing
$ python bert.py
```

For test data, execute preprocessing/bert-test.py
```
$ cd preprocessing
$ python bert-test.py
```

## Train and Test
For fine-tune and test, execute test.py
```
$ python test.py
```

## Reference
[네이버 영화리뷰 감정분석 with Hugging Face BERT colab](https://colab.research.google.com/drive/1tIf0Ugdqg4qT7gcxia3tL7und64Rv1dP)
