# SemEval Task 11 Track A English 2025 

### Task Description
Multi-label emotion classification: Anger, Fear, Joy, Sadness, Surprise

Paths to git ignored dataset files
./public_data
./public_data/dev
./public_data/train

#### Baseline results on validation:

Multi-label accuracy (Jaccard score): 0.3235632183908046
Micro F1 score: 0.4513274336283186
Macro F1 score: 0.38416706699765263

#### Final Best Performing Model on Test Set: /Users/angwang/semeval-task-11-track-b/test/thres-emolex-BERT-EMO.ipynb
1. Dynamic Thresholds for each emotion
2. Emotion BERT Features
3. Emotion Lexicon Features
4. Enhanced Data Preprocessing

Evaluation scores for eng track a:

F1 score: micro=0.6199597841041379, macro=0.5523447943188394

Emotion-level macro F1 scores:
anger: 0.3597
fear: 0.7426
joy: 0.5248
sadness: 0.5489
surprise: 0.5857

On the Development Set, this model achieves:

              precision    recall  f1-score   support

         joy       0.71      0.48      0.58        31
     sadness       0.75      0.69      0.72        35
    surprise       0.45      0.81      0.58        31
        fear       0.61      0.98      0.76        63
       anger       0.57      0.50      0.53        16

   micro avg       0.60      0.76      0.67       176
   macro avg       0.62      0.69      0.63       176
weighted avg       0.63      0.76      0.67       176
 samples avg       0.59      0.69      0.61       176

Per-emotion performance at best thresholds:
joy: F1 = 0.577
sadness: F1 = 0.716
surprise: F1 = 0.581
fear: F1 = 0.756
anger: F1 = 0.533

#### Our initial best performing model is alt4_3 for at Epoch 100 for the validation set:

Multi-label accuracy (Jaccard score): 0.3464080459770116

Micro F1 score: 0.4896551724137931

Macro F1 score: 0.4452320985690972

## Relevancy of Metrics

- If there is class imbalance or the importance of rare classes (e.g., low-frequency emotions) is a concern, Macro F1 is the most relevant.
- If you're working on a multi-label task where exact matches are important, consider using the Jaccard Index, but supplement it with F1 scores for a more nuanced understanding.
- If you are evaluating overall performance without concern for class imbalance, Micro F1 is a good choice.

Thus the best metric for this task would be Macro F1(to provide a fair assessment for each emoticon class and to handle imbalances in data) and Jaccard Index (to handle multi-label classification (i.e. in cases when a single instance can belong to multiple emotions simultaneously))

Epoch 4: 6-1-24

F1 score: micro=0.44985673352435523, macro=0.4323589740292243

Emotion-level macro F1 scores:
anger: 0.2314
fear: 0.6667
joy: 0.3784
sadness: 0.4636
surprise: 0.4218

## To Add:
-ModernBERT
-Adersarial Training 

public_data
--> Original dataset

public_data_dev: Released on the 31st of December
--> Updated dataset, with the same data but ids may have changed due to processing

test set will be released on teh 15th of January and the test phase ends on the 28th of January 

## 30 Jan: Best Performance from Simple-BERT-EMO:
Validation Set Performance:
              precision    recall  f1-score   support

         joy       0.56      0.71      0.63        31
     sadness       0.47      0.77      0.59        35
    surprise       0.36      0.77      0.49        31
        fear       0.55      1.00      0.71        63
       anger       0.35      0.69      0.47        16

   micro avg       0.48      0.84      0.61       176
   macro avg       0.46      0.79      0.58       176
weighted avg       0.49      0.84      0.61       176
 samples avg       0.51      0.77      0.58       176