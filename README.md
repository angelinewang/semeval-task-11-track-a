# semeval-task-11-track-b

Paths to git ignored dataset files
./public_data
./public_data/dev
./public_data/train

Baseline results on validation:

Your results for eng track A are:

Multi-label accuracy (Jaccard score): 0.3235632183908046

Micro F1 score: 0.4513274336283186

Macro F1 score: 0.38416706699765263

## Relevancy of Metrics

- If there is class imbalance or the importance of rare classes (e.g., low-frequency emotions) is a concern, Macro F1 is the most relevant.
- If you're working on a multi-label task where exact matches are important, consider using the Jaccard Index, but supplement it with F1 scores for a more nuanced understanding.
- If you are evaluating overall performance without concern for class imbalance, Micro F1 is a good choice.

Thus the best metric for this task would be Macro F1(to provide a fair assessment for each emoticon class and to handle imbalances in data) and Jaccard Index (to handle multi-label classification (i.e. in cases when a single instance can belong to multiple emotions simultaneously))