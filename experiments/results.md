| Model | Dev Macro F1 | Test Macro F1 |
|----------|----------|----------|
| **Single MLP (0.5 Thrs)**    |||
| BERT (Baseline)    | 0.608     | 0.5418     |
| BERT + Feat ENG    | 0.6114     | 0.5624     |
| BERT + Feat ENG + EMOLex    | 0.6075     | 0.5638     |
| BERT + Feat ENG + EMOLex + POS    | 0.5785     | 0.5547     |
| **Emotion-Specific MLP (0.5 Thrs)**    |||
| BERT    | 0.6162     | 0.5434     |
| BERT + Feat ENG    | 0.6476     | 0.5457     |
| BERT + Feat ENG + EMOLex    | 0.6491     | 0.5538     |
| BERT + Feat ENG + EMOLex + POS   | 0.6125     | 0.5542     |
| **Emotion-Specific MLP (Emotion-Spec Thr)**     |||
| BERT   | 0.6568     | 0.5364     |
| BERT + Feat ENG   | 0.6816     | 0.6079     |
| BERT + Feat ENG + EmoLex   | 0.6433     | 0.5558     |
| BERT + Feat ENG + EMOLex + POS (Final Model)   | 0.6440     | 0.5434     |

Results went down with POS tags.

Emo-Spec MLP 0.5 BERT + Feat ENG AND BERT + Feat ENG + EMOLex performed better than final model on Dev set, but performed worse on Test set.