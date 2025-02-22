| Model | Dev Macro F1 | Test Macro F1 |
|----------|----------|----------|
| Single MLP (0.5 Thrs)    |
| BERT (Baseline)    | 0.608     | 0.5418     |
| BERT + Feat ENG    | 0.6114     | 0.5624     |
| BERT + Feat ENG + EMOLex    | 0.6075     | 0.5638     |
| BERT + Feat ENG + EMOLex + POS    | 0.5785     | 0.5547     |
| Emotion-Specific MLP (0.5 Thrs)    |
| BERT    | 0.6162     | 0.5434     |
| BERT + Feat ENG    | Data     | Data     |
| BERT + Feat ENG + EMOLex    | Data     | Data     |
| BERT + Feat ENG + EMOLex + POS   | Data     | Data     |
| Emotion-Specific MLP (Emotion-Spec Thr)     |
| BERT   | Data     | Data     |
| BERT + Feat ENG   | Data     | Data     |
| BERT + Feat ENG + EmoLex   | Data     | Data     |
| BERT + Feat ENG + EMOLex + POS (Final Model)   | Data     | Data     |

Results went down with POS tags.