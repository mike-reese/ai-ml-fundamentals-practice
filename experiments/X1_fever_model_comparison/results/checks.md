| Check | Condition | Tolerance | Observed | Holds |
| --- | --- | --- | --- | --- |
| Vectorizer vocabulary fitted on the training set only | no term that occurs only in the validation or test set is in it | exact | vocabulary_size 293437; terms_absent_from_training 0 | pass |
| Torch logistic regression against sklearn LogisticRegression, same features, validation macro-F1 | within 0.02 | 0.02 | torch_macro_f1 0.5530; sklearn_macro_f1 0.5715; gap 0.0185; n_train 114883; sklearn_C 1.0000; torch_best_epoch 4 | pass |
| Shuffled training labels, negative control | validation macro-F1 at chance: within 0.05 of 1/3 (uniform) or of the majority-class macro-F1 (prior only) | 0.05 | macro_f1 0.2346; chance_uniform 0.3333; chance_prior_only 0.2295; distance_from_uniform 0.0987; distance_from_prior_only 0.0051; log_loss 1.0358; label_prior_entropy 1.0175; n_train 114883 | pass |
| Seeded rerun | the same seed reproduces the saved validation metrics | 1e-6 | max_abs_difference 0.0000; epochs 7; same_epoch_count True; val_macro_f1 0.5530 | pass |
| Majority-class baseline | training majority class predicted for every validation claim | reported | majority_code 0; majority_label SUPPORTS; macro_f1 0.2295; accuracy 0.5249 | reported |
| Claim-only baseline | torch logistic regression on the claim block alone, validation | reported | macro_f1 0.5431; log_loss 0.8565; macro_f1_supports_refutes 0.7147; n_features 293437; best_epoch 3 | reported |
