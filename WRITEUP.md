# Write-up: Fraud detection with drift monitoring

## Approach

Chronological split -> train supervised + unsupervised models -> evaluate with PR-AUC -> cost-based
threshold -> label-free drift monitoring on later windows -> alert -> retrain.

## Key decisions

* **Class imbalance: class weights, not oversampling.** Frauds are 0.17%. Class weights make mistakes on
fraud count more, without inventing fake rows. SMOTE creates synthetic fraud from only \~490 examples
and can leak into validation if done before splitting; weights are simpler and safe.
Isolation Forest handles imbalance by framing fraud as "rare and different".
* **PR-AUC, not ROC-AUC.** ROC-AUC uses the false-positive *rate*, which stays tiny when there are
\~280k legit rows, so even a poor model looks great. PR-AUC uses *precision*, which directly shows how
many alerts are real fraud. The random baseline for PR-AUC is the fraud rate (0.0017).
* **Chronological split.** Real systems predict the future from the past; shuffling would leak.
* **Drift reference = validation window,** not the training set, because the model's scores on data it
trained on are over-confident.
* **PSI** for feature/score drift: label-free, which matters because fraud labels arrive days or weeks late.
Rule of thumb: <0.1 stable, 0.1-0.25 moderate, >0.25 major.
* **Alert rule:** retrain if many features drift *or* the model's score distribution drifts.

## Cost analysis

Missed fraud (FN) costs the transaction amount; a blocked legitimate transaction (FP) costs an assumed
`FP\_COST` (review, friction, lost sale). The threshold minimises total cost on the validation window and
is then tested on the later period. Compared against threshold 0.5, "block nothing" and "block everything".
Change `FP\_COST` and the best threshold moves: higher FP cost -> stricter (fewer blocks).

## Results 

|Model|PR-AUC|ROC-AUC|
|-|-|-|
|Logistic Regression (class weights)|0.672	|0.935|
|HistGradientBoosting (class weights)|0.726	|0.955|
|Isolation Forest|0.033	|0.947|
|Random baseline|0.0017|0.5|

Drift: score PSI stayed around 0.06-0.09 on the control windows and rose to 0.46-0.49 on the injected windows 3-4, where PR-AUC also fell. However, the RETRAIN alert fired on all windows, including the control, because the control already had 5-6 of 29 features above PSI 0.25 (about 17-21%), over the 15% bound. Feature-level PSI is too sensitive here; score PSI separated drift from no drift much better.
Retrain loop: PR-AUC on window 2 improved from 0.121 (stale model) to 0.439 (retrained).

## Limitations

* Only \~50 frauds per monitoring window, so per-window PR-AUC is noisy.
* Drift is simulated; real drift is messier. Thresholds (0.25 etc.) are conventions, not laws.
* PCA-anonymised features limit how well we can explain *why* drift happened.
* FP\_COST is an assumption, not measured business data.

