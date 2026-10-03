
"""
LightGBM classifier for earthquake damage prediction.

Utilizes a 5-fold stratified cross-validation strategy. To address severe
class imbalance on the Macro F1 metric, the model applies 'balanced' class
weights during training. Post-training, it uses Nelder-Mead optimization
on the out-of-fold (OOF) probabilities to derive custom threshold multipliers,
properly calibrating the decision boundaries before final inference.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from scipy.optimize import minimize


def main():
    SEED = 42
    np.random.seed(SEED)

    public_dir = Path("public")
    train_path = public_dir / "train.csv"
    test_path = public_dir / "test.csv"
    sample_sub_path = public_dir / "sample_submission.csv"

    out_dir = Path("submission")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "submission.csv"

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    target = "damage_grade"
    features = [
        "floor_count_group", "age_group", "area_group",
        "height_group", "foundation_group"
    ]

    for col in features:
        df_train[col] = df_train[col].astype('category')
        df_test[col] = df_test[col].astype('category')

    X = df_train[features]
    y = df_train[target] - 1
    X_test = df_test[features]

    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_error',
        'boosting_type': 'gbdt',
        'class_weight': 'balanced',
        'random_state': SEED,
        'verbose': -1,
        'n_estimators': 150
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    oof_preds_proba = np.zeros((len(df_train), 3))
    test_preds_proba = np.zeros((len(df_test), 3))

    for train_idx, val_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
        )

        oof_preds_proba[val_idx] = model.predict_proba(X_val)
        test_preds_proba += model.predict_proba(X_test) / skf.n_splits

    # Calculate baseline F1 before threshold optimization
    standard_oof_preds = np.argmax(oof_preds_proba, axis=1)
    standard_oof_f1 = f1_score(y, standard_oof_preds, average='macro')

    def f1_loss(weights):
        weighted_oof = oof_preds_proba * weights
        preds = np.argmax(weighted_oof, axis=1)
        return -f1_score(y, preds, average='macro')

    initial_weights = [1.0, 1.0, 1.0]
    result = minimize(f1_loss, initial_weights, method='Nelder-Mead')
    best_weights = result.x

    # Calculate optimized F1 post-threshold optimization
    optimized_oof_preds = np.argmax(oof_preds_proba * best_weights, axis=1)
    final_oof_f1 = f1_score(y, optimized_oof_preds, average='macro')

    # Clean console outputs for tracking pipeline execution
    #
    print(f"Standard argmax OOF Macro F1: {standard_oof_f1:.4f}")
    print(f"Optimized Threshold OOF Macro F1: {final_oof_f1:.4f}")

    final_test_preds = np.argmax(test_preds_proba * best_weights, axis=1) + 1

    sub = pd.read_csv(sample_sub_path)
    sub[target] = final_test_preds.astype(int)
    sub.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()