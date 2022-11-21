class XGBoostAdapter:
    def __init__(self, xgb, early_stopping_rounds=10):
        self.xgb = xgb
        self.early_stopping_rounds = early_stopping_rounds

    def fit_and_validate(self, tra_x, tra_y, val_x, val_y):
        self.xgb.fit(
            tra_x, tra_y,
            eval_set=[(val_x, val_y)],
            early_stopping_rounds=self.early_stopping_rounds,
            eval_metric='merror')

    def predict_proba(self, x):
        return self.xgb.predict_proba(x)