import joblib
m = joblib.load(r'd:\FINAL HUMINT DASH\models\xgb_classifier_reduced.joblib')
print('model class', m.__class__)
print('n_features_in_', getattr(m,'n_features_in_', None))
print('feature_names_in_', getattr(m,'feature_names_in_', None))
print('n_classes_', getattr(m,'n_classes_', None))
print('classes_', getattr(m,'classes_', None))
try:
    print('booster.feature_names', m.get_booster().feature_names)
except Exception as e:
    print('booster.feature_names error', type(e), e)
try:
    print('Booster json dump length', len(m.get_booster().save_raw()))
except Exception as e:
    print('Booster save_raw error', type(e), e)
