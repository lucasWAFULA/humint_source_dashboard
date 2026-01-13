import os
src = r"c:\Users\hp\OneDrive\Dokumenter\MSC. DSA\Intell Sources\FINAL HUMINT DASH\xgb_classifier_reduced.joblib"
dst = r"c:\Users\hp\OneDrive\Dokumenter\MSC. DSA\Intell Sources\FINAL HUMINT DASH\models\xgb_classifier_reduced.joblib"
try:
    os.replace(src, dst)
    print("moved")
except Exception as e:
    print("error:", e)
    raise
