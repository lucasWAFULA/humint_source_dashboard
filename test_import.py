import importlib.util
import sys
import types
# Provide a dummy 'api' module to avoid import-time failures during this isolated import test
api_mod = types.ModuleType('api')
def run_optimization(*args, **kwargs):
    return {"policies": {}, "emv": {}, "evpi": 0.0, "audit_log": {}}

def explain_source(features):
    return {"features": list(features.keys()) if isinstance(features, dict) else [], "shap_values": []}

api_mod.run_optimization = run_optimization
api_mod.explain_source = explain_source
sys.modules['api'] = api_mod

p = r"c:\Users\hp\OneDrive\Dokumenter\MSC. DSA\Intell Sources\FINAL HUMINT DASH\dashboard.py"
spec = importlib.util.spec_from_file_location('dashmod', p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print('Imported OK; xgb_model is', getattr(m, 'xgb_model', None))
print('explainer is', getattr(m, 'explainer', None))
