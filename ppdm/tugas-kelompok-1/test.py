import pickle

def load_model():
    with open("svm_model_test.pkl", "rb") as f:
        model_data = pickle.load(f)
    return model_data

model_data = load_model()
print(f"Accuracy:  {model_data['accuracy']:.5f}")
print(f"Precision: {model_data['precision']:.5f}")
print(f"Recall:    {model_data['recall']:.5f}")
print(f"F1 Score:  {model_data['f1']:.5f}")