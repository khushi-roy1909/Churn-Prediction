import pickle

with open(r"C:\Users\asus\OneDrive\Desktop\churn prediction") as f:
    model = pickle.load(f)

print(model)
