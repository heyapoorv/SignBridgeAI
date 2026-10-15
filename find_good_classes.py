import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
import pandas as pd

print("Loading model and data...")
model = tf.keras.models.load_model('best_hybrid_300.keras')
X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')

print('X_val shape:', X_val.shape, 'y_val shape:', y_val.shape)

# Check if y_val has 300 classes. If it's a smaller dataset, we might get out of bounds
if np.max(y_val) >= 300:
    print("WARNING: y_val contains indices >= 300")

preds = np.argmax(model.predict(X_val), axis=1)
report = classification_report(y_val, preds, output_dict=True, zero_division=0)

good_classes = [int(k) for k, v in report.items() if k.isdigit() and v['f1-score'] >= 0.70]
print(f"Found {len(good_classes)} classes with >= 70% F1 score")

train_df = pd.read_csv('ASL_Citizen/splits/train.csv')
top_words = sorted(train_df['Gloss'].value_counts().head(300).index.tolist())

# It's possible that some classes don't exist in y_val.
good_words = [top_words[i] for i in good_classes if i < len(top_words)]

with open('high_acc_words.txt', 'w') as f:
    f.write(','.join(good_words))

print("Good words saved to high_acc_words.txt")
print('Good words preview:', good_words[:10])
