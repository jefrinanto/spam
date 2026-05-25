Here is a descriptive passage that summarizes what this Python script achieves, followed by a detailed, step-by-step breakdown of how the code works.

The Big Picture: What This Code Does
This script builds an automated Machine Learning pipeline to detect SMS spam. Using a dataset of real text messages, it cleans and transforms raw text into numerical data that a computer can understand. It then trains a probabilistic AI model—specifically, a Multinomial Naive Bayes classifier—to learn the subtle differences between legitimate messages ("ham") and fraudulent ones ("spam"). Finally, the script evaluates the model's accuracy, visualizes its performance using a confusion matrix heatmap, and tests it against a brand-new, unseen sample message to prove it can successfully flag spam in the real world.

Step-by-Step Code Explanation
Step 1: Importing the Toolkit
Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
What's happening: The script starts by importing specialized libraries. pandas and numpy handle data manipulation, while matplotlib and seaborn are used for plotting charts.

The sklearn (scikit-learn) imports bring in the heavy machinery: tools to split the data, convert text to numbers, train the AI model, and measure its performance.

Step 2: Loading and Cleaning the Data
Python
df = pd.read_csv("/kaggle/input/sms-spam-collection-dataset/spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
What's happening: The code reads the raw dataset (spam.csv). Because text data often contains special characters, it uses latin-1 encoding to avoid errors.

It discards unnecessary blank columns, keeps only the columns for the category (v1) and the text (v2), and renames them to 'label' and 'message' for clarity.

Step 3: Encoding Labels into Numbers
Python
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
What's happening: AI models don't inherently understand words like "ham" or "spam"; they require numbers. This line maps the text categories to binary numbers: 0 for legitimate messages (ham) and 1 for spam.

Step 4: Splitting Data for Training and Testing
Python
X_train, X_test, y_train, y_test = train_test_split(
    df['message'],
    df['label'],
    test_size=0.2,
    random_state=42
)
What's happening: To ensure the AI actually learns rather than just memorizes, the dataset is split.

80% of the data goes into the training set (X_train, y_train) to teach the model.

20% is set aside as a test set (X_test, y_test) to grade the model later. random_state=42 ensures that the split is exactly the same every time you run the code.

Step 5: Converting Text to Numbers (Vectorization)
Python
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
What's happening: Computers can't read sentences, so the TfidfVectorizer converts text into a matrix of numbers based on word frequency.

stop_words='english' tells it to ignore common filler words (like "the", "is", "and") that don't add meaning.

.fit_transform() builds a vocabulary from the training data and converts the texts into math vectors. .transform() converts the test data using that exact same vocabulary.

Step 6: Training the AI Model
Python
model = MultinomialNB()
model.fit(X_train_vec, y_train)
What's happening: The script initializes a Multinomial Naive Bayes classifier, a highly efficient algorithm perfectly suited for text classification (based on calculating the probability of words appearing in spam vs. ham).

model.fit() is the actual "learning" step where the model analyzes the training vectors and discovers which words point to spam.

Step 7: Making Predictions and Evaluating
Python
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
What's happening: The trained model is forced to guess whether the unseen test messages (X_test_vec) are spam or ham.

It prints out the Accuracy (overall percentage of correct guesses) alongside a Classification Report detailing deeper metrics like Precision (how many flagged spams were actually spam) and Recall (how many total spams it caught).

Step 8: Visualizing with a Confusion Matrix
Python
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix - Spam Detection")
plt.show()
What's happening: It generates a confusion matrix—a 2x2 grid that maps out:

True Negatives: Correctly identified safe texts.

True Positives: Correctly caught spam.

False Negatives: Spam that slipped through the filter.

False Positives: Safe texts accidentally blocked as spam.

sns.heatmap turns this grid into a color-coded chart, making it easy to see where the model is succeeding or struggling.

Step 9: Testing a Live Sample
Python
sample = ["Congratulations! You have won a free gift card. Click now"]
sample_vec = vectorizer.transform(sample)
prediction = model.predict(sample_vec)

print("\nSample Message Prediction:")
print("Spam" if prediction[0] == 1 else "Ham")
