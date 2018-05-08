from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

""" LINEAR SVC GETS ABOUT 54% ACCURACY, RELATIVE TO THE 33% BASELINE """
with open('./scores') as f1:
    scores = f1.read().strip().split('\n')

with open('./sentences') as f2:
    sentences = f2.read().strip().split('\n')

vec = TfidfVectorizer(ngram_range=(1,3))

v = vec.fit_transform(sentences)

X_train, X_test, y_train, y_test = train_test_split(v, scores, test_size=0.2, random_state=42)

classifier = LinearSVC()

classifier.fit(X_train, y_train)

preds = classifier.predict(X_test)

print('Accuracy of 3 Class Prediction: ', accuracy_score(y_test, preds))

print(classification_report(y_test, preds))
print(confusion_matrix(y_test, preds))

""" CLASSIFY 1'S AS NEGATIVE, 2'S and 3'S AS POSITIVE, AND DO BINARY CLASSIFICATION """
""" BINARY CLASSIFICATION GETS ABOUT 65% ACCURACY """
X_train2 = X_train
y_train2 = ['n' if (y == '1') else 'p' for y in y_train]

X_test2 = X_test
y_test2 = ['n' if (y == '1') else 'p' for y in y_test]

classifier.fit(X_train2, y_train2)
preds2 = classifier.predict(X_test2)

print('Accuracy of Binary Model: ', accuracy_score(y_test2, preds2))
print(classification_report(y_test2, preds2))
print(confusion_matrix(y_test2, preds2))
