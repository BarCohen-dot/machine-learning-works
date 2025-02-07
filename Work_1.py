# -*- summary -*-
"""
The code **trains and evaluates a Naive Bayes classifier** to classify Iris flower species based on attributes like sepal and petal length and width.  

1. **Loads the Iris dataset** and visualizes the frequency of each flower species using a bar chart.  
2. **Splits the data** into training and testing sets (70% training, 30% testing).  
3. **Trains a Naive Bayes classifier (GaussianNB)** on the training data.  
4. **Predicts flower species** on the test set and computes **accuracy**.  
5. **Displays the confusion matrix** to visualize classification performance.  
6. **Performs 5-fold cross-validation** and reports accuracy scores for each fold, alongside the mean accuracy.  
7. **Calculates the conditional probabilities** of each flower species using the normal distribution for each feature.  
8. **Visualizes 5-fold cross-validation accuracy** through a line plot.  
9. **Prints the classification results**, including accuracy and confusion matrix, using a custom function for model evaluation.

The approach provides a comprehensive evaluation of the Naive Bayes classifier on the Iris dataset, offering insights into classification accuracy and model performance.
"""
# -*- summary -*-

from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
print("--------------------------------------------------------------------------------------------------------------------------------")
# חלק א'
# טעינת מערך הנתונים של Iris
dataSet = datasets.load_iris()
data = dataSet.data
targets = dataSet.target
target_names = dataSet.target_names

#התדירות של כל סוג פרח
unique, counts = np.unique(targets, return_counts=True)
plt.bar(target_names, counts)
plt.xlabel('Flower type')
plt.ylabel('Counts')
plt.title('Frequency of flower type of the iris data')
plt.show()

#פיצול הנתונים לקבוצות אימון ובדיקות (70% אימון, 30% בקרה)
x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=0.3)

print("--------------------------------------------------------------------------------------------------------------------------------")
# חלק ב'
#הגדרת מודל בייס נאיבי
gnb = GaussianNB()

# אימון המודל על נתוני האימון
gnb.fit(x_train, y_train)

# חיזוי על נתוני הבדיקה
y_pred = gnb.predict(x_test)

# חישוב מטריצת הבלבול
cm = confusion_matrix(y_test, y_pred)

# הצגת מטריצת הבלבול
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataSet.target_names)
disp.plot()
# הוספת כותרת
plt.title("Confusion Matrix")
plt.show()
print() # לצורך רווח
# הדפסת אחוז הדיוק של המודל
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# הפרח שהמודל ידע לסווג הכי טוב
most_correctly_classified_index = np.argmax(cm.diagonal())
most_correctly_classified_flower = dataSet.target_names[most_correctly_classified_index]
print(f"The flower that the model classified best is: {most_correctly_classified_flower}")

# הרצת המודל עם 5-fold cross-validation ושמירת אחוזי הדיוק
cross_val_scores = cross_val_score(gnb, data, targets, cv=5)
print("Cross-validation scores:",cross_val_scores)
print(f"Mean cross-validation accuracy: {np.mean(cross_val_scores) * 100:.2f}%")

# הדפסת ההסתברויות לקבלת כל סוג פרח
print("Probability for each flower class:")
for i, flower_class in enumerate(dataSet.target_names):
    print(f"{flower_class}: {gnb.theta_[i]}")

#הצגת גרף אחוזי הדיוק של המסווג, עבור כל אחד מ-5 ההרצות
plt.plot(range(1, 6), cross_val_scores, marker='o')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('5-Fold Cross-Validation Accuracies')
plt.show()
print() # לצורך רווח
print("--------------------------------------------------------------------------------------------------------------------------------")
# חלק ג'
#פונקציה לחישוב ההסתברות המותנית לפי ההתפלגות הנורמלית
def normal_distribution(Average, Variance, x):
    exp_term = -((x - Average) ** 2) / (2 * Variance)
    return np.exp(exp_term) / np.sqrt(2 * np.pi * Variance)

#פונקציה לחיזוי משתנה מטרה למדגם בודד מקבוצת הבדיקה
def predict(x_train, y_train, sample_to_classify):
    #חישוב ממוצע ושונות של כל תכונה בעבור כל סוג פרח
    attribute_stats = {}
    for flower_type in np.unique(y_train):
        flower_data = x_train[y_train == flower_type] #  בודקים האם השם של ה-y שווה לשם של הפרח שאנו עומדים עליו כרגע
        attribute_stats[flower_type] = \
            {'mean': np.mean(flower_data, axis=0), #  כאן מבצעים ממוצע לכול הדאטה לפרח הייחודי הזה לפי עמודה
             'variance': np.var(flower_data, axis=0) } # כאן מבצעים שונות לכול הדאטה לפרח הייחודי הזה לפי עמודה

    #חישוב הסתברויות עבור כל סוג פרח
    flower_probs = []
    for flower_type in np.unique(y_train):
        flower_probability = np.sum(
                             np.log(
                             normal_distribution(
                                    attribute_stats[flower_type]['Mean'], # מוציאים ממוצע לפי פרח
                                    attribute_stats[flower_type]['Variance'], # מוציאים שונות לפי פרח
                                    sample_to_classify)))
        flower_probs.append(flower_probability)

    #החזר את סוג הפרח בהסתברות מקסימלית
    return np.argmax(flower_probs)

#פונקציה עיקרית להפעלת המודל Naive Bayes
def main_naive_bayes(x_train, y_train, x_test, y_test):
    model = GaussianNB()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test) # הצבת ה-x בנוסחא לצורך גילוי החיזוי y
    accuracy = accuracy_score(y_test, y_pred)
    print("\nThe Classification Accuracy: ", accuracy)
    conf_matrix = confusion_matrix(y_test, y_pred) # חישוב מטריצת הבלבול
    print("Confusion Matrix:\n", conf_matrix)

# קריאה לפונקציה הראשית
main_naive_bayes(x_train, y_train, x_test, y_test)
