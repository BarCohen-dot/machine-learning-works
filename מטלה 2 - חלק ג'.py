import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

print("============================================================================================================================================================")
df = pd.read_csv('drug200.csv') # טען את קובץ הנתונים
Five_top_rows = pd.read_csv('drug200.csv',nrows = 5) # טען את 5 השורות הראשונות
print(Five_top_rows)
print("============================================================================================================================================================")
print(f"Dataset dimensions: {df.shape}") # הדפס את מידות הקובץ (מספר שורות ועמודות)
print("============================================================================================================================================================")
print(df.describe())

x = df.drop('Drug', axis = 1)  # הגדרת ערכים / הגדרת מטריצת הערכים ללא עמודת Drug
y = df['Drug']                       # הגדרת משתנה מטרה Drug

x_transformed = pd.get_dummies(x) # סיווג המשתנים הקטגוריאלים לערכים כמותיים

x_train,x_test,y_train,y_test = train_test_split(x_transformed ,y ,test_size=0.3) # חלוקת הנתונים לאימון ולבדיקה ,בנוסף נציין את אחוזי נתוני הבדיקה test_size

print("============================================================================================================================================================")

#================================================================================================================================================
# אימן וויזואליזציה של עץ החלטה עם מדד אי - הטיה ״אנטרופיה״
entropy_tree = DecisionTreeClassifier(criterion='entropy')  # הגדרת המודל
entropy_tree.fit(x_train, y_train)                          # ביצוע אימון

# חישוב דיוק המודל עם אנטרופיה
y_pred_entropy = entropy_tree.predict(x_test)
entropy_accuracy = accuracy_score(y_test, y_pred_entropy)
print(f"Entropy Model Accuracy: {entropy_accuracy}")

# הצגת גרף עץ ההחלטה
plt.figure( figsize = (10,10) )   # plt.figure(...), יוצרת אובייקט Figure חדש שמייצג חלון ריק שבו ניתן לצייר את הגרפים
                                  # figsize = (10,10), פרמטר זה מגדיר את גודל החלון שבו יצויר הגרף. הערך (10,10) מציין את גודל החלון באינצ'ים (רוחב x גובה)
plot_tree(entropy_tree, filled = True, max_depth = 5, feature_names = x_transformed.columns, class_names = entropy_tree.classes_)
                                                        #feature_names = X.columns, פרמטר זה מציין את שמות המשתנים הבלתי תלויים (המאפיינים) שמשמשים בעץ ההחלטה, המאפיינים איתם נחשב את המטרה לצומת
                                                        # filled = True, פרמטר זה צובע את הצמתים של העץ בצבעים שונים בהתבסס על המידע שלהם, מה שמקל על ההבנה הוויזואלית של העץ
plt.title('Decision Tree with Entropy')  # הגדרת כותרת
plt.show()

print("============================================================================================================================================================")
# הרצת מודלים בעומק מקסימלי שנע בין 1 ל10
depths = list(range(1, 11)) # נעבור בלולאה בטווח של 1 עד 11 , כלומר מ1 ועד 10 לא כולל האחרון
accuracies = []             # נפתח רשימה חדשה לצורך הזנת אחוזי הדיוק

for depth in depths:
    entropy_tree = DecisionTreeClassifier(criterion='entropy', max_depth=depth)
    entropy_tree.fit(x_train, y_train)
    y_pred = entropy_tree.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    plt.figure(figsize=(10, 10))
    plot_tree(entropy_tree, filled=True, max_depth=depth, feature_names=x_transformed.columns, class_names=entropy_tree.classes_)
    plt.title(f'Decision Tree with Entropy (Depth={depth})')
    plt.show()

# הצגת גרף אחוז הדיוק כפונקציה של עומק העץ
plt.figure(figsize = (10, 6))
plt.plot(depths, accuracies, marker='o', linestyle='-', color='b') # קביעת עיצוב הגרף לאחוזי הדיוק
plt.title('Model Accuracy vs. Tree Depth')
plt.xlabel('Tree Depth')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()












