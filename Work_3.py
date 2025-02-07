import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#==================================================================================================================
# פונקציה לחישוב מרחק אוקלידי בין כל נקודה במערך למרכז נתון
def distance(data, center):
    dist = (np.sum((data - center) ** 2, axis=1)) ** 0.5
    return dist

# פונקציה לשיבוץ כל נקודה בנתונים לקלסטר (= אשכול) הקרוב ביותר
def fit(data, centers):
    distances = np.zeros([len(data), len(centers)])
    for i, c in enumerate(centers):
        distances[:, i] = distance(data, c)
    clusters = np.argmin(distances, axis=1)
    return clusters

# פונקציה לעדכון מרכזי הקלסטרים לפי הממוצע של הנקודות בכל קלסטר
def update_centers(data, clusters):
    rows, cols = np.shape(data)
    unique_clusters = np.unique(clusters)
    new_centers = np.zeros([len(unique_clusters), cols])
    for i, c in enumerate(unique_clusters):
        new_centers[i] = np.mean(data[clusters == c], axis=0)
    return new_centers

# אלגוריתם KMeans
def KMeans(n_clusters, data, max_iter):
    # בחירת מרכזים רנדומליים
    centers = np.random.permutation(data)[:n_clusters]
    clusters = fit(data, centers)
    new_centers = update_centers(data, clusters)
    sub = np.sum(new_centers - centers)
    i = 1

    # עדכון מרכזים ושיבוץ מחדש של הנקודות עד להתכנסות או מקסימום איטרציות (= בחרנו ב300)
    while sub != 0 and i < max_iter:
        centers = new_centers
        clusters = fit(data, centers)
        new_centers = update_centers(data, clusters)
        sub = np.sum(new_centers - centers)
        i += 1
    return clusters, new_centers, i

# פונקציה לביצוע מינימום-מקסימום סקיילינג לנתונים (נרמול הנתונים)
def min_max_scaling(data):
    min_cols = np.min(data, axis=0)
    max_cols = np.max(data, axis=0)
    scaled_data = (data - min_cols) / (max_cols - min_cols)
    return scaled_data

# פונקציה לחישוב SSE (סכום הריבועי השגיאות)
def calc_sse(data, clusters, centers):
    sse = 0
    for i in range(len(centers)):
        current_cluster = data[clusters == i]
        d = distance(current_cluster, centers[i])
        sse += np.sum(d ** 2)
    return sse

def create_scatter_plot(x, y, clusters, x_label, y_label):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, c=clusters, cmap='rainbow', marker='o', alpha=0.5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'{y_label} vs {x_label}')
    plt.show()
    
#==================================================================================================================
# קריאת הנתונים מתוך קובץ CSV
data = pd.read_csv("Family Income and Expenditure.csv")

# חיתוך הנתונים לשתי עמודות: הכנסה כוללת של משק הבית ,והוצאות כוללות על מזון
mini_data = data[['Total Household Income', 'Total Food Expenditure']]

# ביצוע מינימום-מקסימום סקיילינג לנתונים (שולחים לפונקציית הנרמול)
scaled_data = min_max_scaling(mini_data.values)

# הרצת אלגוריתם KMeans עם 15 קלסטרים (לפי ההנחיות)
clusters, centers, num_iter = KMeans(15, scaled_data, 300)
sse = calc_sse(scaled_data, clusters, centers)
print("SSE = ", sse)

# הצגת גרף הפיזור של הנתונים עם האשכולות
plt.scatter(mini_data['Total Household Income'], mini_data['Total Food Expenditure'], c=clusters, cmap='rainbow')
plt.title("The original data with the cluster classification")
plt.xlabel("Total Household Income")
plt.ylabel("Total Food Expenditure")
plt.grid()
plt.show()

k_values = np.arange(1, 16)
sse_values = []

# חישוב SSE לכל מספר קלסטרים בין 1 ל-15
for k in k_values:
    clusters, centers, num_iter = KMeans(k, scaled_data, 300)
    current_sse = calc_sse(scaled_data, clusters, centers)
    sse_values.append(current_sse) # הוספת ה-SSE הנוכחי לרשימה

# הצגת גרף הנתונים עם האשכולות כאשר אנו מסווגים 15 קבוצות
plt.plot(k_values, sse_values)
plt.title("Elbow method with")
plt.xlabel("num of clusters")
plt.ylabel("SSE")
plt.grid()
plt.show()

print("According to the graph, the breaking point is 3 (according to the x-axis)")

# הרצת KMeans עם כמות הקלסטרים שנבחרה והצגת התוצאות
clusters, centers, num_iter = KMeans(3, scaled_data, 300)
print("SSE = ",sse_values[2],", Cluster center:", centers)

# הצגת גרף הנתונים עם האשכולות
plt.scatter(mini_data['Total Household Income'], mini_data['Total Food Expenditure'], c=clusters, cmap='rainbow')
plt.title(f"cluster classification  with {3} clusters")
plt.xlabel("Total Household Income")
plt.ylabel("Total Food Expenditure")
plt.grid()
plt.show()
#==================================================================================================================

# Create scatter plots 
create_scatter_plot(data['Total Household Income'], data['Total Food Expenditure'], clusters, 
                    'Total Household Income', 'Total Food Expenditure')

create_scatter_plot(data['Total Household Income'], data['Restaurant and hotels Expenditure'], clusters, 
                    'Total Household Income', 'Restaurant and hotels Expenditure')

create_scatter_plot(data['Total Household Income'], data['Alcoholic Beverages Expenditure'], clusters, 
                    'Total Household Income', 'Alcoholic Beverages Expenditure')

create_scatter_plot(data['Total Household Income'], data['Medical Care Expenditure'], clusters, 
                    'Total Household Income', 'Medical Care Expenditure')
