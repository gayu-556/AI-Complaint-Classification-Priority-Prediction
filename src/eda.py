import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("dataset/complaints.csv")

# Remove duplicate rows
data = data.drop_duplicates()

# -------------------------------
# 1. Category Distribution
# -------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(data=data, x="category")

plt.title("Complaint Category Distribution")
plt.xlabel("Complaint Category")
plt.ylabel("Number of Complaints")

plt.xticks(rotation=30)

plt.tight_layout()
plt.show()


# -------------------------------
# 2. Priority Distribution
# -------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(data=data, x="priority")

plt.title("Complaint Priority Distribution")
plt.xlabel("Priority")
plt.ylabel("Number of Complaints")

plt.tight_layout()
plt.show()