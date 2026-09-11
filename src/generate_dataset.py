import pandas as pd


# ------------------------------------------------
# Complaint examples with their priority
# ------------------------------------------------

complaints = {

    "Payment": [
        ("Money was deducted from my account but the payment failed", "High"),
        ("I was charged twice for the same transaction", "High"),
        ("My payment was declined", "High"),
        ("The transaction failed after money was deducted", "High"),
        ("The payment is still showing as pending", "Medium"),
        ("I was charged but did not receive confirmation", "Medium"),
        ("How can I get a refund for my payment", "Medium"),
        ("I am unable to complete the payment", "Medium"),
        ("I want to know the available payment methods", "Low"),
        ("Can I change my preferred payment method", "Low")
    ],

    "Account": [
        ("My account has been locked", "High"),
        ("I cannot login to my account", "High"),
        ("I cannot reset my password", "High"),
        ("I need help recovering my account", "High"),
        ("I forgot my password", "Medium"),
        ("I am unable to change my password", "Medium"),
        ("I cannot access my account", "Medium"),
        ("My account login is not working", "Medium"),
        ("I want to change my registered email address", "Low"),
        ("I want to update my profile information", "Low")
    ],

    "Technical": [
        ("The application crashes when I open it", "High"),
        ("The website is completely unavailable", "High"),
        ("I am getting an error while using the application", "High"),
        ("The application stopped working", "High"),
        ("The mobile application is very slow", "Medium"),
        ("The website takes too long to respond", "Medium"),
        ("The website is showing an error", "Medium"),
        ("The app keeps freezing", "Medium"),
        ("The app is working slowly today", "Low"),
        ("I noticed a small issue with the application", "Low")
    ],

    "Delivery": [
        ("My order has not been delivered yet", "High"),
        ("My package has not arrived", "High"),
        ("My order is delayed for several days", "High"),
        ("My order was delivered to the wrong address", "High"),
        ("My delivery is late", "Medium"),
        ("The delivery date has been changed", "Medium"),
        ("The delivery is taking too long", "Medium"),
        ("The delivery person has not contacted me", "Medium"),
        ("I want to track my order", "Low"),
        ("I want to know the expected delivery date", "Low")
    ],

    "Product": [
        ("The product I received is damaged", "High"),
        ("The item I received is defective", "High"),
        ("The product does not work properly", "High"),
        ("The product stopped working after delivery", "High"),
        ("I received the wrong product", "Medium"),
        ("The product is missing some parts", "Medium"),
        ("The product does not match the description", "Medium"),
        ("The product quality is very poor", "Medium"),
        ("The product looks different from the description", "Low"),
        ("I am not satisfied with the product", "Low")
    ],

    "Service": [
        ("The support team did not respond to my complaint", "High"),
        ("Nobody responded to my request", "High"),
        ("I have been waiting too long for customer support", "High"),
        ("The customer service team did not solve my issue", "High"),
        ("The service was not satisfactory", "Medium"),
        ("I am unhappy with the support provided", "Medium"),
        ("The customer support response was very slow", "Medium"),
        ("I did not receive proper support", "Medium"),
        ("I want to contact customer service", "Low"),
        ("I want more information about the available support", "Low")
    ]
}


# ------------------------------------------------
# Extra phrases
# ------------------------------------------------

extra_phrases = [
    "",
    "Please help me with this issue.",
    "I need this problem resolved.",
    "Can you please check this issue?",
    "This issue needs attention.",
    "Please resolve this as soon as possible."
]


# ------------------------------------------------
# Create dataset
# ------------------------------------------------

data = []


for category in complaints:

    for complaint, priority in complaints[category]:

        for extra in extra_phrases:

            if extra:
                final_complaint = complaint + " " + extra
            else:
                final_complaint = complaint

            data.append({
                "complaint": final_complaint,
                "category": category,
                "priority": priority
            })


# ------------------------------------------------
# Convert to DataFrame
# ------------------------------------------------

df = pd.DataFrame(data)


# ------------------------------------------------
# Shuffle dataset
# ------------------------------------------------

df = df.sample(frac=1, random_state=42).reset_index(drop=True)


# ------------------------------------------------
# Save dataset
# ------------------------------------------------

df.to_csv("dataset/complaints.csv", index=False)


# ------------------------------------------------
# Display information
# ------------------------------------------------

print("Dataset generated successfully!")

print("Total complaints:", len(df))

print("\nCategory distribution:")
print(df["category"].value_counts())

print("\nPriority distribution:")
print(df["priority"].value_counts())

print("\nDuplicate rows:")
print(df.duplicated().sum())