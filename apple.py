import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

data = {
    'Date': ['2025-01-10', '2025-02-14', '2025-03-20', '2025-04-25', '2025-05-30'],
    'Subject Line': [
        "Discover the New iPhone Features",
        "Valentine's Day Special Offers",
        "Spring into Innovation with Apple",
        "Exclusive Deals Just for You",
        "Unlock Your Creativity"
    ],
    'Word Count': [250, 200, 220, 180, 210],
    'CTA Count': [3, 4, 2, 3, 2],
    'Personalization': ['Yes', 'No', 'Yes', 'Yes', 'No'],
    'Tone': ['Inspirational', 'Casual', 'Professional', 'Casual', 'Inspirational'],
    'CTR (%)': [5.2, 6.8, 4.5, 7.1, 5.0]
}

df = pd.DataFrame(data)
df['Personalization'] = df['Personalization'].map({'Yes': 1, 'No': 0})
tone_encoder = LabelEncoder()
df['Tone'] = tone_encoder.fit_transform(df['Tone'])

X = df[['Word Count', 'CTA Count', 'Personalization', 'Tone']]
y = df['CTR (%)']

model = LinearRegression()
model.fit(X, y)

intercept = model.intercept_
coefficients = model.coef_
r_squared = model.score(X, y)

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix of Newsletter Features")
plt.show()

print(f"Intercept: {intercept:.3f}")
print("Coefficients:")
for feature, coef in zip(X.columns, coefficients):
    print(f" {feature}: {coef:.3f}")
print(f"R-squared (model fit): {r_squared:.3f}")
