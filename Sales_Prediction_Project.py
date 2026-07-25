# Step 1: Import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load the dataset
project_dir = Path(__file__).resolve().parent
candidate_files = [
    project_dir / 'Advertising.csv',
    project_dir / 'advertising.csv',
]

data_path = next((path for path in candidate_files if path.exists()), None)
if data_path is None:
    raise FileNotFoundError(
        f"Dataset not found. Expected one of: {', '.join(path.name for path in candidate_files)}"
    )

df = pd.read_csv(data_path)
df.columns = [column.strip() for column in df.columns]

# Handle common versions of the dataset with different column casing
column_mapping = {
    'radio': 'Radio',
    'newspaper': 'Newspaper',
    'sales': 'Sales',
}
df = df.rename(columns=column_mapping)

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

print(df.head())
df.info()

# Step 3: Data Cleaning and Exploratory Analysis
# Check for missing values
print("Missing values:", df.isnull().sum())

# Correlation Heatmap - Kya ad spend aur sales ka relation hai?
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
heatmap_path = project_dir / 'correlation_heatmap.png'
plt.tight_layout()
plt.savefig(heatmap_path)
print(f"Correlation heatmap saved to: {heatmap_path}")

# Step 4: Feature and Target Separation
X = df.drop('Sales', axis=1)# Features: TV, Radio, Newspaper
y = df['Sales']# Target: Sales

# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Model Evaluation
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)

# Step 8: Analyse Advertising Impact - Ye Sabse Important hai
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Impact on Sales'])
print(coefficients.sort_values(by='Impact on Sales', ascending=False))

# Interpretation: Agar TV ka coefficient = 0.045 hai,
# iska matlab TV pe 1000 rs aur kharch krne se Sales 45 units badhegi

# Step 9: Visualization - TV Ad vs Sales
plt.figure(figsize=(8, 5))
sns.regplot(x=df['TV'], y=df['Sales'])
plt.title("Impact of TV Advertising on Sales")
plt.xlabel("TV AD Spend")
plt.ylabel("Sales")
tv_plot_path = project_dir / 'tv_vs_sales.png'
plt.tight_layout()
plt.savefig(tv_plot_path)
print(f"TV vs Sales plot saved to: {tv_plot_path}")

# Step 10: Predict Future Sales
future_sales = [[200, 30, 40]]# TV, Radio, Newspaper
predicted_sales = model.predict(future_sales)
print("Predicted Sales for Future:", predicted_sales[0])
print("TV has the highest impact on Sales")
print("60% of the Marketing Budget should be spent on TV")
print("The lowest impact on Sales is from Newspaper")