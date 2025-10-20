import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV data
path = r"C:\Users\ompar\Downloads\math IA\Math IA - Sheet1.csv"
df = pd.read_csv(path)

# Convert columns to numeric and remove missing data
for col in ["Grade","WPM","Accuracy","Consistency"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["WPM","Accuracy"]).reset_index(drop=True)

print(f"Total data points: {len(df)}")
print(f"WPM range: {df['WPM'].min():.1f} - {df['WPM'].max():.1f}")
print(f"Accuracy range: {df['Accuracy'].min():.1f} - {df['Accuracy'].max():.1f}")

# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df["WPM"], df["Accuracy"], alpha=0.6, s=30)
plt.xlabel("WPM (Words Per Minute)")
plt.ylabel("Accuracy (%)")
plt.title("Raw Data: Accuracy vs WPM")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the figure
plt.savefig("raw_scatter_accuracy_vs_wpm.png", dpi=300, bbox_inches='tight')
print("\nFigure saved as 'raw_scatter_accuracy_vs_wpm.png'")

plt.show()