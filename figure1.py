import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = r"C:\Users\ompar\Downloads\math IA\Math IA - Sheet1.csv"
df = pd.read_csv(path)

for col in ["Grade","WPM","Accuracy","Consistency"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["WPM","Accuracy"]).reset_index(drop=True)

wpm_min = int(np.floor(df["WPM"].min()/10)*10)
wpm_max = int(np.ceil(df["WPM"].max()/10)*10)
bins = np.arange(wpm_min, wpm_max+10, 10)
labels = bins[:-1]
df["Bin"] = pd.cut(df["WPM"], bins=bins, right=False, include_lowest=True, labels=labels)

def merge_sparse(d):
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    iteration_count = 0
    max_iterations = 100 
    
    while iteration_count < max_iterations:
        counts = d.groupby("Bin", dropna=True, observed=True).size().sort_index()
        if counts.empty or (counts.min() >= 3) or len(counts) == 1:
            return d
        b = counts[counts<3].index[0]
        idx = list(counts.index)
        i = idx.index(b)
        if i == 0:
            target = idx[1]
        elif i == len(idx)-1:
            target = idx[-2]
        else:
            left = counts.iloc[i-1]
            right = counts.iloc[i+1]
            target = idx[i-1] if left <= right else idx[i+1]
        
    
        d = d.copy()
        d["Bin"] = d["Bin"].replace(b, target)
        
        iteration_count += 1
    
    print(f"Warning: merge_sparse stopped after {max_iterations} iterations")
    return d

df = merge_sparse(df)

summary = (
    df.groupby("Bin", observed=True)
      .agg(MedianAccuracy=("Accuracy","median"),
           P90Accuracy=("Accuracy", lambda s: float(np.percentile(s,90))),
           Count=("Accuracy","size"))
      .reset_index()
      .sort_values("Bin")
)

def plateau_onset(summ, threshold=95, tol=2):
    vals = summ["MedianAccuracy"].values
    bins_le = summ["Bin"].astype(int).values
    for i, m0 in enumerate(vals):
        if m0 >= threshold and np.all(vals[i:] >= m0 - tol):
            return int(bins_le[i]), float(m0)
    return None, None

onset_bin, onset_median = plateau_onset(summary)

print(summary.to_string(index=False))
print("\nPlateau onset (rule-based):")
print({"OnsetBinLowerEdge": onset_bin, "OnsetBinMedianAccuracy": onset_median,
       "Rule": "first bin with median ≥95% and all later medians within 2 points"})

plt.figure()
plt.plot(summary["Bin"].astype(int), summary["MedianAccuracy"], marker="o", label="Median")
plt.plot(summary["Bin"].astype(int), summary["P90Accuracy"], marker="s", label="90th percentile")
plt.axhline(95, linestyle="--")
if onset_bin is not None:
    plt.axvline(onset_bin, linestyle=":")
plt.xlabel("WPM bin (lower edge)")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy by WPM Bins")
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_by_wpm_bins.png", dpi=300, bbox_inches='tight')
print("\nFigure saved as 'accuracy_by_wpm_bins.png'")
plt.show()
