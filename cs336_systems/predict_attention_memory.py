# ChatGPT:

import pandas as pd

# Your benchmark data
data = [
    (16, 256, -0.00030, 0.00110, 25.42188),
    (16, 1024, -0.00197, 0.00645, 148.93750),
    (16, 4096, -0.02839, 0.09697, 2083.00000),
    (16, 8192, -0.11621, 0.40638, 8245.75000),
    (16, 16384, None, None, None),
    (32, 256, -0.00031, 0.00106, 26.54688),
    (32, 1024, -0.00205, 0.00680, 153.43750),
    (32, 4096, -0.03105, 0.10149, 2101.00000),
    (32, 8192, -0.13296, 0.42772, 8281.75000),
    (32, 16384, None, None, None),
    (64, 256, -0.00036, 0.00118, 28.79688),
    (64, 1024, -0.00216, 0.00730, 162.43750),
    (64, 4096, -0.03359, 0.10709, 2137.00000),
    (64, 8192, -0.14530, 0.46977, 8353.75000),
    (64, 16384, None, None, None),
    (128, 256, -0.00036, 0.00106, 33.29688),
    (128, 1024, -0.00272, 0.00905, 180.43750),
    (128, 4096, -0.04378, 0.13543, 2209.00000),
    (128, 8192, -0.19925, 0.61625, 8497.75000),
    (128, 16384, None, None, None),
]

bench_df = pd.DataFrame(
    data, columns=["d_model", "seq_len", "Forward Time (ms)", "Backward Time (ms)", "Memory Before Backward (MB)"]
)


# Predicted memory usage (b=8, fp32)
def mb_for(d, L, bytes_per_el, b=1):
    elements = b * (5 * L * d + 2 * L * L)
    return elements * bytes_per_el / (1024**2)


rows_b8 = []
for d in [16, 32, 64, 128]:
    for L in [256, 1024, 4096, 8192, 16384]:
        rows_b8.append({"d_model": d, "seq_len": L, "Predicted Memory Usage (MB)": round(mb_for(d, L, 4, b=8), 2)})

df_b8 = pd.DataFrame(rows_b8)

# Merge benchmark results with predictions
merged_df = pd.merge(bench_df, df_b8, on=["d_model", "seq_len"], how="left")

print(merged_df.to_markdown(index=False))
