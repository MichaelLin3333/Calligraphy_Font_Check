import os
from collections import Counter, defaultdict
from train_model import font_to_label


DATA_DIR = r'C:\Users\Michael Lin\projects\shufa\chinese_fonts'
print(f"Checking dataset directory: {DATA_DIR}")
exts = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif', '.tiff')

cnt = Counter()
unknown_prefixs = Counter()
samples = defaultdict(list)
total = 0

for root, _, files in os.walk(DATA_DIR):
    for f in files:
        if not f.lower().endswith(exts):
            continue
        total += 1
        prefix = f[0:2]
        lbl = font_to_label(prefix)
        if lbl == -1:
            unknown_prefixs[prefix] += 1
            if len(samples[f'{prefix}_unknown']) < 5:
                samples[f'{prefix}_unknown'].append(os.path.join(root, f))
        else:
            cnt[lbl] += 1
            if len(samples[str(lbl)]) < 5:
                samples[str(lbl)].append(os.path.join(root, f))

print(f"Dataset path: {DATA_DIR}")
print(f"Total images found: {total}")
print("Per-font label counts (label -> count):")
for k, v in sorted(cnt.items()):
    print(f"  {k}: {v}")
if unknown_prefixs:
    print("\nUnknown filename prefixes (first 10):")
    for p, c in unknown_prefixs.most_common(10):
        print(f"  '{p}': {c} files")
print("\nSample files per detected label / unknown:")
for k, v in samples.items():
    print(f"Label {k}:")
    for s in v:
        print(f"  {s}")