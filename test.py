import os,sys
from modules.baseline import Baseline

model_dir = sys.argv[1]

baseline = Baseline()
baseline.load(os.path.join(model_dir, 'brandDict'))

brands = baseline.brands

print(len(brands))

for brand in brands:
    print(brand)
