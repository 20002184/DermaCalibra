import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# 读取测试集
test_df = pd.read_csv('pad-ufes-20_parsed_test.csv')

# 确保数据完整性，移除空行
test_df = test_df.dropna(subset=['diagnostic'])

# 初始化分层划分工具
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)

# 获取特征和标签
X = test_df.drop(columns=['diagnostic'])
y = test_df['diagnostic']

# 进行分层划分
for train_index, test_index in sss.split(X, y):
    subtest_A = test_df.iloc[train_index].copy()
    subtest_B = test_df.iloc[test_index].copy()
    break

# 保存子集
subtest_A.to_csv('subtest_A.csv', index=False)
subtest_B.to_csv('subtest_B.csv', index=False)

# 验证划分意义
# 1. 诊断类别分布
print("Subtest A 诊断类别分布:\n", subtest_A['diagnostic'].value_counts(normalize=True))
print("Subtest B 诊断类别分布:\n", subtest_B['diagnostic'].value_counts(normalize=True))

# 2. 病灶部位分布
region_columns = [col for col in test_df.columns if col.startswith('region_')]
subtest_A_regions = subtest_A[region_columns].sum() / len(subtest_A)
subtest_B_regions = subtest_B[region_columns].sum() / len(subtest_B)
print("\nSubtest A 病灶部位分布:\n", subtest_A_regions)
print("Subtest B 病灶部位分布:\n", subtest_B_regions)

# 3. 年龄分布
print("\nSubtest A 年龄统计:\n", subtest_A['age'].describe())
print("Subtest B 年龄统计:\n", subtest_B['age'].describe())

# 4. 其他特征分布（可选：如 diameter_1, diameter_2）
print("\nSubtest A 病灶直径 (diameter_1) 统计:\n", subtest_A['diameter_1'].describe())
print("Subtest B 病灶直径 (diameter_1) 统计:\n", subtest_B['diameter_1'].describe())