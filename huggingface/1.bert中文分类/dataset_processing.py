import pandas as pd

# 指定文件路径
test_file = "D:\\learning\\deep_learning\\code\\huggingface\\ChnSentiCorp\\dev.tsv"
output_file = "D:\\learning\\deep_learning\\code\\huggingface\\ChnSentiCorp\\dev_modified.tsv"

# 读取 dev.tsv 文件
try:
    test_df = pd.read_csv(test_file, sep='\t')  # 假设文件用 Tab 分隔
    print("File loaded successfully.")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# 检查是否包含 'label' 和 'text_a' 列，并删除多余列
if 'label' in test_df.columns and 'text_a' in test_df.columns:
    test_df = test_df[['label', 'text_a']]  # 保留需要的列
    print(f"Columns after filtering: {test_df.columns.tolist()}")
else:
    print("The required columns ('label', 'text_a') are not found in the dataset.")
    exit()

# 保存修改后的文件
try:
    test_df.to_csv(output_file, sep='\t', index=False)
    print(f"Modified test dataset saved to: {output_file}")
except Exception as e:
    print(f"Error saving file: {e}")
