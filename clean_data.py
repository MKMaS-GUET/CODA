import pandas as pd
import re

def clean_new_dataset(input_path, output_path):
    df = pd.read_csv(input_path, header=None)
    cleaned_lines = []
    skipped_lines = 0

    for i, line in enumerate(df[0]):
        if pd.isna(line):
            print(f"⚠️ 第 {i+1} 行为空，已跳过")
            skipped_lines += 1
            continue

        # 去除不可见字符并规范为标准空格
        raw = str(line).strip().replace('\u3000', ' ').replace('\xa0', ' ').replace('\t', ' ')
        tokens = re.split(r'[,\s]+', raw)
        filtered = [x for x in tokens if x.isdigit()]

        if not filtered:
            print(f"⚠️ 第 {i+1} 行无有效数字，已跳过")
            skipped_lines += 1
            continue

        cleaned_line = ' '.join(filtered)
        cleaned_lines.append(cleaned_line)

    # 保存为新的 CSV 文件
    cleaned_df = pd.DataFrame(cleaned_lines)
    cleaned_df.to_csv(output_path, index=False, header=False)
    print(f"\n✅ 清洗完成：总行数 {len(df)}, 跳过 {skipped_lines}, 保留 {len(cleaned_lines)}")
    print(f"📁 已保存到: {output_path}")


if __name__ == '__main__':
    dataset_name = "wiki"  # 修改为你的数据集名（无后缀）
    input_file = f"/workspace/ACE/data/{dataset_name}_new.csv"
    output_file = f"/workspace/ACE/data/{dataset_name}_new_cleaned.csv"

    clean_new_dataset(input_file, output_file)
