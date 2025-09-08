import re

def extract_avg_win_chips(log_file, output_file=None):
    # 匹配 avg_win_chips 的值
    pattern = re.compile(r"'avg_win_chips':\s*([-]?\d+\.\d+)")
    
    values = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            matches = pattern.findall(line)
            for m in matches:
                values.append(round(float(m), 3))

    # 奇数位（索引 0,2,4...）
    row1 = [str(values[i]) for i in range(0, len(values), 2)]
    # 偶数位（索引 1,3,5...）
    row2 = [str(values[i]) for i in range(1, len(values), 2)]

    result = "\t".join(row1) + "\n" + "\t".join(row2)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result)
    else:
        print(result)


# 示例用法
if __name__ == "__main__":
    import os
    folder = "/home/cuisijia/llm_opponent_modeling/eval_logs"
    # 遍历多个日志文件
    for file in os.listdir(folder):
        if file.endswith(".log") and "eval_parallel" in file:
            log_path = os.path.join(folder, file)
            output_path = os.path.join(folder, "extracted_logs", file.replace(".log", "_avg_win_chips.log"))
            print(f"Processing {log_path} -> {output_path}")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            extract_avg_win_chips(log_path, output_path)
