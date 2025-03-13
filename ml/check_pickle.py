import pickle

# 載入 pickle 文件
file_path = "D:\MLGame/lhu-csie-arkanoid/log/Nomal1_2025_01_15_20_13_00.pickle"  # 替換為您的 pickle 文件路徑
with open(file_path, "rb") as f:
    data = pickle.load(f)

# 檢查文件內容
print(type(data))  # 查看數據的類型
print(data.keys())  # 如果是字典，檢查鍵值
print(data)  # 查看文件的具體內容（適用於小型數據）