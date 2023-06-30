import re

# Hàm load dữ liệu từ file data-train.csv để train model
def load_data(file_data = 'data.csv'):
    v_text = []
    v_label = []
    v_topic = []
    with open(file_data, encoding='utf-8') as f:
        lines = f.readlines()
    
    for idl, line in enumerate(lines):
        if idl == 0:
            continue
        line = line.replace("\n","")
        v_text.append(standardize_data(line[:-4]))
        v_label.append(int(line[-4:-2].replace(",", "")))
        v_topic.append(int(line[-1].replace("\n", "")))
    return v_text, v_label, v_topic

# Hàm chuẩn hoá câu
def standardize_data(row):
    # Xóa dấu chấm, phẩy, hỏi ở cuối câu
    row = re.sub(r"[\.,\?]+$-", "", row)
    # Xóa tất cả dấu chấm, phẩy, chấm phẩy, chấm thang, ... trong câu
    row = row.replace(",", " ").replace(".", " ") \
        .replace(";", " ").replace("“", " ") \
        .replace(":", " ").replace("”", " ") \
        .replace('"', " ").replace("'", " ") \
        .replace("!", " ").replace("?", " ") \
        .replace("-", " ").replace("_", " ")
    # Xóa tất cả 2 khoảng trắng
    while (row.find("  ")!= -1):
        row = row.replace("  ", " ")
    row = row.strip().lower()
    return row

# Hàm load danh sách các từ vô nghĩa: lắm, ạ, à, bị, vì..
def load_stopwords():
    sw = []
    with open("stopwords.txt", encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        sw.append(line.replace("\n",""))
    return sw
