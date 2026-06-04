import psycopg2

# 1. 连接数据库
conn = psycopg2.connect(
    dbname="setdb",
    user="postgres",
    password="123456",   # 替换成你的密码
    host="localhost",     # 或 '127.0.0.1'
    port=5432
)
cur = conn.cursor()

# 2. 建表（如果已建跳过）
cur.execute("""
    CREATE TABLE IF NOT EXISTS set_collection (
        id serial primary key,
        elements integer[]
    );
""")
conn.commit()

# 3. 逐行读取文件，插入数据
with open('/workspace/ACE /data/gn.csv', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for idx, line in enumerate(lines): 
    # 跳过第一行（元素个数）
    if idx == 0:
        continue
    # 分割为整数列表
    numbers = [int(x) for x in line.strip().split()]
    # 插入
    cur.execute(
        "INSERT INTO set_collection (elements) VALUES (%s);",
        (numbers,)
    )

conn.commit()
cur.close()
conn.close()
print("导入完成！")
