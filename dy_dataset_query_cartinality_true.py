import psycopg2
import random
import csv
import time

# ========== 配置 ==========
DB_CONF = dict(
    dbname="setdb",
    user="postgres",
    password="123456",  # <--- 改为你自己的
    host="localhost",
    port=5432
)
QUERY_NUM = 100
QUERY_LEN = (5, 10)
NEW_ELEMENT_RANGE = 10000
NEW_ELEMENT_NUM = 5000
NEW_SET_LEN = (2, 10)  # 新插入集合的长度区间

files = {
    'overlap': 'overlap_queries.txt',
    'subset': 'subset_queries.csv',  # 修改为csv文件
    'superset': 'superset_queries.txt'
}
NEW_SETS_FILE = 'new_sets.csv'
NEW_SETS_WITH_NEW_ELE_FILE = 'new_sets_with_new_elements.csv'

print("1. 正在连接数据库...")
conn = psycopg2.connect(**DB_CONF)
cur = conn.cursor()

print("2. 获取现有集合数据...")
cur.execute("SELECT elements FROM gn;")
regular = [set(row[0]) for row in cur.fetchall()]
print(f"   数据库现有集合数量: {len(regular)}")

# ========== 2. 生成新元素池 ==========
print("3. 生成新元素池...")
old_elements = set()
max_elem = -1
for s in regular:
    old_elements.update(s)
    if s:
        max_elem = max(max_elem, max(s))
new_candidates = list(range(max_elem + 1, max_elem + 1 + NEW_ELEMENT_RANGE))
new_elements = set(random.sample(new_candidates, NEW_ELEMENT_NUM))
all_elements = list(old_elements | new_elements)
print(f"   老元素数量: {len(old_elements)}, 新元素数量: {len(new_elements)}, 合并总数: {len(all_elements)}")

# ========== 3. 插入新混合集合并保存到CSV ==========
print(f"4. 开始插入 {NEW_ELEMENT_NUM} 条新混合集合数据...")

min_new_elements = 3  # ✅ 每个集合至少包含几个新元素

with open(NEW_SETS_FILE, "w", encoding="utf-8", newline="") as f_new, \
     open(NEW_SETS_WITH_NEW_ELE_FILE, "w", encoding="utf-8", newline="") as f_new_ele:
    writer_all = csv.writer(f_new)
    writer_with_new = csv.writer(f_new_ele)
    inserted = 0
    contain_new = 0
    for i in range(NEW_ELEMENT_NUM):
        k = random.randint(*NEW_SET_LEN)
        if k < min_new_elements:
            k = min_new_elements  # 避免采样出错
        num_new = min(min_new_elements, k)
        num_other = k - num_new

        new_part = random.sample(new_elements, num_new)
        other_part = random.sample(all_elements, num_other) if num_other > 0 else []
        sample = sorted(new_part + other_part)

        cur.execute("INSERT INTO gn (elements) VALUES (%s);", (sample,))
        writer_all.writerow(sample)
        writer_with_new.writerow(sample)
        contain_new += 1

        inserted += 1
        if inserted % 1000 == 0 or inserted == NEW_ELEMENT_NUM:
            print(f"   已插入 {inserted}/{NEW_ELEMENT_NUM} 条，包含新元素的集合 {contain_new} 条")

conn.commit()
print(f"   插入完成，共 {inserted} 条新数据。")
print(f"   所有新集合已保存至 {NEW_SETS_FILE}")
print(f"   包含新元素的集合已保存至 {NEW_SETS_WITH_NEW_ELE_FILE}")

# ========== 5. 用SQL高效生成查询负载 ==========
def gen_queries_sql(kind):
    res = []
    tried = set()
    last_report = time.time()
    print(f"   开始生成 {kind} 查询...")

    while len(res) < QUERY_NUM:
        if kind == 'overlap':
            q_len = random.randint(*QUERY_LEN)
        elif kind == 'subset':
            q_len = random.randint(3, 6)  # 更小的集合以提升匹配率
        elif kind == 'superset':
            q_len = random.randint(5, 10)
        else:
            continue

        new_elements_sample = random.sample(new_elements, 2)
        remaining_len = q_len - 2
        old_elements_sample = random.sample(old_elements, remaining_len) if remaining_len > 0 else []
        q = sorted(new_elements_sample + old_elements_sample)
        q_tuple = tuple(q)
        if q_tuple in tried:
            continue
        tried.add(q_tuple)

        if kind == 'overlap':
            sql = "SELECT COUNT(*) FROM gn WHERE elements && %s;"
            cur.execute(sql, (q,))
            count = cur.fetchone()[0]
        elif kind == 'superset':
            sql = "SELECT COUNT(*) FROM gn WHERE %s @> elements;"
            cur.execute(sql, (q,))
            count = cur.fetchone()[0]
        elif kind == 'subset':
            # 加速 subset 查询：先用 overlap 粗筛，再 Python 过滤
            sql = "SELECT elements FROM gn WHERE elements && %s;"
            cur.execute(sql, (q,))
            rows = cur.fetchall()
            count = sum(1 for row in rows if set(row[0]).issubset(set(q)))
        else:
            continue

        if count > 0:
            res.append((q, count))
            now = time.time()
            if len(res) % 10 == 0 or len(res) == QUERY_NUM or now - last_report > 5:
                print(f"      {kind} 查询已生成 {len(res)}/{QUERY_NUM} 条...")
                last_report = now

    print(f"   {kind} 查询生成完成。")
    return res

print("5. 开始批量生成查询负载（overlap/subset/superset）...")

overlap_queries = gen_queries_sql('overlap')
subset_queries = gen_queries_sql('subset')
superset_queries = gen_queries_sql('superset')

cur.close()
conn.close()

# ========== 6. 保存为CSV ==========
print("6. 正在保存查询负载到CSV文件...")
for key, query_list in zip(files.keys(), [overlap_queries, subset_queries, superset_queries]):
    if key == 'subset':
        with open(files[key], "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["query", "cardinality"])
            for q, count in query_list:
                writer.writerow([" ".join(map(str, q)), count])
    else:
        with open(files[key], "w", encoding="utf-8") as f:
            for q, count in query_list:
                f.write(" ".join(map(str, q)) + "," + str(count) + "\n")
    print(f"   {key} 查询负载已保存至 {files[key]}")
print("全部完成！三个查询负载文件已生成：", list(files.values()))
print("所有新集合文件：", NEW_SETS_FILE, "包含新元素集合文件：", NEW_SETS_WITH_NEW_ELE_FILE)
