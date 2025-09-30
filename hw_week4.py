import random

from ortools.sat.python import cp_model
import pandas as pd
import math
import re
from collections import defaultdict, Counter
import os
from datetime import datetime


#  Data Generation 

CSV_PATH = "학급반편성CSP 문제 입력파일.csv"

NUM_CLASSROOMS = 6

def parse_list_cell(cell):
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    if not s:
        return []
    s = s.replace(';', ',')
    s = re.sub(r'[\[\]\(\)\'"]', '', s)
    parts = [p.strip() for p in s.split(',') if p.strip()]
    out = []
    for p in parts:
        if ' ' in p and re.match(r'^\S+\s+\S+$', p):
            out.extend([x.strip() for x in p.split() if x.strip()])
        else:
            out.append(p)
    return out

def normalize_sex(x):
    if pd.isna(x): 
        return 'U'
    s = str(x).strip().upper()
    if s in ('boy', 'BOY', 'BOYS', 'B'):
        return 'M'
    if s in ('girl', 'GIRL', 'GIRLS', 'G'):
        return 'F'
    return 'U'

def make_exact_target_quota(total_cnt, capacities):
    """
    정원 비례 목표치를 반별로 정확히 분할하는 벡터를 만든다.
    기본은 round를 쓰되, 반별 합의 총합이 total_cnt와 불일치하면 마지막 반에서 보정.
    """
    N = sum(capacities)
    quotas = [int(round(total_cnt * cap / N)) for cap in capacities]
    diff = total_cnt - sum(quotas)
    # 보정: 마지막 반에 diff를 더함(음수면 뺌). 필요하면 다른 분배 방식으로 교체 가능.
    if diff != 0:
        quotas[-1] += diff
    assert sum(quotas) == total_cnt, "타겟 쿼터 합계 불일치"
    return quotas

#  Load CSV 
df = pd.read_csv(CSV_PATH)
required_cols = ['id','name','sex','score','24년 학급','클럽','좋은관계','나쁜관계','Leadership','Piano','비등교','운동선호']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"필수 열 '{col}'이(가) CSV 파일에 없습니다.")
    
df['id'] = df['id'].astype(int)
df['name'] = df['name'].astype(str)
df['sex'] = df['sex'].apply(normalize_sex)
df['score'] = pd.to_numeric(df['score'], errors='coerce')
df['24년 학급'] = df['24년 학급'].astype(str)
df['클럽'] = df['클럽'].apply(parse_list_cell)
df['Leadership'] = df['Leadership'].astype(str)
df['Piano'] = df['Piano'].astype(str)
df['비등교'] = df['비등교'].astype(str)
df['운동선호'] = df['운동선호'].astype(str)
df['good_id'] = pd.to_numeric(df['좋은관계'], errors='coerce').astype('Int64')  # 챙겨주는 친구 id
df['bad_id']  = pd.to_numeric(df['나쁜관계'], errors='coerce').astype('Int64')  # 같은 반 금지 id

# yes/blank 플래그 표준화
def is_yes(x):
    if pd.isna(x):
        return False
    return str(x).strip().lower() == 'yes'

df['Leadership_flag'] = df['Leadership'].apply(is_yes)
df['Piano_flag']      = df['Piano'].apply(is_yes)
df['Truancy_flag']    = df['비등교'].apply(is_yes)
df['Sport_flag']      = df['운동선호'].apply(is_yes)

students = df['id'].tolist()
NUM_STUDENTS = len(students)

CLASSROOM_IDS = list(range(NUM_CLASSROOMS))

def generate_capacities(num_students, num_classes):
    base = num_students // num_classes
    remainder = num_students % num_classes
    capacities = [base] * num_classes
    # 나머지를 앞쪽 반에 +1 씩 배분
    for i in range(remainder):
        capacities[i] += 1
    random.shuffle(capacities)  # 실행마다 다르게 하고 싶으면 셔플
    return capacities

print(f"총 학생 수: {NUM_STUDENTS}")

CAPACITIES = generate_capacities(NUM_STUDENTS, NUM_CLASSROOMS)
print(f"반 정원들: {CAPACITIES}")

# 1. 학생들 성적
student_scores = {sid: (0 if pd.isna(sc) else int(sc)) for sid, sc in zip(df['id'], df['score'])}

#  2. 학생들 싫어하는 관계 
student_dislikes = set()
id_set = set(df['id'].to_list())
for _, row in df.iterrows():
    s1 = int(row['id'])
    s2 = row['bad_id']
    if pd.notna(s2):
        s2 = int(s2)
        if s1 != s2 and s2 in id_set and (s2, s1) not in student_dislikes:
            student_dislikes.add((s1, s2))

print(f"Dislikes: {student_dislikes}")

#  전 처리 후 집합/그룹 
leader_ids  = set(df.loc[df['Leadership_flag'], 'id'])
piano_ids   = set(df.loc[df['Piano_flag'], 'id'])
truancy_ids = set(df.loc[df['Truancy_flag'], 'id'])
sport_ids   = set(df.loc[df['Sport_flag'], 'id'])

male_ids    = {sid for sid, sx in zip(df['id'], df['sex']) if sx == 'M'}
female_ids  = {sid for sid, sx in zip(df['id'], df['sex']) if sx == 'F'}

# 이전 학급 그룹 
prev_groups = defaultdict(list)
for sid, pc in zip(df['id'], df['24년 학급'].astype(str)):
    prev_groups[pc].append(sid)
prev_caps = {k: math.ceil(len(v) / NUM_CLASSROOMS) for k, v in prev_groups.items() if len(v) > 0}

# 클럽 그룹 
club_groups = defaultdict(list)
for sid, clubs in zip(df['id'], df['클럽']):
    for c in clubs:
        if c:
            club_groups[c].append(sid)
club_caps = {k: math.ceil(len(v) / NUM_CLASSROOMS) for k, v in club_groups.items() if len(v) > 0}

# 타깃 비율
piano_targets   = make_exact_target_quota(len(piano_ids), CAPACITIES)
truancy_targets = make_exact_target_quota(len(truancy_ids), CAPACITIES)
male_targets    = make_exact_target_quota(len(male_ids), CAPACITIES)
female_targets  = make_exact_target_quota(len(female_ids), CAPACITIES)
sport_targets   = make_exact_target_quota(len(sport_ids), CAPACITIES)

# 점수 균형: 반별 총점 목표 및 허용범위(+/- 0.1%)
total_score = sum(student_scores.values())
score_target = [ total_score * cap / NUM_STUDENTS for cap in CAPACITIES ]
SCORE_TOLERANCE_PERCENT = 0.001
min_class_score = [ st * (1 - SCORE_TOLERANCE_PERCENT) for st in score_target ]
max_class_score = [ st * (1 + SCORE_TOLERANCE_PERCENT) for st in score_target ]

print("반별 점수 목표/허용:", list(zip([round(x,2) for x in score_target],
                                   [round(x,2) for x in min_class_score],
                                   [round(x,2) for x in max_class_score])))

#  CSP -> OR-Tools CP-SAT 
model = cp_model.CpModel()

# Binary variables: x[s, c] = 1 / 만약 학생 s가 반 c에 배정되면 1, 아니면 0
x = {}
for s in students:
    for c in CLASSROOM_IDS:
        x[s, c] = model.NewBoolVar(f"x[{s},{c}]")

# 각 학생은 하나의 반에만 배정
for s in students:
    model.Add(sum(x[s, c] for c in CLASSROOM_IDS) == 1)

# 반 정원
for idx, c in enumerate(CLASSROOM_IDS):
    model.Add(sum(x[s, c] for s in students) == CAPACITIES[idx])

# 제약조건 (1A) 같은 반 금지(나쁜관계)
for s1, s2 in student_dislikes:
    for c in CLASSROOM_IDS:
        model.Add(x[s1, c] + x[s2, c] <= 1)

# 제약조건 (1B) 비등교 학생 / 챙겨주는 친구와 같은 반
buddy_pairs = []
for _, row in df.iterrows():
    if row['Truancy_flag']:
        s = int(row['id'])
        t = row['good_id']
        if pd.notna(t):
            t = int(t)
            if t in id_set and t != s:
                buddy_pairs.append((s, t))
for s, buddy in buddy_pairs:
    for c in CLASSROOM_IDS:
        model.Add(x[s, c] == x[buddy, c])

# 제약조건 (2) 각 반 리더십 최소 1명
for c in CLASSROOM_IDS:
    model.Add(sum(x[s, c] for s in leader_ids) >= 1)

# 제약조건 (3) 피아노 균등
for idx, c in enumerate(CLASSROOM_IDS):
    model.Add(sum(x[s, c] for s in piano_ids) == piano_targets[idx])

# 제약조건 (4) 성적 균형(반별 총점 허용범위)
#  정수화 위해 점수는 int, 허용범위도 int로 변환
for idx, c in enumerate(CLASSROOM_IDS):
    model.Add(sum(student_scores[s] * x[s, c] for s in students) >= int(min_class_score[idx]))
    model.Add(sum(student_scores[s] * x[s, c] for s in students) <= int(max_class_score[idx]))

# 제약조건 (5) 비등교 균등
for idx, c in enumerate(CLASSROOM_IDS):
    model.Add(sum(x[s, c] for s in truancy_ids) == truancy_targets[idx])

# 제약조건 (6) 남녀 균등(정확 타깃)
for idx, c in enumerate(CLASSROOM_IDS):
    model.Add(sum(x[s, c] for s in male_ids)   == male_targets[idx])
    model.Add(sum(x[s, c] for s in female_ids) == female_targets[idx])

# 제약조건 (7) 운동 균등
for idx, c in enumerate(CLASSROOM_IDS):
    model.Add(sum(x[s, c] for s in sport_ids) == sport_targets[idx])

# 제약조건 (8) 전년도 같은 반 분산
for key, members in prev_groups.items():
    cap = prev_caps[key]
    for c in CLASSROOM_IDS:
        model.Add(sum(x[s, c] for s in members) <= cap)

# 제약조건 (9) 클럽 편향 방지
for club, members in club_groups.items():
    cap = club_caps[club]
    for c in CLASSROOM_IDS:
        model.Add(sum(x[s, c] for s in members) <= cap)

#  Solve and Output 
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 120.0  # 시간 제한
solver.parameters.num_search_workers = 8       # 병렬 탐색

print("\nAttempting to find solutions (OR-Tools CP-SAT)...")
status = solver.Solve(model)

if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    classroom_assignments = {c: [] for c in CLASSROOM_IDS}
    for s in students:
        for c in CLASSROOM_IDS:
            if solver.Value(x[s, c]) == 1:
                classroom_assignments[c].append(s)
                # 학생별 배정 결과 df에 기록
                df.loc[df['id'] == s, 'assigned_class'] = c + 1  # 1반~6반으로 표현
                break

    # 요약 출력
    for idx, c_id in enumerate(CLASSROOM_IDS):
        st_list = classroom_assignments[c_id]
        scores = [student_scores[s] for s in st_list]
        avg_sc = sum(scores)/len(scores) if st_list else 0.0
        male = sum(1 for s in st_list if s in male_ids)
        female = sum(1 for s in st_list if s in female_ids)
        piano_cnt = sum(1 for s in st_list if s in piano_ids)
        tru_cnt = sum(1 for s in st_list if s in truancy_ids)
        sport_cnt = sum(1 for s in st_list if s in sport_ids)
        print(f"\nClassroom {c_id+1}:")
        print(f"  Number of Students: {len(st_list)} (cap={CAPACITIES[idx]})")
        print(f"  Avg Score: {avg_sc:.2f}")
        print(f"  Male={male}, Female={female}, Piano={piano_cnt}, Truancy={tru_cnt}, Sport={sport_cnt}")
        print(f"  Students: {st_list}")

    #  CSV 저장 
    output_dir = "or_tools_results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"classroom_assignment_{timestamp}.csv"
    full_path = os.path.join(output_dir, output_filename)

    # 기존 df에 assigned_class 열을 포함해 전체 저장
    df.to_csv(full_path, index=False, encoding='utf-8-sig')
    print(f"\nResults saved to {full_path}")
else:
    print("No solutions found given the current constraints and parameters.")

