import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, Counter
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict, Set, Optional
import os

# === 클래스명, 그룹, 가중치 등 주요 상수 ===
CLASS_NAMES = [
    "bag", "books", "bookshelf", "calendar", "correction-tape", "cosmetic",
    "drink", "earphone", "eraser", "food", "gamepad", "glasses", "glue", "goods",
    "headset", "keyboard-pc", "laptop", "mic-pc", "monitor-pc", "mouse-pc", "organizer",
    "paper", "pen", "pen holder", "pencil case", "phone", "photo", "post-it", "ruler",
    "scissors", "snack", "speakers-pc", "stapler", "stopwatch", "tablet-pc", "tape",
    "tissue", "tower-pc", "trash", "watch"
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "class_usage_frequency_weight_for_algo.csv")
WEIGHTS_DF = pd.read_csv(CSV_PATH)
WEIGHTS_MAP = WEIGHTS_DF.set_index("class").to_dict(orient="index")

GROUPS = {
    "books": ["books", "paper", "post-it"],
    "stationery": ["pen", "pencil case", "scissors", "glue", "tape", "eraser", "stapler", "correction-tape", "pen holder", "ruler"],
    "it": ["laptop", "keyboard-pc", "monitor-pc", "mouse-pc", "tablet-pc", "mic-pc", "headset", "speakers-pc", "tower-pc", "gamepad", "phone", "earphone"],
    "trash": ["food", "drink", "tissue", "trash"],
    "personal": ["glasses", "cosmetic", "bag", "watch"],
    "photo": ["photo"],
    "calendar": ["calendar"],
    "goods": ["goods"],
}

# === 책상 영역 추출 ===
def get_desk_top_dynamic(
    objs: List[Tuple[int,int,int,int,int]], 
    exclude_classes: Set[str], 
    class_names: List[str], 
    img_h: int,
    consider_ratio: float = 0.3,   # 상위 30% 고려
    margin_ratio: float = 0.05
) -> int:
    cy_list = []
    for (x1, y1, x2, y2, cls_id) in objs:
        label = class_names[cls_id]
        if label in exclude_classes:
            continue
        cy = (y1 + y2) // 2
        cy_list.append(cy)

    if not cy_list:
        raise ValueError("monitor-pc 제외 후 유효 객체 없음")

    # 중심점 y좌표 오름차순 정렬
    sorted_cy = sorted(cy_list)

    # 상위 consider_ratio 비율만 사용
    k = max(1, int(len(sorted_cy) * consider_ratio))
    selected_cy = sorted_cy[:k]

    # 평균으로 top 계산
    avg_cy = int(np.mean(selected_cy))

    margin = int((img_h - avg_cy) * margin_ratio)
    desk_top = max(0, avg_cy - margin)
    return desk_top

EXCLUDE_CLASSES_BACKGROUND = {
    "monitor-pc", "photo", "goods", "post-it"
}

# === 책상 영역(그리드) 정의 및 한글 변환 ===
def create_grid_map(rows:int=3, cols:int=4) -> Tuple[Dict[str, List[Tuple[int,int]]], Dict[str, str]]:
    region_map = {
        "left_side": [(r,0) for r in range(rows)],
        "right_side": [(r,cols-1) for r in range(rows)],
        "top": [(0,c) for c in range(1,cols-1)],
        "center": [(r,c) for r in range(1,rows) for c in range(1,cols-1)]
    }
    region_kr = {
        "left_side": "왼쪽",
        "right_side": "오른쪽",
        "top": "상단",
        "center": "중앙"
    }
    return region_map, region_kr

REGION_MAP, REGION_KR = create_grid_map(3, 4)
# HANDED_SENSITIVE_CLASSES = {"drink", "mouse-pc", "pen", "scissors", "stapler"}

def region_to_kr(region:str) -> str:
    return REGION_KR.get(region, region)

def get_region(grid:Tuple[int,int]) -> str:
    for region, grids in REGION_MAP.items():
        if grid in grids:
            return region
    # 인접 영역 자동 선택
    all_grids = [g for grids in REGION_MAP.values() for g in grids]
    closest = min(all_grids, key=lambda g: abs(g[0]-grid[0]) + abs(g[1]-grid[1]))
    for region, grids in REGION_MAP.items():
        if closest in grids:
            return region
    return "center"

def get_user_handedness() -> str:
    while True:
        hand = input("당신은 오른손잡이인가요? (y/n): ").strip().lower()
        if hand in ('y', 'n'):
            handed_str = "오른손잡이" if hand == 'y' else "왼손잡이"
            print(f"\n사용자 선택 : {handed_str}")
            return hand
        print("y 또는 n으로 입력해주세요.")

def load_and_check_image(image_path:str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
    return img

# # YOLO 객체 탐지 수행
# def run_yolo_inference(model, image_path:str, conf_thres:float=0.45):
#     results = model(image_path)
#     boxes = results[0].boxes
#     objs = [
#         (*map(int, box), int(cls_id))
#         for box, cls_id, score in zip(boxes.xyxy, boxes.cls, boxes.conf)
#         if score >= conf_thres
#     ]
#     return objs, results
import cv2
import time

def run_yolo_inference(model, image_path: str, conf_thres: float = 0.45):
    print("🖼️ 이미지 읽는 중...")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"이미지 로딩 실패: {image_path}")

    print("⏱️ YOLO 추론 시작 (model(img))")
    t0 = time.time()
    results = model(img)
    print(f"✅ YOLO 추론 완료 (소요 시간: {time.time() - t0:.2f}초)")

    boxes = results[0].boxes
    objs = [
        (*map(int, box), int(cls_id))
        for box, cls_id, score in zip(boxes.xyxy, boxes.cls, boxes.conf)
        if score >= conf_thres
    ]
    return objs, results


# 객체별 위치를 책상 그리드로 변환
def analyze_objects_by_grid(
    objects:List[Tuple[int,int,int,int,int]], img_h:int, img_w:int, rows:int=3, cols:int=4
) -> Tuple[Dict[Tuple[int,int], List[str]], Dict[str,List[Tuple[int,int]]], List[Tuple[str,Tuple[int,int],Tuple[int,int]]]]:
    desk_top = get_desk_top_dynamic(
        objects, exclude_classes=EXCLUDE_CLASSES_BACKGROUND, class_names=CLASS_NAMES, img_h=img_h
    )
    desk_bottom = img_h
    cell_w, cell_h = img_w / cols, (desk_bottom - desk_top) / rows
    grid_objects, label_grid_map, object_info = defaultdict(list), defaultdict(list), []
    for obj in objects:
        x1, y1, x2, y2, cls_id = obj
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if not (desk_top <= cy <= desk_bottom):
            continue
        grid_r = min(max(int((cy - desk_top) // cell_h), 0), rows-1)
        grid_c = min(max(int(cx // cell_w), 0), cols-1)
        grid, label = (grid_r, grid_c), CLASS_NAMES[cls_id]
        grid_objects[grid].append(label)
        label_grid_map[label].append(grid)
        object_info.append((label, grid, (cx, cy)))
    return grid_objects, label_grid_map, object_info

# 객체 겹침(중복) 감점
def compute_overlap_penalty(boxes:List[List[float]], threshold:float=0.6) -> int:
    heavy_overlap = 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            xi1, yi1, xi2, yi2 = max(boxes[i][0], boxes[j][0]), max(boxes[i][1], boxes[j][1]), min(boxes[i][2], boxes[j][2]), min(boxes[i][3], boxes[j][3])
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            box1_area = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
            box2_area = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])
            union_area = box1_area + box2_area - inter_area
            iou = inter_area / union_area if union_area > 0 else 0
            if iou > threshold:
                heavy_overlap += 1
    return min(heavy_overlap * 2, 20)

# 유저 맞춤 가중치 반영
def get_user_weights(default_map:dict, user_overrides:Optional[dict]=None) -> dict:
    if not user_overrides:
        return default_map
    merged = {k: v.copy() for k, v in default_map.items()}
    for k, v in user_overrides.items():
        if k in merged:
            merged[k].update(v)
        else:
            merged[k] = v
    return merged

# 정돈 점수 산정
def compute_organization_score(label_grid_map, boxes, weights_map) -> Tuple[int,dict]:
    score, breakdown = 100, {}
    for label, info in weights_map.items():
        max_count = info.get("max_acceptable_count", None)
        over_penalty = info.get("over_count_penalty", 5)
        if max_count is not None:
            count = len(label_grid_map.get(label, []))
            if count > max_count:
                penalty = (count - max_count) * over_penalty
                score -= penalty
                breakdown[f"{label} 과다"] = -penalty
    if len(set(label_grid_map.get("books", []))) >= 3:
        score -= 20
        breakdown["책 분산"] = -20
    overlap_penalty = compute_overlap_penalty(boxes)
    if overlap_penalty > 0:
        breakdown["객체 겹침 감점"] = -overlap_penalty
    score -= overlap_penalty
    return max(score, 0), breakdown

# 손잡이 방향에 따른 맞춤 피드백
def csv_based_handed_feedback(label_grid_map, detected_labels, handedness, weights_map):
    weight_col = "right_handed_weight" if handedness == "y" else "left_handed_weight"
    region_for_col = "right_side" if handedness == "y" else "left_side"
    region_kr = region_to_kr(region_for_col)
    feedback = []
    for label in detected_labels:
        info = weights_map.get(label, {})
        handed_weight = info.get(weight_col, 0)
        if handed_weight >= 1:
            wrong_regions = []
            for grid in label_grid_map.get(label, []):
                cur_region = get_region(grid)
                if cur_region != region_for_col:
                    wrong_regions.append(region_to_kr(cur_region))
            n_wrong = len(wrong_regions)
            if n_wrong == 0:
                continue
            wrong_regions_text = ", ".join(sorted(set(wrong_regions)))
            plural = f"{n_wrong}개의" if n_wrong > 1 else "1개의"
            feedback.append(
                f"👉 {plural} '{label}'이(가) {wrong_regions_text}에 있습니다. 모두 {region_kr}으로 옮겨보세요."
            )
    return feedback

# 그룹 단위(책 등) 분산/집중 피드백
def feedback_by_group_and_grid(label_grid_map, grid_objects, detected_labels):
    feedback = []
    if "books" in detected_labels:
        book_grids = set(label_grid_map.get("books", []))
        if len(book_grids) >= 3:
            feedback.append("📚 책이 여러 영역에 흩어져 있습니다. 한 쪽으로 정리해 보세요.")
        else:
            feedback.append("📚 책/노트류 정리가 잘 되어 있습니다!")
    if any(lab in detected_labels for lab in ["drink", "tissue", "trash"]):
        feedback.append("🗑️ 책상에 쓰레기/음료가 조금 있습니다. 필요 없는 건 바로 치우면 더 깨끗해져요.")
    return list(dict.fromkeys(feedback))

# 기타(자주 어지르는 패턴, 권장 배치 등) 커스텀 피드백
def feedback_custom_rules(label_grid_map, grid_objects, detected_labels):
    feedback = []
    grid_counter = Counter()
    for label in GROUPS["stationery"]:
        if label not in detected_labels:
            continue
        for g in label_grid_map.get(label, []):
            grid_counter[g] += 1
    if grid_counter:
        common_grid, _ = grid_counter.most_common(1)[0]
        common_region = region_to_kr(get_region(common_grid))
        for label in GROUPS["stationery"]:
            if label not in detected_labels:
                continue
            for grid in label_grid_map.get(label, []):
                if grid != common_grid:
                    feedback.append(f"📎 '{label}'은 흩어져 있어요. {common_region} 쪽으로 모아 정리해보세요.")
    if "pen" in detected_labels and "pen holder" in detected_labels:
        pen_grids = label_grid_map.get("pen", [])
        holder_grids = label_grid_map.get("pen holder", [])
        if pen_grids and holder_grids:
            for g in pen_grids:
                if all(g != hg for hg in holder_grids):
                    feedback.append("✏️ 펜은 펜꽂이에 정리하면 더 깔끔하고 분실 걱정이 없어요.")
                    break
    elif "pen" in detected_labels and "pen holder" not in detected_labels:
        feedback.append("✏️ 펜이 흩어져 있어요. 통이나 한곳에 정리해보세요.")
    for label in ["drink", "food", "snack", "tissue"]:
        if label in detected_labels and len(label_grid_map.get(label, [])) >= 2:
            feedback.append(f"🧃 '{label}'이 많아요. 정리하거나 치워주세요.")
    if "trash" in detected_labels:
        feedback.append("🚮 쓰레기는 바로 버려주세요.")
    if "bag" in detected_labels:
        for grid in label_grid_map.get("bag", []):
            if get_region(grid) != "left_side":
                feedback.append("🎒 가방은 책상 위에 두지 말고 치워보세요.")
    for label in ["cosmetic", "glasses"]:
        if label in detected_labels:
            for grid in label_grid_map.get(label, []):
                if get_region(grid) not in ["left_side", "right_side"]:
                    feedback.append(f"🧴 '{label}'은 구석(좌/우)에 두는 것이 더 깔끔해 보여요.")
    if "calendar" in detected_labels:
        cal_grids = label_grid_map.get("calendar", [])
        cal_region = region_to_kr(get_region(cal_grids[0]))
        for label in ["stopwatch", "watch", "earphone"]:
            if label in detected_labels:
                for g in label_grid_map.get(label, []):
                    if get_region(g) != get_region(cal_grids[0]):
                        feedback.append(f"🕒 '{label}'은 calendar가 있는 {cal_region} 쪽에 같이 두는 걸 추천해요.")
    return list(dict.fromkeys(feedback))

# 박스 컬러 고정
def fixed_color(cls_id:int):
    np.random.seed(cls_id)
    return tuple(np.random.uniform(0.2, 0.8, 3))

# 박스 시각화(Matplotlib)
def draw_boxes_matplotlib(img:np.ndarray, objs:List[Tuple[int]], class_names:List[str], scores:Optional[List[float]]=None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for i, (x1, y1, x2, y2, cls_id) in enumerate(objs):
        color = fixed_color(cls_id)
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        label = class_names[cls_id]
        text = f"{label}"
        if scores is not None:
            text += f" {scores[i]:.2f}"
        ax.text(x1, y1 - 5, text, fontsize=10, color=color, weight='bold',
                bbox=dict(facecolor='none', edgecolor=color, boxstyle='round,pad=0.2', lw=1))
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.show()

# 피드백 요약 출력
def summarize_feedback(custom_feedback, user_feedback, fb_group, score, breakdown, handed_str):
    def print_unique(title, items):
        if items:
            print(f"\n{title}")
            for item in items:
                print(item)
    print_unique("🧠 [사용자 배치 기준 피드백] 🧠", custom_feedback)
    print(f"\n👉사용자 선택 : {handed_str}👈")
    print_unique("🤚 [사용자 맞춤 피드백] 🤚", user_feedback)
    print_unique("📦 [객체 분포 기반 피드백] 📦", fb_group)
    print(f"\n📊 정돈 점수: {score}/100")
    for k, v in breakdown.items():
        if v != 0:
            print(f"  - {k}: {v}")

def draw_objects_with_boxes(
    img: np.ndarray,
    objs: List[Tuple[int]],
    class_names: List[str]
):
    """
    객체 바운딩 박스와 라벨만 시각화합니다. (그리드 없이)
    """
    img_disp = img.copy()

    for (x1, y1, x2, y2, cls_id) in objs:
        color = (0, 180, 255)

        # 바운딩 박스
        cv2.rectangle(img_disp, (x1, y1), (x2, y2), color, thickness=4)

        # 라벨 - 외곽선(검정)
        cv2.putText(
            img_disp, class_names[cls_id],
            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 6, cv2.LINE_AA
        )
        # 라벨 - 내부(흰색)
        cv2.putText(
            img_disp, class_names[cls_id],
            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA
        )

    # 시각화
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB))
    plt.title("Detected Object Bounding Boxes", fontsize=18)
    plt.axis("off")
    plt.show()

def visualize_grid_and_centers(img, objs, class_names, desk_left, desk_top, desk_right, desk_bottom, grid_coords):
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    for (r, c), (x1, y1, x2, y2) in grid_coords.items():
        grid_rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1.5, edgecolor='lime', facecolor='none', linestyle='--')
        ax.add_patch(grid_rect)

    for (x1, y1, x2, y2, cls_id) in objs:
        cx, cy = (x1+x2)//2, (y1+y2)//2
        color = np.random.rand(3,)
        ax.plot(cx, cy, 'o', color=color, markersize=8)
        ax.text(cx+5, cy-10, class_names[cls_id], color=color, fontsize=10, weight='bold')

    ax.set_title("Desk Area + Grid + Object Centers", fontsize=16)
    plt.axis("off")
    plt.show()

# 객체 탐지 결과 이미지 저장 경로
def draw_boxes_and_save(img_path: str, objs: List[Tuple[int]], output_path: str):
    """
    객체 바운딩 박스만 그려서 output_path에 저장.
    """
    img = cv2.imread(img_path)
    for (x1, y1, x2, y2, cls_id) in objs:
        color = (0, 180, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
        cv2.putText(img, CLASS_NAMES[cls_id], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(img, CLASS_NAMES[cls_id], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.imwrite(output_path, img)


# # 메인 프로세스: 1) 탐지 → 2) 분석 → 3) 피드백 → 4) 시각화
# def recommend_for_image(image_path:str, model, user_overrides:Optional[dict]=None):
#     try:
#         handedness = get_user_handedness()
#         handed_str = "오른손잡이" if handedness == "y" else "왼손잡이"
#         img = load_and_check_image(image_path)
#         h, w, _ = img.shape
#         objs, results = run_yolo_inference(model, image_path)
#         if not objs:
#             print("인식된 객체가 없습니다.")
#             return
#         scores = [float(s.item()) if hasattr(s, "item") else float(s) for s in results[0].boxes.conf]
#         draw_boxes_matplotlib(img, objs, CLASS_NAMES, scores)
#         print('YOLO 결과:', [CLASS_NAMES[int(cls_id)] for cls_id in results[0].boxes.cls])
#         draw_objects_with_boxes(img, objs, CLASS_NAMES)
#         grid_objects, label_grid_map, object_info = analyze_objects_by_grid(objs, h, w)
#         detected_labels = set(label for label, _, _ in object_info)
#         print('detected_labels:', detected_labels)
#         weights_map = get_user_weights(WEIGHTS_MAP, user_overrides)
#         custom_feedback = feedback_custom_rules(label_grid_map, grid_objects, detected_labels)
#         user_feedback = csv_based_handed_feedback(label_grid_map, detected_labels, handedness, weights_map)
#         fb_group = feedback_by_group_and_grid(label_grid_map, grid_objects, detected_labels)
#         boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
#         score, breakdown = compute_organization_score(label_grid_map, boxes, weights_map)
#         summarize_feedback(custom_feedback, user_feedback, fb_group, score, breakdown, handed_str)
#     except Exception as e:
#         print("오류 발생:", e)


import os
from ultralytics import YOLO

def recommend_for_image(image_path: str, handedness: str, user_overrides: dict):
    print("📌 [recommend_for_image] 시작")
    print(f"📷 입력 이미지 경로: {image_path}")
    print(f"🧍 사용자 설정 - 손: {handedness}, 가중치: {user_overrides}")

    try:
        # 1. 모델 로딩
        MODEL_PATH = os.path.join(BASE_DIR, "models/best.pt")
        print(f"📦 모델 경로 확인: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        print("✅ 모델 로딩 완료")

        # 2. 이미지 로딩
        print("🖼️ 이미지 로딩 시도")
        img = load_and_check_image(image_path)
        print("✅ 이미지 로딩 성공")
        h, w, _ = img.shape
        print(f"📐 이미지 크기: {w} x {h}")

        # 3. YOLO 추론
        print("🔎 YOLO 추론 시작")
        objs, results = run_yolo_inference(model, image_path)
        print("✅ YOLO 추론 완료")
        print(f"🔍 탐지된 객체 수: {len(objs)}")

        if not objs:
            print("⚠️ 객체 없음 → 분석 종료")
            return {"score": 0, "feedback": ["객체가 탐지되지 않았습니다."], "boxes": []}

        # 4. 그리드 분석
        print("🧭 그리드 분석 중...")
        grid_objects, label_grid_map, object_info = analyze_objects_by_grid(objs, h, w)
        print("✅ 그리드 분석 완료")

        # 5. 라벨 집합 추출
        detected_labels = set(label for label, _, _ in object_info)
        print(f"🏷️ 탐지된 라벨: {detected_labels}")

        # 6. 사용자 가중치 통합
        weights_map = get_user_weights(WEIGHTS_MAP, user_overrides)
        print("⚖️ 사용자 가중치 반영 완료")

        # 7. 피드백 생성
        print("💡 피드백 생성 시작")
        custom_feedback = feedback_custom_rules(label_grid_map, grid_objects, detected_labels)
        user_feedback = csv_based_handed_feedback(label_grid_map, detected_labels, handedness, weights_map)
        fb_group = feedback_by_group_and_grid(label_grid_map, grid_objects, detected_labels)
        print("✅ 피드백 생성 완료")

        # 8. 바운딩 박스 처리
        boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
        print(f"📦 바운딩 박스 수: {len(boxes)}")

        # 9. 정돈 점수 산정
        print("📊 점수 계산 중...")
        score, breakdown = compute_organization_score(label_grid_map, boxes, weights_map)
        print(f"✅ 점수 산정 완료 → {score}")

        # 10. 최종 결과 반환
        return {
            "score": score,
            "feedback": list(dict.fromkeys(custom_feedback + user_feedback + fb_group)),
            "breakdown": breakdown,
            "boxes": objs
        }

    except Exception as e:
        print("❌ [recommend_for_image] 오류 발생:", str(e))
        return {
            "score": 0,
            "feedback": [f"⚠️ 분석 중 오류 발생: {str(e)}"],
            "breakdown": {"시스템 오류": -100},
            "boxes": []
        }
