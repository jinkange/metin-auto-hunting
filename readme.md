
# Metin1 (윈도우 창모드) 디버그 자동화 봇

> ⚠️ 이 프로젝트는 **본인이 만든 게임(디버깅/테스트 목적)**에서만 사용하세요.

## 설치
```bash
pip install -r requirements.txt
```

## 준비(필수)

### 1) 창 제목
- 게임 창 제목이 `Metin1`이어야 합니다. 다르면 `config.py`의 `WINDOW_TITLE` 수정.

### 2) 좌표 숫자 템플릿
- `assets/digits/`에 숫자 템플릿을 넣습니다.

권장(복수 템플릿):
```
assets/digits/0/*.png
assets/digits/1/*.png
...
assets/digits/9/*.png
```

단일 파일도 지원:
```
assets/digits/0.png ... assets/digits/9.png
```

### 3) 몬스터/캐릭터/특수 이미지 템플릿
- `assets/monster/` : 몬스터 판별용 템플릿들(.png)
- `assets/character/` : 캐릭터 중심점 템플릿들(.png)
- `assets/special/1/`, `assets/special/2/`, `assets/special/3/` : 감시할 이미지 템플릿들(.png)

## 실행
```bash
python main.py
```

- `F1` 시작
- `F2` 정지
- 콘솔:
  - `step 1~5` : 이동 단계 변경
  - `stride 1~10` : 순회 목표 간격
  - `status` : 상태
  - `quit` : 종료

## 튜닝 포인트
- `config.py`
  - `COORD_REGION_REL` : 좌표 표시 영역(창 기준 상대좌표)
  - `COORD_BIN_THRESH` / `DIGIT_MATCH_THRESH` : 좌표 인식 튜닝
  - `MOVE_PIXEL_PER_TILE` : 우클릭 이동 클릭 거리(타일당 픽셀)

## 금지 좌표
- `FORBIDDEN_POINTS`에 (x,y) 추가
- `FORBIDDEN_PAD=2`이면 해당 좌표 기준 2칸 이내 접근 금지(대각 포함 기본)

