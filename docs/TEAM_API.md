# 팀원 연결 규약

목적은 렌더러에 **실제 영상과 픽셀 좌표가 일치하는 깊이맵**을 넘기는 것입니다. 깊이맵은 현재 화면의 최신 RGB와 임의로 섞지 말고, 같은 패킷의 기준 RGB에 합성합니다.

## 정합 담당자용 REST API · 0~1 float32

**계산 입력은 카메라 1·2, 반환 시점은 카메라 1 하나**입니다. 여러 영상을 이어 붙이지 않습니다. 반환 RGB의 픽셀 `(u,v)`와 깊이 `[v,u]`가 같은 원본 카메라 1 좌표를 가집니다. 등록 가능한 카메라는 `cam1`, `cam2` 두 대입니다.

```http
GET /api/rgbd/frame
X-Session-Key: 접속키
```

응답은 `application/json`, `Cache-Control: no-store`입니다. 이미지와 배열을 **한 HTTP 응답**에 묶습니다. JSON의 숫자 배열 대신 Base64로 감싼 **실제 float32 바이트**를 반환하여 자료형을 유지하고 일반적인 JSON 클라이언트에서도 받을 수 있게 했습니다.

CPU와 GPU의 반환 형식은 같습니다. `depth_device`는 `cpu` 또는 `cuda`, `depth_backend`는 실제 계산 방식입니다. CPU는 `OpenCV StereoSGBM / CPU`, GPU는 `C-Fast-FoundationStereo / ONNX Runtime CUDA`입니다. CPU 방식의 유효 영역과 정확도가 GPU 모델과 같다는 보장은 없습니다. 기본 `auto`는 CUDA를 준비하지 못하면 CPU로 전환하며, `--depth-device cpu`로 명시할 수도 있습니다.

| 필드 | 의미 |
| --- | --- |
| `schema`, `frame_id`, `captured_ms`, `age_ms` | 규약 `ar-lava.rgbd.v1`, 중복 프레임 확인용 ID, 촬영 시각 추정치와 서버 인코딩 요청 시점의 나이 |
| `image`, `depth`, `valid_mask` | 원본 JPEG와 동일 크기·시점의 정규화 Z 및 유효 마스크, 각각 `data_base64`에 저장 |
| `camera`, `tracking` | 카메라 1의 보정값과 같은 프레임에서 구한 발 3D 위치. 전신 관절·사람 ID 추적은 미제공 |

### 깊이 복원과 바이트 순서

정규화는 **고정 범위 0.1~15m의 선형 Z-depth**입니다. 프레임별 최솟값·최댓값으로 정규화하지 않으므로 같은 거리는 장면이 바뀌어도 같은 값을 유지합니다.

```text
d = (Z_m - 0.1) / (15.0 - 0.1)
Z_m = depth.near_m + d * (depth.far_m - depth.near_m)
```

- `depth.data_base64`: Base64 해제 후 **little-endian float32**, **행 우선(row-major)**, `shape=[height,width]`입니다. 왼쪽 위가 원점이며 인덱스는 `v * width + u`입니다. 바이트 길이는 `height * width * 4`입니다.
- `valid_mask.data_base64`: Base64 해제 후 `height * width`개의 **uint8**, **1=유효, 0=무효**입니다. 무효 깊이의 정규화 값은 0이므로 **항상 마스크를 함께 사용**합니다. 깊이를 얻지 못한 영역을 가까운 물체로 해석하지 않습니다.
- 유효한 실제 Z 범위는 **0.1m 초과~15m 이하**입니다. 색상 미리보기의 8m 포화와 데이터의 15m 범위는 서로 다릅니다.
- 이 값은 **광축 Z 거리**입니다. 카메라에서 픽셀까지의 유클리드 거리, 시차, 역깊이, 그래픽스 API의 비선형 depth buffer 값과 다릅니다. 가상 장면의 깊이도 같은 카메라 좌표의 미터 Z로 변환한 뒤 비교합니다.

### 선택적 사람 마스크

`segmentation.available=true`이면 `segmentation.data_base64`에 원본 RGB·깊이와 동일 크기·동일 `frame_id`의 **단일 채널 PNG8**이 있습니다. `model=mediapipe_pose_landmarker_lite`, `semantics=person_foreground_probability`이며, PNG의 0~255 값을 `probability_scale=1/255`와 곱하면 0~1 사람 영역 확률입니다. 합성 담당자가 임계값 또는 부드러운 가중치로 사용할 수 있습니다. 이 확률은 광학적으로 정확한 알파 매트나 깊이 유효 마스크와 다릅니다.

사람 한 명이 확인되지 않거나 마스크 형식이 맞지 않으면 `available=false`이며 데이터를 생략합니다. 이를 ‘확실히 사람이 없는 검은 마스크’로 해석하지 마세요. 마스크 부재가 필수 RGB·깊이 응답을 막지는 않습니다. 수신 예제는 마스크가 있으면 `person_mask.png`도 저장합니다.

현재는 포즈 추론이 이미 만드는 마스크를 재사용합니다. SAM 등 추가 GPU 모델을 실행하지 않습니다. 카메라 재접속·보정 중에는 이전 마스크를 재사용하지 않으며, 카메라 1의 샘플 영상과 마스크를 함께 깊이 결과에 고정해 REST 인코딩 중 새 영상과 섞이지 않도록 했습니다.

### 보정값과 추적 좌표

`camera.K`는 원본 카메라 1의 내부 행렬, `camera.distortion`은 OpenCV 왜곡 계수, `camera.world_to_camera`는 미터 단위 `[R|t]`입니다. `X_camera = R X_world + t`로 사용합니다. 카메라 좌표는 X 오른쪽·Y 아래·Z 전방이며, 세계 좌표는 바닥 기준판 X 오른쪽·Y 판 아래·Z 바닥 안쪽입니다.

RGB와 깊이는 **왜곡이 남아 있는 원본 영상 좌표**입니다. 3D 복원은 픽셀을 `K`와 왜곡 계수로 먼저 역왜곡해 광선 `(x_n,y_n)`을 구한 뒤 `P_camera=(x_n*Z, y_n*Z, Z)`로 계산합니다. 가상 공간이 왜곡 없는 핀홀 카메라라면 RGB와 깊이를 함께 같은 좌표로 역왜곡해야 합니다. 깊이만 보정하거나 이미 보정된 영상에 원본 행렬을 중복 적용하지 않습니다.

`tracking.kind`는 `feet_3d`입니다. `joints_world_m`는 현재 처리하는 발 관련 랜드마크 **29·30·31·32**, `feet_world_m`는 바닥 부근 발의 평균 위치입니다. REST 응답에서는 Z 부호를 세계 좌표에 맞춰 변환했으므로 `world_to_camera`에 바로 사용할 수 있습니다. 기존 게임용 추적 상태의 Z-up 표현과 구별합니다. `tracking.valid=false`여도 RGB-D는 제공될 수 있으며, 사람 ID 유지나 전신 모션 캡처를 제공한다는 뜻은 아닙니다.

### 같은 시점과 실시간 조건

이 API는 요청마다 추론을 새로 시작하지 않고 **가장 최근에 계산한 프레임**을 가져옵니다. 직전과 `frame_id`가 같으면 중복 결과이므로 클라이언트에서 생략할 수 있습니다. RGB·깊이·추적값·보정값을 별도 요청의 결과와 섞지 마세요.

좌우 촬영 시각 차이 허용치는 **40ms**, 서버에서 유효한 결과 나이는 **350ms 이하**입니다. 응답 인코딩 중 보정이 바뀌거나 연결이 끊기거나 만료되어도 **409**를 반환합니다. 인코딩 중 새 프레임이 나와도 응답의 구성 요소는 처음 잡은 프레임 하나로 유지합니다.

`captured_ms`는 서버 단조 시계 기준의 **브라우저 그리기 시각 추정치**입니다. Unix 시각이나 센서 노출 시각이 아닙니다. 네트워크 전송 이후 지연은 `age_ms`에 들어 있지 않으며, HTTP 응답 수신까지 350ms를 보장하지 않습니다. 클라이언트에서 이 값을 자기 시계와 직접 빼지 마세요.

REST 수신은 **응답 후 다음 요청을 보내는 방식**으로 시작하세요. 깊이·마스크만으로도 640×480에서 Base64 약 **2.05MB/프레임**이며 JPEG·메타데이터가 추가됩니다. 카메라 입력 FPS와 REST 결과 FPS는 같지 않습니다. 브라우저의 다른 출처에서 호출할 경우 별도의 명시적 CORS 설정이 필요하며, 현재 서버에 임의 출처 허용은 추가하지 않았습니다.

### 수신 예제

`scripts/fetch_rgbd.py`는 Python 표준 HTTP 라이브러리와 NumPy로 한 프레임을 받아 원본 JPEG, 정규화 float32 NPY, bool 마스크, 미터 깊이 NPY, 보정·추적 메타데이터를 저장합니다. 무효 미터 깊이는 NaN입니다.

```powershell
$env:LAVA_SESSION_KEY = '서버 접속 키'
.venv/Scripts/python.exe scripts/fetch_rgbd.py --url https://노트북IP:8443/api/rgbd/frame --ca certs/lava-local-ca.crt --output data/rgbd-capture
```

실시간 사용자는 이 예제의 `decode_frame()`처럼 디코딩해 메모리에서 바로 정합하면 됩니다. C#/Unity에서는 Base64를 해제해 little-endian `float` 배열로 읽고 동일한 복원식을 적용합니다.

**성능 제한:** 장치와 장면에 따라 350ms를 넘는 처리 지연이 생기면 이 API도 409를 반환합니다. CPU StereoSGBM은 공식 정지 샘플로 RGB-D REST 200 응답을 확인했지만 실제 폰 영상의 전체 지연·정확도를 보장하는 것은 아닙니다. 과거 GPU 통과 기록이나 REST 계약 검사 통과를 모든 환경의 실시간 성능 보장으로 해석하지 마세요.

## 실시간 수신

연결 주소는 `wss://노트북IP:8443/ws`입니다. 첫 메시지에 아래 객체를 보냅니다.

```json
{"role":"depth","token":"서버 실행 창의 접속 키"}
```

이후 서버에서 이진 WebSocket 메시지가 옵니다.

```text
[4 bytes big-endian unsigned JSON length N]
[N bytes UTF-8 JSON metadata]
[depth_png_bytes bytes PNG16]
[reference_jpeg_bytes bytes JPEG]
[preview_jpeg_bytes bytes JPEG]
```

PNG는 **단일 채널 unsigned 16-bit**, 값에 **0.001**을 곱하면 미터입니다. **0은 미측정**입니다. PNG는 무손실입니다. 시각화 JPEG는 거리 계산에 사용하지 않습니다.

메타데이터에는 아래 값이 있습니다.

- `camera_id`: 기준 카메라 `cam1`.
- `width`, `height`: 정렬된 원본 좌측 카메라 좌표의 깊이 이미지 크기.
- `captured_ms`: 서버 단조 시계로 환산한 기준 영상의 브라우저 그리기 시각 추정치. Unix 시간이 아니며 실제 센서 노출 시각도 아닙니다.
- `source_seq`, `right_captured_ms`, `pair_skew_ms`: 기준 프레임·우측 시각·좌우 시각 차이.
- `K`: 원본 카메라 3×3 내부 행렬. `distortion`: OpenCV 왜곡 계수.
- `world_to_camera`: 3×4 `[R|t]`, `X_camera=R X_world+t`. 길이 단위는 미터.
- `depth_definition`: `original_left_camera_optical_z`. 카메라 좌표의 X는 오른쪽, Y는 아래, Z는 전방입니다.
- `rectified_K`, `rectification_R`, `rectified_size`: 스테레오 내부 처리에 사용한 보정 행렬·해상도입니다. 반환 깊이를 원본에 다시 정합할 때 중복 적용하지 않습니다.
- `processing_ms`: 모델 추론 시간. 전체 왕복 지연이 아닙니다.
- `preview_space`: `original_left`. 색상 JPEG도 기준 RGB 및 PNG16과 같은 원본 좌측 카메라 좌표·크기입니다.
- `preview_range_m`: `[0.1, 8.0]`. 빨강→노랑(2m)→초록(4m)→파랑(8m 이상) 고정 범례이며, 8~15m는 같은 파랑입니다. 무효 픽셀은 검정입니다.
- `center_depth_m`: 원본 영상 중앙 픽셀의 광축 Z 깊이(m), 무효이면 `null`.
- `valid_percent`, `depth_range_m`: 정렬된 원본 깊이의 유효 픽셀 비율(정확도 점수가 아님)과 최소·최대(m). 모두 무효이면 범위는 `null`입니다.

카메라 세계 좌표는 바닥 ChArUco 보드의 X 오른쪽, Y 보드 아래쪽, Z 보드 안쪽입니다. 게임에서 높이를 표시할 때는 `height=-Z_world`를 사용합니다.

## 단건 수신

```python
from io import BytesIO
import json
import numpy as np
import requests

response = requests.get(
    "https://노트북IP:8443/api/depth/frame.npz",
    headers={"X-Session-Key": "접속키"},
    verify="certs/lava-local-ca.crt",
    timeout=3,
)
response.raise_for_status()
with np.load(BytesIO(response.content), allow_pickle=False) as result:
    depth_m = result["depth_m"]  # float32 H×W, NaN invalid
    reference_jpeg = result["reference_jpeg"].tobytes()
    metadata = json.loads(str(result["metadata"]))
```

이 예시의 `requests`는 팀원 소비자 측 패키지입니다. 서버는 요청 처리를 위해 필요하지 않습니다. 아직 유효한 깊이 결과가 없거나 결과가 350ms를 넘으면 HTTP **409**가 옵니다.

`/api/depth/depth.npy`, `/api/depth/reference.jpg`, `/api/depth/preview.jpg`, `/api/depth/metadata`도 제공합니다. 개별 요청은 서로 다른 프레임일 수 있으므로 결합할 때 `X-Capture-Ms` 또는 `captured_ms`를 확인합니다.

## 가림 처리

1. 가상 장면을 기준 카메라와 같은 투영·왜곡·영상 크기로 렌더링합니다. 또는 실제 기준 영상을 보정한 뒤 깊이도 같은 방식으로 보정합니다.
2. 가상 장면의 비선형 GPU depth buffer는 카메라 광축 Z의 미터 값으로 바꿉니다. near/far 값이 다른 정규화 depth와 직접 비교하면 안 됩니다.
3. 유효한 실제 깊이가 `real_z < virtual_z - epsilon`이면 그 픽셀의 가상 물체를 가립니다. `epsilon`은 실측으로 정하며 데모 바닥 합성은 **0.05m**를 사용합니다.
4. 무효 픽셀은 별도로 처리합니다. 무효 값을 0m 물체로 간주하거나 보간한 값을 실측처럼 취급하지 않습니다.
5. 사람을 반투명하게 보일 때는 사람 마스크·빈 배경·가상 장면 합성 정책을 적용합니다. 깊이맵 자체를 알파 마스크처럼 쓰지 않습니다.

현재 데모는 실제 깊이가 가상 바닥보다 앞서는 픽셀을 찾아 원본 영상을 불투명도 45%로 혼합합니다. 게임 충돌은 별도 관절 추적 결과를 같은 시차맵에서 샘플링합니다. 깊이 서비스는 사람 0명이어도 동작해야 합니다.

## 카메라 교체

프로필·렌즈·해상도가 바뀌면 내부 보정부터 다시 수행합니다. 위치만 바뀌어도 외부 보정은 다시 수행합니다. 이동·교체 버튼으로 기존 깊이 결과를 무효화합니다. 이전 보정의 결과는 새 보정에 재사용하지 않습니다.


## 연결·장치 상태

카메라 정보의 `guide`와 WSS 카메라 처리 응답 헤더의 `guide`는 같은 자동 보정 상태입니다. `stage`는 `collect`, `place`, `floor`, `background`, `ready`, `manual`이며, `checks`와 `coverage`의 `done`이 실제 만족 여부를 나타냅니다. `message`는 다음 동작, `outline`은 정규화된 기준판 외곽 네 점, `action`과 `can_act`는 사용자 확인 버튼입니다. 안내는 각 카메라의 첫 영상부터 독립적으로 시작됩니다.

`POST /api/cameras/{cam1|cam2}/{action}`에 `X-Session-Key` 인증을 사용합니다.

- `guide-floor`: 유효한 렌즈 보정과 새 영상이 필요합니다. 사용자가 최종 고정을 확인한 뒤 안정된 기준판 영상으로 외부 보정합니다.
- `guide-background`: 연결된 모든 카메라의 외부 보정과 추적 모델이 필요합니다. 사용자가 빈 화면을 확인한 뒤 배경을 저장하고 추적을 시작합니다.
- `guide-resume`: 저장된 렌즈 보정을 유지하되 외부 위치·배경·사진 수집 상태를 초기화하고 자동 안내로 돌아갑니다.
- `guide-restart`: 저장된 렌즈 보정 파일과 현재 렌즈·외부 위치·배경·사진 수집 상태를 초기화하고 처음부터 보정합니다.

기존 수동 보정 액션도 유지합니다. 수동 액션이 성공하면 자동 수집을 중지합니다. 재접속 시 내부 보정은 프로필·크기 일치 여부 확인 후 재사용하며 외부 보정과 배경은 새로 확인합니다.

기본 화면은 HTTPS/WSS로 TCP 8443을 사용합니다. 노트북 자동 로그인 및 카메라 초대 QR의 접속 키는 URL fragment에만 포함되고 페이지 로드 직후 주소에서 제거합니다. `GET /api/pairing`, `GET /api/pairing/{cam1|cam2}.png?address=노트북IP`는 `X-Session-Key` 인증이 필요합니다.

`GET /api/state`와 viewer WebSocket의 카메라 정보:

- `connection_status`: `disconnected`(미연결), `waiting`(인증 연결·영상 미수신), `receiving`(최근 2초 내 처리된 영상), `stalled`(연결은 남았으나 영상 끊김).
- `received_frames`: 현재 연결에서 처리한 프레임 수. `fps`: 최근 2초 내 처리된 프레임의 전송률.
- `connected`만으로 영상 성공을 판단하지 않습니다. 카메라 연결과 보정 완료·깊이 계산 준비도 별도로 판단합니다.
- WSS 카메라 입력은 기존 `pack` 바이트 형식, 헤더 `seq`, `captured_ms`, `profile`와 JPEG입니다. 서버가 반환한 이진 frame 응답이 실제 수신·처리 확인입니다. 클라이언트는 응답 전 다음 영상을 전송하지 않습니다.

QR로 사이트에 처음 접속하는 과정과 브라우저 카메라 권한 허용은 생략할 수 없습니다. 브라우저가 아직 접속하지 않은 임의의 주변 핸드폰을 자동 탐색한다고 가정하지 않습니다.
