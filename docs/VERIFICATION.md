# 검증 기록

검증일: 2026-11-15, Asia/Seoul. GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4096MiB. 이 문서는 소프트웨어 검증과 실제 핸드폰 검증을 구분합니다.

| 실행 명령 | 관찰 결과 |
|---|---|
| `.venv/Scripts/python -m pytest -q` | 28 passed. 보정·좌표 정합·시각 동기화·API·게임·인증서·CA 유지·자동 로그인·QR·실제 TLS WebSocket 프레임·연결 상태 검사 |
| `.venv/Scripts/python -m ruff check ar_lava scripts tests` | exit 0, All checks passed |
| `.venv/Scripts/python -m ruff format --check ar_lava scripts tests` | exit 0, 20 files already formatted |
| `npm run lint` | exit 0, anti-slop 전체 일반 규칙 활성 |
| `npm run typecheck` | exit 0, TypeScript의 JavaScript 검사 |
| `node node_modules/@playwright/test/cli.js test --reporter=line --repeat-each=2` | 8 passed, 4개 시나리오를 2회 반복. Chrome 가상 카메라 2개의 WebSocket 영상·QR 자동 입력·연결 검사 실패·재연결·1440/390px 화면 검사 |
| `.venv/Scripts/python scripts/smoke_depth.py` | 실제 CUDA 모델, 사람 검출 없이 HTTP NPZ와 WebSocket PNG16 반환 성공 |
| `.venv/Scripts/python -m pip check` | No broken requirements found |

초기 WebRTC 경로는 해상도 자동 축소 문제를 수정했습니다. 현재 기본 화면은 HTTPS/WSS로 전환했고, WebRTC API는 별도로 유지합니다. 기본 화면은 UDP 연결을 요구하지 않습니다. WebSocket 연결 종료 중 취소 처리도 수정하고 재검사했습니다. 원본 모델의 단순 FP16 변환은 그래프 타입 오류로 사용하지 않습니다.

로컬 HTTPS 서버를 실제로 시작해 Python 3.13의 엄격한 CA·호스트 이름 검증으로 `/api/health`의 HTTP 200을 확인하고 서버를 정상 종료했습니다. 최초 인증서의 키 식별자 누락을 수정하고, CA/서버 인증서 식별자 및 CA 유지 회귀 검사를 추가했습니다. `setup.ps1`도 exit 0으로 실행했습니다.

`cmd.exe /d /c start.cmd`도 실행해 환경 확인·모델/인증서 준비·HTTPS 서버 시작·브라우저 실행 전 health HTTP 200을 확인했습니다. 첫 설치가 필요한 분기는 기존 `setup.ps1`을 호출하며, 이번 실행에서는 설치된 가상 환경을 사용했습니다.

GPU 통합 검사에서 모델 추론은 약 **141.2ms**, 출력 깊이 크기는 **960×540**, 해당 공식 샘플에서 유효 깊이 비율은 **약 84.1%**였습니다. 유효 비율은 오차나 정확도 점수가 아닙니다. 출력 기준 영상과 깊이의 크기·단위·무효 픽셀·PNG 비트 깊이·보정 메타데이터를 검사했습니다.

검사 로그의 Starlette TestClient/httpx 지원 변경 경고와 Node의 type stripping 실험 기능 경고는 남아 있습니다. 실행 실패는 아닙니다. MediaPipe는 빈 프레임을 실제 추론하여 사람 0명으로 반환했고 한글 절대 경로의 모델 파일을 읽었습니다.

lint 자동 수정 두 번째 실행에서 파일 해시가 바뀌지 않았습니다. anti-slop 원본 파일 38개의 SHA256를 비교했고, 원형 ZIP과 라이선스·출처 정보를 보관했습니다.

**미검증:** 실제 핸드폰 2~3대, 실물 기준판 보정 정확도, 기기별 노출 지연과 롤링 셔터, 실제 플레이 중 깊이 오차·전체 왕복 지연·충돌 판정, Android/iOS 인증서 등록. 브라우저 시각 보정은 하드웨어 동기화를 대체하지 않습니다. TensorRT 최적화도 적용하지 않았습니다.


현재 연결 흐름은 노트북 자동 로그인 → 화면 상단에 카메라별 QR 자동 표시 → 핸드폰의 ‘연결하기’입니다. QR 입력 키는 주소 fragment에서 읽은 뒤 주소에서 제거합니다. 연결 대기·서버 인증 연결·영상 수신 중·영상 끊김을 구분하고, 끊어진 영상과 FPS가 계속 표시되지 않는지 확인했습니다. 주소를 바꾸는 동안 오래된 QR 요청이 새 QR을 덮던 경쟁 조건도 수정했습니다.

TLS 검사는 임시 CA를 신뢰한 실제 `wss://localhost:임시포트/ws` 연결로 수행했습니다. WebRTC 없이 JPEG를 보내고 이진 처리 응답 및 `HTTPS/WSS` 수신 상태를 확인했습니다. 기본 WSS는 응답 대기 프레임 한 개, JPEG 품질 0.8, 최대 15FPS 촬영 주기를 사용합니다. 전송량·학교망·실제 폰 성능은 미측정입니다.


브라우저 가상 카메라가 여러 시크릿 컨텍스트 사이에서 간헐적으로 ‘장치를 찾지 못함’을 반환해, 실제 여러 핸드폰처럼 각 폰을 독립 Chrome 프로세스로 실행하도록 검사 환경을 수정했습니다. 수정 후 4개 시나리오를 2회 반복해 **8개 모두 통과**했습니다. 앱은 장치 찾기/읽기 실패에 한해 500ms 후 기본 카메라로 한 번 재시도하고, 권한 거부·장치 없음·다른 앱 점유를 구분해 알립니다. 실제 폰에서 같은 증상이 재현됐다는 뜻은 아닙니다.

QR 표시 수정 후 `node node_modules/@playwright/test/cli.js test --reporter=line --grep "desktop shows three QR"`는 **1 passed**였습니다. 버튼 조작 없이 QR 3개가 로딩되고 첫 QR이 첫 화면 안에 있는지 검사했습니다. 전체 4개 시나리오 재실행은 **3 passed, 1 failed**였고, 실패 지점은 촬영 종료 후 가상 카메라 재연결의 ‘카메라를 찾지 못함’이었습니다. 해당 시나리오만 재실행해도 같은 지점에서 실패했습니다. QR 표시·초기 영상 수신은 통과했지만, 위의 이전 8 passed를 현재 전체 통과 결과로 해석하면 안 됩니다. 실제 폰 재연결은 미검증입니다.

## 자동 보정과 거리별 색상 추가 검증 (2026-11-16)

이번 작업의 최신 결과이며, 위의 과거 통과 기록을 현재 전체 통과로 해석하면 안 됩니다.

- `.venv/Scripts/ruff.exe check ar_lava tests`, `npm run typecheck`, `npm run lint`: exit 0.
- `.venv/Scripts/python.exe -m pytest -q`: **35 passed**. 실제 ChArUco 검출로 중복·움직임·기준판 소실을 거부하고, 서로 다른 합성 투영 사진 12장 이상과 촬영 범위를 모아 내부 보정을 계산했습니다. 수동 고정 확인 전에는 외부 위치 보정을 하지 않는지, 모든 연결 카메라의 외부 보정 전에 배경 저장을 거부하는지, 재접속·프로필 변경·초기화, 인증 및 색상·깊이 좌표 정합도 검사했습니다.
- `node node_modules/@playwright/test/cli.js test --reporter=line`: **5 passed, 1 failed**, exit 1. Chrome 프로세스 3개의 실제 가상 카메라 영상 수신과 각 폰의 독립 체크리스트, QR 표시, 오류 표시, 1440px·390px 가로 넘침, 새 체크 상태 전환·색상 픽셀·표시 모드 변경은 통과했습니다. 실패는 기존에 기록된 촬영 종료 후 가상 카메라 재연결의 ‘카메라를 찾지 못함’입니다. 검사를 약화하거나 생략하지 않았습니다.
- 새 UI 시나리오만 별도 포트에서 실행한 검사도 **1 passed**, exit 0이었습니다. 스크린샷의 UI 검사용 깊이와 완료 상태는 합성 데이터이며 실물 보정 성공 증거가 아닙니다.
- 실제 CUDA 모델과 기존 공식 좌우 샘플로 **960×540 원본 좌표의 색상 깊이 JPEG**를 만들었습니다. `docs/depth-color-demo.jpg`가 출력입니다. 측정 가능 픽셀 **84.1%**, 중앙 깊이 **0.58m**이며 정확도 점수가 아닙니다. 직접 계산 두 회의 전체 처리 시간은 **5183.0ms / 1664.5ms**, 두 번째 모델 추론은 **897.8ms**였습니다.
- 기존 `.venv/Scripts/python.exe scripts/smoke_depth.py`는 **실패**, exit 1입니다. 모델이 준비됐지만 이번 실행에서 추론 지연이 약 **4319.6ms**로 **350ms** 유효시간을 넘어 실시간 결과를 받지 못했습니다. 기본 실행에서도 한 번 실패했고 샌드박스 밖 재검사에서도 실패했습니다. 색상 파일 생성 성공을 실시간 성능 통과로 해석하면 안 됩니다. 기존 최신 결과 유효시간과 게임 추적 조건은 완화하지 않았습니다.

**남은 확인:** 실제 폰 2~3대와 출력 기준판으로 보정 정확도, 밝기·선명도 임계값의 기기별 사용성, 빈 배경 확인, 실시간 GPU 처리·전송 성능을 확인해야 합니다. 최종 고정과 빈 배경은 사용자가 확인하며, 설치 후 카메라 이동을 자동 탐지하지 않습니다.

## REST RGB-D 정합 응답 추가 (2026-11-16)

`GET /api/rgbd/frame`으로 원본 카메라 1의 RGB, 고정 0.1~15m 범위의 0~1 float32 Z-depth, 유효 마스크, 보정값, 같은 프레임의 발 3D 추적을 반환합니다. 새 프레임 계산은 기존 좌우 카메라 파이프라인에서 수행하며 REST는 완료된 결과 하나를 가져옵니다. HTTP 응답 인코딩은 별도 스레드에서 수행하고, 인코딩 뒤에도 350ms 만료·연결·보정 세대를 다시 확인합니다.

- `.venv/Scripts/ruff.exe check ar_lava scripts tests`: exit 0.
- `.venv/Scripts/python.exe -m pytest -q`: **45 passed**. 인증, 원본 RGB 바이트, little-endian float32 정규화·미터 복원, 무효 마스크, 고정 범위 유지, 발 좌표 Z 부호 변환, 사람 미확인 상태의 깊이 제공, 중복 프레임 ID, 만료·연결 끊김·보정 변경, 인코딩 중 프레임 교체·만료, 실제 REST 응답의 수신 예제 디코딩, 기존 미터 NPZ 호환성을 검사합니다.
- `.venv/Scripts/python.exe scripts/fetch_rgbd.py --help`: exit 0. 수신 예제는 표준 HTTP 라이브러리와 NumPy만 사용합니다. 실제 폰 서버를 대상으로 한 예제 실행은 아직 하지 않았습니다.

이번 검증은 API 계약과 일관성 검증입니다. 새 GPU 성능 통과나 실제 폰·가상 공간 정합 성공을 뜻하지 않습니다. 위에 기록한 GPU 실시간 지연과 가상 카메라 재연결 실패는 해결했다고 주장하지 않습니다. 프런트엔드는 이번 REST 추가에서 수정하지 않았으며 브라우저 전체 검사를 재실행하지 않았습니다.

## 사람 마스크 반환과 현재 환경의 모델 선택 (2026-11-16)

현재 기본 모델은 **MediaPipe Pose Landmarker Lite**를 유지합니다. 이미 발 추적에 쓰는 모델의 마스크를 함께 반환하므로 별도 세그멘테이션 추론·GPU 모델을 추가하지 않습니다. [MediaPipe 공식 설명](https://developers.google.com/edge/mediapipe/solutions)은 Pose Landmarker의 선택적 세그멘테이션 출력을 설명합니다. SAM 3.1은 최신 여부만 확인했으며 기본 모델로 설치하거나 적용하지 않았습니다.

검토 후보로 [RVM MobileNetV3의 공식 ONNX 모델](https://github.com/PeterL1n/RobustVideoMatting)을 내려받아 **CPUExecutionProvider, 2스레드**로 비교했습니다. 테스트 이미지는 Google의 `https://storage.googleapis.com/mediapipe-assets/pose.jpg`, 입력은 640×480입니다. 첫 2회 제외 후 같은 사진 10회 반복의 중앙값은 다음과 같습니다.

- 기존 포즈+마스크 처리: **22.85ms**.
- RVM MobileNetV3 FP32, 내부 축소 비율 0.5: **53.28ms**.
- RVM MobileNetV3 FP32, 내부 축소 비율 0.8: **140.97ms**.

RVM을 추가하면 기존 발 추적 계산에 위 마스크 계산 비용이 더해집니다. 샘플 합성을 눈으로 비교했고, 이 한 장에서는 현재 마스크 재사용을 포기할 만한 근거를 확인하지 못했습니다. 이 판단은 정답 마스크를 이용한 품질 평가나 실제 플레이 동영상의 경계 안정성 평가가 아닙니다. RVM 모델은 로컬 비교용이며, 기본 실행·모델 준비·배포 ZIP에는 추가하지 않았습니다. 비교 수치와 이미지는 `data/segmentation-comparison.json`, `data/segmentation-comparison.jpg`에 있습니다.

현재 REST의 `segmentation`은 원본 RGB·깊이와 같은 프레임의 사람 영역 확률을 PNG8로 반환하며, 실제 제공자는 `model`에 표시합니다. MediaPipe가 실제로 `(H,W,1)` 마스크를 반환하는 것을 확인해 단일 채널 `(H,W)`로 변환했습니다. 확률 마스크를 깊이 유효 마스크나 광학 알파 매트로 표시하지 않습니다. 마스크가 없으면 필수 RGB·깊이를 유지하면서 `available=false`를 반환합니다.

`.venv/Scripts/ruff.exe check ar_lava scripts tests`는 exit 0, `.venv/Scripts/python.exe -m pytest -q`는 **54 passed**, exit 0입니다. 새 검사는 마스크 PNG 디코딩과 확률 범위, 제공 모델 이름, 크기 정합, 인코딩 중 새 프레임과 마스크가 섞이지 않는지, 누락·NaN·범위 밖 마스크의 선택적 생략, MediaPipe 채널 축과 크기 변환을 포함합니다. 기존 GPU 350ms 지연 제한 미달은 해결하지 않았으며 실제 플레이 영상의 품질·전체 지연은 남아 있습니다.

## 기존 웹 모니터·REST 로그와 실행 파일 분리

UI 확인용은 루트 `run_ui.py`, 브라우저 자동 실행 없이 같은 REST·카메라 수신 서버를 제공하는 연결용은 `run_api.py`입니다. 공통 코드와 준비된 환경·모델을 함께 사용하며 두 파일만 복사해 독립 실행하는 방식은 아닙니다. 같은 포트이므로 둘 중 하나만 실행합니다. 새 비공개 저장소는 `wellseekogi/ar-rgbd-camera`입니다. 접속 키·인증서·모델·촬영 데이터와 로컬 환경은 업로드 대상에서 제외합니다.

- `npm run lint`, `npm run typecheck`, `.venv/Scripts/ruff.exe check ar_lava scripts tests run_ui.py run_api.py`: exit 0.
- `.venv/Scripts/python.exe -m pytest -q`: **54 passed**, exit 0.
- `.venv/Scripts/python.exe run_ui.py --help`, `.venv/Scripts/python.exe run_api.py --help`: exit 0. 실행 진입점이 로드되는지 확인했으며 실제 인증서 서버의 브라우저 자동 열기는 이번 작업에서 재검사하지 않았습니다.
- `node node_modules/@playwright/test/cli.js test --reporter=line --grep 'three phone|live monitor|guided checks|desktop shows three QR'`: **4 passed**, exit 0. 두 카메라 영상 캔버스·수신 상태, 기존 QR 표시와 자동 접힘, 깊이 색상 전환, REST의 HTTP 200·409·JSON·Base64 표시 축약, 로그 중지 후 요청 중단·비우기·재시작, 1440px·390px 가로 넘침을 포함합니다. REST 성공 응답은 UI 테스트 fixture이며 실제 GPU 성공을 뜻하지 않습니다. 전체 브라우저 검사 중 기존 가상 카메라 재연결 실패는 이번 변경에서 해결하지 않았습니다.

REST 로그는 화면 표시용이고 응답 형식이나 350ms 유효시간을 완화하지 않습니다. 현재 GPU 지연 상태에서는 실제 REST에 409가 표시될 수 있습니다.
