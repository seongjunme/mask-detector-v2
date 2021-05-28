# USAGE
# python3 detect_mask_webcam_______.py

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tensorflow as tf
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os


def detect_and_predict_mask(frame, faceNet, maskNet):
	# 프레임 높이 너비
	(h, w) = frame.shape[:2]
	# 비디오 프레임 blob 객체로 전처리
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# 얼굴 인식 모델에 blob 객체 삽입
	faceNet.setInput(blob)
	# 얼굴인식 추론 값을 detections에 저장
	detections = faceNet.forward()

	faces = [] # 인식된 얼굴들
	locs = [] # 얼굴 좌표
	preds = [] # 마스크 착용 여부 추측

	# loop over the detections
	for i in range(0, detections.shape[2]):

		# 얼굴이라고 인식된 부분의 신뢰도를 confidence에 저장
		confidence = detections[0, 0, i, 2]

		# 신뢰도 값을 최소 신뢰도 값과 비교
		if confidence > 0.5:
			# box 얼굴 인식 좌표 값 설정 후 int 형 변환
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# 박스 위치가 화면에 벗어나지 않도록 조정
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# 박스 영역을 RGB로 변경하고 resizing 후 전처리
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# 인식된 얼굴들을 하나의 faces list로 통합 (face 와 loc index로 연관)
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# faces 넘파이 (float32) 변환
		faces = np.array(faces, dtype="float32")
		# 마스크 인식 모델에 faces 값 전달 후 예측
		# 얼굴 수 만큼, 모델 input의 형상 resize
		maskNet.resize_tensor_input(input_details[0]['index'], [len(faces), 224, 224, 3])
		# 텐서 할당
		maskNet.allocate_tensors()
		# faces를 모델에 전달
		maskNet.set_tensor(input_details[0]['index'], faces)
		maskNet.invoke()
		# 모델로 부터 예측 값 받기
		preds = maskNet.get_tensor(output_details[0]['index'])

	# 얼굴 좌표, 예측 값 리턴
	return (locs, preds)


# 얼굴 인식 모델 로드
faceNet = cv2.dnn.readNet('caffe_model/deploy.prototxt.txt',
                             'caffe_model/res10_300x300_ssd_iter_140000.caffemodel')

# my_model = load_model('model_2clas.tflite', compile=False)


# 마스크 인식 모델 로드
maskNet = tf.lite.Interpreter(model_path='model_2class_qua.tflite')
input_details = maskNet.get_input_details()
output_details = maskNet.get_output_details()

# 비디오스트림 열기
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# 비디오 시작
while True:
	# 영상을 프레임으로 할당
	frame = vs.read()
	frame = imutils.resize(frame, width=500)

	# 마스크 착용 예측 locs: 얼굴 좌표, preds: 마스크 착용 여부 예측
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# 얼굴 좌표와 예측 값 하나씩 불러와서 처리
	for (box, pred) in zip(locs, preds):
		# 박스 좌표와 예측 값 가져오기
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# 마스크 착용 여부 판단
		if mask > 0.9:
			label = "Thank You. Mask On."
			color = (0, 255, 0)
		else:
			label = "No Face Mask Detected"
			color = (0, 0, 255)
		
		# 박스 위에 글자 출력
		cv2.putText(frame, label, (startX-50, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
		# 박스 그리기
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# 비디오 창 띄우기
	cv2.imshow("Face Mask Detector", frame)
	# 무슨 키가 눌렸는지 확인
	key = cv2.waitKey(1) & 0xFF

	# q 입력 시 종료
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
