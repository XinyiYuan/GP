import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

with mp_face_mesh.FaceMesh(
	max_num_faces = 1,
	min_detection_confidence = 0.5,
	min_tracking_confidence = 0.5) as face_mesh:
	
	file_name = 'test'
	frame = cv2.imread(file_name + '.JPG')
	frame = cv2.cvtColor(frame, 1, cv2.COLOR_BGR2RGB)
	results = face_mesh.process(frame)
	
	shape = []
	x_min, x_max = 1, 0
	y_min, y_max = 1, 0
	z_min, z_max = 1, 0
	if results.multi_face_landmarks:
		for face_landmarks in results.multi_face_landmarks:
			for id,lm in enumerate(face_landmarks.landmark):
				x_min, x_max = min(x_min, lm.x), max(x_max, lm.x)
				y_min, y_max = min(y_min, lm.y), max(y_max, lm.y)
				z_min, z_max = min(z_min, lm.z), max(z_max, lm.z)
				shape.append([lm.x, lm.y, lm.z])
				# ih, iw, ic = frame.shape
				# x, y = int(lm.x*iw), int(lm.y*ih)
				# shape.append([id, x, y, lm.z])
	
	print(type(shape))
	shape = np.array(shape)
	face = [x_min, y_min, z_min, x_max, y_max, z_max]
	print(type(shape))
	print(shape.shape)
	print(face)
	print(len(face))