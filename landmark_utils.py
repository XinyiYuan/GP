from tqdm import tqdm
import numpy as np
from imutils import face_utils
# import dlib
from collections import OrderedDict
import cv2
from calib_utils import track_bidirectional
import mediapipe as mp

def shape_to_face(shape, face, width, height):
    """
    Recalculate the face bounding box based on coarse landmark location(shape)
    :param
    shape: landmark locations
    :return:
    face_new: new bounding box of face (1*4 list [x1, y1, x2, y2])
    """
    x_min, x_max = face[0], face[3]
    y_min, y_max = face[1], face[4]
    z_min, z_max = face[2], face[5]
    
    x_min = int(x_min * width)
    x_max = int(x_max * width)
    y_min = int(y_min * height)
    y_max = int(y_max * height)

    face_new = [x_min, y_min, z_min, x_max, y_max, z_max]
    face_size = (x_max-x_min) * (y_max-y_min)
    
    return face_new, face_size

def predict_single_frame(frame):
    """
    :param frame: A full frame of video
    :return:
    shape: landmark locations
    """
  
  # use mediapipe
    mp_face_mesh = mp.solutions.face_mesh
    
    with mp_face_mesh.FaceMesh(
        max_num_faces = 1,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5) as face_mesh:
        # Convert the BGR image to RGB before processing.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame)
    
    shape = []
    x_min, x_max = 1.0, 0.0
    y_min, y_max = 1.0, 0.0
    z_min, z_max = 1.0, 0.0
    
    for face_landmarks in results.multi_face_landmarks:
        for id,lm in enumerate(face_landmarks.landmark):
            x_min, x_max = min(x_min, lm.x), max(x_max, lm.x)
            y_min, y_max = min(y_min, lm.y), max(y_max, lm.y)
            z_min, z_max = min(z_min, lm.z), max(z_max, lm.z)
            shape.append([lm.x, lm.y, lm.z])
    
    shape = np.array(shape)
    face = [x_min, y_min, z_min, x_max, y_max, z_max]
    # print(shape)
    # print(type(shape)) # numpy.ndarray
    # print(shape.shape) # (468, 3)
    # print(len(face)) # 6
    return face, shape

def detect_frames_track(frames, fps, use_visualization, visualize_path, video):

    frames_num = len(frames)
    frame_height, frame_width = frames[0].shape[:2]
    """
    Pre-process:
    To detect the original results,
    and normalize each face to a certain width, 
    also its corresponding landmarks locations and 
    scale parameter.
    """
    face_size_normalized = 400
    faces = []
    locations = []
    shapes_origin = []
    shapes_para = []  # Use to recover the shape in whole frame. ([x1, y1, scale_shape])
    face_size = 0
    skipped = 0

    """
    Use single frame to detect face on Mediapipe (CPU)
    """
    # ----------------------------------------------------------------------------#

    print("Detecting:")
    # for i in range(frames_num):
    for i in tqdm(range(frames_num)):
        frame = frames[i]
        face, shape = predict_single_frame(frame) # face: [0.0, 1.0] (normalized)

        face_new, face_size = shape_to_face(shape, face, frame_width, frame_height) # face_new: original size
        
        faceFrame = frame[face_new[1]: face_new[4], # y_min : y_max
                          face_new[0]: face_new[3]] # x_min : x_max
        if face_size < face_size_normalized:
            inter_para = cv2.INTER_CUBIC
        else:
            inter_para = cv2.INTER_AREA
        
        face_norm = cv2.resize(faceFrame, (face_size_normalized, face_size_normalized), interpolation=inter_para)
        scale_shape = face_size_normalized/face_size

        faces.append(face_norm)
        shapes_para.append([face_new[0], face_new[1], scale_shape])
        shapes_origin.append(shape)
        shape = shape.ravel()
        shape = shape.tolist()
        locations.append(shape)
    
    # print(type(locations))
    # print(len(locations))
    return locations

    """
    Calibration module.
    """
    locations_sum = len(locations)
    locations_track = locations
    
    """
    Visualization module.
    """
    # TODO
    
    # return locations
    