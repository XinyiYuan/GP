from tqdm import tqdm
import numpy as np
from imutils import face_utils
# import dlib
from collections import OrderedDict
import cv2
from calib_utils import track_bidirectional
import mediapipe as mp

# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# TODO:
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("inner_mouth", (60, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])


def shape_to_face(shape, width, height, scale=1.2):
    """
    Recalculate the face bounding box based on coarse landmark location(shape)
    :param
    shape: landmark locations
    scale: the scale parameter of face, to enlarge the bounding box
    :return:
    face_new: new bounding box of face (1*4 list [x1, y1, x2, y2])
    # face_center: the center coordinate of face (1*2 list [x_c, y_c])
    face_size: the face is rectangular( width = height = size)(int)
    """
    x_min, y_min, z_min = np.min(shape, axis=0)
    x_max, y_max, z_max = np.max(shape, axis=0)
    
    x_min = np.rint(x_min * width)
    x_max = np.rint(x_max * width)
    y_min = np.rint(y_min * height)
    y_max = np.rint(y_max * height)
    
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2
    
    face_size = np.rint(max(x_max - x_min, y_max - y_min) * scale)
    face_size = face_size // 2 * 2
    
    x1 = max(x_center - face_size // 2, 0)
    y1 = max(y_center - face_size // 2, 0)
  
    face_size = min(width - x1, face_size)
    face_size = min(height - y1, face_size)
    
    x2 = x1 + face_size
    y2 = y1 + face_size
    
    face_new = [int(x1), int(y1), z_min, int(x2), int(y2), z_max]
    
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
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        shape = results.multi_face_landmarks
    shape = str(shape).split()
    shape.pop(0)
    shape.pop(-1)
    
    j = 0
    for i in range(0, len(shape)):
        if shape[j] == 'x:' or shape[j] == 'y:' or shape[j] == 'z:' or shape[j] == '{' or shape[j] == '}' or shape[j] == 'landmark':
            shape.pop(j)
        else:
            shape[j] = float(shape[j])
            j += 1
    
    shape_new = [[0 for i in range(0, 3)]for j in range(0, len(shape)//3)]
    for j in range(0, len(shape)//3):
        for i in range(0, 3):
            shape_new[j][i] = shape[3*j+i]
        
    shape = np.array(shape_new)
    # print('shape:')
    # print(shape)
    # print(type(shape))
    
    x_min, y_min, z_min = np.min(shape, axis=0)
    x_max, y_max, z_max = np.max(shape, axis=0)
    face = [float(x_min), float(y_min), float(z_min), float(x_max), float(y_max), float(z_max)]
    return face, shape

def landmark_align(shape):
    desiredLeftEye = (0.35, 0.25)
    desiredFaceWidth=2
    desiredFaceHeight=2
    ## TODO
    (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]

    leftEyePts = shape[lStart:lEnd]
    rightEyePts = shape[rStart:rEnd]

    # compute the center of mass for each eye
    leftEyeCenter = leftEyePts.mean(axis=0)#.astype("int")
    rightEyeCenter = rightEyePts.mean(axis=0)#.astype("int")
    # compute the angle between the eye centroids
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX))  # - 180

    # compute the desired right eye x-coordinate based on the
    # desired x-coordinate of the left eye
    desiredRightEyeX = 1.0 - desiredLeftEye[0]

    # determine the scale of the new resulting image by taking
    # the ratio of the distance between eyes in the *current*
    # image to the ratio of distance between eyes in the
    # *desired* image
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desiredDist = (desiredRightEyeX - desiredLeftEye[0])
    desiredDist *= desiredFaceWidth
    scale = desiredDist / dist

    # compute center (x, y)-coordinates (i.e., the median point)
    # between the two eyes in the input image
    eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                  (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

    # update the translation component of the matrix
    tX = 0#desiredFaceWidth * 0.5
    tY = desiredFaceHeight * desiredLeftEye[1]
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

    n, d = shape.shape
    temp = np.zeros((n, d), dtype="int")
    temp[:, 0:3] = shape
    # temp[:, 2] = 1
    # print(M.shape) # (2, 3)
    # print(temp.T.shape) # (3, 468)
    aligned_landmarks = np.matmul(M, temp.T) # (2,468)
    return aligned_landmarks.T #.astype("int"))
'''
def check_and_merge(location, forward, feedback, P_predict, status_fw=None, status_fb=None):
    num_pts = 468 #68
    check = [True] * num_pts

    target = location[1]
    forward_predict = forward[1]

    # To ensure the robustness through feedback-check
    forward_base = forward[0]  # Also equal to location[0]
    feedback_predict = feedback[0]
    feedback_diff = feedback_predict - forward_base
    feedback_dist = np.linalg.norm(feedback_diff, axis=1, keepdims=True)

    # For Kalman Filtering
    detect_diff = location[1] - location[0]
    detect_dist = np.linalg.norm(detect_diff, axis=1, keepdims=True)
    predict_diff = forward[1] - forward[0]
    predict_dist = np.linalg.norm(predict_diff, axis=1, keepdims=True)
    predict_dist[np.where(predict_dist == 0)] = 1  # Avoid nan
    P_detect = (detect_dist / predict_dist).reshape(num_pts)

    for ipt in range(num_pts):
        if feedback_dist[ipt] > 2:  # When use float
            check[ipt] = False

    if status_fw is not None and np.sum(status_fw) != num_pts:
        for ipt in range(num_pts):
            if status_fw[ipt][0] == 0:
                check[ipt] = False
    if status_fw is not None and np.sum(status_fb) != num_pts:
        for ipt in range(num_pts):
            if status_fb[ipt][0] == 0:
                check[ipt] = False
    location_merge = target.copy()
    # Merge the results:
    """
    Use Kalman Filter to combine the calculate result and detect result.
    """

    Q = 0.3  # Process variance

    for ipt in range(num_pts):
        if check[ipt]:
            # Kalman parameter
            P_predict[ipt] += Q
            K = P_predict[ipt] / (P_predict[ipt] + P_detect[ipt])
            location_merge[ipt] = forward_predict[ipt] + K * (target[ipt] - forward_predict[ipt])
            # Update the P_predict by the current K
            P_predict[ipt] = (1 - K) * P_predict[ipt]
    return location_merge, check, P_predict
'''

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
    Use single frame to detect face on Dlib (CPU)
    """
    # ----------------------------------------------------------------------------#

    print("Detecting:")
    for i in tqdm(range(frames_num)):
        frame = frames[i]
        face, shape = predict_single_frame(frame) # face: [0.0, 1.0] (normalized)

        face_new, face_size = shape_to_face(shape, frame_width, frame_height, 1.2) # face_new: original size
        
        faceFrame = frame[face_new[1]: face_new[4], face_new[0]: face_new[3]]
        if face_size < face_size_normalized:
            inter_para = cv2.INTER_CUBIC
        else:
            inter_para = cv2.INTER_AREA
        
        face_norm = cv2.resize(faceFrame, (face_size_normalized, face_size_normalized), interpolation=inter_para)
        scale_shape = face_size_normalized/face_size
        
        # shape_norm = np.rint((shape-np.array([face_new[0], face_new[1]])) * scale_shape).astype(int)
        
        face_array = np.array([face_new[0], face_new[1]])
        face_array = np.pad(face_array, (0,1), 'constant', constant_values=(0,0))
        shape_norm = np.rint((shape-face_array)*scale_shape)
        
        faces.append(face_norm)
        shapes_para.append([face_new[0], face_new[1], scale_shape])
        shapes_origin.append(shape)
        locations.append(shape_norm)
    
      # print('success!')
    """
    Calibration module.
    """
    locations_sum = len(locations)
    locations_track = locations
    
    """
    If us visualization, write the results to the visualize output folder.
    """
    if locations_sum != frames_num:
        print("INFO: Landmarks detection failed in some frames. Therefore we disable the "
              "visualization for this video. It will be optimized in future version.")
    else:
        if use_visualization:
            fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
            frame_size = (frames[0].shape[1], frames[0].shape[0])
            origin_video = cv2.VideoWriter(visualize_path+video+"_origin.avi",
                                           fourcc, fps, frame_size)
            track_video = cv2.VideoWriter(visualize_path+video+"_track.avi",
                                          fourcc, fps, frame_size)

            print("Visualizing")
            for i in tqdm(range(frames_num)):
                frame_origin = frames[i].copy()
                frame_track = frames[i].copy()
                shape_origin = shapes_origin[i]
                para_shift = shapes_para[i][0:2]
                para_scale = shapes_para[i][2]
                shape_track = np.rint(locations_track[i] / para_scale + para_shift).astype(int)
                for (x, y) in shape_origin:
                    cv2.circle(frame_origin, (x, y), 2, (0, 0, 255), -1)
                for (x, y) in shape_track:
                    cv2.circle(frame_track, (x, y), 2, (0, 255, 0), -1)
                origin_video.write(frame_origin)
                track_video.write(frame_track)
            origin_video.release()
            track_video.release()

    aligned_landmarks = []
    for i in locations_track:
        shape = landmark_align(i)

        shape = shape.ravel()
        shape = shape.tolist()
        aligned_landmarks.append(shape)

    return aligned_landmarks


