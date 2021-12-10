from tqdm import tqdm
import numpy as np
from imutils import face_utils
# import dlib
from collections import OrderedDict
import cv2
from calib_utils import track_bidirectional
import mediapipe as mp

def shape_to_face(shape, width, height):
    """
    Recalculate the face bounding box based on coarse landmark location(shape)
    :param
    shape: landmark locations
    :return:
    face_new: new bounding box of face (1*4 list [x1, y1, x2, y2])
    """
    x_min, y_min, z_min = np.min(shape, axis=0)
    x_max, y_max, z_max = np.max(shape, axis=0)
    
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
    
    x_min, y_min, z_min = np.min(shape, axis=0)
    x_max, y_max, z_max = np.max(shape, axis=0)
    face = [float(x_min), float(y_min), float(z_min), float(x_max), float(y_max), float(z_max)]
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
    for i in range(frames_num):
    # for i in tqdm(range(frames_num)):
        frame = frames[i]
        face, shape = predict_single_frame(frame) # face: [0.0, 1.0] (normalized)

        face_new, face_size = shape_to_face(shape, frame_width, frame_height) # face_new: original size
        
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
        

    """
    Calibration module.
    """
    locations_sum = len(locations)
    locations_track = locations
    
    """
    If us visualization, write the results to the visualize output folder.
    """
  # TODO
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
                # cv2.circle 第一项frame必须为灰度图
                for (x, y) in shape_origin:
                    cv2.circle(frame_origin, (x, y), 2, (0, 0, 255), -1)
                for (x, y) in shape_track:
                    cv2.circle(frame_track, (x, y), 2, (0, 255, 0), -1)
                origin_video.write(frame_origin)
                track_video.write(frame_track)
            origin_video.release()
            track_video.release()


    return locations
