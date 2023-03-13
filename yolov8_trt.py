# import torch
import cv2
import numpy as np
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta, date
from warnings import filterwarnings

import argparse

from config import CLASSES, COLORS
from models.utils import blob, det_postprocess, letterbox, path_to_list

# from models.experimental import attempt_load

filterwarnings('ignore')

class SlagAI:

    def __init__(self):
        self.parse_args()

        # -----------------------------------------------------------------
        # load model

        # self.yolo_weights = Path('./yolov7.pt')

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # self.model = attempt_load(self.yolo_weights, map_location=self.device)  # load FP32 model
        # self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # Get the class names

        self.names = ['dropping', 'success']

        # Yolov8 TRT

        if self.args.method == 'cudart':
            from models.cudart_api import TRTEngine
        elif self.args.method == 'pycuda':
            from models.pycuda_api import TRTEngine
        else:
            raise NotImplementedError

        self.Engine = TRTEngine(self.args.engine)
        self.H, self.W = self.Engine.inp_info[0].shape[-2:]

        # Yolov8 TRT



        # load model
        # -----------------------------------------------------------------

        self.log_dir = Path('./SlagAI_log')
        self.save_dir = Path('./runs')

        self.check_dir_exists()
        
        self.save_vid = True
        self.save_txt = True

        # -- video size

        self.windows_w, self.windows_h = (1280, 720)

        # -- video size


        # -----------------------------------------------------------------
        # Set logger

        self.date_today_sys = ''
        self.date_today = ''

        self.detect_logger = None
        self.logger = None

        # self.set_save_config()
        # self.set_sys_logger()

        # Set logger
        # -----------------------------------------------------------------

        # -----------------------------------------------------------------
        # Set threshold

        self.threshold_frame = 100

        # Set threshold
        # -----------------------------------------------------------------

    def check_dir_exists(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # @torch.no_grad()
    def detect_objects_realtime(self):

        # Start the video capture
        #cap = cv2.VideoCapture(0)
        #cap = cv2.VideoCapture("slagOK.mp4")
        cap = cv2.VideoCapture("rtsp://admin:Admin1357@192.168.0.190:554/profile2/media.smp")
        time.sleep(1)  # Wait for the camera to warm up
        
        self.frame_counter = -1
        self.empty_coutner = -1

        self.droped = False

        # Loop through the frames and detect objects in real-time
        while True:
            # Read a frame from the video capture
            success, frame = cap.read()

            if not success:
                break

            if str(date.today()) != self.date_today_sys:
                self.date_today_sys = str(date.today())
                self.set_sys_logger()

            if self.windows_w is None and self.windows_h is None:
                self.windows_h, self.windows_w, _ = frame.shape


            # Extract the ROI from the frame
            # x, y, w, h = self.roi
            # roi_frame = frame[y:y+h, x:x+w]
            drawed_img = frame.copy()
            
            # Convert the ROI image to a PyTorch tensor and move it to the GPU
            # blob = torch.from_numpy(cv2.resize(blob, (640, 640))).to(self.device)
            # blob = blob.permute(2, 0, 1).float().unsqueeze(0) / 255.0
            
            # Detect objects in the ROI using the YOLOv7 model
            # with torch.no_grad():
            # detections = self.model(blob)[0][0]
            # detections = detections[detections[:, 4] > 0.5]  # Keep only detections with a confidence score > 0.5

            # Draw the bounding boxes and labels on the ROI image
            font = cv2.FONT_HERSHEY_SIMPLEX

            blob_frame, ratio, dwdh = letterbox(drawed_img, (self.W, self.H))

            tensor = blob(blob_frame, return_seg=False)
            dwdh = np.array(dwdh * 2, dtype=np.float32)
            tensor = np.ascontiguousarray(tensor)

            # inference
            data = self.Engine(tensor)

            bboxes, scores, labels = det_postprocess(data)
            bboxes -= dwdh
            bboxes /= ratio

            if len(labels) > 0:
                for (bbox, score, label) in zip(bboxes, scores, labels):
                    bbox = bbox.round().astype(np.int32).tolist()
                    cls_id = int(label) - 1

                    class_str = self.names[cls_id]
                    cls = CLASSES[cls_id]
                    color = COLORS[cls]

                    cv2.rectangle(drawed_img, bbox[:2], bbox[2:], color, 2)
                    cv2.putText(drawed_img,
                                f'{class_str}:{score:.3f}', (bbox[0], bbox[1] - 2),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, [225, 255, 255],
                                thickness=2)
                    

            #     for detection in detections:
            #         detection = detection.cpu()

            #         classid, score, box = detection[5:], detection[4], detection[0:4]
            #         box = box.numpy().astype(np.int)
            #         left, top, width, height = box

            #         xmin = int(left / 640.0 * self.windows_w)
            #         ymin = int(top / 640.0 * self.windows_h)
            #         xmax = int((left + width) / 640.0 * self.windows_w)
            #         ymax = int((top + height) / 640.0 * self.windows_h)

            #         classid = torch.argmax(classid)
            #         class_str = self.names[classid]

                    label = f"{class_str} {score:.2f}"

            #         cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            #         cv2.putText(frame, label, (xmin, ymin - 5), font, 0.5, (0, 0, 255), 1)
            
                self.logger.info(f"Detected objects: {label}")

                if class_str == 'dropping':
                    if not self.droped:
                        self.set_save_config()
                    
                    self.droped = True

                if self.droped:
                    self.video_writer.write(cv2.resize(drawed_img, (self.windows_w, self.windows_h)))
                    self.detect_logger.debug(class_str)
                
            else:
                if self.droped:
                    self.empty_coutner += 1

                if self.droped and self.empty_coutner == self.threshold_frame:
                    if isinstance(self.video_writer, cv2.VideoWriter):
                        self.video_writer.release()
                        self.video_writer = None

                    self.droped = False
                    self.empty_coutner = 0

                    self.logger.info(f"Save video to: {self.save_exp_dir}")

            # Display the ROI image with bounding boxes
            cv2.imshow("ROI", cv2.resize(drawed_img, (self.windows_w, self.windows_h)))
            
            # Save the ROI image with bounding boxes to a log file
            # logging.basicConfig(filename='detected_objects.log', level=logging.INFO)
            
            # self.logger.info(f"Detected objects: {detections.cpu().numpy()}")
            
            # Wait for a key press to exit
            if cv2.waitKey(1) == ord('q'):
                break

        # Release the video capture and close all windows
        cap.release()
        cv2.destroyAllWindows()

    def set_save_config(self) -> None:
        if str(date.today()) != self.date_today:
            self.date_today = str(date.today())

            self.save_date_dir = self.save_dir / f"{self.date_today}"
            self.save_date_dir.mkdir(parents=True, exist_ok=True)
        
        new_idx = len(list(self.save_date_dir.iterdir())) + 1

        self.save_exp_dir = self.save_date_dir / f"exp{new_idx}"
        self.save_exp_dir.mkdir(parents=True, exist_ok=True)
        
        if self.save_vid:
            self.video_path = self.save_exp_dir / f"{self.date_today}-exp{new_idx}.mp4"
            fps = 30.0
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(str(self.video_path), fourcc, fps, (self.windows_w, self.windows_h))

        if self.save_txt:
            if not isinstance(self.detect_logger, logging.Logger):
                self.detect_logger = logging.getLogger('Detected_objects')
                self.detect_logger.setLevel(logging.DEBUG)

            formatter = logging.Formatter("%(asctime)s :: %(message)s")

            stream_handler_exists = False
            for hdl in self.detect_logger.handlers:
                if isinstance(hdl, logging.StreamHandler):
                    stream_handler_exists = True
        
            if not stream_handler_exists:
                self.stream_handler = logging.StreamHandler()
                self.stream_handler.setFormatter(formatter)
                self.stream_handler.setLevel(logging.INFO)
                self.detect_logger.addHandler(self.stream_handler)

            if hasattr(self, 'file_handler') and isinstance(self.detect_logger, logging.Logger):
                self.detect_logger.removeHandler(self.file_handler)

            self.file_handler = logging.FileHandler(str(self.save_exp_dir / f"Detected_objects.log"))
            self.file_handler.setFormatter(formatter)
            self.detect_logger.addHandler(self.file_handler)
    
    def set_sys_logger(self) -> None:
        if hasattr(self, 'file_handler_sys') and isinstance(self.logger, logging.Logger):
            self.logger.removeHandler(self.file_handler_sys)
        else:
            self.logger: logging.Logger = logging.getLogger(name='System')
            self.logger.setLevel(logging.DEBUG)

        stream_handler_exists = False
        for obj in self.logger.handlers:
            if isinstance(obj, logging.StreamHandler):
                stream_handler_exists = True
                break
        
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)-10s :: %(message)s")

        if not stream_handler_exists:
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.INFO)
            stream_handler.setFormatter(formatter)
            
            self.logger.addHandler(stream_handler)

        self.file_handler_sys = logging.FileHandler(str(self.log_dir / f"{self.date_today_sys}.log"))
        self.file_handler_sys.setLevel(logging.DEBUG)
        self.file_handler_sys.setFormatter(formatter)

        self.logger.addHandler(self.file_handler_sys)

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--engine', type=str, default='yolov8n.engine', help='Engine file')
        parser.add_argument('--method',
                            type=str,
                            default='cudart',
                            help='CUDART pipeline')
        self.args = parser.parse_args()


if __name__ == "__main__":
    app: SlagAI = SlagAI()
    app.detect_objects_realtime()
