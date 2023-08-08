"""
A Module which binds Yolov7 repo with Deepsort with modifications
"""
import os
import torchvision
import matplotlib.pyplot as plt
from calc_speed import calcSpeed
from scipy.stats import multivariate_normal

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # comment out below line to enable tensorflow logging outputs
import time
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)



from tensorflow.compat.v1 import \
    ConfigProto  # DeepSORT official implementation uses tf1.x so we have to do some modifications to avoid errors

# deep sort imports
from deep_sort.deep_sort import preprocessing, nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from detection_helpers import *
from tracking_helpers import read_class_names

# load configuration for object detector
config = ConfigProto()
config.gpu_options.allow_growth = True


def get_gaussian_mask():
    # 128 is image size
    x, y = np.mgrid[0:1.0:128j, 0:1.0:128j]
    xy = np.column_stack([x.flat, y.flat])
    mu = np.array([0.5, 0.5])
    sigma = np.array([0.22, 0.22])
    covariance = np.diag(sigma ** 2)
    z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
    z = z.reshape(x.shape)

    z = z / z.max()
    z = z.astype(np.float32)

    mask = torch.from_numpy(z)

    return mask


class DeepSORT:
    """
    Class to Wrap ANY detector  of YOLO type with DeepSORT
    """

    def __init__(self, reID_model_path: str, detector, max_cosine_distance: float = 0.4, nn_budget: float = None,
                 nms_max_overlap: float = 1.0):
        """
        args:
            reID_model_path: Path of the model which uses generates the embeddings for the cropped area for Re identification
            detector: object of YOLO models or any model which gives you detections as [x1,y1,x2,y2,scores,class]
            max_cosine_distance: Cosine Distance threshold for "SAME" person matching
            nn_budget:  If not None, fix samples per class to at most this number. Removes the oldest samples when the budget is reached.
            nms_max_overlap: Maximum NMs allowed for the tracker
        """
        self.detector = detector
        self.nms_max_overlap = nms_max_overlap
        self.class_names = read_class_names()
        self.total_objects = 0

        # initialize deep sort
        # self.encoder = create_box_encoder(reID_model_path, batch_size=128)
        device = select_device("0" if torch.cuda.is_available() else 'cpu')
        self.encoder = torch.load(reID_model_path, map_location=torch.device(device))
        self.encoder = self.encoder.eval()
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance,
                                                                nn_budget)  # calculate cosine distance metric
        self.tracker = Tracker(self.metric)  # initialize tracker
        self.gaussian_mask = get_gaussian_mask()
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.ToTensor()])

    def reset_tracker(self):
        """

        Returns
        -------

        """
        self.tracker = Tracker(self.metric)

    def pre_process(self, frame, detections):
        """

        Parameters
        ----------
        frame
        detections

        Returns
        -------

        """
        crops = []
        for d in detections:
            img_h, img_w, img_ch = frame.shape

            xmin, ymin, w, h = d

            if xmin > img_w:
                xmin = img_w

            if ymin > img_h:
                ymin = img_h

            xmax = xmin + w
            ymax = ymin + h

            ymin = abs(int(ymin))
            ymax = abs(int(ymax))
            xmin = abs(int(xmin))
            xmax = abs(int(xmax))

            try:
                crop = frame[ymin:ymax, xmin:xmax, :]
                crop = self.transforms(crop)
                crops.append(crop)
            except:
                continue

        crops = torch.stack(crops)
        return crops

    def run_deep_sort(self, frame, detections):
        """

        Parameters
        ----------
        frame
        detections

        Returns
        -------

        """
        if detections is None:
            self.total_objects = 0
            self.tracker.predict()
            # print('No detections')
            # trackers = self.tracker
            return self.tracker

        else:
            bboxes = detections[:, :4]
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]  # convert from xyxy to xywh
            bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]

            scores = detections[:, 4]
            classes = detections[:, -1]
            num_objects = bboxes.shape[0]
            self.total_objects = num_objects

        names = []
        for i in range(num_objects):  # loop through objects and use class index to get class name
            class_idx = int(classes[i])
            class_name = self.class_names[class_idx]
            names.append(class_name)
        names = np.array(names)

        processed_crops = self.pre_process(frame, bboxes.copy()).cuda()
        processed_crops = self.gaussian_mask.cuda() * processed_crops

        features = self.encoder.forward_once(processed_crops)
        features = features.detach().cpu().numpy()

        if len(features.shape) == 1:
            features = np.expand_dims(features, 0)

        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                      zip(bboxes, scores, names, features)]  # [No of BB per frame] deep_sort.detection.Detection object

        boxs = np.array([d.tlwh for d in detections])  # run non-maxima suppression below
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        self.tracker.predict()  # Call the tracker
        self.tracker.update(detections)  # update using Kalman Gain

        return self.tracker

    def track_video(self, video: str, output: object = None, skip_frames: int = 0, show_live: bool = False,
                    count_objects: bool = False, verbose: int = 0):
        """
        Track any given webcam or video
        args:
            video: path to input video or set to 0 for webcam
            output: path to output video
            skip_frames: Skip every nth frame. After saving the video, it'll have very visuals experience due to skipped frames
            show_live: Whether to show live video tracking. Press the key 'q' to quit
            count_objects: count objects being tracked on screen
            verbose: print details on the screen allowed values 0,1,2
        """
        try:  # begin video capture
            vid = cv2.VideoCapture(int(video))
        except:
            vid = cv2.VideoCapture(video)

        if output:  # get video ready to save locally if flag is set
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))  # by default VideoCapture returns float instead of int
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(output, codec, fps, (width, height))
        else:
            out = None

        frame_num = 0
        while True:  # while video is running
            return_value, frame = vid.read()
            if not return_value:
                # print('Video has ended or failed!')
                break
            frame_num += 1
            if skip_frames and not frame_num % skip_frames:
                continue  # skip every nth frame. When every frame is not important, you can use this to fasten the
                # process
            if verbose >= 1:
                start_time = time.time()

            # -----------------------------------------PUT ANY DETECTION MODEL HERE ------------------------------------
            yolo_dets = self.detector.detect(frame.copy(), plot_bb=False)  # Get the detections
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # run deepsort
            self.tracker = self.run_deep_sort(frame, yolo_dets)
            count = self.total_objects

            # ---------------------------------------- DETECTION PART COMPLETED ---------------------------------------
            # cv2.putText(frame, "FPS: {}".format(round(fps, 2)), (5, 80), cv2.FONT_HERSHEY_COMPLEX_SMALL,
            #             1.5, (255, 0, 0), 2)
            if count_objects:
                cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            1.5, (255, 0, 0), 2)

            # ---------------------------------- DeepSORT tracker work starts here ------------------------------------
            for track in self.tracker.tracks:  # update new findings AKA tracks
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                class_name = track.get_class()

                # Calculate speed of object
                track = calcSpeed(track, bbox, frame_num, 30)

                # initialize color map
                cmap = plt.get_cmap('tab20b')
                colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
                color = colors[int(track.track_id) % len(colors)]  # draw bbox on screen
                color = [i * 255 for i in color]

                # draw bounding box of objects
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)

                # draw speed above bounding box of object
                text_header_bbox: str = class_name + ":" + str(track.track_id)
                if track.speed > 0:
                    print(f"{track.class_name} {track.track_id}: {track.speed} km/h")
                    text_header_bbox += "-" + str(round(track.speed, 1)) + "km/h"

                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 30)),
                              (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])), color,
                              -1)
                cv2.putText(frame, text_header_bbox, (int(bbox[0]), int(bbox[1] - 11)), 0, 0.6,
                            (255, 255, 255), 1, lineType=cv2.LINE_AA)

                if verbose == 2:
                    print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(
                        str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

            # -------------------------------- Tracker work ENDS here --------------------------------------------------
            if verbose >= 1:
                fps = 1.0 / (time.time() - start_time)  # calculate frames per second of running detections
                if not count_objects:
                    print(f"Processed frame no: {frame_num} || Current FPS: {round(fps, 2)}")
                else:
                    print(
                        f"Processed frame no: {frame_num} || Current FPS: {round(fps, 2)} || Objects tracked: {count}")

            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if output:
                out.write(result)  # save output video

            if show_live:
                cv2.imshow("Output Video", result)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()
