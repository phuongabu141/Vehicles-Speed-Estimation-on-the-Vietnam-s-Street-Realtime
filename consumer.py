import json
import random
import time
from pathlib import Path
from queue import Queue
from threading import Thread
from time import sleep
import matplotlib.pyplot as plt
from flask import Flask, render_template, make_response, request, jsonify
from flask import Response
from kafka import KafkaConsumer
from bridge_wrapper import DeepSORT
from calc_speed import calcSpeed
from detection_helpers import *


class KafkaVideoView():
    def __init__(self, bootstrap_servers, topic, client_id, group_id, poll=500, frq=0.01):
        self.topic = topic
        self.client_id = client_id
        self.group_id = group_id
        self.bootstrap_servers = bootstrap_servers
        self.poll = poll
        self.frq = frq
        self.frame_num = 0


    def setConsumer(self):
        self.consumer = KafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers.split(','),
            fetch_max_bytes=52428800,
            fetch_max_wait_ms=1000,
            fetch_min_bytes=1,
            max_partition_fetch_bytes=1048576,
            value_deserializer=None,
            key_deserializer=None,
            max_in_flight_requests_per_connection=10,
            client_id=self.client_id,
            group_id=self.group_id,
            auto_offset_reset='earliest',
            max_poll_records=self.poll,
            max_poll_interval_ms=300000,
            heartbeat_interval_ms=3000,
            session_timeout_ms=10000,
            enable_auto_commit=True,
            auto_commit_interval_ms=5000,
            reconnect_backoff_ms=50,
            reconnect_backoff_max_ms=500,
            request_timeout_ms=305000,
            receive_buffer_bytes=32768,
        )

    def playStream(self, queue):
        while self.keepPlaying:
            try:
                msg = queue.get(block=True, timeout=20)
                self.queue_status = True
            except:
                print("WARN: Timed out waiting for queue. Retrying...")
                self.queue_status = False

            if self.queue_status:
                self.frame_num += 1
                print(f"Processing frame {self.frame_num}...")
                nparr = np.frombuffer(msg, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                isSuccess, result_frame = self.detector_and_tracker.detect_and_tracking(frame, self.frame_num)
                cv2.imshow('Streaming Video', result_frame if isSuccess else frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.keepConsuming = False
                    cv2.destroyAllWindows()
                    break

                sleep(self.frq)

    def run(self):
        self.keepPlaying = True
        self.setConsumer()
        self.videoQueue = Queue()
        self.keepConsuming = True

        self.playerThread = Thread(target=self.playStream, args=(self.videoQueue,), daemon=False)
        self.playerThread.start()

        try:
            Path("./detected_frame").mkdir(parents=True, exist_ok=True)
            while self.keepConsuming:
                payload = self.consumer.poll(self.poll)
                for bucket in payload:
                    for msg in payload[bucket]:
                        self.videoQueue.put(msg.value)

        except KeyboardInterrupt:
            self.keepConsuming = False
            self.keepPlaying = False
            cv2.destroyAllWindows()
            print("WARN: Keyboard Interrupt detected. Exiting...")

        self.playerThread.join()


frame_num = 0
detector = Detector()
detector.load_model('./weights/best.pt', trace=False)
detector.device = 0
tracker = DeepSORT(reID_model_path='./weights/deep_sort.pt', detector=detector)

id_objects = {'car': [], 'van': [], 'bus': [], 'truck': []}


def get_video_stream(detector, tracker):
    global frame_num
    global id_objects
    global count_truck, count_bus, count_van, count_car
    global labels_line, values_line_car, values_line_van, values_line_bus, values_line_truck
    for message in consumer:
        nparr = np.frombuffer(message.value, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        yolo_dets = detector.detect(frame.copy(), plot_bb=False)  # Get the detections
        tracker.run_deep_sort(frame, yolo_dets)

        car, van, bus, truck = 0, 0, 0, 0
        for track in tracker.tracker.tracks:  # update new findings AKA tracks
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()
            if class_name == 'car': car += 1
            if class_name == 'van': van += 1
            if class_name == 'bus': bus += 1
            if class_name == 'truck': truck += 1

            if track.track_id not in id_objects[class_name]:
                id_objects[class_name].append(track.track_id)

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
                # print(f"{track.class_name} {track.track_id}: {track.speed} km/h")
                text_header_bbox += "-" + str(round(track.speed, 1)) + "km/h"

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 30)),
                          (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])), color,
                          -1)
            cv2.putText(frame, text_header_bbox, (int(bbox[0]), int(bbox[1] - 11)), 0, 0.6,
                        (255, 255, 255), 1, lineType=cv2.LINE_AA)

        frame_num += 1
        count_car = len(id_objects['car'])
        count_van = len(id_objects['van'])
        count_bus = len(id_objects['bus'])
        count_truck = len(id_objects['truck'])

        if frame_num % 30 == 0:
            labels_line.append(str(frame_num))
            values_line_car.append(car)
            values_line_van.append(van)
            values_line_bus.append(bus)
            values_line_truck.append(truck)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + buffer.tobytes() + b'\r\n\r\n')
        # ret, buffer = cv2.imencode('.jpg', frame)
        # yield (b'--frame\r\n'
        #        b'Content-Type: image/jpg\r\n\r\n' + buffer.tobytes() + b'\r\n\r\n')


topic = "KafkaVideoStream"
consumer = KafkaConsumer(
    topic,
    bootstrap_servers=['localhost:9092'])

app = Flask(__name__)

count_car = len(id_objects['car'])
count_van = len(id_objects['van'])
count_bus = len(id_objects['bus'])
count_truck = len(id_objects['truck'])

labels_line = ['0']
values_line_car = [0]
values_line_van = [1]
values_line_bus = [2]
values_line_truck = [5]


@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        conf_thres = request.data
        detector.conf_thres = int(conf_thres) / 100
        print("Confidence threshold", conf_thres)
    return render_template('index1.html',
                           count_car=count_car,
                           count_van=count_van,
                           count_bus=count_bus,
                           count_truck=count_truck,
                           labels_line=labels_line[-10:],
                           values_line_car=values_line_car[-10:],
                           values_line_van=values_line_van[-10:],
                           values_line_bus=values_line_bus[-10:],
                           values_line_truck=values_line_truck[-10:])


@app.route('/refreshData')
def refresh_graph_data():
    global count_car, count_van, count_bus, count_truck
    global labels_line, values_line_car, values_line_van, values_line_bus, values_line_truck
    return jsonify(count_car=count_car,
                   count_van=count_van,
                   count_bus=count_bus,
                   count_truck=count_truck,
                   labels_line=labels_line,
                   values_line_car=values_line_car,
                   values_line_van=values_line_van,
                   values_line_bus=values_line_bus,
                   values_line_truck=values_line_truck)


@app.route('/data', methods=["GET", "POST"])
def data():
    data = [time.time() * 1000, random.random() * 100]
    data1 = [time.time() * 1000, random.random() * 100]
    data2 = [time.time() * 1000, random.random() * 100]
    data3 = [time.time() * 1000, random.random() * 100]
    response = make_response(json.dumps([data, data1, data2, data3]))
    response.content_type = 'application/json'
    return response


@app.route('/slider_update', methods=['POST', 'GET'])
def slider():
    received_data = request.data
    print(received_data)
    return received_data


@app.route('/video_feed', methods=['GET'])
def video_feed():
    return Response(
        get_video_stream(detector, tracker),
        mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
