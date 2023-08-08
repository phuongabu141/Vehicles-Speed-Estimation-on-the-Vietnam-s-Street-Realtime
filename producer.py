import sys
import time
import cv2
from time import sleep
from kafka import KafkaProducer


class KafkaVideoStreaming():
    def __init__(self, bootstrap_servers, topic, videoFile: str, client_id, batch_size=65536, frq=0.001):
        self.videoFile = videoFile
        self.topicKey = videoFile
        self.topic = topic
        self.batch_size = batch_size
        self.client_id = client_id
        self.bootstrap_servers = bootstrap_servers
        self.frq = frq

    def setProducer(self):
        self.producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            api_version=(0, 10, 1),
            client_id=self.client_id,
            acks=1,
            value_serializer=None,
            key_serializer=str.encode,
            batch_size=self.batch_size,
            compression_type='gzip',
            linger_ms=0,
            buffer_memory=67108864,
            max_request_size=1048576,
            max_in_flight_requests_per_connection=1,
            retries=1,
        )

    def report_callback(self, record_metadata):
        print("Topic Record Metadata: ", record_metadata.topic)
        print("Parition Record Metadata: ", record_metadata.partition)
        print("Offset Record Metatada: ", record_metadata.offset)

    def err_callback(self, excp):
        print('Errback', excp)

    def publishFrames(self, payload):
        self.producer.send(
            topic=self.topic, key=self.topicKey, value=payload
        ).add_callback(
            self.report_callback
        ).add_errback(
            self.err_callback
        )

    def run(self):
        try:
            print("Opening file %s" % self.videoFile)
            __VIDEO_FILE = cv2.VideoCapture(self.videoFile)
        except:
            raise

        self.setProducer()

        print(
            "Publishing: %{v}\n\
            \tBatch Size: {b},\n\
            \tSleep ({t}) \n\
            \tTarget Topic: {t} \n\
            \tHost: {h}".format(
                v=self.topicKey,
                b=self.batch_size,
                t=self.topic,
                h=self.bootstrap_servers
            )
        )

        self.keep_processing = True

        try:
            while (__VIDEO_FILE.isOpened()) and self.keep_processing:
                readStat, frame = __VIDEO_FILE.read()

                if not readStat:
                    self.keep_processing = False

                if frame is not None:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    self.publishFrames(buffer.tostring())

                sleep(self.frq)

            if self.keep_processing:
                print('Finished processing video %s' % self.topicKey)
            else:
                print("Error while reading %s" % self.topicKey)

            __VIDEO_FILE.release()
        except KeyboardInterrupt:
            __VIDEO_FILE.release()
            print("Keyboard interrupt was detected. Exiting...")


topic = "KafkaVideoStream"


def publish_video(video_file):
    producer = KafkaProducer(bootstrap_servers='localhost:9092')
    video = cv2.VideoCapture(video_file)
    print('Publishing video...')

    frame_num = 0
    while (video.isOpened()):
        success, frame = video.read()

        print(f"Sending frame {frame_num}...")
        if not success:
            print("Error while reading video!")
            break

        ret, buffer = cv2.imencode('.jpg', frame)
        producer.send(topic, buffer.tobytes())

        frame_num += 1
        # time.sleep(0.01)

    video.release()
    print('publish complete')


if __name__ == '__main__':
    video_path = 'video/1.mp4'
    publish_video(video_path)
    
    # if (len(sys.argv) > 1):
    #     video_path = sys.argv[1]

    #     publish_video(video_path)

        # video_stream = KafkaVideoStreaming(
        #     bootstrap_servers='localhost:9092',
        #     topic='KafkaVideoStream',
        #     videoFile=video_path,
        #     client_id='KafkaVideoStreamClient',
        # )
        # video_stream.run()
