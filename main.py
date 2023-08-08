from bridge_wrapper import DeepSORT
from detection_helpers import Detector

detector = Detector()
# pass the path to the trained weight file
detector.load_model('./weights/best.pt', trace=False)
detector.device = 0

# Initialise  class that binds detector and tracker in one class
tracker = DeepSORT(reID_model_path='./weights/deep_sort.pt', detector=detector)
tracker.track_video(video='./IO_data/input/video/1.mp4', show_live=True, count_objects=True, verbose=1)

