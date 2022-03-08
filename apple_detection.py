from imageai.Detection.Custom import CustomObjectDetection
#from keras.layers import LayerNormalization
import os

execution_path = os.getcwd()

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(detection_model_path="detection_model-ex-028--loss-8.723.h5")
detector.setJsonPath(configuration_json="detection_config.json")
detector.loadModel()

detections = detector.detectObjectsFromImage(input_image="test_apple_image5.jpeg", minimum_percentage_probability=60, output_image_path="new-test_apple_image5.jpeg")

for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
