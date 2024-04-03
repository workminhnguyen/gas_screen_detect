import cv2
from ultralytics import YOLO
import os

model = YOLO("models/best.pt")

def check_input(input):
    if not isinstance(input, str):
        raise TypeError("Đầu vào phải là một đường dẫn")
    
def check_image(input):
    image = cv2.imread(input)
    if image is None: return False, None
    return True, image

def draw_box(result, x, y, w, h, text):

    top_right_x = x + (w / 2)
    top_right_y = y - (h / 2)
    top_right_x = int(top_right_x.item())
    top_right_y = int(top_right_y.item())

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2

    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    text_bottom_left_x = top_right_x - text_width
    text_bottom_left_y = top_right_y + text_height
    rect_top_left = (text_bottom_left_x, top_right_y)
    rect_bottom_right = (top_right_x, text_bottom_left_y)

    cv2.rectangle(result, rect_top_left, rect_bottom_right, (0, 255, 0), cv2.FILLED)
    cv2.putText(result, text, (text_bottom_left_x, text_bottom_left_y), font, font_scale, (255, 255, 255), thickness)

    return result

def save_image(results: list, input: str , num, save_folder = "SAVE_IMAGES"):
    os.makedirs(save_folder, exist_ok = True)
    basename = os.path.basename(input)
    name, ext = os.path.splitext(basename)
    save_path = os.path.join(save_folder, name) + "_result" + ext
    result = results[0].plot(conf = False)
    
    if num == 1:
        x, y, w, h = results[0].boxes.xywh[0]
        draw_box(result, x, y, w, h, "center")

    elif num == 2:
        x1, y1, w1, h1 = results[0].boxes.xywh[0]
        x2, y2, w2, h2 = results[0].boxes.xywh[1]
        label1, label2 = ("left", "right") if x1 < x2 else ("right", "left")
        draw_box(result, x1, y1, w1, h1, label1)
        draw_box(result, x2, y2, w2, h2, label2)
        
    cv2.imwrite(save_path, result)
    directory = os.path.join(os.getcwd(), save_path)
    print("The image has been saved in the directory:", directory)

def result():
    pass

def YOLOv8_gas_screen_predict(input: str, act: int, cond: int = 0):
    check_input(input)
    success, image = check_image(input)
    if not success: 
        print("Image unreadable, please check path again")
        return
    if act == 0:
        print("Image without any screen")
        return
    results = model(source = image, iou = 0, conf = 0.5)
    num_detect = len(results[0])
    
    if num_detect == 0:
        print("No detections, reality differs")
        print("Not passed")
        return

    if num_detect != act:
        print("Detections don't match actual count.")
        print("Not passed")
    
    if act == num_detect:
        print("Passed")
        save_image(results = results, input = input, num = num_detect)
