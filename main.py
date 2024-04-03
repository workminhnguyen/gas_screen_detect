from func import YOLOv8_gas_screen_predict
import argparse
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type = str, default = 'images/2.jpg', help = "image path")
    parser.add_argument('--actual', type = int, help = 'number of detection')
    # parser.add_argument('--method', default=0.5, type=float, help='method')
    args = parser.parse_args()
    if args.actual is None: 
        print("The actual number of screens has not been provided.")
        print("Please try again.")
        sys.exit()

    if args.actual < 0: 
        print("The number of screens cannot be negative.")
        print("Please try again.")
        sys.exit()
    
    YOLOv8_gas_screen_predict(input = args.imgpath, act = args.actual)    
        