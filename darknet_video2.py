from ctypes import *
import random
import os
import cv2
import time
import darknet
import argparse
from threading import Thread, enumerate
from queue import Queue
from darknet_parser import parser
# import tello_live_video_3
from threading import Thread, enumerate
from queue import Queue
import sys
accList =[]
fpsList = []

def str2int(video_path):
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise(ValueError("Invalid video path {}".format(os.path.abspath(args.input))))


def set_saved_video(input_video, output_video, size):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video


def convert2relative(bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h  = bbox
    _height     = darknet_height
    _width      = darknet_width
    return x/_width, y/_height, w/_width, h/_height


def convert2original(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_x       = int(x * image_w)
    orig_y       = int(y * image_h)
    orig_width   = int(w * image_w)
    orig_height  = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted


def convert4cropping(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_left    = int((x - w / 2.) * image_w)
    orig_right   = int((x + w / 2.) * image_w)
    orig_top     = int((y - h / 2.) * image_h)
    orig_bottom  = int((y + h / 2.) * image_h)

    if (orig_left < 0): orig_left = 0
    if (orig_right > image_w - 1): orig_right = image_w - 1
    if (orig_top < 0): orig_top = 0
    if (orig_bottom > image_h - 1): orig_bottom = image_h - 1

    bbox_cropping = (orig_left, orig_top, orig_right, orig_bottom)

    return bbox_cropping


def video_capture(frame_queue, darknet_image_queue):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            #break
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height),
                                   interpolation=cv2.INTER_LINEAR)
        print("@#$%^^&*()","darknet_width=",darknet_width,"darknet_height=",darknet_height,"@#$%^^&*()")
        frame_queue.put(frame)
        #frame_queue.put(frame_resized)
        img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
        darknet_image_queue.put(img_for_detect)
        #print("Can't receive*")
    cap.release()


def inference(darknet_image_queue, detections_queue, fps_queue):
    while cap.isOpened():
        darknet_image = darknet_image_queue.get()
        prev_time = time.time()
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh)
        detections_queue.put(detections)
        fps = int(1/(time.time() - prev_time))
        fps_queue.put(fps)
        fpsList.append(fps)
        #print("FPS: {}".format(fps))
        if detections and isinstance(detections,list):
            dete_tuple =detections[0]
            print(dete_tuple)
            object_found = dete_tuple[0]
            acc = dete_tuple[1]
            accList.append(float(acc))
            print(object_found,"#$%^&",acc)
        # acc = darknet.print_detections(args.ext_output) 
        darknet.print_detections(detections, args.ext_output)
        darknet.free_image(darknet_image)
        if fpsList:
            print("fpslistmax=",max(fpsList))
            print("fpslistmin=",min(fpsList))
        if accList:

            print("accListMAX=",max(accList))
            print("accListMin=",min(accList))

    cap.release()


def drawing(frame_queue, detections_queue, fps_queue):
    random.seed(3)  # deterministic bbox colors
    video = set_saved_video(cap, args.out_filename, (darknet_width, darknet_height))
    print(args.out_filename)
    while cap.isOpened():
        frame = frame_queue.get()
        detections = detections_queue.get()
        fps = fps_queue.get()
        detections_adjusted = []
        if frame is not None:
            for label, confidence, bbox in detections:
                bbox_adjusted = convert2original(frame, bbox)
                detections_adjusted.append((str(label), confidence, bbox_adjusted))
            image = darknet.draw_boxes(detections_adjusted, frame, class_colors)
            if not args.dont_show:
                cv2.imshow('Inference', image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if args.out_filename is not None:
                video.write(image)
                cv2.imwrite("Detected_Target.jpeg",image)
                
                print (f"__So, FPS are__  {fps}")
            if cv2.waitKey(25) & 0xFF == ord('q'):
                time.sleep(1)
                #break
                cap.release()
                video.release()
                cv2.destroyAllWindows()
                break
                sys.exit("Values do match")  
                
    cap.release()
    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    frame_queue = Queue()
    darknet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)

    args = parser()
    check_arguments_errors(args)
    network, class_names, class_colors = darknet.load_network(
            args.config_file,
            args.data_file,
            args.weights,
            batch_size=1
        )
    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)
    input_path = str2int(args.input)

    # cap = cv2.VideoCapture(input_path) # stored video
    cap = cv2.VideoCapture(0)          # webcam video
    #cap.set(3, 1280)
    cap.set(4, 720)
########################live video from tello################################
   
    #from tello_live_video_3 import udp_video_address
    #cap = cv2.VideoCapture(udp_video_address)
    #if not cap.isOpened():
    #     cap.open(udp_video_address) 
   
############################################################################

    time.sleep(1)
    Thread(target=video_capture, args=(frame_queue, darknet_image_queue)).start()
     
    Thread(target=inference, args=(darknet_image_queue, detections_queue, fps_queue)).start()

    Thread(target=drawing, args=(frame_queue, detections_queue, fps_queue)).start()
 
