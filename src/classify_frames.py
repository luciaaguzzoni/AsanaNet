import cv2
import numpy as np
import time as t
import matplotlib.pyplot as plt


import load_images as im

def get_positions_from_video(path, model, img_size, legend_labels):
    frames = []
    guessed_position = []

    if path == 0:
        #webcam
        pass

    else:
        video_cap = cv2.VideoCapture(path) 
        tmp = ''
        i=1
        start_time = t.time()

        while True:
                # Read a frame from the video source
                ret, frame = video_cap.read()
                end_time = t.time()

                if end_time - start_time > 0.7:    
                    #ret, frame = cap.read()
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
                    img = im.resize_without_squeezing(img,img_size)
                    img_np = np.array([img])/255  
                    frames.append(img_np)
                
                    # Your image processing and classification code goes here
                    label = np.argmax(model.predict(img_np), axis=1)[0]
                    probability = model.predict_proba(img_np)
                    prob = round(max(probability[0]), 2)
                    guessed_position.append(legend_labels[label])
                    try:
                        new_pose = legend_labels[label]
                        if new_pose!=tmp:
                            tmp=new_pose  
                            plt.imshow(img)
                            plt.savefig(f"{i} - {new_pose} - p{prob}")
                            i+=1
                    except KeyError:
                        print(f"Key Error: {label[0]}")

                    start_time = t.time()
                
                # Break the loop if 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    return frames, guessed_position
