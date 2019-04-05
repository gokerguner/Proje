import cv2
import os
import glob
img_dir = "" # Enter Directory of all images
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
n = 0
for f1 in files:
    print(n, f1)
    n = n + 1
    if(not os.path.isfile("/mnt/data/data/summer_2018/resized_target_files/" + f1)):
        img = cv2.imread(f1)
    #print(img)
        img = cv2.resize(img,(256,256))
    #1print(img)
    #data.append(img)
        try:
            cv2.imwrite( "/mnt/data/data/summer_2018/resized_target_files/" + f1, img );
        except:
            print('error', f1)
~                                                                                      
