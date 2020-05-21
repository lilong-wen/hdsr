import cv2
import os

root1 = "./assets/cvl_dataset/train/"
root2 = "./assets/cvl_dataset/train3/"

for item in os.listdir(root1):
    image_array = cv2.imread(root1 + item, 0)
    blur = cv2.GaussianBlur(image_array,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #image_array = cv2.copyMakeBorder(th3, 20, 3, 20, 5, cv2.BORDER_CONSTANT, value=[255,255,255])
    image = cv2.cvtColor(th3, cv2.COLOR_GRAY2RGB)

    cv2.imwrite(root2 + item, image)
