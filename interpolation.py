import numpy as np
import cv2

### parameter ###
INPUT_FILE1 = 'img1.jpg'
INPUT_FILE2 = 'img2.jpg'
OUTPUT_FILE = './hoge'

def Liner_interpolation(a,b,c,d,dx,dy):
    pix_e = (b-a)*dx - a
    pix_f = (d-c)*dx - c
    pix_g = (pix_f-pix_e)*dy - pix_f
    return pix_g

img1 = cv2.imread(INPUT_FILE1)
img2 = cv2.imread(INPUT_FILE2)
img_arrow = img1.copy()

# padding
width,height,ch = img1.shape
# resize
# flownet

delta_x = res[;,;,0]    # (128,128)
delta_y = res[;,;,1]    # (128,128)
delta_x = cv2.resize(delta_x, (height,width))
delta_y = cv2.resize(delta_y, (height,width))
img_trans = np.zeros((x,y,ch))
for x in range(width):
    for y in range(height):
        current_dx = delta_x[x]
        current_dy = delta_y[y]
        if (int(np.floor(x+delta_x))>=0)
        and(int(np.floor(x+delta_x)+1)<width)
        and(int(np.floor(y+delta_y))>=0)
        and(int(np.floor(y+delta_y+1))<height):
            pix_a = img2[int(np.floor(x+current_dx)),   int(np.floor(y+current_dy))]
            pix_b = img2[int(np.floor(x+current_dx+1)), int(np.floor(y+current_dy))]
            pix_c = img2[int(np.floor(x+current_dx)),   int(np.floor(y+current_dy+1))]
            pix_d = img2[int(np.floor(x+current_dx+1)), int(np.floor(y+current_dy+1))]
            pix_g = Liner_interpolation(pix_a,pix_b,pix_c,pix_d,current_dx,current_dy)
            img_trans[x,y] = pix_g
            # arraw vector
            cv2.arrowedLine(img_arrow,(x,y),(x+current_dx,y+current_dy), (0,255,0), thickness=1, tipLength=0.05)
        else:
            img_trans[x,y] = 0
            # arraw vector
            cv2.arrowedLine(img_arrow,(x,y),(x+current_dx,y+current_dy), (0,255,0), thickness=1, tipLength=0.05)

# error map            
img_diff = abs(img1 - img_trans)
cv2.imwrite(OUTPUT_FILE + '_img_diff.jpg', img_diff)
cv2.imwrite(OUTPUT_FILE + '_img_trans.jpg', img_trans)
