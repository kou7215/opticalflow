import chainer
from chainer import serializers, Variable, cuda
from flownets import FlowNetS
import cv2
import numpy as np
import argparse

### parameter ###
INPUT_FILE1 = 'samples/0000000-imgL.ppm'
INPUT_FILE2 = 'samples/0000000-imgR.ppm'
OUTPUT_FILE = './results/test'
ARROW_FREQ = 16

def preprocessing(img):
    img = img.astype('f')
    img = img / 255.0
    img = img.transpose((2, 0, 1))
    return img

def Padding(img1,img2): 
    assert (img1.shape == img2.shape), 'Not equal img1.shape & img2.shape'
    height,width = img1.shape[0], img1.shape[1]
    if height >= width: 
        pad = int((height-width)/2)
        img1 = cv2.copyMakeBorder(img1,0,0,pad,pad,cv2.BORDER_CONSTANT,value=0)
        img2 = cv2.copyMakeBorder(img2,0,0,pad,pad,cv2.BORDER_CONSTANT,value=0)
    elif height <= width:
        pad = int((width-height)/2)
        img1 = cv2.copyMakeBorder(img1,pad,pad,0,0,cv2.BORDER_CONSTANT,value=0)
        img2 = cv2.copyMakeBorder(img2,pad,pad,0,0,cv2.BORDER_CONSTANT,value=0)
        return img1, img2
        

def Liner_interpolation(a,b,c,d,dx,dy):
    pix_e = (b-a)*dx - a
    pix_f = (d-c)*dx - c
    pix_g = (pix_f-pix_e)*dy - pix_f
    return pix_g

def main():
    parser = argparse.ArgumentParser(
        description='Test FlownetS')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument("--load_model", '-m', default='flownets.npz', help='load model')
    parser.add_argument("--method", default='dnn', help='cv2 or dnn')
    args = parser.parse_args()
    ### FlowNet (DNN)  ###   
    if args.method == 'dnn':
        if args.gpu >= 0:
            chainer.cuda.get_device(0).use()
        f = FlowNetS()
        serializers.load_npz('flownets.npz', f)
    
        if args.gpu >=0:
            f.to_gpu()
    
        row_img1 = cv2.imread(INPUT_FILE1)
        row_img2 = cv2.imread(INPUT_FILE2)
        # Padding
        row_img1, row_img2 = Padding(row_img1,row_img2)
        row_img1 = cv2.resize(row_img1, (512,512), cv2.INTER_AREA)
        row_img2 = cv2.resize(row_img2, (512,512), cv2.INTER_AREA)
        img_arrow = row_img1.copy()
        height,width,ch = row_img1.shape
        img1 = preprocessing(row_img1)
        img2 = preprocessing(row_img2)
        xin = np.zeros((1, 6, 512, 512), dtype=np.float32)
        xin[0, 0:3, :] = img1
        xin[0, 3:6, :] = img2
        if args.gpu>=0:
            xin = cuda.to_gpu(xin)
    
        res = f(Variable(xin)).data
    
        if args.gpu>=0:
            res = cuda.to_cpu(res)
        img=np.zeros((128,128,3))
        img[:,:,0]=res[0, 0] + 128
        img[:,:,2]=res[0, 1] + 128
        img=img.astype(np.uint8)
        cv2.imwrite('samples/out.jpg', img)

        # flownet
        delta_x = res[0,0]    # (128,128)
        delta_y = res[0,1]    # (128,128)
        delta_x = cv2.resize(delta_x, (height,width))
        delta_y = cv2.resize(delta_y, (height,width))
        img_trans = np.zeros_like(row_img1)
        for x in range(width):
            for y in range(height):
                current_dx = delta_x[x,y]
                current_dy = delta_y[x,y]
                if (np.floor(x+current_dx)>=0)\
                and(np.floor(x+current_dx)+1<width)\
                and(np.floor(y+current_dy)>=0)\
                and(np.floor(y+current_dy+1)<height):
                    # wander if row_img1 or row_img2?
                    pix_a = row_img1[int(np.floor(x+current_dx)),   int(np.floor(y+current_dy)),:]
                    pix_b = row_img1[int(np.floor(x+current_dx+1)), int(np.floor(y+current_dy)),:]
                    pix_c = row_img1[int(np.floor(x+current_dx)),   int(np.floor(y+current_dy+1)),:]
                    pix_d = row_img1[int(np.floor(x+current_dx+1)), int(np.floor(y+current_dy+1)),:]
                    pix_g = Liner_interpolation(pix_a,pix_b,pix_c,pix_d,current_dx,current_dy)
                    img_trans[x,y,:] = pix_g
                    # arraw vector
                    if (x % ARROW_FREQ == 0) and (y % ARROW_FREQ == 0):
                        cv2.arrowedLine(img_arrow,(x,y),(int(np.floor(x+current_dx)),int(np.floor(y+current_dy))), (0,255,0), thickness=1, tipLength=0.05)
                else:
                    img_trans[x,y,:] = 0
                    # arraw vector
                    if (x % ARROW_FREQ == 0) and (ARROW_FREQ % 8 == 0):
                        cv2.arrowedLine(img_arrow,(x,y),(int(np.floor(x+current_dx)),int(np.floor(y+current_dy))), (0,255,0), thickness=1, tipLength=0.05)
        
        # error map            
        img_diff = abs(row_img1 - img_trans)
        cv2.imwrite(OUTPUT_FILE + '_img_diff_dnn.jpg', img_diff)
        cv2.imwrite(OUTPUT_FILE + '_img_trans_dnn.jpg', img_trans)
        cv2.imwrite(OUTPUT_FILE + '_img_vector_dnn.jpg', img_arrow)

    ### Dense optical flow (opencv) ###
    if args.method == 'cv2':
        img1_rgb = cv2.imread(INPUT_FILE1)
        img2_rgb = cv2.imread(INPUT_FILE2)
        img1_gray= img1_rgb.copy()
        img2_gray= img2_rgb.copy()
        img1_gray= cv2.cvtColor(img1_gray,cv2.COLOR_BGR2GRAY)
        img2_gray= cv2.cvtColor(img2_gray,cv2.COLOR_BGR2GRAY)
        
        img1_rgb, img2_rgb = Padding(img1_rgb, img2_rgb)
        img1_gray, img2_gray = Padding(img1_gray, img2_gray)
        img1_rgb = cv2.resize(img1_rgb, (512,512), cv2.INTER_AREA)
        img2_rgb = cv2.resize(img2_rgb, (512,512), cv2.INTER_AREA)
        img1_gray = cv2.resize(img1_gray, (512,512), cv2.INTER_AREA)
        img2_gray = cv2.resize(img2_gray, (512,512), cv2.INTER_AREA)
        flow = cv2.calcOpticalFlowFarneback(img1_gray,img2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0) # (512,512,2)
        img_arrow = img1_rgb.copy()
        delta_x, delta_y = flow[:,:,0], flow[:,:,1]
        #delta_y, delta_x = flow[:,:,0], flow[:,:,1]
        img_trans = np.zeros_like(img1_rgb)
        height,width,ch= img1_rgb.shape
        # NOTE i don't know which is correct, (x,y) or (y,x) to plot vector map
        for x in range(width):
            for y in range(height):
                current_dy = delta_x[x,y]
                current_dx = delta_y[x,y]
                if (np.floor(x+current_dx)>=0)\
                and(np.floor(x+current_dx)+1<width)\
                and(np.floor(y+current_dy)>=0)\
                and(np.floor(y+current_dy+1)<height):
                    # wander if row_img1 or row_img2?
                    pix_a = img1_rgb[int(np.floor(x+current_dx)),   int(np.floor(y+current_dy)),:]
                    pix_b = img1_rgb[int(np.floor(x+current_dx+1)), int(np.floor(y+current_dy)),:]
                    pix_c = img1_rgb[int(np.floor(x+current_dx)),   int(np.floor(y+current_dy+1)),:]
                    pix_d = img1_rgb[int(np.floor(x+current_dx+1)), int(np.floor(y+current_dy+1)),:]
                    pix_g = Liner_interpolation(pix_a,pix_b,pix_c,pix_d,current_dx,current_dy)
                    img_trans[x,y,:] = pix_g
                    # arraw vector
                    if (x % ARROW_FREQ == 0) and (y % ARROW_FREQ == 0):
                        #cv2.arrowedLine(img_arrow,(x,y),(int(np.floor(x+current_dx)),int(np.floor(y+current_dy))), (0,255,0), thickness=1, tipLength=0.05)
                        cv2.arrowedLine(img_arrow,(y,x),(int(np.floor(y+current_dy)),int(np.floor(x+current_dx))), (0,255,0), thickness=1, tipLength=0.05)
                else:
                    img_trans[x,y,:] = 0
                    # arraw vector
                    if (x % ARROW_FREQ == 0) and (ARROW_FREQ % 8 == 0):
                        #cv2.arrowedLine(img_arrow,(x,y),(int(np.floor(x+current_dx)),int(np.floor(y+current_dy))), (0,255,0), thickness=1, tipLength=0.05)
                        cv2.arrowedLine(img_arrow,(y,x),(int(np.floor(y+current_dy)),int(np.floor(x+current_dx))), (0,255,0), thickness=1, tipLength=0.05)
        
        # error map            
        img_diff = abs(img1_rgb - img_trans)
        cv2.imwrite(OUTPUT_FILE + '_img_diff_cv2.jpg', img_diff)
        cv2.imwrite(OUTPUT_FILE + '_img_trans_cv2.jpg', img_trans)
        cv2.imwrite(OUTPUT_FILE + '_img_vector_cv2.jpg', img_arrow)

if __name__ == '__main__':
    main()
