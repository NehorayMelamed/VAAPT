import cv2
img = cv2.imread('/home/dudy/Nehoray/SHABACK_POC_NEW/data/photos/netivey_israel/Screenshot from 2023-02-23 10-21-15.png') # load a dummy image
while(1):
    cv2.imshow('img',img)
    k = cv2.waitKey(33)
    if k==27:    # Esc key to stop
        break
    elif k==-1:  # normally -1 returned,so don't print it
        continue
    else:
        print (k )# else print its value