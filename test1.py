import cv2

img_path = "C:/Users/DELL/Desktop/test.png"
original_img = cv2.imread(img_path)
new_img = cv2.resize(original_img, (640,640))
cv2.imshow("original", original_img)
cv2.imshow("new", new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()