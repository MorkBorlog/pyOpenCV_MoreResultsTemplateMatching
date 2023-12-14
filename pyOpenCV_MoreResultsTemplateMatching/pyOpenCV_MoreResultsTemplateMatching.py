# Quelle: https://stackoverflow.com/questions/50579050/template-matching-with-multiple-objects-in-opencv-python/58514954#58514954

import cv2
from cv2.gapi import convertTo
import numpy as np
from matplotlib import pyplot as plt
import time

image = cv2.imread(r'Smiley.png', cv2.IMREAD_COLOR )
template = cv2.imread(r'Smiley-Auge.png', cv2.IMREAD_COLOR)

# Es ginge auch [::-1]
h, w = template.shape[:2]

method = cv2.TM_CCOEFF_NORMED

# Genauigkeit
threshold = 0.90 # 90%

start_time = time.time()
print("Start Zeit: " +  str(start_time) + "\n")

res = cv2.matchTemplate(image, template, method)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# fake out max_val for first run through loop
max_val = 1 # 1 = 100%

while max_val > threshold:
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if max_val > threshold:
        res[max_loc[1]-h//2:max_loc[1]+h//2+1, max_loc[0]-w//2:max_loc[0]+w//2+1] = 0   
        image = cv2.rectangle(image,(max_loc[0],max_loc[1]), (max_loc[0]+w+1, max_loc[1]+h+1), (255,0,0) )

end_time = time.time() - start_time

print("Zeit benoetigt: %.2f Sekunden\n" % end_time)

#cv2.imwrite('output.png', image)
plt.imshow(image)
plt.show()