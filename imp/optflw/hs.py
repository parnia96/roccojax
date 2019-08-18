
# -*- coding: utf-8 -*-


# ---------------------------------
# OPTICAL FLOW HORN-SCHUNCK METHOD
# ---------------------------------



from lib import *


def App(stem):
	
	flist, ext = getimgfiles(stem)
	for i in range(len(flist)-1):
		try:
			fn1 = str(stem) +'.'+ str(i) + ext
			m1 = cv2.imread(fn1, 0)
			m1 = cv2.resize(m1,None,fx=0.5,fy=0.5) # 0.5 means 50% of image to be scalling
			m1 = smoothImage(m1, FILTER)

			fn2 = str(stem) + '.' + str(i+1) + ext
			m2 = cv2.imread(fn2, 0)
			m2 = cv2.resize(m2,None,fx=0.5,fy=0.5) # 0.5 means 50% of image to be scalling
			m2 = smoothImage(m2, FILTER)

			[U, V] = HornSchunck(m1, m2, 1, 100)
			
			# each pixel of our u and v vector is something 
			# between 0 and 1 , we multiply each one of them by 255 to get
			# a range of 0 to 255 pixels  			
			U = U * 255
			V = V * 255
			
			# change the type of u and v 
			# cause we don't have a pixel lile 124.5 ;-)	 			
			U = U.astype(int)
			V = V.astype(int)
			
			# plot u and v for each image which their 
			# pixel range is something between 0 to 255 			
			plottyGraph(U, V, m2)
		
		except KeyboardInterrupt:
		    sys.exit(0)


if __name__ == '__main__':
	
	App("pics/shoes")
