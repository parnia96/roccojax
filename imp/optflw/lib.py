
import cv2, sys, numpy as np
from pathlib import Path
from matplotlib import pyplot as plt


QUIVER = 5
FILTER = 1

def getimgfiles(stem):
	stem = Path(stem).expanduser()
	path = stem.parent
	name = stem.name
	exts = ['.ppm','.bmp','.png','.jpg']
	for ext in exts:
		pat = name + '.*' + ext
		print('[*] searching {}/{}'.format(path, pat))
		flist = sorted(path.glob(pat))
		if flist:
			break

	if not flist:
		raise FileNotFoundError('[-] no files found under {} with {}'.format(stem,exts))

	print('[+] analyzing {} files {}.*{}'.format(len(flist), stem, ext))

	return flist, ext 



def HornSchunck(im1, im2, landa, ite):
	uInitial = np.zeros([im1.shape[0], # row 
						 im1.shape[1]] # col
						)
	vInitial = np.zeros([im1.shape[0], # row 
						 im1.shape[1]] # col
						)

	# cause our imag is a 255 value matrix
	# we'll make our u and v vector range 
	# something between 0 and 1 	
	u = uInitial / 255
	v = vInitial / 255

	[fx, fy, ft] = computeDerivatives(im1, im2)

	# cause our imag is a 255 value matrix
	# we'll make our f matrices range 
	# something between 0 and 1 
	fx = fx / 255
	fy = fy / 255
	ft = ft / 255

	# plotting each fx , fy and ft 	
	fg, ax = plt.subplots(1, 3, figsize=(18,5))
	for f, a, t in zip((fx, fy, ft), ax, ('$f_x$','$f_y$','$f_t$')):
		h = a.imshow(f, cmap = 'bwr')
		a.set_title(t)
		fg.colorbar(h, ax = a)


	# We use an odd kernel size to ensure there 
	# is a valid integer (x, y)-coordinate at the center of the image

	mask = np.array([[1/6, 1/3, 1/6],
					  [1/3,    0, 1/3],
					  [1/6, 1/3, 1/6]], float)

	for i in range(ite):
		# cause our imag is a 255 value matrix
		# we'll make our u and v vector range 
		# something between 0 and 1 
		uAvg = cv2.filter2D(u,-1, mask) / 255
		vAvg = cv2.filter2D(v,-1, mask) / 255
		derv = (fx*uAvg + fy*vAvg + ft) / (landa**2 + fx**2 + fy**2)
		u = uAvg - fx * derv
		v = vAvg - fy * derv

	return u,v




def computeDerivatives(im1, im2):
	# our result img pixel range conved with our mask should be
	# something between 0 to 255 and for that reason we multiply 
	# our 2x2 mask by 0.25	
	maskX = np.matrix([[-1,1],[-1,1]])*.25
	maskY = np.matrix([[-1,-1],[1,1]])*.25
	maskT = np.ones([2,2])*.25
	# because of adding two conv will raise a pixel range exception error
	# here we are dividing the result for each f by 2 to make the result
	# less than or equal to 255  	
	fx = (cv2.filter2D(im1, -1, maskX) + cv2.filter2D(im2, -1, maskX))/2
	fy = (cv2.filter2D(im1, -1, maskY) + cv2.filter2D(im2, -1, maskY))/2
	ft = (cv2.filter2D(im1, -1, maskT) + cv2.filter2D(im2, -1, -maskT))/2

	return (fx,fy,ft)




def smoothImage(img, gaussSigma):
	"""
	HS scheme assumes that the velocity field varies 
	globally smoothly and neighboring pixels
	have almost the same velocity
	"""
	G = gaussFilter(gaussSigma)
	smoothedImage = cv2.filter2D(img, -1, G)
	smoothedImage = cv2.filter2D(smoothedImage, -1, G.T)
	return smoothedImage



def plottyGraph(u, v, Inew, scale=3):
	"""
	plotting u and v vector on each image
	"""
	ax = plt.figure().gca()
	ax.imshow(Inew, cmap = 'gray')
	for i in range(0, len(u), 3):
		for j in range(0, len(v), 3):
			ax.arrow(j,i, v[i,j]*scale, u[i,j]*scale, color='red')
	plt.draw()
	plt.pause(0.01)



def gaussFilter(segma):
	"""
	G(x) = 1/2*pi*segma^2 * (e^-x^2/2*segma^2)
	x : is the distance from the origin in the horizontal axis,
	σ : is the standard deviation of the Gaussian distribution
	"""
	kSize = 2*(segma*3)
	x = range(int(-kSize/2), int(kSize/2), int(1+1/kSize))
	x = np.array(x)
	G = (1/(2*np.pi)**.5*segma) * np.exp(-x**2/(2*segma**2))
	return G
  

