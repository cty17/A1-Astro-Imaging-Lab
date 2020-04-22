# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 10:28:21 2020

@author: cty17
"""

from astropy.io import fits
import numpy as np
from skimage.io import imread
import os
import matplotlib.pyplot as plt
import cv2 as cv2
#cv2 is mainly for the package we used for the blob discovery 
import skimage as ski
from skimage.measure import label, regionprops
import scipy as sp
#%%
# set path
os.chdir(r"C:\Users\Alan Yung")
#%%
#for testing purposes only, using a cropped image with about 10 galaxies without blooming
# cropping image
#file='A1_mosaic.fits'
#data=fits.open(file)[0].data
#header=fits.open(file)[0].header

#from astropy.nddata import Cutout2D
#from astropy import units as u
#position = (1280,2305)
#size = (4180*u.pixel,2120*u.pixel) 
#cutout = Cutout2D(data, position, size,copy=True)
#plt.imshow(cutout.data, origin='lower')
#header=fits.open(file)[0].header
#fits.writeto('deletee.fits', cutout.data, header)
#data=cutout.data
#%%
# read file
#we will use the cropped smaller image for testing purposes 
file='a2crop.fits'
data=fits.open(file)[0].data
header=fits.open(file)[0].header
plt.figure('origi')
plt.imshow(data,origin='lower')
# reference magnitude
ref_mag=header['MAGZPT']
#%%
# find SINGLE max! therefore even if they have the same magnitude, only give the FIRST max! 
#we mask the first found one and it will find the next one!
#i.e. one at a time
def findmax(data):
    pos=list(np.unravel_index(np.argmax(data, axis=None), data.shape))
    mag=data[pos[0],pos[1]]
    return pos,mag
        
findmax(data)
#%%
# draw a rectangle around pos and calculate the background within
def local_noise(data,pos,length1,length2):
    # drawing the square
    left=int(pos[0]-length1/2)
    if left<0:                  # make sure it is inside the image
        left=0
    right=int(pos[0]+length1/2)
    if right<0:
        left=0
    down=int(pos[1]-length2/2)
    if down<0:
        left=0
    up=int(pos[1]+length2/2)
    if up<0:
        left=0
    
    fluxlist=[]
    poslist=[]
    for i in range(len(data)):
        if i >=left and i <=right:
            for j in range(len(data[i])):
                if j >= down and j <= up:
                    fluxlist.append(data[i][j])
                    poslist.append([i,j])
    
    plt.figure('hist')
    plotlist=[]
    for x in fluxlist:
        if x<=3450:         # use <3450 to fit the Gaussian
            plotlist.append(x)
    nbins=max(plotlist)-min(plotlist)+1         # one bin for one interger
    hist,edge=np.histogram(plotlist,bins=nbins)
#        print(hist)
    
    
    # fitting a Gaussian for noise    
    mean = np.mean(plotlist)      
                                                           
    pdf_x = np.linspace(min(plotlist),3450,100)
    pdf_y = 310000/np.sqrt(2*np.pi*np.var(plotlist))*np.exp(-0.5*(pdf_x-mean)**2/np.var(plotlist))      # change magnitude for the G to fit nicely
    
    # plotting the histogram with different(longer range)
    plotlist2=[]
    for x in fluxlist:
        if x<=3460:
            plotlist2.append(x)
    nbins=max(plotlist2)-min(plotlist2)+1
    plt.hist(plotlist2,nbins)     # ,log=True
    
    plt.plot(pdf_x,pdf_y,'k--')
    plt.xlim(min(fluxlist),3450)
    print('data length=',len(plotlist),'min',min(plotlist),'max=',max(plotlist),'mean=',mean,'std=',np.std(plotlist))
    
    plt.ylabel('Number count')
    plt.xlabel('Pixel value')

local_noise(data,[300,900],300,900)
#%%
# binarization
def binarize(image,mean,n,sigma):
    # n: threshold=how many sigma away from mean?
    #we have decided to use 3sigma in the end
    bi=np.zeros((len(image),len(image[0])))
    
    thresh=mean+n*sigma
    
    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j] >= thresh:
                bi[i][j]=1
            else:
                bi[i][j]=0
                
    return bi
            
#mask=binarize(data,3420,2,9.68)
#plt.figure('mask2')
#plt.imshow(mask)

# use two different threshold on the whole image & a defined area
def binarize2(image,mean1,mean2,n,sigma1,sigma2,xran,yran):
    # n: threshold=how many sigma away from mean?
    bi=np.zeros((len(image),len(image[0])))
    
    thresh1=mean1+n*sigma1
    thresh2=mean2+n*sigma2
    
    print(thresh1,thresh2)
    for i in range(len(image)):
        for j in range(len(image[i])):
            if i<xran[0] or i>xran[1] or j<yran[0] or j>yran[1]:  # inside the area
                if image[i][j] >= thresh1:
                    bi[i][j]=1
                else:
                    bi[i][j]=0
            else:
                if image[i][j] >= thresh2:
                    bi[i][j]=1
                else:
                    bi[i][j]=0
            
    return bi

            
mask=binarize2(data,3414,3425,3,9,9,[0,700],[300,1300])
#mask=binarize2(data,3414,3425,2,9,9,[2400,3600],[450,1800])
plt.figure('mask2')
plt.imshow(mask,origin='lower')
#%%
'''=========================blooming================================='''
# manually masking blooming
def mcircle(point,cen,r):
    val=(point[0]-cen[0])**2+(point[1]-cen[1])**2
    if val<r**2:
        return True
    else:
        return False
    
def rect(point):
    if point[1]>1200 and point[1]<1232:
        return True
    if point[1]>1165 and point[1]<1251:
        if point[0]>=0 and point[0]<53:
            return True
    if point[1]>888 and point[1]<1430:
        if point[0]>=209 and point[0]<219:
            return True
    if point[1]>798 and point[1]<1480:
        if point[0]>=99 and point[0]<111:
            return True
    if point[0]>110 and point[0]<245:
        if point[1]>1105 and point[1]<1333:
            return True
    if point[0]>2124 and point[0]<2142:  
        if point[1]>681 and point[1]<688:
            return True
    if point[0]>3169 and point[0]<3203:
        if point[1]>546 and point[1]<565:
            return True
    if point[0]>2984 and point[0]<3038:
        if point[1]>547 and point[1]<564:
            return True
    if point[0]>2483 and point[0]<2509:
        if point[1]>801 and point[1]<830:
            return True
    if point[0]>3150 and point[0]<3185:
        if point[1]>600 and point[1]<642:
            return True
            
def maskbloom(mask,cen,r):
    nmask=mask.copy()
    
    nmask[1213][0]=1    # mysterious pixel
    for i in range(len(nmask)):
        for j in range(len(nmask[i])):
            pixel=[i,j]
            if mcircle(pixel,cen,r)==True:
                nmask[i][j]=0
            if rect(pixel)==True:
                nmask[i][j]=0
    return nmask


nbmask=maskbloom(mask,[3000,1200],500)     # OPPOSITE: imshow position=[a,b], then need to use [b,a]
plt.figure('nobloom')
plt.imshow(nbmask,origin='lower')   
#%%
# masking small areas 
# convert to8 bits image before using label
from scipy.misc import bytescale
mask8 = bytescale(nbmask)
num_labels, img_label = cv2.connectedComponents(mask8,connectivity=8)       # connect components as a whole
print(num_labels)

def clear_noise(mask,img_label,mini):
    regions = regionprops(img_label)
    n=0
    for props in regions:
        if props.area<=mini:
            poslist=list(props.coords)
            for pos in poslist:
                mask[pos[0],pos[1]]=0
                n+=1

    return mask,n


cmask,n=clear_noise(nbmask,img_label,5)
plt.figure('clearmask')
plt.imshow(cmask,origin='lower')
plt.plot()
#%%
'''===========================create mask data=============================='''

pymask=np.zeros([len(cmask),len(cmask[0])])
for i in range(len(cmask)):
    for j in range(len(cmask[0])):
        if cmask[i][j]==1:
            pymask[i][j]=0
        else:
            pymask[i][j]=1          # opposite way: 1 means True in np mask

pymasked=np.ma.masked_array(data,pymask)
plt.figure('pymasked')
plt.imshow(pymasked,origin='lower')
#%%
# find the position of the pixel with maximum flux of the MASKED data
#with manual data rejection now, these discovered sources should be the true galaxies/stars, not blooming pixels 
def findmax(data):
    pos=list(np.unravel_index(np.argmax(data, axis=None), data.shape))
    mag=data[pos[0],pos[1]]
    return pos,mag
        
findmax(pymasked)   # the max point is different from before masking as expected
#%%
# calculate corresponding magnitude from flux
def cal_mag(count,zp,background,n):
    if count-n*background<=0:       # avoid infinity
        return 999
    m=zp-2.5*np.log10(count-n*background) 
    return m
#%%
'''===========================================aperture=========================================================='''
# verify if a point is inside a defined circle
def circle(point,cen,r):
    val=(point[0]-cen[0])**2+(point[1]-cen[1])**2
    if val<r**2:
        return True
    else:
        return False
        
# draw a circle aperture with specific size at a specific pixel
# sum up the flux of pixels inside the circle and mask them
# return the new masked array
def circle_aperture(masked,pixel,size,ran):
    
    nmasked=masked.copy()
    
    flux_list=[]
    pnum=0
    irange=np.linspace(pixel[0]-size,pixel[0]+size,size*2+1)
    jrange=np.linspace(pixel[1]-size,pixel[1]+size,size*2+1)
    for i in irange:
        if i <0:
            i=0
        elif i>(ran[0]):
            i=ran[0]
        for j in jrange:
            if j <0:
                j=0
            elif j>(ran[1]):
                j=ran[1]
            i=int(i)
            j=int(j)
            pos=[i,j]
            if circle(pos,pixel,size)==True:
                flux_list.append(nmasked.data[i][j])   # DO NOT USE np.ma.getdata, which gives 0 when mask=True. This gives the data value regardlessly
                nmasked[i][j]=np.ma.masked
                pnum+=1
    flux=sum(flux_list)
    return nmasked,flux,pnum

tmasked,tflux,pnum=circle_aperture(pymasked,findmax(pymasked)[0],50,[4180-1,2120-1])    # constraint: MUST BE size MINUS 1, otherwise out of range
#plt.figure('first step')
#plt.imshow(tmasked,origin='lower')
#%%
# manually masking some bright sources in the clusters. 
def aperture_manual(masked,poslist,rlist):
    nmasked=masked.copy()
    fluxlist=[]
    pnumlist=[]
    for i in range(len(poslist)):
        nmasked,flux,pnum=circle_aperture(nmasked,poslist[i],rlist[i],[4180-1,2120-1])
        fluxlist.append(flux)
        pnumlist.append(pnum)
        
    return nmasked,fluxlist,pnumlist


plist=[[451,2139],[578,2150],[490,2035],[978,2044],[1063,2128],[755,2200],[407,2348],[592,2450],[414,2561],[767,2553],[911,2481],[681,2663],[686,2069],[900,2572],[652,2992],[561,3097],[384,3083],[337,3116],[417,3138],[473,3134],[627,3200],[688,3170],[300,3327]]
rlist=[25,24,36,23,25,26,21,41,55,84,39,69,82.758,55,36,83,45,28,35,21,35,42,37]

poslist=[]
for p in plist:
    poslist.append([p[1],p[0]])

mmasked,mfluxlist,mpbumlist=aperture_manual(pymasked,poslist,rlist)
plt.figure('tt')
plt.imshow(mmasked.mask,origin='lower')
#%%
# main function
# variable aperture size
def aperture_more(masked,num,zp,background,ran,plot=True): 
    # masked: np masked array
    # num: number of source detection required
    # zp: reference magnitude
    # background: threshold (mean+n*std)
    # ran: range of whole img
    # plot: plotting aperture?
    
    nmasked=masked.copy()       # keep input array untouched
   
    star_mag_list=[]
    pos_list=[]
    size_list=[]
    
    n=0
    
    
    for n in range(num):
        now_max=findmax(nmasked)[0]
        
        
        if plot==True:
            plt.figure('afteraperture')
            plt.plot(now_max[1],now_max[0],'.',color='limegreen')
#            print(n,'-----------------------',now_max,findmax(nmasked)[1])
        n+=1
        
        length=[]
        
        i=now_max[0]
        j=now_max[1]
        pos_list.append(now_max)
        
        
        while np.ma.getmask(nmasked[i][j])==False:       
            if i==0:
                break           # if it reaches edge
            else:
                i-=1            # goes down
        length.append(abs(i-now_max[0]))        # find the difference from centre to edge
        
        i=now_max[0]
        while np.ma.getmask(nmasked[i][j])==False: 
            if i==ran[0]:
                break
            else:
                i+=1            # goes up
        length.append(abs(i-now_max[0]))        
        
        i=now_max[0]
        while np.ma.getmask(nmasked[i][j])==False:
            if j==0:
                break
            else:
                j-=1            # goes left
        length.append(abs(j-now_max[1])) 
                
        j=now_max[1]
        while np.ma.getmask(nmasked[i][j])==False:
            if j==ran[1]:
                break
            else:
                j+=1            # goes right
        length.append(abs(j-now_max[1])) 
        
#        print(length)
        
        # TBC：1.take max（too large） 2.take mean（large or small） 3.compare，ignore too large/small ones
        
        if np.std(length)>10:       # large std means detecting multiple objects
            length.remove(min(length))
            length.remove(max(length))
            size=int(np.mean(length)*1.1)       # *1.1: slightly larger?
        else:
            size=int(max(length)*1.1)   # *1.1: slightly larger?
#        print(size)
        
        if length.count(0)>1:
            print('all is found')
            break       
        
        size_list.append(size)
        
        nmasked,flux,pnum=circle_aperture(nmasked,now_max,size,ran)         
        
        star_mag_list.append(cal_mag(flux,zp,background,pnum))
        
        if plot==True:
            fig = plt.gcf()
            ax = fig.gca()
            circle1 = plt.Circle([now_max[1],now_max[0]], size, color='r',fill=False)
            ax.add_artist(circle1)
    
    return nmasked,star_mag_list,[pos_list,star_mag_list,size_list]


#tttmask,tttml,catalogue=aperture_more(pymasked,5000,ref_mag,3420,[4180-1,2120-1])
tttmask,tttml,catalogue=aperture_more(mmasked,15000,ref_mag,3425,[4180-1,2120-1],plot=False)
plt.figure('afteraperture')
plt.imshow(tttmask,origin='lower')
plt.imshow(cmask,origin='lower')
plt.show()
#%%
# for plotting
mfloor=[]
#plotting galaxy count as integer bins in magnitude
count=[]
for m in tttml:
    mfloor.append(np.floor(m))      # to do statistics
    
maglist=np.linspace(min(mfloor),max(mfloor),max(mfloor)-min(mfloor)+1)      # a list of magnitudes appeared
#print(maglist)

for m in maglist:
    num=mfloor.count(m)     # count number
    count.append(num)
    
print(len(mfloor))
#%%
# plotting
plt.figure('mag')
def expo(x,a,b):
    return a*np.exp(b*x)
popt,pcov=sp.optimize.curve_fit(expo,maglist[:-5],count[:-5])
xdata=np.linspace(min(maglist),16,100)
ydata=[expo(x,*popt) for x in xdata]
plt.plot(xdata,ydata,'--',label='fit')

#%%
plt.plot(maglist[:-5],count[:-5],'.',markersize=10,label='valid')
plt.plot(maglist[10:],count[10:],'.',markersize=5,color='red',label='discarded')
plt.yscale('log')
plt.xlabel('Corrected Integer Magnitude')
plt.ylabel('Log Number')
plt.errorbar(maglist[:-5],count[:-5],yerr=[np.sqrt(c) for c in count[:-5]],fmt='none')
plt.show()

plt.figure('mag')
plt.legend()
#%%
# calculating gradient
a,b=np.polyfit(maglist[:-5],[np.log(c) for c in count[:-5]],1)
print(a,b)
#%%
# export???
import xlwt
from tempfile import TemporaryFile
book = xlwt.Workbook()
sheet1 = book.add_sheet('sheet1')


for i,n in enumerate(['magnitude','count','x coord','y coord','magnitude','size']):
    sheet1.write(0,i,n)

for i,e in enumerate(maglist):
    sheet1.write(i+1,0,e)
for i,e in enumerate(count):
    sheet1.write(i+1,1,e)    
    
for i in range(len(catalogue[0])):
    sheet1.write(i+1,2,np.float64(catalogue[0][i][0]))
    sheet1.write(i+1,3,np.float64(catalogue[0][i][1]))
    sheet1.write(i+1,4,catalogue[1][i])
    sheet1.write(i+1,5,catalogue[2][i])


name = "full catalogue.xls"
book.save(name)
book.save(TemporaryFile())
#%%
catalogue[0][1]