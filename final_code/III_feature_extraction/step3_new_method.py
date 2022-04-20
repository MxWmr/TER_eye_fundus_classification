import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from tqdm import tqdm
import pandas as pd
from scipy import ndimage
from scipy.stats import kde




Df=pd.read_csv('Partie III/glaucoma.csv',sep=',')
cdr=Df['ExpCDR'].values
glauc=Df['Glaucoma'].values
N=len(cdr)
cdrpos=[]
cdrneg=[]
for i in range(N):
    if glauc[i]==1:
        cdrpos.append(cdr[i])
    else:
        cdrneg.append(cdr[i])



# fct qui permet de ne garder que les contours extérieurs.

def exte(disc_edge):
    l_corde=[]
    for i in range(len(disc_edge)):
        start=False
        stop_j=0
        start_j=0
        for j in range(len(disc_edge[0])):
            if start==True and disc_edge[i,j]!=0:
                disc_edge[i,j]=0
                stop_j=j         
            if disc_edge[i,j]!=0 and start==False:
                disc_edge[i,j]=255
                start_j=j
                start=True
        if stop_j-start_j!=0:
            l_corde.append(stop_j-start_j)
        disc_edge[i,stop_j]=255

    return l_corde



"""i=13
optic_cup=np.load('segmented_dataset/train/glaucoma_negative/cup_segmented'+str(i)+'.npy', )
optic_disc=np.load('segmented_dataset/train/glaucoma_negative/disc_segmented'+str(i)+'.npy')
disc_edge = filters.roberts(optic_disc)
cup_edge= filters.roberts(optic_cup)
l_corde_disc=exte(disc_edge)
l_corde_cup=exte(cup_edge)
mymodel = np.poly1d(np.polyfit(range(len(l_corde_disc)),l_corde_disc, 6))
myline = np.linspace(1, len(l_corde_disc), 1000)
plt.figure(1)
plt.scatter(range(len(l_corde_disc)),l_corde_disc)
plt.plot(myline,mymodel(myline))
plt.show()
mymodel = np.poly1d(np.polyfit(range(len(l_corde_cup)),l_corde_cup, 6))
myline = np.linspace(1, len(l_corde_cup), 1000)
plt.figure(1)
plt.scatter(range(len(l_corde_cup)),l_corde_cup)
plt.plot(myline,mymodel(myline))
plt.show()"""


r_cup=[]
for i in tqdm(range(36,596)):
    try:
        optic_disc=np.load('segmented_dataset/train/glaucoma_positive/cup_segmented'+str(i)+'.npy')
        optic_disc=np.transpose(optic_disc)
        disc_edge = filters.roberts(optic_disc)
        #disc_edge = ndimage.binary_dilation(disc_edge).astype(np.float32)
        #disc_edge = ndimage.binary_erosion(disc_edge).astype(np.float32)
        l_corde_disc=exte(disc_edge)
        mymodel = np.poly1d(np.polyfit(range(len(l_corde_disc)),l_corde_disc, 6))
        myline = np.linspace(1, len(l_corde_disc), 1000)
        l=mymodel(myline)
        r_cup.append(max(l))
        #plt.plot(myline,mymodel(myline),linewidth=0.5)
    except:
        pass
print(len(r_cup))

r_disc=[]
for i in tqdm(range(36,596)):
    try:
        optic_disc=np.load('segmented_dataset/train/glaucoma_positive/disc_segmented'+str(i)+'.npy')
        optic_disc=np.transpose(optic_disc)
        disc_edge = filters.roberts(optic_disc)
        #disc_edge = ndimage.binary_dilation(disc_edge).astype(np.float32)
        #disc_edge = ndimage.binary_erosion(disc_edge).astype(np.float32)
        l_corde_disc=exte(disc_edge)
        mymodel = np.poly1d(np.polyfit(range(len(l_corde_disc)),l_corde_disc, 6))
        myline = np.linspace(1, len(l_corde_disc), 1000)
        l=mymodel(myline)
        r_disc.append(max(l))
        #plt.plot(myline,mymodel(myline),linewidth=0.5)
    except:
        pass

r_cupn=[]
for i in tqdm(range(500)):
    try:
        optic_disc=np.load('segmented_dataset/train/glaucoma_negative/cup_segmented'+str(i)+'.npy')
        optic_disc=np.transpose(optic_disc)
        disc_edge = filters.roberts(optic_disc)
        #disc_edge = ndimage.binary_dilation(disc_edge).astype(np.float32)
        #disc_edge = ndimage.binary_erosion(disc_edge).astype(np.float32)
        l_corde_disc=exte(disc_edge)
        mymodel = np.poly1d(np.polyfit(range(len(l_corde_disc)),l_corde_disc, 6))
        myline = np.linspace(1, len(l_corde_disc), 1000)
        l=mymodel(myline)
        r_cupn.append(max(l))
        #plt.plot(myline,mymodel(myline),linewidth=0.5)
    except:
        pass

r_discn=[]
for i in tqdm(range(500)):
    try:
        optic_disc=np.load('segmented_dataset/train/glaucoma_negative/disc_segmented'+str(i)+'.npy')
        optic_disc=np.transpose(optic_disc)
        disc_edge = filters.roberts(optic_disc)
        #disc_edge = ndimage.binary_dilation(disc_edge).astype(np.float32)
        #disc_edge = ndimage.binary_erosion(disc_edge).astype(np.float32)
        l_corde_disc=exte(disc_edge)
        mymodel = np.poly1d(np.polyfit(range(len(l_corde_disc)),l_corde_disc, 6))
        myline = np.linspace(1, len(l_corde_disc), 1000)
        l=mymodel(myline)
        r_discn.append(max(l))
        #plt.plot(myline,mymodel(myline),linewidth=0.5)
    except:
        pass


cdr=[]
fn=0
for i in range(len(r_cup)):
    cd=r_cup[i]/r_disc[i]
    cdr.append(cd)
    if cd<0.65:
        fn+=1
    
cdrn=[]
fp=0
for i in range(len(r_cupn)):
    cd=r_cupn[i]/r_discn[i]
    cdrn.append(cd)
    if cd>0.65:
        fp+=1

plt.figure(1)
plt.plot(cdrn,'b*',label='negatif')
plt.plot(cdr,'r*',label='positif')
plt.xlabel('image')
plt.ylabel('cdr')
plt.legend()
plt.grid()
plt.show()  

tot=len(cdr)+len(cdrn)
print(fp/len(cdrn),fn/len(cdr),1-(fp+fn)/tot)

density = kde.gaussian_kde(cdr)
density2 =  kde.gaussian_kde(cdrn)
x = np.linspace(0,1,300)
y=density(x)
y2=density2(x)
plt.plot(x,y,label='positive')
plt.plot(x,y2,'r',label='negative')
plt.legend()
plt.xlabel('cdr')
plt.grid()
plt.title("Density Plot of CDR")
plt.show()