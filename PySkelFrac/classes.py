from scipy import misc
from skimage import io
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pylab as pl
import time
import PySkelFrac.Miscfunc as M
from scipy.signal import argrelextrema
import os
import shutil
import cv2
from pathlib import Path
from skimage import measure

### LIBRAIRIES

"""
Contains all the Objects we are using to describe a gorgonian
_Image
_Contours, containing all Contour
_Arcs    , containing all Arc
_Places  , containing all Place
_Voies   , containing all Voie
"""

class Void:
    '''
    for empty class if needed (lambda : none also work)
    '''
    def __init__(self): pass
    def __repr__(self): return(M.dirP(self))


class Image:
    """
    Object containing all the image information, and 2D Maps of properties

    X,Y size of the image

    Maps from init :
    _binary     # Thresholded image
    _original   # Original Image (GBR)
    _dist       # Distance to closest border
    _enveloppe  # Filled Holes
    _envdist    # Distance to exterior contours
    """

    def __init__(self,p):
        print('\n### IMAGE EXTRACTION ### ### ### ### ### ###')
        # Original image Load
        print(str( p['workfold'] /  ( p['image']+'.'+p['imageformat'])))
        print('process...',end='')
        t=time.time()

        self.original= cv2.imread(str( p['workfold'] /  ( p['image']+'.'+p['imageformat'])))
        GRAY = cv2.blur(cv2.cvtColor(self.original,cv2.COLOR_BGR2GRAY),(5,5))
        ret,self.binary=cv2.threshold(GRAY,5,255,cv2.THRESH_BINARY)
        self.X,self.Y=np.shape(self.binary)[:2]

        ### REDUCE BORDER SIZE
        #Dims=M.reduceIMGsize(self.binary)
        #self.binary=self.binary[Dims[2]:Dims[3],Dims[0]:Dims[1]]
        #print('Removed ',self.X-(Dims[3]-Dims[2]), 'pixels in X,',self.Y-(Dims[1]-Dims[0]), 'pixel in y')
        self.X,self.Y=np.shape(self.binary)[:2]

        ### ENVELOPPE filling
        im_floodfill = np.copy(self.binary)
        h, w = self.binary.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(im_floodfill, mask, (0,0), 255);
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        self.enveloppe = self.binary | im_floodfill_inv
        self.envdist    = ndimage.distance_transform_edt( self.enveloppe    ).astype(np.int16)

        ### ROTATION IF NECESSARY
        """
        if self.X<self.Y :
            print('Rotation of image for verticality')
            self.binary      =self.binary     .T
            self.enveloppe   =self.enveloppe  .T
            self.envdist     =self.envdist    .T
            self.X,self.Y=self.Y,self.X
        """

        ### INVERSION IF NECESSARY
        self.dist    = ndimage.distance_transform_edt( self.binary    ).astype(np.int16)
        foot =divmod(np.argmax(self.dist),self.Y)
        """
        if foot[0]<self.X/2 :
            print('Miror on image (foot at the bottom)')
            self.binary      =np.flipud(self.binary   )
            self.dist        =np.flipud(self.dist     )
            self.enveloppe   =np.flipud(self.enveloppe)
            self.envdist     =np.flipud(self.envdist  )
        """
        print('Image Loaded. Time :',time.time()-t,'s')
        print('Porosity :', np.sum(self.binary/np.amax(self.binary))/np.sum(self.enveloppe/np.amax(self.enveloppe)))
        print('Size :',str(self.X)+'*'+str(self.Y))

    def __repr__(self):
        return( str(self.X)+'*'+str(self.Y)+' image')

####################################################################################################################
####################################################################################################################
class Contour:
    """
    For each hole in the image, its properties

    OBJECTS
    _XY         : all the coordinates of their points
    _X          : same with only X
    _Y          : same with only Y
    _curvature  : curvature calculated without smoothing
    _perimeter  : Perimeter
    _surface    : Surface
    _isexterior : True only for the enveloppe
    _artefact   : Too small hole

    METHOD
    _None
    """
    def __init__(self,p,XY):
        """Extract all classical properties"""

        self.len       =len(XY)
        self.XY        =XY/p.get('resize',1)
        self.X         =XY[:,0]/p.get('resize',1)
        self.Y         =XY[:,1]/p.get('resize',1)
        self.Center    =[np.mean(self.X),np.mean(self.Y)]
        self.curvature=M.curvature(XY)
        self.perimeter =np.sum(np.sqrt( ( self.X-np.roll(self.X,1) )**2 +
                                        ( self.Y-np.roll(self.Y,1) )**2 ))
        self.surface   =0.5*np.abs(np.dot(self.X,np.roll(self.Y,1))-
                                   np.dot(self.Y,np.roll(self.X,1)))
        if self.surface < p.get('typicallength',10)**2: self.artefact  =True
        else: self.artefact = False

        self.isexterior=False
        self.Arcs=   []
        self.Voies=  []
        self.Points= []
        self.ArcsInside=[]

    def __repr__(self):
        return('ext :'+str( self.isexterior)+' | surface :'+str(int(self.surface))+' | perimeter :'+str(int(self.perimeter)) )
class Contours:
    """
    Get all the contours from the image, to get all the holes

    OBJECTS :
    _list : all the contours and their informations
    _XY   : all the points of all contours, their label and their splitted label
    _labmax : the enveloppe of the structure

    Method :
    _Split contours : add contours
    """
    def __init__(self,IMG,p):
        print('\n### CONTOURS EXTRACTION ### ### ### ### ### ###')
        t=time.time()
        # Get from the parameters the interesting values
        treshold=p.get('treshold',0.5)

        if p.get('resize',1)!=1: print('Downscaling of contours with a factor of ',p['resize'])
        print('Find Contours...',end='')

        # Get all the points (rough information)
        contours = measure.find_contours( cv2.resize(IMG.binary, (0,0), fx=p.get('resize',1), fy=p.get('resize',1)  ), treshold)

        # Give all Contour its information
        self.ncontours=len(contours)
        self.list=self.ncontours*[None]

        Npts=0 #Number of points taken
        for i in range(self.ncontours):
            self.list[i]=Contour(p,contours[i])
            Npts+=int(len(contours[i]))

        self.XY=np.zeros((Npts,4))
        Npt=0
        for i in range(self.ncontours):
            l=len(contours[i])
            self.XY[Npt:Npt+l,0]=contours[i][:,0]/p.get('resize',1)
            self.XY[Npt:Npt+l,1]=contours[i][:,1]/p.get('resize',1)
            self.XY[Npt:Npt+l,2]=i
            self.XY[Npt:Npt+l,3]=i
            Npt+=int(l)

        self.Surface   = np.zeros(self.ncontours)
        for i in range(self.ncontours):
            self.Surface   [i]=self.list[i].surface

        # Find the biggest one
        self.labmax=np.argmax(self.Surface)
        self.list[self.labmax].isexterior=True
        print('Done. t=',time.time()-t)
        print('Created', len(self.list), 'Contours')

    def SplitContours(self,p):
        t=time.time()
        print('Split Contours...',end='')
        Cdiff=p.get('Cdiff',0.9)
        NCont=self.XY[-1,2]
        j=0
        for C in self.list:
            OK=False
            if (C.isexterior and p.get('Splitexterior',False)): OK=True
            elif p.get('Splitall',False): OK=True

            if OK:
                for k in range(p.get('Ndiff',100)): C.curvature=Cdiff*C.curvature+(0.5-Cdiff/2.)*(np.roll(C.curvature,1)+np.roll(C.curvature,-1))

                maxs=argrelextrema(C.curvature, np.less)[0]                           #Position of all the extrenums that interest us

                index=0                                                     #Index in max
                stay=len(maxs)                                                      #0 when we can leave the contours
                NCont+=1
                index2=0
                while stay :                                                     #While we stay in the contours
                    if j==len(self.XY[:,0]):break
                    self.XY[j,3]=NCont                                          #We give a new label
                    if maxs[index]==index2:                                        #If we find an extrenum
                        NCont+=1                                                #Next label will be +1
                        index+=1                                                #We load the next extrenum
                    if (index==len(maxs)-1):                                   #When we reach the last maximum
                        stay=0                                                  #We stop
                    j+=1                                                        #Else we go to the next point
                    index2+=1
            NCont+=len(C.X)
        print('Done. t=',time.time()-t)

    def __repr__(self):
        return(str(len(self.list))+' contours ,'+str(self.labmax)+' exterieur')

####################################################################################################################
####################################################################################################################
class Arcs:
    """
    Get all the portions of skeleton existing with a Voronoi Algorithm on the holes, and with a lot of clean-up
    """
    def __repr__(self):
        return str(self.Narcs)+' Arcs in list'

    def __init__(self,IMG,AllContours,p):
        print('\n### ARCS EXTRACTION ### ### ### ### ### ###')
        self.Vertices, Arclist ,self.Segments= M.Voronoi_Clean(AllContours.XY,IMG,p)

        self.RoughList=Arclist
        self.Arclist= M.Arcs_nettoyage(Arclist)
        self.Narcs=len(self.Arclist)
        self.list=self.Narcs*[None]
        for i in range(self.Narcs):
            self.list[i]=Arc(self.Vertices,self.Arclist[i],IMG,i,p)

    def ArcsAsVoies(self,IMG,AllArcs,AllVoies,p):
        for Arc in self.list: Arc.XYasVoies(IMG,AllArcs,AllVoies,p)
class Arc :
    '''
    All the Arcs with all their properties
    '''
    def __repr__(self):
        return 'Arc :'+str(self.index)+' | InaPlace :'+str(self.IsInaPlace)+' | length :'+str(int(self.lengthBubble))+' | XY :'+str(self.XY[:,0])+'| links :'+self.FirstLink+' '+self.LastLink+'| Voie :'+str(self.Usedinvoie)

    def __init__(self,XY,Arc,IMG,index,p):
        self.Vertices       = Arc
        self.XY             = XY[self.Vertices][:,0:2]
        self.Connectivitypts= XY[self.Vertices][:,  2]
        self.lengthBubble   = np.sum(np.sqrt( ( self.XY[:-1,0]-self.XY[1 :,0])**2 +
                                              ( self.XY[:-1,1]-self.XY[1 :,1])**2 ))
        self.lengthBird     =        np.sqrt( ( self.XY[  0,0]-self.XY[ -1,0])**2 +
                                              ( self.XY[  0,1]-self.XY[ -1,1])**2 )

        self.First          = Arc[0]
        self.FirstXY        = XY[self.First,0:2]
        self.FirstSize      = IMG.dist[int(self.FirstXY[0]),int(self.FirstXY[1])]

        self.Last           = Arc[-1]
        self.LastXY         = XY[self.Last,0:2]
        self.LastSize       = IMG.dist[int(self. LastXY[0]),int(self. LastXY[1])]

        self.FirstReferee   = self.XY[np.argmin(np.abs( np.sqrt( (self.XY[:,0]-self.FirstXY[0] )**2 +
                                                                 (self.XY[:,1]-self.FirstXY[1] )**2 ) - p.get('refereedist',1.1)*self.FirstSize))]
        self.LastReferee    = self.XY[np.argmin(np.abs( np.sqrt( (self.XY[:,0]- self.LastXY[0] )**2 +
                                                                 (self.XY[:,1]- self.LastXY[1] )**2 ) - p.get('refereedist',1.1)*self.LastSize ))]
        self.ScalarCorrect = False

        if p.get('ExtendedPlaces',False): self.IsInaPlace     = (self.FirstSize+self.LastSize) > self.lengthBird
        else :                            self.IsInaPlace     = np.amax((self.FirstSize,self.LastSize)) > self.lengthBird
        self.Contours=[]
        self.index=index
        if not self.IsInaPlace :
            self.RemovePointsInPlace(IMG)
            self.XYNoPlace =XY[self.VerticesNoPlace,0:2]
            self.Usedinvoie = False
            self.FirstLink=['Extremity']
            self.LastLink =['Extremity']

            self.R=IMG.dist[self.XYNoPlace[:,0].astype(np.int32),self.XYNoPlace[:,1].astype(np.int32)]
            self.Rmin = np.amin(self.R)
            self.Rmean= np.mean(self.R)
            self.Rmax = np.amax(self.R)
            self.Rstd = np.std (self.R)
        else :
            self.VerticesNoPlace=[None]
            self.XYNoPlace =[None,None]
            self.Usedinvoie=True
            self.FirstLink=["No Links"]
            self.LastLink= ["No Links"]
            self.R=     IMG.dist[self.XY[:,0].astype(np.int32),self.XY[:,1].astype(np.int32)]
            self.Rmean= np.mean(self.R)
            self.Rmin=  np.amin(self.R)
            self.Rmax = np.amax(self.R)
            self.Rstd = np.std (self.R)

    def ChangeInaPlace(self,IMG):
        self.VerticesNoPlace=[None]
        self.XYNoPlace =[None,None]
        self.Usedinvoie=True
        self.FirstLink=["No Links"]
        self.LastLink= ["No Links"]
        self.R=     IMG.dist[self.XY[:,0].astype(np.int32),self.XY[:,1].astype(np.int32)]
        self.Rmean= np.amin(self.R)
        self.Rmin=  np.mean(self.R)
        self.Rmax = np.amax(self.R)
        self.Rstd = np.std (self.R)

    def RemovePointsInPlace(self,IMG):
        ArcC=[]
        ### Calculation of distance to places
        Dist=np.zeros((len(self.XY),2))
        KeptRDist= np.zeros(len(self.XY))
        Dist[:,0]=np.sqrt((self.XY[:,0]-self.FirstXY[0])**2+
                          (self.XY[:,1]-self.FirstXY[1])**2)/self.FirstSize
        Dist[:,1]=np.sqrt((self.XY[:,0]-self.LastXY[0] )**2+
                          (self.XY[:,1]-self.LastXY[1] )**2)/self.LastSize
        for i in range(len(self.XY)): KeptRDist[i]=np.amin((Dist[i,0],Dist[i,1]))

        # Two cases :
        # 1) Some point are out of both extremities influence
        if np.amax(KeptRDist)>1:
            for i in range(len(self.XY)):
                if KeptRDist[i]> 1:   ArcC.append(self.Vertices[i])
        # 2) All the points are in one of the extremity influence
        else:            ArcC=[self.Vertices[np.argmax(KeptRDist)]]
        self.VerticesNoPlace=ArcC

        if self.Connectivitypts[0]== 1:
            self.FirstPlace     = []
            self.FirstLink      = ['Extremity']

        if self.Connectivitypts[-1]== 1:
            self.LastPlace      = []
            self.LastLink       = ['Extremity']

    def XYasVoies(self,IMG,AllArcs,AllVoies,p):
        """
        Add the extremities of arcs (as corrected in voies)
        """
        self.XYasVoies=np.zeros((len(self.XYNoPlace)+2,2))
        self.XYasVoies[1:-1,:]=self.XYNoPlace
        if not self.IsInaPlace:
            ### Premier point
                ### Si c'est une extremité
            if self.FirstLink == ['Extremity']:
                if AllVoies.list[self.Usedinvoie].Arc[0]==self.index:
                    self.XYasVoies[0,:] = AllVoies.list[self.Usedinvoie].XY[0,:]
                else:
                    self.XYasVoies[0,:] = AllVoies.list[self.Usedinvoie].XY[-1,:]

                ### Si c'est un lien
            else:
                if (self.FirstLink[1]=='first' or self.FirstLink[1]=='First'):
                    self.XYasVoies[0,:]=( self.XYNoPlace[0,:] + AllArcs.list[self.FirstLink[0]].XYNoPlace[ 0,:] )/2
                else:
                    self.XYasVoies[0,:]=( self.XYNoPlace[0,:] + AllArcs.list[self.FirstLink[0]].XYNoPlace[-1,:] )/2
            ### Dernier point
                ### Si c'est une extremité
            if self.LastLink == ['Extremity']:
                if AllVoies.list[self.Usedinvoie].Arc[0]==self.index:
                    if    (np.sqrt( ( AllVoies.list[self.Usedinvoie].XY[-1,0]-self.XYasVoies[-2,0]  )**2 +
                                    ( AllVoies.list[self.Usedinvoie].XY[-1,1]-self.XYasVoies[-2,1]  )**2 )
                        <  np.sqrt( ( AllVoies.list[self.Usedinvoie].XY[0,0] -self.XYasVoies[-2,0]  )**2 +
                                    ( AllVoies.list[self.Usedinvoie].XY[0,1] -self.XYasVoies[-2,1]  )**2 ) ):
                        self.XYasVoies[-1,:] = AllVoies.list[self.Usedinvoie].XY[-1,:]
                    else :
                        self.XYasVoies[-1,:] = AllVoies.list[self.Usedinvoie].XY[ 0,:]
                else:
                    self.XYasVoies[-1,:] = AllVoies.list[self.Usedinvoie].XY[-1,:]

                ### Si c'est un lien
            else:
                if (self.LastLink[1]=='first' or self.LastLink[1]=='First'):
                    self.XYasVoies[-1,:]=(self.XYNoPlace[-1,:]+AllArcs.list[self.LastLink[0]].XYNoPlace[ 0,:])/2
                else:
                    self.XYasVoies[-1,:]=(self.XYNoPlace[-1,:]+AllArcs.list[self.LastLink[0]].XYNoPlace[-1,:])/2

            self.XYasVoies=M.removeuselesspoints(self.XYasVoies,IMG.X/100)
            self.Angle= np.arctan2(np.abs(self.XYasVoies[-1,1]-
                                          self.XYasVoies[0 ,1] ),
                                   np.abs(self.XYasVoies[-1,0]-
                                          self.XYasVoies[0 ,0]))

###################################################################################################################
####################################################################################################################
class Places :
    def __repr__(self):
        return str(len(self.list))+' Places'

    def __init__(self,IMG,Arcs,p):
        t=time.time()
        print('\n### PLACES EXTRACTION ### ### ### ### ### ###')
        print('Creation of Places and centers...',end='')
        PlacesArcs,PlacesCenters = M.PreparePlaces(Arcs,p)
        ## Now we have every places with : their Arcs, and their centers
        self.list=len(PlacesArcs)*[None]
        for i in range(len(PlacesArcs)):
            self.list[i]=Place()

        # We create our object
        for i,P in enumerate(PlacesCenters.keys()):
            self.list[i].Centers  = PlacesCenters[P]
            self.list[i].Arcs     = PlacesArcs   [P]

        # We inform every arc to what place is it linked to (temporary label)
        for i,P in enumerate(self.list):
            # We take all arcs
            for ArcInfo in P.Arcs :
                if ArcInfo[1]=='first': Arcs.list[ArcInfo[0]].FirstPlace=i
                if ArcInfo[1]=='last' : Arcs.list[ArcInfo[0]]. LastPlace=i

        # We remove the arc inside a place from the dictionnary
        for i,Arc in enumerate(Arcs.list) :
            if Arc.FirstPlace == Arc.LastPlace :
                self.list[Arc.FirstPlace].ArcsInside.append(i)
                self.list[Arc.FirstPlace].Arcs.remove([i,'first'])
                self.list[Arc.FirstPlace].Arcs.remove([i,'last' ])

        #We reactualize the labels of the Arcs FirstPlace and LastPlace
        for i,ArcsInfo in enumerate(PlacesArcs.values()):
            # We take all arcs
            for ArcInfo in ArcsInfo :
                if ArcInfo[1]=='first': Arcs.list[ArcInfo[0]].FirstPlace=i
                if ArcInfo[1]=='last' : Arcs.list[ArcInfo[0]]. LastPlace=i

        for i,P in enumerate(self.list):
            self.list[i].Radius=np.zeros(len(P.Centers))
            self.list[i].XY=np.zeros((len(P.Centers),2),dtype=np.uint32)
            for j,V in enumerate(P.Centers):
                self.list[i].XY[j,0]  = Arcs.Vertices[V,0]
                self.list[i].XY[j,1]  = Arcs.Vertices[V,1]
                self.list[i].Radius[j]=IMG.dist[self.list[i].XY[j,0],self.list[i].XY[j,1]]
            P.ModifiedPlace = False
        print('Done. t=',time.time()-t)
        print('We have',len(self.list),'big places')

    def AddExtremities(self,AllArcs,AllVoies,p):
        for index,P in enumerate(self.list) :
            Allsegments = np.zeros((len(P.Links),4))
            for i in range(len(P.Links)):
                if P.Links[i][2]=='first' : Allsegments[i,0:2] = AllArcs.list[P.Links[i][0]].XYNoPlace[ 0,:]
                else                      : Allsegments[i,0:2] = AllArcs.list[P.Links[i][0]].XYNoPlace[-1,:]
                if P.Links[i][3]=='first' : Allsegments[i,2:4] = AllArcs.list[P.Links[i][1]].XYNoPlace[ 0,:]
                else                      : Allsegments[i,2:4] = AllArcs.list[P.Links[i][1]].XYNoPlace[-1,:]
            #############################################################

            if p.get('debugExtremities',False):
                plt.figure('test',figsize=(10,10))
                ax=plt.gca()
                for j in range(len(P.Centers)):
                    plt.plot(P.XY[j,0],P.XY[j,1],'*',c='k')
                    Pol=mpl.patches.Circle((P.XY[j,0],P.XY[j,1]),P.Radius[j]  ,color='k',fill=False  )
                    ax.add_patch(Pol)
                for i in range(len(P.Links)):
                    plt.plot([Allsegments[i,0],Allsegments[i,2]],[Allsegments[i,1],Allsegments[i,3]],'k')
                for Arcs in P.Arcs:
                    plt.plot(AllArcs.list[Arcs[0]].XYNoPlace[:,0],AllArcs.list[Arcs[0]].XYNoPlace[:,1],'*',c='b')
            ############################################################
            #### REMPLIR ALLSEGMENTS
            if len(P.Links)>0:
                for E in P.Extremities :
                    if not AllArcs.list[E[0]].IsInaPlace :
                        V=AllVoies.list[AllArcs.list[E[0]].Usedinvoie]
                        #M.dirP(AllArcs.list[E[0]])
                        if (E[1]=='first' or E[1]=='First'):
                            Pente= 0

                            Pt   = AllArcs.list[E[0]].XYNoPlace[ 0,:]
                        else :
                            Pente= 0
                            Pt   = AllArcs.list[E[0]].XYNoPlace[-1,:]
                        Dist        = np.zeros(len(P.Links))
                        CoordsInter = np.zeros((len(P.Links),2))
                        for i in range(len(P.Links)):
                            S=Allsegments[i,:]
                            CoordsInter[i,:] = (S[0:2]+S[2:4])/2
                            Dist[i] = np.sqrt((CoordsInter[i,0]-Pt[0])**2 + (CoordsInter[i,1]-Pt[1])**2)

                        kept=np.argmin(Dist)

                        D1=np.sqrt((CoordsInter[i,0]-V.XY[ 0,0])**2 + (CoordsInter[i,1]-V.XY[ 0,1])**2)
                        D2=np.sqrt((CoordsInter[i,0]-V.XY[-1,0])**2 + (CoordsInter[i,1]-V.XY[-1,1])**2)
                        if  min(D1,D2)<300:
                            if E[0]==V.Arc[0]: V.FirstExtremity = True
                            else :             V.LastExtremity  = True
                            if D1<D2 :
                                V.XY=np.vstack((CoordsInter[kept,:],V.XY))
                                if p.get('debugExtremities',False): plt.plot(V.XY[0:2,0],V.XY[:2,1],'r')
                            else     :
                                V.XY=np.vstack((V.XY,CoordsInter[kept,:]))
                                if p.get('debugExtremities',False): plt.plot(V.XY[-2:,0],V.XY[-2:,1],'r')

                        if p.get('debugExtremities',False):
                            print('E',E)
                            print('Pente',Pente)
                            print('Pt',Pt)
                            print('Dist',Dist)
                            print('CoordsInter',CoordsInter)
                            print('V',V)
                            print('kept',kept)
                            print('\n\n\n')
                    if p.get('debugExtremities',False):print(Allsegments)
                    if p.get('debugExtremities',False):plt.show()
            else:
                if len(P.Extremities)>1:
                    pass
                    #print(index,'Place with no link !')
class Place :
    def __init__(self):
        self.Centers=[]
        self.Radius=[]
        self.Arcs   =[]
        self.ArcsInside = []
        self.Potentiallink = []
        self.Potentialscore = []
        self.Links= []
        self.Extremities = []
        self.XY=[]
        self.Voies=[]
        self.Modified= False

    def __repr__(self):
        return str(len(self.Centers))+" Centers |"+str(len(self.Arcs))+" Arcs |"+str(self.Voies)+" Voie"

####################################################################################################################
####################################################################################################################
class Voies :
    def __repr__(self):
        return str(len(self.list))+" Voies"

    def __init__(self,AllArcs,p):
        """
        ActualArc : Arc Explored, and written in NewVoie and Lecture [ArcNumber,'ElementWebeganwith']
        NextArc   : Arc which has not been explored yet              [ArcNumber,'ElementWebeganwith']
        """
        self.list=[]
        indexvoie=0
        t=time.time()
        print('\n### WAYS CREATION ### ### ### ### ###')
        print('Construction of ways...',end='')
        for i,Arc in enumerate(AllArcs.list):
###############################################################################
            if p.get('debugVoies',False):                                   ###
                print('############### Arc number',i,'################')     ##
                print('InaPlace :',Arc.IsInaPlace)                            #
                print('AlreadyUsed :',Arc.Usedinvoie)
                print('Extremities First :', Arc.FirstLink)
                print('Extremities Last  :', Arc.LastLink,'\n')
                if Arc.IsInaPlace   :
                    print('\n We skip it, its in a place')
                elif Arc.Usedinvoie : print('\n We skip it, alreadyused')
###############################################################################
            Explore=False

### IF The Arc is acceptable for a new link
            if not (Arc.IsInaPlace or Arc.Usedinvoie) :
    ### IF it's alone, we add it and go to the other one
                if (Arc.FirstLink[0]=='Extremity' and Arc.LastLink[0]=='Extremity'):
###############################################################################
                    if p.get('debugVoies',False):                           ###
                        print('\nThis Arc is Not linked to anything')        ##
                        print('We record it as Street',indexvoie)           ###
###############################################################################
                    self.list.append(Voie(AllArcs,[i],[1],p))
                    AllArcs.list[i].Usedinvoie=indexvoie
                    indexvoie+=1

    ### ELIF it has the first side at an extremity, we begin a link
                elif Arc.FirstLink[0]=='Extremity':
                    NewVoie=[i]
                    Lecture=[1]
                    Explore=True

                    ActualArc=[i,'first']
                    NextArc=AllArcs.list[ActualArc[0]].LastLink

    ### ELIF it has the last side at an axtremity, we begin a link
                elif Arc.LastLink[0]=='Extremity':
                    NewVoie=[i]
                    Lecture=[-1]
                    Explore=True

                    ActualArc=[i,'last']                         #Where we come from
                    NextArc=AllArcs.list[ActualArc[0]].FirstLink #Where we go

    ### IF we are on a more than one arc street, we follow all the links
    ### The procedure is the following :
    #   _We look at the nextlink (stocked), if it's an extremity we record
                check= ['','']
                if Explore: # IF we are on a more than one arc street, we follow all the links
                    while True:
                        if NextArc[0]=='Extremity': #   We look at the nextlink, if it's an extremity we record  and stop
                            if p.get('debugVoies',False):
                                for N in NewVoie : print(N,AllArcs.list[N].FirstLink,AllArcs.list[N].LastLink)
                                print('We record this Street as :')
                                print('number', indexvoie)
                                print('NewVoie',NewVoie,Lecture)
                            for N in NewVoie :
                                AllArcs.list[N].Usedinvoie=indexvoie
                            self.list.append(Voie(AllArcs,NewVoie,Lecture,p))
                            indexvoie+=1
                            break
                        else : #If it's not an extremity,
                            #print(ActualArc[0],NextArc[0])
                            if ActualArc[0]==AllArcs.list[NextArc[0]].FirstLink[0]: #If we arrived on NextArc by the first
                                ActualArc=NextArc*1
                                NextArc=AllArcs.list[NextArc[0]].LastLink
                                NewVoie.append(ActualArc[0])
                                Lecture.append(1)
                            elif ActualArc[0]==AllArcs.list[NextArc[0]].LastLink[0]:
                                ActualArc=NextArc*1
                                NextArc=AllArcs.list[NextArc[0]].FirstLink
                                NewVoie.append(ActualArc[0])
                                Lecture.append(-1)
                        if check == [NextArc,ActualArc]:
                            self.list.append(Voie(AllArcs,NewVoie,Lecture,p))
                            indexvoie+=1
                            print('Problem voies',[NextArc,ActualArc])
                            break
                        check= [NextArc,ActualArc]
        print('Done. t=',time.time()-t)
        print('We created', len(self.list),' ways')

    def shareneighbors(self,AllPlaces,p):
        ### We give every places the arcs connected to it
        for i,V in enumerate(self.list):
            for P in V.Places:
                AllPlaces.list[P].Voies.append(i)
        for V in self.list:
            for P in V.PlacesInside :
                for X in AllPlaces.list[P].Voies:
                    if X not in V.VoiesNeighbor: V.VoiesNeighbor.append(X)

    def Reducepts(self,AllArcs):
        for j,V in enumerate(self.list):
            self.list[j].reducept(AllArcs)
        self.lengthBubble=[]
        self.lengthBird=[]
        for i,V in enumerate(self.list):
            self.lengthBubble.append(V.lengthBubble)

    def Heredite(self,AllPlaces,p):

    ### Fills the Mere,Filles,Arret,Arretes and Begin fields. If the position of the pied is defined, for each Voie, the end close to the pied is a Fille of the previous Voie
    ### (the one whos has in it the place at that extremity, if it exists), and the end far from the pied becomes an Arretes (stoped) of the following Voie
    ### (same, if it exists). Mere and arret are defined reciprocally. Begin is the position of where the Voies started to grow


        pied =  p['Pied']

        for j,V in enumerate(self.list):    #For each Voie, identify the extrmity (place) closer to the pied and the farer one

            first       =   AllPlaces.list[V.FirstPlace]
            last        =   AllPlaces.list[V.LastPlace]

            distfirst   =   np.sqrt((pied[0]-first.XY[0,0])**2 + (pied[1]-first.XY[0,1])**2)
            distlast    =   np.sqrt((pied[0]-last.XY[0,0])**2  + (pied[1]-last.XY[0,1])**2)

            if (distfirst < distlast) :                                                                     #Here FirstPlace is the closer to the pied, Begin is the coordinate of FirstPlace
                V.Begin     =   [first.XY[0,0]+0.2*np.random.rand(1), first.XY[0,1]+0.2*np.random.rand(1)]   #The random is for plotting, it buggs when it has same x or y...

                for id,VoieCherche in enumerate(self.list):
                    if id != j:                                                   #Looking for Mere and Arrêt Voie
                        for place in VoieCherche.PlacesInside:
                            if place == V.FirstPlace:                                                              #Finding the Mere Voie (has FirstPlace in it)
                                V.Mere  =   id
                                VoieCherche.Filles.append(j)
                            elif place == V.LastPlace:                                                             #Finding the Arret Voie (has LastPlace in it)
                                V.Arret =   id
                                VoieCherche.Arretes.append(j)

            else:                                                                                           #Same with LastPlace closer to the pied
                V.Begin     =   [last.XY[0,0]+0.2*np.random.rand(1),last.XY[0,1]+0.2*np.random.rand(1)]

                for id,VoieCherche in enumerate(self.list):
                    if id != j:
                        for p in VoieCherche.PlacesInside:
                            if p == V.LastPlace:
                                V.Mere  =   id
                                VoieCherche.Filles.append(j)
                            elif p == V.FirstPlace:
                                V.Arret =   id
                                VoieCherche.Arretes.append(j)

        Heredite =[]                                                                     # To put some Hierarchy in the stucture, the "first" Voies are close to the feet
        for id,Voie in enumerate(self.list):
            if Voie.Mere ==-1:
                if (len(Voie.Filles) == 0):
                    Voie.Heredite = -1
                else:
                    Heredite.append(Voie)

        NIter = 0
        while ( len(Heredite)>0):
            Mere = Heredite.pop()                                                            #Propagates with poping/adding Filles
            for idFilles in Mere.Filles:
                Fille = self.list[idFilles]
                Fille.Heredite = Mere.Heredite + 1
                Heredite.append(Fille)
            NIter+=1
            if (NIter > 10*len(self.list)):
                raise Exception('Maybe there is a cycle problem! Too long to run')
                break

    def HerediteEnds(self):
        Heredite =[]                                                                     # To put some Hierarchy in the stucture, the "first" Voies are close to the feet
        for id,Voie in enumerate(self.list):
            if Voie.Arret ==-1:
                if (len(Voie.Arretes) == 0):
                    Voie.HerediteEnds = -1
                else:
                    Heredite.append(Voie)

        NIter = 0
        while ( len(Heredite)>0):
            Stop = Heredite.pop()                                                            #Propagates with poping/adding Filles
            for idArrete in Stop.Arretes:
                Arrete = self.list[idArrete]
                Arrete.HerediteEnds = Stop.HerediteEnds + 1
                Heredite.append(Arrete)
            NIter+=1
            if (NIter > 10*len(self.list)):
                raise Exception('Maybe there is a cycle problem! Too long to run')
                break





####################################################################################################################

class Voie :
    def __repr__(self):
        return str(len(self.Arc))+' Arcs | length :'+str(self.lengthBubble)

    def __init__(self,AllArcs,Liste,lecture,p):
        self.Vertices=[]
        self.Arc=Liste
        self.Contours=[]
        self.lecture=lecture
        self.Places=[]
        self.VoiesNeighbor=[]
        self.Filles=[]
        self.Arretes=[]
        self.Mere = -1
        self.Arret = -1
        self.Heredite = 0
        self.Begin=[]

        for i,Elem in enumerate(Liste):
            self.Vertices += AllArcs.list[Elem].VerticesNoPlace[::lecture[i]]
            self.Places   += [AllArcs.list[Elem].FirstPlace]
            self.Places   += [AllArcs.list[Elem].LastPlace]


        self.Vertices=self.Vertices
        self.Places=list(set(self.Places))
        self.XY=np.zeros((len(self.Vertices),2))
        for i,V in enumerate(self.Vertices):
            self.XY[i,:]=AllArcs.Vertices[V,0:2]
        self.lengthBubble   = np.sum(np.sqrt( ( self.XY[:-1,0]-self.XY[1 :,0])**2 +
                                              ( self.XY[:-1,1]-self.XY[1 :,1])**2 ))
        self.lengthBird     =        np.sqrt( ( self.XY[  0,0]-self.XY[ -1,0])**2 +
                                              ( self.XY[  0,1]-self.XY[ -1,1])**2 )
        self.Curvature=M.curvature(self.XY)
        self.Curvature[ 0] =0
        self.Curvature[-1] =0
        self.CurvMax       =np.amax(np.abs(self.Curvature))
        if lecture[0]==1 :
            self.FirstPlace = AllArcs.list[Liste[0]].FirstPlace
            self.FirstPt    = AllArcs.list[Liste[0]].XYNoPlace[ 0,:]
        else             :
            self.FirstPlace = AllArcs.list[Liste[0]].LastPlace
            self.FirstPt    = AllArcs.list[Liste[0]].XYNoPlace[-1,:]


        if lecture[-1]==1 :
            self.LastPlace = AllArcs.list[Liste[-1]].FirstPlace
            self.LastPt    = AllArcs.list[Liste[-1]].XYNoPlace[ 0,:]
        else              :
            self.LastPlace = AllArcs.list[Liste[-1]].LastPlace
            self.LastPt    = AllArcs.list[Liste[-1]].XYNoPlace[-1,:]

        if len(self.XY[:,0])>=2:
            self.FirstAng = np.arctan2((self.XY[ 1,1]-self.XY[ 0,1]),(self.XY[ 1,0]-self.XY[ 0,0]))
            self.LastAng  = np.arctan2((self.XY[-1,1]-self.XY[-2,1]),(self.XY[-1,0]-self.XY[-2,0]))
        else :
            self.FirstAng = False
            self.LastAng  = False
        self.PlacesInside=list(set(self.Places)-set([self.FirstPlace,self.LastPlace]))
        self.FirstExtremity= False
        self.LastExtremity = False

    def reducept(self,AllArcs):
        A= [ AllArcs.list[self.Arc[i]].XYasVoies[::self.lecture[i]] for i,elem in enumerate(self.Arc)]
        self.Vertices2 = np.vstack((a for a in A))
