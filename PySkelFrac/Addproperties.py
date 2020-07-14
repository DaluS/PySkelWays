# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 16:52:00 2017

@author: Paul
"""

import numpy as np
from skimage import measure
from collections import defaultdict
import PySkelFrac.classes as c
from scipy import ndimage
import copy
import time
import cv2

def NewAssociatedContours(AllContours,AllArcs,AllPlaces,AllVoies,p,Exterior=False):
    """
    For every Contours, we look at the other structures associated
    """

    print('\n### Associations between classes ### ### ### ###')
    #print('Association of Ways with contours...')
    t=time.time()
    # PTCont contains all Contours linked to the point
    #print('All Contours Linked to a point')
    PtCont=defaultdict(list)
    for S in AllArcs.Segments:
        PtCont[int(S[0])].extend(S[2:4].astype(np.int))
        PtCont[int(S[1])].extend(S[2:4].astype(np.int))
    for key,value in PtCont.items():
        PtCont[key]=list(set(value))


    #Now for every arcs we want to do the list of Contours around it
    #print('All Contours linked to an arc')
    for i,Al in enumerate(AllArcs.Arclist):
        for Pt in Al[1:-1] :
            AllArcs.list[i].Contours.extend(PtCont[Pt])
    for i,A in enumerate(AllArcs.list):
        A.Contours=list(set(A.Contours))
        for C in A.Contours:
            AllContours.list[C].Arcs.append(i)

    #Now for every voie we to the list of Contours around it
    #print('All Contours linked to a voie')
    for V in AllVoies.list:
        for A in V.Arc:
            V.Contours.extend(AllArcs.list[A].Contours)
    for i,V in enumerate(AllVoies.list):
        V.Contours=list(set(V.Contours))
        for C in V.Contours:
            AllContours.list[C].Voies.append(i)

    #print('All theses link to the contours')
    for i,C in enumerate(AllContours.list):
        for A in C.Arcs:
            if min(AllArcs.list[A].Connectivitypts)>=2:
                AllContours.list[i].Points.extend(AllArcs.list[A].Vertices)
            else :
                 AllContours.list[i].ArcsInside.append(A)

    #print('Calculation of new contours')
    #for i,C in enumerate(AllContours.list):
        #if C.Points:
        #    if (i==AllContours.labmax and Exterior):
        #        AllContours.list[i].Points=ReorganizePolygon(C.Points,AllArcs.Vertices)
        #    elif i!=AllContours.labmax:
        #        AllContours.list[i].Points=ReorganizePolygon(C.Points,AllArcs.Vertices)
        #        AllContours.list[i].PointsXY=AllArcs.Vertices[ AllContours.list[i].Points,0:2]
        #else:
        #AllContours.list[i].Points=[]
        #AllContours.list[i].PointsXY=[]

    for A in AllArcs  .list: A.Contours=[]
    for V in AllVoies .list: C.Voies   =[]


    for i,V in enumerate(AllVoies.list):
        V.VoiesLink=[]
        V.ArcsLink =[]
        for Pid in V.Places:
            P=AllPlaces.list[Pid]
            for E in P.Extremities:
                V.VoiesLink.append(AllArcs.list[E[0]].Usedinvoie)
                V.ArcsLink. append(E[0])
            P.Voie=i
    for i,C in enumerate(AllContours.list):

        for Aid in C.Arcs:
            C.Voies.append(AllArcs.list[Aid].Usedinvoie)
            AllArcs.list[Aid].Contours.append(i)
        C.Voies=list(set(C.Voies))



    for i,C in enumerate(AllContours.list) : C.index=i
    for i,A in enumerate(AllArcs    .list) : A.index=i
    for i,P in enumerate(AllPlaces  .list) : P.index=i
    for i,V in enumerate(AllVoies   .list) : V.index=i

    ### CORRECT WAY IN THE PLACE
    for P in AllPlaces.list:
        if (len(P.Arcs)>2 and len(P.Links)):
            ind1 = AllArcs.list[P.Links[0][0]].Usedinvoie
            ind2 = AllArcs.list[P.Links[0][1]].Usedinvoie
            if ind1==ind2 : P.Voie = ind1
            else : print("Place", P.index, "has a ill defined way")


    ### RIGHT DIRECTION (WAYS TO BRANCHES )
    for i,V in enumerate(AllVoies.list):
        if  ((V.FirstPt[0]-p['Pied'][0])**2 + (V.FirstPt[1]-p['Pied'][1])** 2) > (( V.LastPt[0]-p['Pied'][0])**2 + ( V.LastPt[1]-p['Pied'][1])** 2):
            V.lecture=-np.array(V.lecture)[::-1]
            V.Arc= V.Arc[::-1]
            V.Vertices = V.Vertices[::-1]
            V.XY = V.XY[::-1,:]
            V.FirstPlace, V.LastPlace = V.LastPlace, V.FirstPlace
            V.FirstPt   , V.LastPt    = V.LastPt   , V.FirstPt
        del V.Arret, V.Arretes, V.Begin, V.CurvMax, V.Curvature, V.Filles, V.Heredite, V.Mere, V.VoiesNeighbor, V.FirstAng, V.LastAng

        ### CORRECT FIRST PLACE AND LASTPLACE
        A = AllArcs.list[V.Arc[0]]
        if AllPlaces.list[A.FirstPlace].Voie != i    : V.FirstPlace=A.FirstPlace
        else                                         : V.FirstPlace=A.LastPlace
        if len(AllPlaces.list[A.FirstPlace].Arcs)==1 : V.FirstPlace=A.FirstPlace
        if len(AllPlaces.list[A. LastPlace].Arcs)==1 : V.FirstPlace=A. LastPlace

        A = AllArcs.list[V.Arc[-1]]
        if AllPlaces.list[A.FirstPlace].Voie != i    : V.LastPlace=A.FirstPlace
        else                                         : V.LastPlace=A.LastPlace
        if len(AllPlaces.list[A.FirstPlace].Arcs)==1 : V.LastPlace=A.FirstPlace
        if len(AllPlaces.list[A. LastPlace].Arcs)==1 : V.LastPlace=A. LastPlace

        ### CORRECT VOIESLINKED TO REMOVE EXTREME PLACES




        V.ArcsLink = list(set([A for A in V.ArcsLink if A not in V.Arc]))              ### LIEN FILLES-MORTES TOUTES
        V.VoiesLink = list(set([AllArcs.list[Ai].Usedinvoie for Ai in V.ArcsLink]))    ### LIEN FILLES-MORTES TOUTES
        V.VoiesExtremes = [AllArcs.list[Ai[0]].Usedinvoie for Ai in AllPlaces.list[V.FirstPlace].Arcs]+[AllArcs.list[Ai[0]].Usedinvoie for Ai in AllPlaces.list[V.LastPlace].Arcs]
        V.VoiesIn = [V2 for V2 in V.VoiesLink if V2 not in V.VoiesExtremes ]

        if     V.index==AllPlaces.list[V.FirstPlace].Voie: V.Mother = False
        else : V.Mother=AllPlaces.list[V.FirstPlace].Voie
        if     V.index==AllPlaces.list[V.LastPlace ].Voie: V.Killer = False
        else : V.Killer=AllPlaces.list[V.LastPlace ].Voie
        V.Killed = []
        V.Daughter = []

    for V in AllVoies.list :
        if V.Mother : AllVoies.list[V.Mother].Daughter.append(V.index)
        if V.Killer : AllVoies.list[V.Killer].Killed  .append(V.index)




    print('Done. t=',time.time()-t)
    return AllContours,AllArcs,AllPlaces,AllVoies



def RegulatedContours2(IMG,AllContours,AllArcs,AllPlaces,AllVoies,p):
    NOWIDTH=c.Void()
    NOWIDTH.binary=np.zeros((IMG.Y,IMG.X,3),dtype=np.uint8)
    for A in AllVoies.list:
        pts=np.vstack((  A.XY[:,0], A.XY[:,1])).T.reshape(-1,2).astype(np.int32)
        cv2.polylines(NOWIDTH.binary, [pts], 0, (255,255,255),2)
    NOWIDTH.binary   = NOWIDTH.binary[:,:,0].T                                      # CARTE SANS DISTANCE
    #IMG.DistNoborder = cv2.distanceTransform(NOWIDTH.binary,cv2.DIST_L2,3)          # DISTANCE DANS LA STRUCTURE

    ContoursNowidth= c.Contours(NOWIDTH,p)                                          # CONTOURS ASSOCIES
    #IMG.HolesNoWidth = ((255-NOWIDTH.binary)*IMG.enveloppe).astype(np.uint8)        # UNIQUEMENT LES TROUS
    #IMG.HolesDistNoWidth = cv2.distanceTransform(IMG.HolesNoWidth,cv2.DIST_L2,3)      # DISTANCE DANS LES TROUS
    #IMG.NoWidth= NOWIDTH.binary

    ### CREATION DU REGULEE
    REGU=c.Void()
    REGU.binary=np.zeros((IMG.Y,IMG.X,3),dtype=np.uint8)
    for A in AllVoies.list:
        pts=np.vstack((  A.XY[:,0], A.XY[:,1])).T.reshape(-1,2).astype(np.int32)
        cv2.polylines(REGU.binary, [pts], 0, (255,255,255),int(p['branchwidth']))
    REGU.binary  = REGU.binary[:,:,0].T                                             # CARTE REGULEE
    #IMG.DistRegu = cv2.distanceTransform(NOWIDTH.binary,cv2.DIST_L2,3)              # DISTANCE DANS LA STRUCTURE
    ContoursRegu   = c.Contours(REGU   ,p)                                          # CONTOURS ASSOCIES
    #IMG.HolesRegu= ((255-REGU.binary)*IMG.enveloppe).astype(np.uint8)               # UNIQUEMENT LES TROUS
    #IMG.HolesDistRegu = cv2.distanceTransform(IMG.HolesRegu,cv2.DIST_L2,3)          # DISTANCE DANS LES TROUS
    #IMG.Regu     = REGU.binary

    """
    for _ in range()
        AllContours.C_XY      = [ C.Center for C in AllContours.list     if not C.isexterior]
        ContoursRegu.C_XY     = [ C.Center for C in ContoursRegu.list    if not C.isexterior]
        ContoursNowidth.C_XY  = [ C.Center for C in ContoursNowidth.list if not C.isexterior]

        from scipy.spatial import distance_matrix
        ALLDIST1 = distance_matrix(AllContours.C_XY,ContoursRegu.C_XY )
        ALLDIST2 = distance_matrix(AllContours.C_XY,ContoursNowidth.C_XY )
        ALLDIST3 = distance_matrix(ContoursRegu.C_XY,ContoursNowidth.C_XY )

        MIN1 = np.argmin(ALLDIST1, axis=1) # Donne pour chaque trou de
        MIN11= np.argmin(ALLDIST1, axis=0)

        MIN2 = np.argmin(ALLDIST2, axis=1)
        MIN22= np.argmin(ALLDIST2, axis=0)

        for C in AllContours.list :
            C.Regcontours = False
            C.NoWContours = False
        for C in ContoursRegu.list : C.Taken = False
        for C in ContoursNowidth.list : C.Taken = False
        for i,val in enumerate(MIN1[MIN11]-np.linspace(0,len(MIN11)-1,len(MIN11))) :
            if val==0. :
                AllContours.list[i+1].RegContours=  ContoursRegu.list[ MIN1[i]]
                ContoursRegu.list[ MIN1[i]].Taken = True
        for i,val in enumerate(MIN2[MIN22]-np.linspace(0,len(MIN22)-1,len(MIN22))) :
            if val==0. :
                AllContours.list[i+1].NoWcontours=  ContoursNowidth.list[ MIN2[i]]
                ContoursNowidth.list[ MIN2[i]].Taken= True
    """
    return IMG,ContoursNowidth,ContoursRegu,REGU.binary

################################################################################
################################################################################
###############################################################################

def ContoursCaracteristics(IMG,AllContours,AllArcs,p):
    """
    On ajoute quelques propriétés aux contours, notamment en supprimant les epaisseurs
    """
    print('### ADDING NEW CONTOURS PROPERTIES ###')
    IMG.enveloppe         = np.copy(IMG.binary).astype(np.int32)
    IMG.Holes             = np.zeros((IMG.X,IMG.Y),dtype=np.int32)
    IMG.HolesDistRegu     = np.zeros((IMG.X,IMG.Y),dtype=np.int32)
    IMG.HolesDistNoborder = np.zeros((IMG.X,IMG.Y),dtype=np.int32)
    IMG.regu              = np.zeros((IMG.X,IMG.Y),dtype=np.int32)

    pourcent=0
    t=0
    print('Calculation off AllContours Properties...')
    for i,C in enumerate(AllContours.list):
        if i != AllContours.labmax:
            if 100*i/len(AllContours.list)>pourcent:
                pourcent+=1
                print(10*'\r',end='')
                print(int(100*i/len(AllContours.list)),'%',end='')
            ### Classical Contours
            xmin=np.amin(C.XY[:,0]);xmax=np.amax(C.XY[:,0])
            ymin=np.amin(C.XY[:,1]);ymax=np.amax(C.XY[:,1])
            XY=np.copy(C.XY)
            XY[:,0]-=xmin
            XY[:,1]-=ymin
            XY=XY.reshape((-1,1,2)).astype(np.int32)

            IMG_POLY=np.zeros((int(xmax)-int(xmin)+2,(int(ymax)-int(ymin)+2)),dtype=np.int32)
            cv2.fillPoly(IMG_POLY,[XY],1)
            IMG_POLY=ndimage.distance_transform_edt(IMG_POLY)

            AllContours.list[i].DistMax = np.amax(IMG_POLY)
            if p.get('Polypsize',False):
                AllContours.list[i].SurfPolyp= sum(1 for e in np.reshape(IMG_POLY,(len(IMG_POLY[:,0])*len(IMG_POLY[0,:]),1)) if e > p['Polypsize'])

            #ContoursNoborder :
            C.Noborder=c.Void()
            CN=C.Noborder
            CN.XY = np.copy(AllArcs.Vertices[C.Points,0:2])
            welldetermined=True
            if len(CN.XY)==0:
                CN.XY=C.XY
                welldetermined=False
            CN.perimeter=   np.sum(np.sqrt( ( CN.XY[:,0]-np.roll(CN.XY[:,0],1) )**2 +
                                            ( CN.XY[:,1]-np.roll(CN.XY[:,1],1) )**2 ))
            CN.surface  =   0.5*np.abs(np.dot(CN.XY[:,0],np.roll(CN.XY[:,1],1))-
                                       np.dot(CN.XY[:,1],np.roll(CN.XY[:,0],1)))

            xmin=np.amin(CN.XY[:,0]);xmax=np.amax(CN.XY[:,0])
            ymin=np.amin(CN.XY[:,1]);ymax=np.amax(CN.XY[:,1])
            XY=copy.copy(CN.XY)
            XY[:,0]-=xmin
            XY[:,1]-=ymin
            XY=XY.reshape((-1,1,2)).astype(np.int32)

            IMG_POLY=np.zeros((int(xmax)-int(xmin)+2,(int(ymax)-int(ymin)+2)),dtype=np.int32)

            cv2.fillPoly(IMG_POLY,[XY],1)
            ### AJOUTER POUR CHAQUE A L'INTERIEUR
            # L'ENSEMBLE DES POINTS QUI LE CONSTITUENT : ON MET 0 avec LA LINE MAISON
            IMG_POLY=ndimage.distance_transform_edt(IMG_POLY)

            CN.DistMax = np.amax(IMG_POLY)
            if p.get('Polypsize',False):
                CN.SurfPolyp= sum(1 for e in np.reshape(IMG_POLY,(len(IMG_POLY[:,0])*len(IMG_POLY[0,:]),1)) if e > p['Polypsize'])
            #print(np.shape(IMG.HolesDistNoborder[int(xmin):int(xmin)+np.shape(IMG_POLY)[0]) ,int(ymin):int(ymin)+np.shape(IMG_POLY)[0] ]) ,np.shape(IMG_POLY.astype(np.int32) ))

            try:
                IMG.HolesDistNoborder[int(xmin):int(xmin)+np.shape(IMG_POLY)[0] ,int(ymin):int(ymin)+np.shape(IMG_POLY)[1] ]+=IMG_POLY.astype(np.int32)
                IMG.enveloppe        [int(xmin):int(xmin)+np.shape(IMG_POLY)[0] ,int(ymin):int(ymin)+np.shape(IMG_POLY)[1] ]+=IMG_POLY.astype(np.int32)
            except BaseException:
                welldetermined=False
            #ContoursRegu :
            if p.get('branchwidth',False):
                C.Regulated=c.Void()
                CR=C.Regulated
                Regcont=measure.find_contours(IMG_POLY,p['branchwidth']/2)
                if len(Regcont)>1:
                    lens=np.zeros(len(Regcont))
                    for z,elems in enumerate(Regcont):
                        lens[z]=len(elems)
                    CR.XY=Regcont[np.argmax(lens)]
                elif len(Regcont)==1: CR.XY=Regcont[0]


                if (len(Regcont)==0 or not welldetermined):
                    CR.XY=[[None],[None]]
                    CR.perimeter= 0
                    CR.surface  = 0
                    CR.DistMax  = 0
                    CR.DistMax  = 0
                else:
                    CR.perimeter=   np.sum(np.sqrt( ( CR.XY[:,0]-np.roll(CR.XY[:,0],1) )**2 +
                                                    ( CR.XY[:,1]-np.roll(CR.XY[:,1],1) )**2 ))
                    CR.surface  =   0.5*np.abs(np.dot(CR.XY[:,0],np.roll(CR.XY[:,1],1))-
                                           np.dot(CR.XY[:,1],np.roll(CR.XY[:,0],1)))

                    XY=CR.XY.astype(np.int32)
                    IMG_POLY=np.zeros((int(xmax)-int(xmin)+2,(int(ymax)-int(ymin)+2)),dtype=np.int32)
                    cv2.fillPoly(IMG_POLY,[XY],1)
                    IMG_POLY=ndimage.distance_transform_edt(IMG_POLY)

                    CR.DistMax = np.amax(IMG_POLY)
                    if p.get('Polypsize',False):
                        CR.SurfPolyp= sum(1 for e in np.reshape(IMG_POLY,(len(IMG_POLY[:,0])*len(IMG_POLY[0,:]),1)) if e > p['Polypsize'])
                    IMG.HolesDistRegu[int(xmin):int(xmin)+len(IMG_POLY[:,0]) ,int(ymin):int(ymin)+len(IMG_POLY[0,:]) ]+=IMG_POLY.astype(np.int32)
                    CR.XY[:,0]+=xmin
                    CR.XY[:,1]+=ymin

    print('Done, t=',time.time()-t)

    pourcent=0
    t=time.time()
    return IMG,AllContours

def AllPropertiesCompilation(IMG,AllContours,AllArcs,p):
    """
    On ajoute pleeeeeeeins de propriétés
    """
    print('\n### PROPERTIES COMPILATION')
    print('Compilation of properties to "Images"...')
    pourcent=0
    t=time.time()

    IMG.enveloppe[IMG.enveloppe>1]=1
    IMG.enveloppe                 = ndimage.binary_fill_holes(IMG.enveloppe)
    IMG.distenveloppe             = ndimage.distance_transform_edt(IMG.enveloppe)
    IMG.regu[IMG.HolesDistRegu>1]=1
    IMG.regu                      = (1-IMG.regu)*IMG.enveloppe
    IMG.distregu                  = ndimage.distance_transform_edt(IMG.regu)

    IMG.Holes                     = (1-IMG.binary)*IMG.enveloppe
    IMG.HolesDist                 = ndimage.distance_transform_edt( IMG.Holes )
    IMG.Holesregu                 = (1-IMG.regu)*IMG.enveloppe
    IMG.HolesDistRegu             = ndimage.distance_transform_edt( IMG.Holesregu )

    print('Done, t=',time.time()-t)
    if not p.get('stayinint32',False):
        print('Conversion of IMG arry in int16...')
        IMG.enveloppe    =IMG.enveloppe.astype(np.int16)
        IMG.distenveloppe=IMG.distenveloppe.astype(np.int16)
        IMG.Regu         =IMG.regu.astype(np.int16)
        IMG.DistRegu     =IMG.distregu.astype(np.int16)
        IMG.Holes        =IMG.Holes.astype(np.int16)
        IMG.HolesDist    =IMG.HolesDist.astype(np.int16)
        IMG.Holesregu    =IMG.Holesregu.astype(np.int16)
        IMG.HolesDistRegu=IMG.HolesDistRegu.astype(np.int16)

        IMG.HolesDistNoborder = IMG.HolesDistNoborder.astype(np.int16)
        IMG.binary            = IMG.binary.astype(np.int16)
        IMG.dist              = IMG.dist.astype(np.int16)




    print('Compilation of properties to "AllContours"...',end='')
    t=time.time()
    AllContours.Properlists= {}
    AllContours.Properlists['Perimetres']  = np.zeros(AllContours.ncontours-1)
    AllContours.Properlists['Surface']     = np.zeros(AllContours.ncontours-1)
    AllContours.Properlists['AspectRatio'] = np.zeros(AllContours.ncontours-1)
    AllContours.Properlists['DistMax']     = np.zeros(AllContours.ncontours-1)
    AllContours.Properlists['SsupDpolyps'] = np.zeros(AllContours.ncontours-1)
    AllContours.Properlists['ratiosurfac'] = np.zeros(AllContours.ncontours-1)

    if p.get('branchwidth',False):
        AllContours.Properlists['RegPerimetres']  = np.zeros(AllContours.ncontours-1)
        AllContours.Properlists['RegSurface']     = np.zeros(AllContours.ncontours-1)
        AllContours.Properlists['RegAspectRatio'] = np.zeros(AllContours.ncontours-1)
        AllContours.Properlists['RegDistMax']     = np.zeros(AllContours.ncontours-1)
        AllContours.Properlists['RegSsupDpolyps'] = np.zeros(AllContours.ncontours-1)
        AllContours.Properlists['Regratiosurfac'] = np.zeros(AllContours.ncontours-1)

    AllContours.Properlists['NoWPerimetres']  = np.zeros(AllContours.ncontours-1)
    AllContours.Properlists['NoWSurface']     = np.zeros(AllContours.ncontours-1)
    AllContours.Properlists['NoWAspectRatio'] = np.zeros(AllContours.ncontours-1)
    AllContours.Properlists['NoWDistMax']     = np.zeros(AllContours.ncontours-1)
    AllContours.Properlists['NoWSsupDpolyps'] = np.zeros(AllContours.ncontours-1)
    AllContours.Properlists['NoWratiosurfac'] = np.zeros(AllContours.ncontours-1)

    AllContours.Properlists['nbvoisins']   = np.zeros(AllContours.ncontours-1).astype(np.int32)
    AllContours.Properlists['Arcsinside']  = np.zeros(AllContours.ncontours-1).astype(np.int32)
    AllContours.Properlists['Voies']       = np.zeros(AllContours.ncontours-1).astype(np.int32)
    AllContours.Properlists['X']    = np.zeros(AllContours.ncontours-1)
    AllContours.Properlists['Y']    = np.zeros(AllContours.ncontours-1)
    AllContours.Properlists['Foot'] = np.zeros(AllContours.ncontours-1)
    AllContours.Properlists['Ext']  = np.zeros(AllContours.ncontours-1)

    index=0
    for i,C in enumerate(AllContours.list[:]):

        if i!=AllContours.labmax:
            #print(i)
            if 100*i/len(AllContours.list)>pourcent:
                pourcent+=1
                print(10*'\r',end='')
                print(int(100*i/len(AllContours.list)),'%',end='')
            AllContours.Properlists['Perimetres'] [index]=(C.perimeter  )
            AllContours.Properlists['Surface']    [index]=(C.surface  )
            AllContours.Properlists['AspectRatio'][index]=(np.sqrt(C.surface)/(C.perimeter*2*np.sqrt(np.pi))  )
            AllContours.Properlists['DistMax']    [index]=(C.DistMax  )
            if p.get('Polypsize',False):
                AllContours.Properlists['SsupDpolyps'][index]=(C.SurfPolyp  )
                AllContours.Properlists['ratiosurfac'][index]=(C.SurfPolyp/C.surface   )

            if p.get('branchwidth',False):
                if C.Regulated.perimeter >0 :
                    AllContours.Properlists['RegPerimetres'] [index]=(C.Regulated.perimeter  )
                    AllContours.Properlists['RegSurface']    [index]=(C.Regulated.surface  )
                    AllContours.Properlists['RegAspectRatio'][index]=(np.sqrt(C.Regulated.surface)/(C.Regulated.perimeter*2*np.sqrt(np.pi))   )
                    AllContours.Properlists['RegDistMax']    [index]=(C.Regulated.DistMax  )
                    if p.get('Polypsize',False):
                        AllContours.Properlists['RegSsupDpolyps'][index]=(C.Regulated.SurfPolyp  )
                        AllContours.Properlists['Regratiosurfac'][index]=(C.Regulated.SurfPolyp/C.Regulated.surface  )
                else:
                    AllContours.Properlists['RegPerimetres'] [index]=(0)
                    AllContours.Properlists['RegSurface']    [index]=(0)
                    AllContours.Properlists['RegAspectRatio'][index]=(0)
                    AllContours.Properlists['RegDistMax']    [index]=(0)
                    if p.get('Polypsize',False):
                        AllContours.Properlists['RegSsupDpolyps'][index]=(0)
                        AllContours.Properlists['Regratiosurfac'][index]=(0)

            AllContours.Properlists['NoWPerimetres'] [index]=(C.Noborder.perimeter  )
            AllContours.Properlists['NoWSurface']    [index]=(C.Noborder.surface  )
            AllContours.Properlists['NoWAspectRatio'][index]=(np.sqrt(C.Noborder.surface)/(C.Noborder.perimeter*2*np.sqrt(np.pi))   )
            AllContours.Properlists['NoWDistMax']    [index]=(C.Noborder.DistMax  )
            if p.get('Polypsize',False):
                AllContours.Properlists['NoWSsupDpolyps'][index]=(C.Noborder.SurfPolyp  )
                AllContours.Properlists['NoWratiosurfac'][index]=(C.Noborder.SurfPolyp/C.Noborder.surface  )

            AllContours.Properlists['nbvoisins']  [index]=(len(C.Arcs)  )
            AllContours.Properlists['Arcsinside'] [index]=(len(C.ArcsInside)  )
            AllContours.Properlists['Voies']      [index]=(len(C.Voies))

            AllContours.Properlists['X']   [index]=(C.Center[0]  )
            AllContours.Properlists['Y']   [index]=(C.Center[1]  )
            AllContours.Properlists['Foot'][index]=(np.sqrt( (C.Center[0] - p['Pied'][0]) ** 2 + (C.Center[1] - p['Pied'][1]) ** 2 ) )
            AllContours.Properlists['Ext'] [index]=(IMG.distenveloppe[int(C.Center[0]),int(C.Center[1]) ] )
            index+=1
    print('Done, t=',time.time()-t)


    print('Compilation of properties to "AllArcs"...',end='')
    t=time.time()
    AllArcs.Properlists= {}
    AllArcs.Properlists['rmean']=np.zeros(AllArcs.Narcs)
    AllArcs.Properlists['rmin'] =np.zeros(AllArcs.Narcs)
    AllArcs.Properlists['rmed'] =np.zeros(AllArcs.Narcs)
    AllArcs.Properlists['rmax'] =np.zeros(AllArcs.Narcs)
    AllArcs.Properlists['r-std']=np.zeros(AllArcs.Narcs)

    AllArcs.Properlists['lbird']  =np.zeros(AllArcs.Narcs)
    AllArcs.Properlists['lBubble']=np.zeros(AllArcs.Narcs)
    AllArcs.Properlists['r-std']  =np.zeros(AllArcs.Narcs)

    AllArcs.Properlists['ArcFoot'] =np.zeros(AllArcs.Narcs)
    AllArcs.Properlists['ArcAngle']=np.zeros(AllArcs.Narcs)
    for i,A in enumerate(AllArcs.list):
        AllArcs.Properlists['rmean']  [i]=A.Rmean
        AllArcs.Properlists['rmin']   [i]=A.Rmin
        AllArcs.Properlists['rmed']   [i]=np.median(A.R)
        AllArcs.Properlists['rmax']   [i]=A.Rmax
        AllArcs.Properlists['r-std']  [i]=A.Rstd

        AllArcs.Properlists['lbird']  [i]=A.lengthBird
        AllArcs.Properlists['lBubble'][i]=A.lengthBubble
        AllArcs.Properlists['ArcFoot'][i]=np.sqrt( (np.mean(A.XY[:,0]) - p['Pied'][0])**2
                                                  +(np.mean(A.XY[:,1]) - p['Pied'][1])**2)
        AllArcs.Properlists['ArcAngle'][i] = np.arctan2(A.XY[-1,1]-A.XY[0,1],A.XY[-1,0]-A.XY[0,0])
    return IMG,AllContours,AllArcs

def ReorganizePolygon(List,Vertices):
    '''
    Prends tous les points des contours et les lies de manière la plus logique possible
    '''
    Ptstemp=List[1:]
    Pts=     [List[0]]
    lastPts=  Vertices[List[0],0:2]
    while len(Ptstemp)>0:
        Dist= np.sqrt( (lastPts[0]-Vertices[Ptstemp,0])**2+(lastPts[1]-Vertices[Ptstemp,1])**2   )
        Newpt=np.argmin(Dist)
        Pts+=[Ptstemp[Newpt]]
        lastPts=Vertices[Ptstemp[Newpt],0:2]
        Ptstemp.pop(Newpt)
    return Pts

def CreateHierarchy(AllVoies,AllArcs,firstArc):
    '''
    Pas actualisé depuis un sacré bout de temps !
    '''
    FirstVoie=AllArcs.list[firstArc].Usedinvoie
    L=len(AllVoies.list)
    for i,V in enumerate(AllVoies.list):
        AllVoies.list[i].Hierarchy = L
    AllVoies.list[FirstVoie].Hierarchy= 0

    VoiesToExplore=[]
    ActualIter=0

    for VTE in AllVoies.list[FirstVoie].VoiesNeighbor:
        if AllVoies.list[VTE].Hierarchy > ActualIter :
            VoiesToExplore.append(VTE)
            AllVoies.list[VTE].Hierarchy=ActualIter+1
    ActualIter+=1
    while len(VoiesToExplore)>0:
        L=len(VoiesToExplore)
        for V in VoiesToExplore[::-1]:
            for VTE in AllVoies.list[V].VoiesNeighbor:
                if AllVoies.list[VTE].Hierarchy > ActualIter :
                    VoiesToExplore.append(VTE)
                    AllVoies.list[VTE].Hierarchy=ActualIter+1
            VoiesToExplore.pop(L-1)
            L-=1
        ActualIter+=1


    L=len(AllVoies.list)
    for i,V in enumerate(AllVoies.list):
        if AllVoies.list[i].Hierarchy == L:
            for VTE in AllVoies.list[i].VoiesNeighbor:
                if AllVoies.list[VTE].Hierarchy < AllVoies.list[i].Hierarchy:
                    AllVoies.list[i].Hierarchy =  AllVoies.list[VTE].Hierarchy+1
            if AllVoies.list[i].Hierarchy==L:
                AllVoies.list[i].Hierarchy=ActualIter+1
        for A in V.Arc:
            AllArcs.list[A].HierarchyVoies= V.Hierarchy
    return AllVoies,AllArcs
