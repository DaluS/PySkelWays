# -*- coding: utf-8 -*-
import shapefile
import numpy as np

"""
Transformation des données en format QGIS friendly
"""

### WITHOUT ALL THE ADDS #######################################################
def AllQGISsimple(Folder,AllArcs,AllContours,AllPlaces,AllVoies,p):
    """
    Creates all the simple .shp and associated files for a first visualisation.
    Check the code to add your own data types.
    """
    QGISHoles   (Folder,AllContours ,p)
    QGISEnvelope(Folder,AllContours ,p)
    QGISArcs    (Folder,AllArcs     ,p)
        #QGISPlaces  (AllPlaces   ,p)
    QGISVoies   (Folder,AllVoies    ,p)
    return('Simple QGIS save done !')


### SORTIE DES TROUS
def QGISHoles(Folder,AllContours,p):
    w = shapefile.Writer(Folder+'Holes',shapefile.POLYGON)
    w.field('numero'      ,'C',size=250)
    w.field('Perimetre',   'N',decimal=5)
    w.field('Surface',     'N',decimal=5)
    w.field('AspectRatio', 'N',decimal=5)
    w.field('Neighbors',   'N',decimal=0)

    for j,C in enumerate(AllContours.list):
        if j!=AllContours.labmax:
            A = [ list(C.XY[i][0:2]) for i in np.arange(C.len) ]
        else: A=[ [0,0]]
        if len(A)>1:
            w.poly([A])
            w.record(C.perimeter,
                     C.surface,
                     np.sqrt(C.surface)*np.pi*2/C.perimeter,
                     C.len,
                     len(C.Arcs))
    w.close()
    print('Holes Simple done')

### SORTIE DE l'ENVELOPPE
def QGISEnvelope(Folder,AllContours,p):
    env=AllContours.labmax
    w = shapefile.Writer(Folder+'Envelope',shapefile.POLYGON)
    w.field('Perimetre',   'N',decimal=5)
    w.field('Surface',     'N',decimal=5)
    w.field('Neighbors',   'N',decimal=0)
    w.field('Npoints',     'N',decimal=0)

    A = [ list(AllContours.list[env].XY[i][0:2]) for i in np.arange(AllContours.list[env].len) ]
    if len(A)>1:
        w.poly([A])
        w.record(AllContours.list[env].perimeter,
                 AllContours.list[env].surface,
                 len(AllContours.list[env].Arcs),
                 AllContours.list[env].len)
    w.close()

### SORTIE DES ARCS
def QGISArcs(Folder,AllArcs,p):
    w = shapefile.Writer(Folder+'ArcsAsVoies',shapeType=3)
    w.field('number','N')
    w.field('len' ,'N', decimal=5)
    w.field('twist','N',decimal=5)
    w.field('Rmin','N',decimal=5)
    w.field('Rmean','N',decimal=5)
    #w.field('Ordre','N')
    for i,A in enumerate(AllArcs.list):
        ### Normal Arc
        C=[]
        for j in range(len(A.XYasVoies)):
            C.append([A.XYasVoies[j,0],A.XYasVoies[j,1]])
        w.line([C])
        w.record(i,A.lengthBubble,A.lengthBubble/A.lengthBird,A.Rmin,A.Rmean)#,A.scoreOrdre)
    #w.save(str(p['savefold'] / 'QGIS/Arcs'))
    w.close()

    w = shapefile.Writer(Folder+'Arcs',shapeType=3)
    w.field('number','N')
    w.field('len' ,'N', decimal=5)
    w.field('twist','N',decimal=5)
    w.field('Rmin','N',decimal=5)
    w.field('Rmean','N',decimal=5)
    #w.field('Ordre','N')
    for i,A in enumerate(AllArcs.list):
        ### Normal Arc
        C=[]
        for j in range(len(A.XY)):
            C.append([A.XY[j,0],A.XY[j,1]])
        w.line([C])
        w.record(i,A.lengthBubble,A.lengthBubble/A.lengthBird,A.Rmin,A.Rmean)#,A.scoreOrdre)
    #w.save(str(p['savefold'] / 'QGIS/Arcs'))
    w.close()

### SORTIE DES VOIES
def QGISVoies(Folder,AllVoies,p):
    w = shapefile.Writer(Folder+'Ways',shapeType=3)
    w.field('number','N')
    w.field('len' ,'N', decimal=5)
    w.field('twist','N',decimal=5)
    #w.field('Ordre','N')
    for i,V in enumerate(AllVoies.list):
        ### Normal Arc
        C=[]
        for j in range(len(V.XY)):
            C.append([V.XY[j,0],V.XY[j,1]])
        w.line([C])
        w.record(i,V.lengthBubble,V.lengthBubble/V.lengthBird)#,A.scoreOrdre)
    #w.save(str(p['savefold'] / 'QGIS/Arcs'))
    w.close()

### SORTIE DES PLUMES

























'''
def QGISHoles (AllContours,p):
    ### ENVELOPPE
    env=AllContours.labmax
    w = shapefile.Writer(shapefile.POLYGON)
    w.field('Perimetre',   'N',decimal=5)
    w.field('Surface',     'N',decimal=5)
    w.field('Neighbors',   'N',decimal=0)
    w.field('Npoints',     'N',decimal=0)

    A = [ list(AllContours.list[env].XY[i][0:2]) for i in np.arange(AllContours.list[env].len) ]
    if len(A)>1:
        w.poly([A])
        w.record(AllContours.list[env].perimeter,
                 AllContours.list[env].surface,
                 len(AllContours.list[env].Arcs),
                 AllContours.list[env].len)
    #print(p['savefold']+'/QGIS/Enveloppe')
    w.save(str(p['savefold'] / '/Enveloppe'))

    ### TROUS CLASSIQUES
    w = shapefile.Writer(shapefile.POLYGON)
    w.field('numero'      ,'C',size=250)
    w.field('Perimetre',   'N',decimal=5)
    w.field('Surface',     'N',decimal=5)
    w.field('Neighbors',   'N',decimal=0)
    for j,C in enumerate(AllContours.list):
        if j!=AllContours.labmax:
            A = [ list(C.XY[i][0:2]) for i in np.arange(C.len) ]
        else: A=[ [0,0]]
        if len(A)>1:
            w.poly([A])
            w.record(C.perimeter,
                     C.surface,
                     C.len,
                     len(C.Arcs))
    w.save(str(p['savefold'] / '/AllHoles'))
    print('Holes Simple done')

def QGISARCS  (AllArcs    ,p):
    w = shapefile.Writer(shapeType=3)
    w.field('number','N')
    w.field('len' ,'N', decimal=5)
    w.field('twist','N',decimal=5)
    w.field('Rmin','N',decimal=5)
    w.field('Rmean','N',decimal=5)
    for i,A in enumerate(AllArcs.list):
        ### Normal Arc
        C=[]
        for j in range(len(A.XYasVoies)):
            C.append([A.XYasVoies[j,0],A.XYasVoies[j,1]])
        w.line(parts=[C])
        w.record(i,A.lengthBubble,A.lengthBubble/A.lengthBird,A.Rmin,A.Rmean)
    w.save(str(p['savefold'] / '/Arcs'))
    print('Arcs Simple done')

def QGISPLACES(AllPlaces  ,p):
    w = shapefile.Writer(shapefile.POLYGON)
    theta=np.arange(0,2*np.pi,2*np.pi/30)
    w.field('number','N')
    w.field('Size','N',decimal=2)
    w.field('NumberofArcs','N')
    for i,P in enumerate(AllPlaces.list) :
        for j,Pt in enumerate(P.Centers):
            X=P.XY[j,0]
            Y=P.XY[j,1]
            Dist1=P.Radius[j]
            XY=np.zeros((len(theta),2))
            XY[:,0]=X+Dist1*np.cos(theta)
            XY[:,1]=Y+Dist1*np.sin(theta)
            A2= [ list(XY[k,0:2]) for k in range(len(theta)) ]
            w.poly([A2])
            w.record(i,P.Radius[j],len(P.Arcs))
    w.save(str(p['savefold'] / '/Places'))
    print('Places Simple done')

def QGISVOIES (AllVoies   ,p):
    ### Voie
    w = shapefile.Writer(shapeType=3)
    w.field('number','N')
    w.field('len' ,'N', decimal=5)
    w.field('twist','N',decimal=5)
    #w.field('Hierarchy','N',decimal=0)
    ### Extremities
    x = shapefile.Writer(shapeType=3)
    x.field('number','N')

    ### Filling
    for i,V in enumerate(AllVoies.list):
        C=[]
        for j in range(len(V.Vertices2)):
            C.append([V.Vertices2[j,0],V.Vertices2[j,1]])
        w.line(parts=[C])
        w.record(i,V.lengthBubble,V.lengthBubble/V.lengthBird)#,V.Hierarchy)
        if V.FirstExtremity :
            D=[]
            D.append([V.Vertices2[ 0,0],V.Vertices2[ 0,1]])
            D.append([V.Vertices2[ 1,0],V.Vertices2[ 1,1]])
            x.line(parts=[D])
            x.record(i)
        if V.LastExtremity  :
            D=[]
            D.append([V.Vertices2[-2,0],V.Vertices2[-2,1]])
            D.append([V.Vertices2[-1,0],V.Vertices2[-1,1]])
            x.line(parts=[D])
            x.record(i)
    ### Save
    w.save(str(p['savefold'] / '/Voies'))
    x.save(str(p['savefold'] / '/Extremities'))
    print('Voies Simple done')

### WITH THEM ##################################################################
def QGISARC2(AllArcs,p):
    ### Arc Normal
    w = shapefile.Writer(shapeType=3)
    w.field('number','N')
    w.field('len' ,'N', decimal=5)
    w.field('twist','N',decimal=5)
    w.field('Rmin','N',decimal=5)
    w.field('Rmean','N',decimal=5)
    ### ArcsWithoutPlaces
    x = shapefile.Writer(shapeType=3)
    x.field('number','N')
    x.field('len' ,'N', decimal=5)
    x.field('twist','N',decimal=5)
    x.field('Rmin','N',decimal=5)
    x.field('Rmean','N',decimal=5)
    ### ArcsAsPoints
    y = shapefile.Writer(shapeType=1)
    y.field('Number','N')
    ### ArcsInPlaces
    z = shapefile.Writer(shapeType=3)
    z.field('Number','N')

    ### ArcsAsVoies
    v = shapefile.Writer(shapeType=3)
    v.field('Number','N')

    ### Filling
    for i,A in enumerate(AllArcs.list):
        ### Normal Arc
        C=[]
        for j in range(len(A.XY)):
            C.append([A.XY[j,0],A.XY[j,1]])
        w.line(parts=[C])
        w.record(i,A.lengthBubble,A.lengthBubble/A.lengthBird,A.Rmin,A.Rmean)
        if not A.IsInaPlace:
            ### ArcsWithoutPlaces (Arc)
            if len(A.XYNoPlace)>1:
                C=[]
                for j in range(len(A.XYNoPlace)):
                    C.append([A.XYNoPlace[j,0],A.XYNoPlace[j,1]])
                x.line(parts=[C])
                x.record(i,A.lengthBubble,A.lengthBubble/A.lengthBird,A.Rmin,A.Rmean)
            ### ArcsWithoutPlaces (Points)
                y.point(A.XYNoPlace[0,0],A.XYNoPlace[0,1])
                y.record(i)

            D=[]
            for j in range(len(A.XYasVoies)):
                D.append([A.XYasVoies[j,0],A.XYasVoies[j,1]])
            v.line(parts=[D])
            v.record(i)
        ### Arcs in a Place
        else:
            z.line(parts=[C])
            z.record(i,A.lengthBubble,A.lengthBubble/A.lengthBird,A.Rmin,A.Rmean)


    ### Save
    v.save(p['savefold']+'/ArcsAsVoies')
    w.save(p['savefold']+'/AllArcs')
    x.save(p['savefold']+'/ArcsWOPlace')
    y.save(p['savefold']+'/ArcsWOPlacePts')
    z.save(p['savefold']+'/ArcsInPlace')

def QGISVOIES2(AllVoies,p):
    ### Voie
    w = shapefile.Writer(shapeType=3)
    w.field('number','N')
    w.field('len' ,'N', decimal=5)
    w.field('twist','N',decimal=5)
    #w.field('Hierarchy','N',decimal=0)
    ### Extremities
    x = shapefile.Writer(shapeType=3)
    x.field('number','N')

    ### Filling
    for i,V in enumerate(AllVoies.list):
        C=[]
        for j in range(len(V.XY)):
            C.append([V.XY[j,0],V.XY[j,1]])
        w.line(parts=[C])
        w.record(i,V.lengthBubble,V.lengthBubble/V.lengthBird)#,V.Hierarchy)
        if V.FirstExtremity :
            D=[]
            D.append([V.XY[ 0,0],V.XY[ 0,1]])
            D.append([V.XY[ 1,0],V.XY[ 1,1]])
            x.line(parts=[D])
            x.record(i)
        if V.LastExtremity  :
            D=[]
            D.append([V.XY[-2,0],V.XY[-2,1]])
            D.append([V.XY[-1,0],V.XY[-1,1]])
            x.line(parts=[D])
            x.record(i)
    ### Save
    w.save(p['savefold']+'/Voies')
    x.save(p['savefold']+'/Extremities')
'''
