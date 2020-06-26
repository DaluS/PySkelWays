'''
############################### PYSKELWAYS #####################################

A software for hypergraph extraction from a binarised image

TODOLIST :
* Put small functions in miscfunc, or create two func files
* Put CORRECTION IN A NEW MODULE
* REPAIR SPLIT (Connexion unbreakable, wrong connection broken when clicked)
* Reduce pickle size ?

note : the structure for _ in range(1): is just to wrap the code with a proper IDE
use M.dirP to see what is in an object !

WEIRD BUGS :
* Sometimes there are arcs that are not created. Reducing the size of the image ( or p['resize'] to a different value ) solves this problem for the moment
* Some splits can never be clicked, and some are impacted even when not clicked on. This should be corrected also.
'''
######################### IMPORTATIONS #########################################
for _ in range(1): # Permet simplement de faire un repliement
    ### PYSKELFRAC ###
    import PySkelFrac.classes as c       ### All Objects and their properties
    import PySkelFrac.Miscfunc as M      ### Most functions I coded
    import PySkelFrac.QGISsave as Q      ### Save datas as QGIS SHAPE
    #import PySkelFrac.Addproperties as AdP

    ### LIBRAIRIES SCIENTIFIQUES UTILES DANS LE MAIN ###
    import numpy as np
    import cv2

    ### LIBRAIRIES UTILITAIRES (MANIPULATION FICHIERS) ###
    import pickle             # enregistrement de variables python, via "dump" et "load"
    import os                 # exploration de fichiers
    from pathlib import Path  # Permet de gérer le fait d'être sous PC et LINUX
    from copy import deepcopy # permet d'éviter les problemes de "pointeurs", en recopiant les vlaeurs lcas

    ### PRINTS ###
    import matplotlib as mpl
    import matplotlib.pylab as pl
    import matplotlib.pyplot as plt

    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    from matplotlib.collections import LineCollection

    from matplotlib.ticker import FormatStrFormatter
    import matplotlib.ticker as ticker
    SIZETICKS=20
    SIZEFONT=25
    LEGENDSIZE=20
    LEGENDHANDLELENGTH=2
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=SIZEFONT)
    plt.rc('xtick', labelsize=SIZETICKS)
    plt.rc('ytick', labelsize=SIZETICKS)
    plt.rc('figure',figsize = (10, 10))
    plt.rcParams.update({'figure.autolayout': True})
    plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath} \usepackage{libertine}"]
    ticksfontProperties = {'family':'sans-serif','sans-serif':['Helvetica'],   'weight' : 'medium', 'size' : SIZETICKS}
    params = {'legend.fontsize': LEGENDSIZE,
              'legend.handlelength': LEGENDHANDLELENGTH}
    plt.rcParams.update(params)
    CMAP= mpl.cm.get_cmap('jet')
    CMAP.set_under('w',1.)

    import warnings
    #warnings.filterwarnings("ignore") #Uncomment to remove warning messages
    print('warning ignored !')

for _ in range(1): # fonctions en plus

    from scipy.optimize import curve_fit
    def split(event,x,y,flags,param):
        '''
        Action when there is a click on the split interface
        '''
        global AllPlaces
        global AllArcs
        global img
        global voiesWidth
        if event == cv2.EVENT_LBUTTONDOWN:
            a,b,c = cv2.resize(np.rot90(imglink[xm:xM,ym:yM,:],k=flip),(winsize,winsize))[y,x,:]
            if a + b*255 + c* 255 *255 -1>=0:
                P=AllPlaces.list[a + b*255 + c* 255 *255 -1]
                if not P.ModifiedPlace:
                    L =P.Links[0]
                    ### AJOUTE DES EXTREMITES
                    P.Extremities.append([L[0],L[2]])
                    P.Extremities.append([L[1],L[3]])

                    ### AJOUT DE l'INFO DANS LES ARCS
                    A = AllArcs.list[L[0]]
                    if L[2]=='first': A.FirstLink= ['Extremity']
                    else :            A.LastLink = ['Extremity']

                    A = AllArcs.list[L[1]]
                    if L[3]=='first': A.FirstLink= ['Extremity']
                    else :            A.LastLink = ['Extremity']

                    ### SUPPRESSION DU LIEN
                    if L[2] == 'first': pt1 = AllArcs.list[L[0]].XYNoPlace[0 ,:]
                    else              : pt1 = AllArcs.list[L[0]].XYNoPlace[-1,:]

                    if L[3] == 'first': pt2 = AllArcs.list[L[1]].XYNoPlace[0 ,:]
                    else              : pt2 = AllArcs.list[L[1]].XYNoPlace[-1,:]
                    pts = np.array([pt1,pt2]).astype(np.int32)
                    img     = cv2.polylines(img    ,[pts],0,(255,255,255),4*voiesWidth)
                    P.Links=[]
                    P.ModifiedPlace=True


    def link(event,x,y,flags,param):
        '''
        Action when click on link function
        '''
        global listpts
        global P
        global Nimg
        global winsize
        if event == cv2.EVENT_LBUTTONDOWN:
            v = cv2.resize(imglink[ym:yM,xm:xM,:],(winsize,winsize) )[y,x,0]
            if v!=0:
                listpts.append(int(v-1))
                E= P.Extremities[v-1]
                if E[1].lower()=='first': pt = AllArcs.list[E[0]].XYNoPlace[0,:]
                else            : pt = AllArcs.list[E[0]].XYNoPlace[-1,:]
                Nimg     = cv2.circle(Nimg   ,(int(pt[0]),int(pt[1])),2*voiesWidth,(255  ,255,255),-1)

    def MKDIR(fold):
        try: os.mkdir(fold)
        except BaseException as E : print('no folder created :',E)
######################## INITIALISATION DES DONNEES ############################

### CASES A REMPLIR OBLIGATOIREMENT
'''
imgname  = ADDTHENAMEAS"IMAGE.FORMAT"  # The name of the image we are analyzing
Workfold = Path(ADDTHEFOLDERADRESS)    # The folder the image is stocked
SavedBase =Path(ADDTHEFOLDERADRESS)    # Where the result will be placed
'''
imgname  = "IMG_0021_DxO.png"  # The name of the image we are analyzing
Workfold = Path("D:/These")    # The folder the image is stocked
SavedBase =Path("Results/Gorgones/")    # Where the result will be placed



### CREATION DU DICTIONNAIRE DE PARAMETRES "p" ###
for _ in range(1):

    p={}
    ### PARAMETRES IMAGE
    p['resize']=1.01            # Réduit la dimension de l'image si nécessaire. ça va beaucoup plus vite ainsi, mais on perd en précision. >1 pour réduire
    p['typicallength']=10    # Taille typique (en pixel) des "artefacts", soit les erreurs de binarisation. 10 c'est raisonnable
    p['threshold']=15
    #p['treshold']=127        # Threshold "réappliqué" (meme sur une image déja thresholdé). 127 (la moitiée) c'est bien

    p['Splitexterior']=True  # Permet de prendre les branches qui composent l'enveloppe
    p['Splitall']=True      # Permet aussi de récuperer les toutes branches dans les trous. ça ralenti pas mal
    p['Cdiff']=0.9           # Fait diffuser la courbure pour trouver les vrais extrenums dans le split.
    p['Ndiff']=100           # Pareil

    ### PARAMETRES ARCS
    p['Kmin']= 1.1           # Rapport de taille minimum pour les arcs-impasse. En dessous de 1 ils sont tous gardé, plus c'est grand moins y'en a

    ### PARAMETRE SCORE LINK
    p['refereepoint']=True  # Si True, on considère un point à 'refereedist' fois la distance de la place comme référent, on s'écarte plus quoi
    p['refereedist'] =1.1    # Voir au dessus
    p['delta']=0             # Si !=0 (doit etre <1), déplace un peu le point pour le 'raytracing' dans les calculs de score

    ### CHOIX DU SCORE
    p['ScoreMode']="+" # NOT USED ANYMORE, SCORE ONLY WITH p['crit']
    p['coeffs']   = {'D1'          : 0, # DEPRECIATED
                    'D2'           : 0,
                    'DCenter'      : 0,
                    'Rarc1'        : 0,
                    'Rarc2'        : 0,
                    'inLen'        : 0,
                    'inMinD'       : 0,
                    'inVarD'       : 0,
                    'Ang12'        : 0,
                    'Ang1'         : 0,
                    'Ang2'         : 0,
                    'inDintegrale' : 0,
                    'Outlen'       : 1,
                    'OutDintegrale': 0,}
    p['crit']='Outlen'

    ### SPLIT VISUALISATION
    winsize=1150
    p['voiesWidth']=0.25
    p['rotate'] = False    # Not sur if used anymore ?
    rotate=True;flip = -1  # Same

    for _ in range(1):
        Savefold = Path(str(SavedBase)+imgname.split('.')[0])
        p['image']      =   imgname.split('.')[0]
        p['imageformat']=   imgname.split('.')[-1]
        p['workfold']   = Workfold
        p['savefold']   = Savefold
        MKDIR(SavedBase)
        MKDIR(         str(p['savefold']             ))
        MKDIR(str(Path(str(p['savefold'] /'PicklePRE'))))
        MKDIR(str(Path(str(p['savefold'] /'Pickle'   ))))
        MKDIR(str(Path(str(p['savefold'] /'QGIS'     ))))


### PREPARATION OF THE DATAS BEFORE A SPLIT ###
try :
    print('Try to reload old datas...')
    IMG             = pickle.load(open(str(p['savefold'] / 'PicklePRE/IMG.p'        ),'rb'))
    AllContours     = pickle.load(open(str(p['savefold'] / 'PicklePRE/AllContours.p'),'rb'))
    try:
        AllArcs     = pickle.load(open(str(p['savefold'] / 'PicklePRE/AllArcs1.p'   ),"rb"))
        AllPlaces   = pickle.load(open(str(p['savefold'] / 'PicklePRE/AllPlaces1.p' ),"rb"))
    except BaseException:
        AllArcs     = pickle.load(open(str(p['savefold'] / 'PicklePRE/AllArcs.p'   ),"rb"))
        AllPlaces   = pickle.load(open(str(p['savefold'] / 'PicklePRE/AllPlaces.p' ),"rb"))
    print('loaded !')
except BaseException :
    print('DATA NOT EXISTING YET ! Recreate base datas')
    IMG         =c.Image(p)                               # FIRST IMAGE EXTRACTION
    AllContours =c.Contours(IMG,p)                        # CONTOURS    EXTRACTION
    AllContours.SplitContours(p)                          # CONTOURS    SPLIT
    AllArcs     = c.Arcs(IMG,AllContours,p)               # ARCS        EXTRACTION


    AllPlaces= c.Places(IMG,AllArcs,p)                    # PLACES      EXTRACTION
    AllPlaces=M.PrepareScore(IMG,AllArcs,AllPlaces,p)     # POTENTIALLINK SCORE EXTRACTION

    pickle.dump(AllArcs,      open(str(p['savefold'] / 'PicklePRE/AllArcs.p')    ,"wb"))
    pickle.dump(IMG,          open(str(p['savefold'] / 'PicklePRE/IMG.p')        ,"wb"))
    pickle.dump(AllContours , open(str(p['savefold'] / 'PicklePRE/AllContours.p'),"wb"))
    pickle.dump(AllPlaces   , open(str(p['savefold'] / 'PicklePRE/AllPlaces.p')  ,"wb"))

    for P in AllPlaces.list : P.ThereisnoLink=False
    AllArcs,AllPlaces = M.LinkByScore(IMG,AllArcs,AllPlaces,p)

################################################################################
#################### CORRECTION (REPEAT THIS PART !)############################
################################################################################
for _ in range(1):### SPLIT ###
    voiesWidth=int(np.median([A.Rmean for A in AllArcs.list if A.Usedinvoie])*p['voiesWidth'])
    repair = True
    while repair :
        # ACTUALISATION DES VOIES
        for A in AllArcs.list: A.Usedinvoie=False
        AllVoies = c.Voies(AllArcs,p)
        AllPlaces.AddExtremities(AllArcs,AllVoies,p)

        # CREATION DE LA CARTE
        for _ in range(1):
            # AJOUT DU FOND
            img      = np.zeros((IMG.Y,IMG.X,3),np.uint8)
            img[:,:,0]=IMG.original[:,:,0].T
            img[:,:,1]=IMG.original[:,:,1].T
            img[:,:,2]=IMG.original[:,:,2].T

            imglink = np.zeros((IMG.Y,IMG.X,3),np.uint8) # Image des liens
            # AJOUT DES LIENS
            for i,P in enumerate(AllPlaces.list) :
                if len(P.Links):
                    L = P.Links[0]
                    if L[2] == 'first': pt1 = AllArcs.list[L[0]].XYNoPlace[ 0,:]
                    else              : pt1 = AllArcs.list[L[0]].XYNoPlace[-1,:]
                    if L[3] == 'first': pt2 = AllArcs.list[L[1]].XYNoPlace[ 0,:]
                    else              : pt2 = AllArcs.list[L[1]].XYNoPlace[-1,:]

                    pts = np.array([pt1,pt2]).astype(np.int32)
                    img     = cv2.polylines(img    ,[pts],0,(0         ,              0   ,          0    ),voiesWidth*4)
                    imglink = cv2.polylines(imglink,[pts],0,((i+1) % 255, (i+1) // 255%255 ,(i+1) // (255*255)),voiesWidth*4)

            # AJOUT DES VOIES
            colormax  = pl.cm.jet( np.linspace(0,1,1+len(AllVoies.list) ) )
            for i,V in enumerate(AllVoies.list) :
                pts=np.vstack((  V.XY[:,0], V.XY[:,1])).T.reshape(-1,2).astype(np.int32)
                G = int(colormax[i,1]*255)
                B = int(colormax[i,2]*255)
                R = int(colormax[i,0]*255)
                img = cv2.polylines(img, [pts],0,(R,G,B),voiesWidth)

        # INITIALISATION DE L'INTERFACE
        for _ in range(1):
            d = int(np.amin((IMG.X,IMG.Y))/2)
            xm=int(IMG.Y/2)-d+1
            xM=int(IMG.Y/2)+d-1
            ym=int(IMG.X/2)-d+1
            yM=int(IMG.X/2)+d-1
            exit = False
            cv2.namedWindow(     'image')
            cv2.setMouseCallback('image',split)

        while not exit: # SPLITTING
            # Retournement
            IMGSHOW= img[xm:xM,ym:yM,0:]*1
            if rotate : IMGSHOW=np.rot90(IMGSHOW,k=flip)
            cv2.imshow('image',cv2.resize(IMGSHOW,(winsize,winsize)))
            k = cv2.waitKey(1) & 0xFF

            # Gestion des touches de direction
            for _ in range(1):
                plus = int((xM-xm)*0.1)
                if k == ord('p') :
                    xm +=  plus;
                    xM -=  plus;
                    ym +=  plus;
                    yM -=  plus;

                if k == ord('m') :
                    xm -=  plus
                    xM +=  plus
                    ym -=  plus
                    yM +=  plus

                up    = ord('s')
                left  = ord('d')
                right = ord('q')
                down  = ord('z')

                deltaT= int((xM-xm)*0.1 )
                if k == left  : xm-=deltaT;xM-=deltaT                # left
                if k == up    : ym+=deltaT;yM+=deltaT                # up
                if k == right : xm+=deltaT;xM+=deltaT                # right
                if k == down  : ym-=deltaT;yM-=deltaT                # down
                if k == ord('h') :
                    xm=int(IMG.Y/2)-d+1
                    Xm=int(IMG.Y/2)+d-1
                    ym=int(IMG.X/2)-d+1
                    Ym=int(IMG.X/2)+d-1

                if k == 27 : exit = True;repair = False                 # exit
                if xm<= 0    :  xm=1
                if xM>=IMG.Y :  xM=IMG.Y-1
                if ym<= 0    :  ym=1
                if yM>=IMG.X :  yM=IMG.X-1
        cv2.destroyAllWindows()

    pickle.dump(AllArcs  , open(str(p['savefold'] / 'PicklePRE/AllArcs1.p'),"wb"))
    pickle.dump(AllPlaces, open(str(p['savefold'] / 'PicklePRE/AllPlaces1.p')  ,"wb"))

for _ in range(1):### LINK ###
    # Refresh des voies
    for A in AllArcs.list: A.Usedinvoie=False
    AllVoies = c.Voies(AllArcs,p)
    AllPlaces.AddExtremities(AllArcs,AllVoies,p)

    # Creation de l'image
    for _ in range(1):
        # image orginale en fond
        img = np.zeros((IMG.Y,IMG.X,3),np.uint8)+255
        img[:,:,0]=IMG.original[:,:,0].T
        img[:,:,1]=IMG.original[:,:,1].T
        img[:,:,2]=IMG.original[:,:,2].T
        '''
        img[:,:,0]=REAL[:,:,0].T
        img[:,:,1]=REAL[:,:,1].T
        img[:,:,2]=REAL[:,:,2].T
        '''
        colormax  = pl.cm.jet( np.linspace(0,1,1+len(AllVoies.list) ) )
        for i,V in enumerate(AllVoies.list) :
            pts=np.vstack((  V.XY[:,0], V.XY[:,1])).T.reshape(-1,2).astype(np.int32)
            G = int(colormax[i,1]*255)
            B = int(colormax[i,2]*255)
            R = int(colormax[i,0]*255)
            img = cv2.polylines(img, [pts],0,(R,G,B),voiesWidth)

    # CORRECTION PAR PLACE
    print('Number of place to correct :',len([P for P in AllPlaces.list if (not len(P.Links) and len(P.Extremities)>1 and not P.ThereisnoLink)]))
    for index,P in enumerate(                [P for P in AllPlaces.list if (not len(P.Links) and len(P.Extremities)>1 and not P.ThereisnoLink)]) :
        Nimg = np.copy(img)
        imglink = np.zeros_like(img)
        for i,E in enumerate(P.Extremities):
            # Ajout des cercles
            if E[1].lower()=='first': pt = AllArcs.list[E[0]].XYNoPlace[0,:]
            else            : pt = AllArcs.list[E[0]].XYNoPlace[-1,:]
            Nimg     = cv2.circle(Nimg   ,(int(pt[0]),int(pt[1])),2*voiesWidth,(127+30*i  ,127-50*i,255),-1)
            imglink  = cv2.circle(imglink,(int(pt[0]),int(pt[1])),2*voiesWidth,(i+1,0,0),-1)

        # Coordonnées
        xm = np.amax((int(P.XY[0,0]-IMG.X/10), 0    ))
        xM = np.amin((int(P.XY[0,0]+IMG.X/10),IMG.X ))
        ym = np.amax((int(P.XY[0,1]-IMG.Y/10), 0    ))
        yM = np.amin((int(P.XY[0,1]+IMG.Y/10),IMG.Y ))

        listpts = []
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',link)
        exit = False
        while not exit:

            IMGSHOW= Nimg[ym:yM,xm:xM,:]*1
            cv2.imshow('image',cv2.resize(IMGSHOW,(winsize,winsize)))
            k = cv2.waitKey(1) & 0xFF

            # Gestion des touches de direction
            for _ in range(1):
                deltaT= int((xM-xm)*0.1 )
                deltaZ=  0.05
                if k == ord('p'):
                    xm = xm + IMG.X*deltaZ
                    xM = xM - IMG.X*deltaZ
                    ym = ym + IMG.Y*deltaZ
                    yM = yM - IMG.Y*deltaZ
                if k == ord('m'):
                    xm = xm - IMG.X*deltaZ
                    xM = xM + IMG.X*deltaZ
                    ym = ym - IMG.Y*deltaZ
                    yM = yM + IMG.Y*deltaZ
                if k == ord('z') : xm-=deltaT;xM-=deltaT                # left
                if k == ord('d') : ym+=deltaT;yM+=deltaT                # up
                if k == ord('s') : xm+=deltaT;xM+=deltaT                # right
                if k == ord('q') : ym-=deltaT;yM-=deltaT                # down
                if k == ord('h') : xm = 0;xM = IMG.X;ym = 0;yM = IMG.Y  # home

                xm = int(np.amin((np.amax((0    ,xm)),IMG.X)))
                ym = int(np.amin((np.amax((0    ,ym)),IMG.Y)))
                xM = int(np.amax((np.amin((IMG.X,xM)),0    )))
                yM = int(np.amax((np.amin((IMG.Y,yM)),0    )))

                if k == ord('o'):
                    P.ThereisnoLink=True
                    exit=True
                if k == ord('r'):
                    try:
                        E= P.Extremities[listpts[0]]
                        if E[1].lower()=='first': pt = AllArcs.list[E[0]].XYNoPlace[0,:]
                        else                    : pt = AllArcs.list[E[0]].XYNoPlace[-1,:]
                        Nimg     = cv2.circle(Nimg   ,(int(pt[0]),int(pt[1])),2*voiesWidth,(0  ,0,0),-1)
                        listpts=[]
                    except BaseException:
                        print('No Reinitialisation possible !')
                if k == 27 :
                    exit = True

            # Ajout du lien
            if len (listpts)==2:
                P.Links=[[P.Extremities[listpts[0]][0],
                         P.Extremities[listpts[1]][0],
                         P.Extremities[listpts[0]][1].lower(),
                         P.Extremities[listpts[1]][1].lower()]]
                L=P.Links[0]
                if L[2] == 'first': pt1 = AllArcs.list[L[0]].XYNoPlace[0 ,:]
                else              : pt1 = AllArcs.list[L[0]].XYNoPlace[-1,:]

                if L[3] == 'first': pt2 = AllArcs.list[L[1]].XYNoPlace[0 ,:]
                else              : pt2 = AllArcs.list[L[1]].XYNoPlace[-1,:]
                pts = np.array([pt1,pt2]).astype(np.int32)
                img     = cv2.polylines(img    ,[pts],0,(0,0,0),voiesWidth)

                if  L[2]=='first': AllArcs.list[ L[0] ].FirstLink=[L[1],L[3]]
                else             : AllArcs.list[ L[0] ]. LastLink=[L[1],L[3]]
                if  L[3]=='first': AllArcs.list[ L[1] ].FirstLink=[L[0],L[2]]
                else             : AllArcs.list[ L[1] ]. LastLink=[L[0],L[2]]
                del P.Extremities[np.amax(listpts)]
                del P.Extremities[np.amin(listpts)]
                break
    cv2.destroyAllWindows()

    pickle.dump(AllArcs     , open(str(p['savefold'] / 'PicklePRE/AllArcs1.p'),"wb"))
    pickle.dump(AllPlaces   , open(str(p['savefold'] / 'PicklePRE/AllPlaces1.p')  ,"wb"))

################################################################################
################## CONGRATULATION, YOU CREATED YOUR DATAS, #####################
###################### ANALYZE THEM NOW IN ANALYZE.PY ##########################
################################################################################

#### GENERATION OF A MAP OF THE RESULT #########################################
for _ in range(1):
    for A in AllArcs.list: A.Usedinvoie=False
    AllVoies = c.Voies(AllArcs,p)
    AllPlaces.AddExtremities(AllArcs,AllVoies,p)
    AllArcs, AllVoies = M.UpgradeArcsVoies(IMG,AllArcs,AllVoies)

    for _ in range(1):### CARTE COMPLETE
        voiesWidth=int(np.median([A.Rmean for A in AllArcs.list if A.Usedinvoie])*p['voiesWidth'])
        ### MAP WITH IMAGE
        img = np.zeros((IMG.X,IMG.Y,3),np.uint8)+255
        img[:,:,0]=IMG.original[:,:,0]
        img[:,:,1]=IMG.original[:,:,1]
        img[:,:,2]=IMG.original[:,:,2]
        colormax  = pl.cm.jet( np.linspace(0,1,1+len(AllVoies.list) ) )
        for i,V in enumerate(AllVoies.list) :
            pts=np.vstack((  V.XY[:,1], V.XY[:,0])).T.reshape(-1,2).astype(np.int32)
            G = int(colormax[i,1]*255)
            B = int(colormax[i,2]*255)
            R = int(colormax[i,0]*255)
            img = cv2.polylines(img, [pts],0,(R,G,B),voiesWidth)
        cv2.imwrite(str(p['savefold'] / 'VoiesMAP.jpg'),img)

#### BASIC GENERATION OF QGISDATA ##############################################
Q.AllQGISsimple(str(Path(str(p['savefold'] /'QGIS/'))),AllArcs,AllContours,AllPlaces,AllVoies,p)
