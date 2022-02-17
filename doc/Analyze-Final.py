######################### IMPORTATIONS #########################################
for _ in range(1): # Permet simplement de faire un repliement
    ### PYSKELFRAC ###
    import PySkelFrac.classes as c       ### All Objects and their properties
    import PySkelFrac.Miscfunc as M      ### Most functions I coded
    import PySkelFrac.QGISsave as Q      ### Save datas as QGIS SHAPE
    import PySkelFrac.Addproperties as AdP
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

    print('Importation Bibliothèque Finie')

    import warnings
    warnings.filterwarnings("ignore")
    print('warning ignored !')

for _ in range(1): # fonctions en plus
    def MKDIR(fold):
        try: os.mkdir(fold)
        except BaseException as E : print('folder creation error :',E)

    from scipy.optimize import curve_fit
    def split(event,x,y,flags,param):
        global AllPlaces
        global AllArcs
        global message
        global img
        global voiesWidth
        if event == cv2.EVENT_LBUTTONDOWN:
            a,b,c = cv2.resize(np.rot90(imglink[xm:xM,ym:yM,:],k=flip),(winsize,winsize))[y,x,:]
            if a + b*255 + c* 255 *255 -1>=0:
                P=AllPlaces.list[a + b*255 + c* 255 *255 -1]
                if not P.ModifiedPlace:
                    print(a + b*255 + c* 255 *255 -1 )
                    L =P.Links[0]
                    ### AJOUTE DES EXTREMITES
                    P.Extremities.append([L[0],L[2]])
                    P.Extremities.append([L[1],L[3]])

                    ### AJOUT DE l'INFO DANS LES ARCS
                    A = AllArcs.list[L[0]]
                    if L[2]=='first': A.FirstLink= ['Extremity']
                    else :                  A.LastLink = ['Extremity']

                    A = AllArcs.list[L[1]]
                    if L[3]=='first': A.FirstLink= ['Extremity']
                    else :                  A.LastLink = ['Extremity']

                    ### SUPPRESSION DU LIEN
                    if L[2] == 'first': pt1 = AllArcs.list[L[0]].XYNoPlace[0 ,:]
                    else              : pt1 = AllArcs.list[L[0]].XYNoPlace[-1,:]

                    if L[3] == 'first': pt2 = AllArcs.list[L[1]].XYNoPlace[0 ,:]
                    else              : pt2 = AllArcs.list[L[1]].XYNoPlace[-1,:]
                    pts = np.array([pt1,pt2]).astype(np.int32)
                    img     = cv2.polylines(img    ,[pts],0,(255,255,255),4*voiesWidth)

                    P.Links=[]
                    P.ModifiedPlace=True

    def nothing(x):    pass
    def restart(x):    exit = True

    def dist(XY):
        return np.sum(np.sqrt( (XY[:-1,0]-XY[1 :,0])**2 +
                               (XY[:-1,1]-XY[1 :,1])**2 ))
    def pow2(x,a,b):
        return a*x**b
    def pow3(x,a,b):
        return a-x*b


    def Normalizescore(AllPlaces,p):
        p['maxval'] = {}                                                     # Creation of a dictionnary containing only biggest values
        for key, value in p['coeffs'].items(): p['maxval'][key] = 0          # Initialisation with all the Keys

        for i,P in enumerate([P for P in AllPlaces.list if len(P.Extremities)>=2]):
            for key,val in P.Criteria.items(): p['maxval'][key]=np.amax((p['maxval'][key],np.amax(np.abs(val))))    # Actualisation of the value
        for P in [P for P in AllPlaces.list if len(P.Extremities)>=2]:                                                  # Application
            for key,val in P.Criteria.items(): P.Criteria[key]=np.array(P.Criteria[key])/p['maxval'][key]
        return(AllPlaces)

    def link(event,x,y,flags,param):
        global listpts
        global P
        global Nimg
        global winsize
        if event == cv2.EVENT_LBUTTONDOWN:
            v = cv2.resize(imglink[ym:yM,xm:xM,:],(winsize,winsize) )[y,x,0]
            print(v,listpts)
            if v!=0:
                listpts.append(int(v-1))
                E= P.Extremities[v-1]
                if E[1].lower()=='first': pt = AllArcs.list[E[0]].XYNoPlace[0,:]
                else            : pt = AllArcs.list[E[0]].XYNoPlace[-1,:]
                Nimg     = cv2.circle(Nimg   ,(int(pt[0]),int(pt[1])),2*voiesWidth,(255  ,255,255),-1)

    from matplotlib.collections import LineCollection
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

for _ in range(1): # Fonctions locales

    def PaulHist(Val,CMAP='jet',MIN=False, MAX=False ,NBINS=False,MODE='lin',normed=False):
        if MIN  ==False : MIN=np.amin(Val)
        if MAX  ==False : MAX=np.amax(Val)
        if NBINS==False : NBINS=20

        if MODE=='log':
            bins = np.logspace(np.log10(MIN),np.log10(MAX),NBINS)
            bins2= 0.5 * (bins[:-1] + bins[1:])
            col = np.log10(bins2) -np.log10(np.amin(bins2))
        if MODE=='lin':
            bins = np.linspace(         MIN ,         MAX ,NBINS)
            bins2= 0.5 * (bins[:-1] + bins[1:])
            col = bins2 -np.amin(bins2)
        n,bins,patches=plt.hist(Val,bins=bins,normed=normed)

        col /= np.amax(col)
        cm=plt.cm.get_cmap('jet')
        for colors, rect in zip(col, patches): plt.setp(rect, 'facecolor', cm(colors))
        return(n,bins2)
    def afflaw(x,a,b): return b-a*x
    def powlaw(x,a,b): return b*x**(-a)
    def lognorm3(x,mu,sigma):
        a=-1/(sigma**2)
        b=mu/sigma
        c=-mu**2 / (2*sigma**2)
        return a*x**2+b*x+c
    from scipy.optimize import fmin
    from scipy.interpolate import CubicSpline
    from scipy.interpolate import interp1d
    def SfromXY(XY):
        dX = XY[:,0]-np.roll(XY[:,0],-1)
        dY = XY[:,1]-np.roll(XY[:,1],-1)
        dS=np.sqrt(dX**2+dY**2)
        XY2=np.array([xy for i,xy in enumerate(XY) if dS[i]!=0 ])
        dS= [ s for s in dS if s!=0]
        S = [np.sum(dS[:i]) for i in range(len(dS)-1) ]
        return(XY2,S)
    def curvfrom(X,Y):
        dX  = np.gradient(X)
        dY  = np.gradient(Y)
        ddX = np.gradient(dX)
        ddY = np.gradient(dY)
        Curv=(dX*ddY - dY*ddX ) / (dX**2 + dY**2)**(1.5)
        return Curv
    from scipy.misc import comb
    def bernstein_poly(i, n, t):
        """
         The Bernstein polynomial of n, i as a function of t
        """

        return comb(n, i) * ( t**(n-i) ) * (1 - t)**i
    def bezier_curve(points, nTimes=1000):
        """
           Given a set of control points, return the
           bezier curve defined by the control points.

           points should be a list of lists, or list of tuples
           such as [ [1,1],
                     [2,3],
                     [4,5], ..[Xn, Yn] ]
            nTimes is the number of time steps, defaults to 1000
            See http://processingjs.nihongoresources.com/bezierinfo/
        """
        nPoints = len(points)
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])

        t = np.linspace(0.0, 1.0, nTimes)

        polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

        xvals = np.dot(xPoints, polynomial_array)
        yvals = np.dot(yPoints, polynomial_array)

        return xvals, yvals
    def fitbez(Pts):
        Xb,Yb = bezier_curve(Pts.reshape(-1,2),len(Xs))
        return(np.mean((Xb-Xs)**2+(Yb-Ys)**2))
    def fitfast(Pts2,debut,fin):

        Pts2 = np.vstack(( debut,np.array(Pts2).reshape(-1,2) ,fin))
        Xb,Yb = bezier_curve(Pts2,len(Xs))
        #CbezExtr  = curvfrom(Xb      ,Yb     )

        return(np.mean( ((Xb-Xs)**2+(Yb-Ys)**2)))#[2:-2]*(1+te*np.abs(CbezExtr[2:-2])*ArcMed**2   )))
    def fitSpline(XYs):
        XY2=np.array(XYs).reshape(2,-1).T
        XY2,s=SfromXY(XY2)
        Xs = XY2[:,0]
        Ys = XY2[:,1]
        fx2 = CubicSpline(s,Xs[:-1]) # FONCTION D'INTERPOLATION
        fy2 = CubicSpline(s,Ys[:-1])
        Xs2=fx2(S)
        Ys2=fy2(S)
        return(np.mean((Xs2-XY[:-1,0])**2+(Ys2-XY[:-1,1])**2))

################################################################################
######################## INITIALISATION DES DONNEES ############################
################################################################################
p={}
### NOM DE L'IMAGE ###
type = 'Gorgones'#'Cracks'          # WHICH FOLDER
Workfold = Path('/media/pa/Storage/Paul_VALCKE_THESE_FULL/0--IMAGES/QUALITYSQUEL/'+type)
FOLDS = [f for f in os.listdir('/home/pa/Bureau/FOOOOOOOOOORME/Results/'+type) if ('.' not in f and 'QGIS' not in f)] # To get the list with every images
SavedBase ='Results/'+type+'/'
im="Cochon-Moyenne"#'10'
im='11'
Dic={}
#Dic =  pickle.load(open('DIC-B.p','rb'))


### CHARGEMENT DE L'IMAGE ######################################################
#for _ in range(1):
for im in FOLDS:
    FigureFold='FiguresFINAL/'+type+'/'+im
    #QGISFold =  'ResultsQGIS/'+type+'/'+im+'/'
    #MKDIR(QGISFold)
    MKDIR(FigureFold)
    #MKDIR('ResultsQGIS/'+type+'/'+im)

    print(30*'###')
    print(im)
    print(30*'###')


    for _ in range(1): #UPDATED LOADING
        AllContours = pickle.load(open(str(SavedBase+im+ '/Pickle/AllContoursFINAL.p'),'rb'))
        IMG         = pickle.load(open(str(SavedBase+im+ '/PicklePRE/IMG.p'     ),'rb'))
        AllArcs     = pickle.load(open(str(SavedBase+im+ '/Pickle/AllArcsFINAL.p'    ),'rb'))
        AllPlaces   = pickle.load(open(str(SavedBase+im+ '/Pickle/AllPlacesFINAL.p'  ),'rb'))
        AllVoies    = pickle.load(open(str(SavedBase+im+ '/Pickle/AllVoiesFINAL.p'   ),'rb'))


    for _ in range(0):
        ### ECHELLES SPATIALES
        p['Pied']=divmod(np.argmax(IMG.dist),IMG.Y)
        Xx,Yy=np.mgrid[-1:IMG.X-1,-1:IMG.Y-1]
        IMG.FootDist=np.sqrt( (Xx - p['Pied'][0]) ** 2 + (Yy - p['Pied'][1]) ** 2 ).astype(np.uint16)
        AngMap=np.arctan2(-Yy+ p['Pied'][1],-Xx+ p['Pied'][0])

        AllAR    = [ A.Rmean        for A in AllArcs.list  if (A.lengthBubble>0 and A.Usedinvoie)]
        AllALens = [ A.lengthBubble for A in AllArcs.list  if (A.lengthBubble>0 and A.Usedinvoie)]
        ArcMed= np.median(AllALens)
        AllALens/=ArcMed
        p['branchwidth']=np.median(AllAR)
        ORDR = np.argsort([V.lengthBubble for V in AllVoies.list])[::-1]
        AllVoies.ORDR=ORDR
        Lens = np.array([ V.lengthBubble for V in AllVoies.list ])/ArcMed
        NArcs= np.array([ len(V.Arc)     for V in AllVoies.list ])
        sizes = [A.Rmax for A in AllArcs.list]
        order = np.argsort(sizes)
        Lmed=ArcMed
        Rmed=np.median(AllAR)
        delta = Rmed*2/3
        millimeter = 2*Rmed

        if im.lower() == "cochon-canonique":
            delta=0
            millimeter=3*Rmed
        for _ in range(1): # COLLECTION OF WAY, FASTER FOR PLOT
            Lines_Voies=[]
            for A in AllVoies.list:
                Lines_Voies+=list(zip(list(zip(A.XY[0:-1,0],A.XY[0:-1,1]))
                                     ,list(zip(A.XY[1:  ,0],A.XY[1:  ,1]))))


    ############## DEGREE MAP ##################################################
    for _ in range(0):
        #DEGREE MAP
        for _ in range(1):
            img = np.zeros((IMG.Y,IMG.X,3),np.uint8)
            #img=np.zeros((5000,5000,3),np.uint8)
            img+=255

            lab= AllContours.labmax
            C=AllContours.list[lab]

            cols  = plt.cm.jet_r(np.linspace(0,1,100*np.log10(np.amax(NArcs))+1))

            for iv in C.Voies :
                V=AllVoies.list[iv]
                pts=np.vstack((  V.XY[:,0], V.XY[:,1])).T.reshape(-1,2).astype(np.int32)
                ang = int(100*np.amax((np.log10(len(V.Arc)),0)))
                G = int(cols[ang ,1]*255)
                B = int(cols[ang ,2]*255)
                R = int(cols[ang ,0]*255)
                img = cv2.polylines(img, [pts],0,(0,0,0),30)
                img = cv2.polylines(img, [pts],0,(R,G,B),20)

            for i,V in enumerate(AllVoies.list) :
                    pts=np.vstack((  V.XY[:,0], V.XY[:,1])).T.reshape(-1,2).astype(np.int32)
                    ang = int(100*np.amax((np.log10(len(V.Arc)),0)))
                    G = int(cols[ang ,1]*255)
                    B = int(cols[ang ,2]*255)
                    R = int(cols[ang ,0]*255)

                    img = cv2.polylines(img, [pts],0,(R,G,B),10)
            C=AllContours.list[0]
            pts=np.vstack((  C.XY[:,0], C.XY[:,1])).T.reshape(-1,2).astype(np.int32)
            img = cv2.polylines(img, [pts],0,(0,0,0),20)
            cv2.imwrite(FigureFold+'/'+im+'Arcs.png',img)
            print('Degree Map')

        # DEGREE DISTRIBUTION
        for _ in range(1):
            try:
                plt.figure('',figsize=(10,10))
                plt.subplot(121)
                MAX=np.amax(NArcs)
                plt.hist(NArcs,bins = np.logspace(0,np.log10(MAX),10 ),label='Full Distribution')
                n,bins=np.histogram(NArcs,bins = np.logspace(0,np.log10(MAX),10 ))
                bins2 = (bins[:-1]+bins[1:])/2

                ####
                ret=curve_fit(afflaw,np.log10(bins2[n!=0]),np.log10(n[n!=0]),p0=(2,np.log10(np.amax(n) )))
                plt.plot(bins2[n!=0][:],10**afflaw( np.log10(bins2[n!=0][:]),ret[0][0],ret[0][1]), label ='  a = '+str( int(ret[0][0]*10)/10)+' $\pm$ '+str(int(100*ret[1][0][0])/100))
                #plt.plot(bins2[n!=0],10**afflaw(bins2[n!=0],ret0[0][0],ret0[0][1]))

                plt.xlabel('Degree')
                plt.ylabel('Population')
                plt.legend()
                ####
                plt.ylim([1,np.amax(n)])
                plt.yscale('log');plt.xscale('log')
                plt.axis('scaled')
                plt.ylim(bottom=1)

                print('PENTE DEGREE FULL :','  a = '+str( int(ret[0][0]*10)/10)+' $\pm$ '+str(int(100*ret[1][0][0])/100))
                ########################################################################
                #plt.subplot(122)
                NArcs2=[ len(AllVoies.list[iv].Arc) for iv in AllContours.list[AllContours.labmax].Voies ]
                MAX=np.amax(NArcs)
                plt.hist(NArcs2,bins = np.logspace(0,np.log10(MAX),10 ),label='Growing ways')
                n,bins=np.histogram(NArcs2,bins = np.logspace(0,np.log10(MAX),10 ))
                bins2 = (bins[:-1]+bins[1:])/2

                ####
                ret =curve_fit(powlaw,         bins2[n!=0][:] ,         n[n!=0][:] ,p0=(2,         np.amax(n)  ))
                ret =curve_fit(afflaw,         np.log10(bins2[n!=0][:]) ,         np.log10(n[n!=0][:]) ,p0=(2,         np.log10(np.amax(n))  ))
                plt.plot(bins2[n!=0][:],10**afflaw( np.log10(bins2[n!=0][:]),ret[0][0],ret[0][1]), label ='  a = '+str( int(ret[0][0]*10)/10)+' $\pm$ '+str(int(100*ret[1][0][0])/100))
                #plt.plot(bins2[n!=0],10**afflaw(bins2[n!=0],ret0[0][0],ret0[0][1]))

                plt.legend()
                plt.xlabel('Degree')
                plt.ylabel('Population')
                ####
                plt.ylim([1,np.amax(n)])
                plt.yscale('log');plt.xscale('log')
                plt.axis('scaled')
                plt.ylim(bottom=1)

                print('PENTE DEGREE OUT  :','  a = '+str( int(ret[0][0]*10)/10)+' $\pm$ '+str(int(100*ret[1][0][0])/100))
                plt.savefig(FigureFold+'/'+im+'WaysDegree.svg')
                plt.show()
            except BaseException:
                pass

    ############## TREE MAP ####################################################
    for _ in range(0): # Tree Representation
        print('Tree Representation !')
        Lines_ArcStop=[]
        for V in AllVoies.list:
            if len(V.Arc)>1:
                for Ai in V.Arc[:-1] :
                    A = AllArcs.list[Ai]
                    Lines_ArcStop+=list(zip(list(zip(A.XYasVoies[0:-1,0],A.XYasVoies[0:-1,1]))
                                           ,list(zip(A.XYasVoies[1:  ,0],A.XYasVoies[1:  ,1]))))

        plt.figure('AsTrees',figsize=(20,20))
        ax=plt.gca()
        ax.add_collection(LineCollection(Lines_ArcStop,color='k',linewidth=2))
        C0=AllContours.list[AllContours.labmax]
        plt.plot(C0.XY[:,0],C0.XY[:,1],c='k',lw=.1)
        plt.axis('scaled')
        plt.axis('off')
        plt.subplots_adjust(left = 0.,right = 1  ,bottom = 0  ,top = 1  ,wspace = 0,hspace = 0)
        plt.savefig(FigureFold+'/AsTrees.png',dpi=100)
        plt.show()

    ############## PLUMES ######################################################
    NbWays=int(len(AllVoies.list)/50)
    for _ in range(0):
        plt.figure('',figsize=(20,20));ax=plt.gca()
        ax=plt.gca()
        #ax.add_collection(LineCollection(Lines_Contours,color='k',linewidth=.51))
        #ax.add_collection(LineCollection(Lines_Voies,color='k',linewidth=.51))
        clr= plt.cm.jet(np.linspace(0,1,NbWays+1))

        C=AllContours.list[0];plt.plot(C.Y,C.X,c='k')
        for i in np.arange(NbWays)[::-1]:
            V=AllVoies.list[ORDR[i]]
            plt.plot(     V.XY[ :,1],V.XY[ :,0],c=clr[i,:],lw=3)
            for V2 in [AllVoies.list[j] for j in V.VoiesLink if AllVoies.list[j].lengthBubble<0.5*V.lengthBubble]:
                plt.plot(V2.XY[ :,1],V2.XY[:,0],'--',c=clr[i,:],lw=1)


        for i in range(NbWays):
            V=AllVoies.list[ORDR[i]]
            plt.plot(     V.XY[ 0,1],V.XY[ 0,0],'.',ms=20,c='g')
            if ORDR[i] not in C.Voies:
                plt.plot( V.XY[-1,1],V.XY[-1,0],'.',ms=15,c='r')
        plt.axis('scaled')
        plt.axis('off')
        plt.savefig(FigureFold+'/'+im+'Plumes2.png',dpi=100)
        plt.show()

    ############# HIERARCHY ####################################################
    ### HIERARCHY PER WAY EXTERIOR-ALL  AUTHORIZED)  ###########
    for _ in range(1):
        for V in AllVoies.list : V.HierarchyExtFull=-1

        Voiestoexplore=[[]]
        for Vi in AllContours.list[AllContours.labmax].Voies:
            V=AllVoies.list[Vi]
            V.HierarchyExtFull=0
            Voiestoexplore[0].extend(V.Daughter)
            Voiestoexplore[0].extend(V.Killed)
            if V.Mother : Voiestoexplore[0].append(V.Mother)
            if V.Killer : Voiestoexplore[0].append(V.Killer)

        ### RECCURSION
        Hierarchie=0
        index=0
        while len(Voiestoexplore[index])>0:
            Hierarchie+=1
            Voiestoexplore.append([])
            for Vi in Voiestoexplore[index]:
                V=AllVoies.list[Vi]
                if V.HierarchyExtFull<=0:
                    V.HierarchyExtFull=Hierarchie
                    for V2 in V.Daughter:
                        if AllVoies.list[V2].HierarchyExtFull<=0:
                            Voiestoexplore[index+1].append(V2)
                    for V2 in V.Killed:
                        if AllVoies.list[V2].HierarchyExtFull<=0:
                            Voiestoexplore[index+1].append(V2)
                    if V.Killer :
                        if AllVoies.list[V.Killer].HierarchyExtFull<=0: Voiestoexplore[index+1].append(V.Killer)
                    if V.Mother :
                        if AllVoies.list[V.Mother].HierarchyExtFull<=0: Voiestoexplore[index+1].append(V.Mother)
            index+=1
            if index>40:break
        AllVoies.HierarchyExtFullMax = index*1

    ### HIERARCHY PER WAY (EXTERIOR-MOTHER) ####################
    for _ in range(1):

        for V in AllVoies.list : V.HierarchyExt=-1
        Voiestoexplore=[[]]

        for Vi in AllContours.list[AllContours.labmax].Voies:
            AllVoies.list[Vi].HierarchyExt=0
            Voiestoexplore[0].append(AllVoies.list[Vi].Mother)

        ### RECCURSION
        Hierarchie=0
        index=0
        while len(Voiestoexplore[index])>0:
            print(Hierarchie)
            Hierarchie+=1
            Voiestoexplore.append([])
            for Vi in Voiestoexplore[index]:
                V=AllVoies.list[Vi]
                if V.HierarchyExt<0:
                    V.HierarchyExt=Hierarchie*1
                    if AllVoies.list[V.Mother].HierarchyExt<=0:
                        Voiestoexplore[index+1].append(V.Mother)
            index+=1
            if index>40:break
        AllVoies.HierarchyExtMax = index*1
        HierarchyExtFull=[V.HierarchyExtFull for V in AllVoies.list]
        Length= [len(V.Arc) for V in AllVoies.list]

    for _ in range(0):# LA CARTE
        plt.figure('',figsize=(10,10))

        #ax=plt.axes([0.05,0.25,.9,.7])
        col=plt.cm.jet(np.linspace(0,1,AllVoies.HierarchyExtFullMax+2))
        C0=AllContours.list[AllContours.labmax]
        #plt.plot(P.XY[:,0],P.XY[:,1],"*",ms=10,c='k')
        plt.plot(C0.XY[:,1],-C0.XY[:,0],c='k')
        for V in AllVoies.list :
            if V.HierarchyExtFull>=0:
                if V.HierarchyExt>=0: plt.plot(V.XY[:,1],-V.XY[:,0],color=col[V.HierarchyExtFull,:],lw=3)
                else :                plt.plot(V.XY[:,1],-V.XY[:,0],color=col[V.HierarchyExtFull,:])
            else:                  plt.plot(V.XY[:,1],-V.XY[:,0],c='k',lw=.1)
        plt.axis('off')
        plt.axis('scaled')
        plt.savefig(FigureFold+'/'+im+'Hierarchie-ExtFull.png',dpi=150)
        plt.show()

    for _ in range(1):#LA STAT###
        '''
        plt.figure('Distrib',figsize=(10,10))
        ax=plt.gca()
        Xval =np.array(HierarchyExtFull)
        Yval =np.array(Length)

        binsX=np.linspace(-1.5,np.amax(Xval)+.5,num=np.amax(Xval)+3)
        binsY=np.linspace(0,np.amax(Yval),num=20)
        X, Y = np.meshgrid(binsX[1:], binsY[1:])

        H,binsX,binsY = np.histogram2d(Xval,Yval,bins=(binsX,binsY))

        Stat = [np.sum(H[i,:]) for i in range(len(H[:,0]))]

        plt.pcolormesh(X[:,:],Y[:,:],H[:,:].T,vmin=1,cmap='viridis',
                   norm=mpl.colors.LogNorm(vmin=1, vmax=np.amax(H[:,:])))
        plt.colorbar()
        ax.set_xlabel('HierarchyNumber (Exterior Full)')
        ax.spines['left'].set_color('green')
        ax.yaxis.label.set_color('blue')
        ax.tick_params(axis='y', colors='green')
        ax.set_ylabel('Degree')
        ax2=plt.twinx(ax)
        ax2.plot(1+(binsX[1:-1]+binsX[:-2])/2,Stat[:-1],'*',ms=10,c='r',label='Experiment')
        ax2.set_ylabel('Hierarchy Population')
        Cs = len(Xval)
        N0 = 2*Cs/np.sqrt(1+np.log2(Cs))
        mu = 2 + (np.log2(Cs)-2)/6
        sigma=.25*np.sqrt(np.log2(Cs)+1)

        x = np.linspace(0,binsX[-1],100)
        Ni=Cs/(sigma*np.sqrt(2*np.pi))*np.exp( -((x+1-mu)**2) /(2*sigma**2))
        ax2.plot(x,Ni,'--',c='k',lw=2,label='Theory')
        ax2.yaxis.label.set_color('red')
        plt.savefig(FigureFold+'/FullExtHierarchy.svg')
        plt.show()
        '''

        Val =np.array(HierarchyExtFull)
        ### TRACE DE L'HISTOGRAMME
        plt.figure('',figsize=(10,5))
        bins = np.linspace(-1.5,np.amax(Val)+.5,num=np.amax(Val)+3)
        bins2= 0.5 * (bins[:-1] + bins[1:])
        col = bins2 -np.amin(bins2)
        n,bins,patches=plt.hist(Val,bins=bins)

        col /= np.amax(col)
        cm=plt.cm.get_cmap('jet')
        for colors, rect in zip(col, patches): plt.setp(rect, 'facecolor', cm(colors))

        ### TRACE DE LA GAUSSIENNE
        x = np.linspace(-1.5,np.amax(HierarchyExtFull)+.5,100)
        Cs = len(Val)
        N0 = 2*Cs/np.sqrt(1+np.log2(Cs))
        mu = 1+ (np.log2(Cs)-2)/6
        sigma=.25*np.sqrt(np.log2(Cs)+1)
        Ni=Cs/(sigma*np.sqrt(2*np.pi))*np.exp( -((x+1-mu)**2) /(2*sigma**2))
        plt.plot(x,Ni,'--',c='k',lw=2,label='Theory')



        plt.xlabel('Hierarchy number')
        plt.savefig(FigureFold+'/'+im+'Hierarchie-ExtStat.png',dpi=150)
        plt.show()










    plt.close('all')
    del IMG, AllContours,AllPlaces,AllVoies, AllArcs
print('Finish !')
