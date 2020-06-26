# What is PySkelWays ?

PyskelWays takes an image as input, and gives an hyper-graph of ways in multiple python objects as an output, that can be connected to other softwares such as QGIS.
Ways are an efficient element of description for networks corresponding to a branch growth mechanisms. 

More informations about this in chapter 3 and 4 of my thesis here : (https://frama.link/PaulVThesisHD for the HD version or https://frama.link/PaulVThesisCompressed if you want to save data). A lot of illustrations and detail on how it works are there (Chapter 3) and typical properties (Chapter 4)

I developed this to study the shape of *Gorgonia ventalina*, but it can be used on many different systems

# How to use PySkelWays

## Needed libraries

The library needed are :
- Numpy and scipy for numerical storage and calculation
- cv2 (openCV) for image analysis treatment
- os,pickle,copy,pathlib for storage and adresses management
- matplotlib for figure plot
 shapefile (pyshp) for conversion in GIS datas


## The software

PySkelWays is the python software we have created for image analysis of spatial reticulated networks. 

The input is an image (binarized or cleaned for a fast binarization), and the output is a serie of classes objects which contains the different elements, in python. The output can also be vizualized with QGIS, on the informations we have selected. 

The version we worked with (0.99) is named *PySkelFrac*, and the released version PySkelWays. It is possible that datas extracted previously require that the library is renamed PyskelFrac in consequence to load the different classes. 

All codes are made to be userfriendly using an IDE based on jupyter (interactive python) such as Anaconda or Atom. The two required functions for a better readability are : 


- code folding : there is a lot of "for $\_$ in range(1):" which are only here to fold sections
- Partial execution, to execute relevant part of the code and not everything (although possible)


## Composition of the library

It is composed of two type of files :


- The library itself *Pyskelways*
- The executable code **Extract.py**, and **Analyze.py**


The file **Extract.py** is the code to create the new classes, generate the ways and correct the links. The file **Analyze.py} contains all the tools to generate the graphs and the statistics on the different objects.

The library is composed of the files : 


- **Classes.py** containing all the classes properties
- **MiscFunc.py** containing all the functions we have created for network generation
- **Addproperties.py** containing the functions to add properties on a cleaned network
- **QGISsave.py** containing the function to generate QGIS files (.shp, .shx,.dbf)


The code began as a class-oriented code but has been mostly developped using functions. 

To scan the properties of objects, it is recommended to use the function (M.dirP) on the selected object, which allow the scan of all their properties. 

Objects are save as pickle files (.p) and can be loaded using pickle.load, and save using pickle.dump function. 

## Folders and locations

*PySkelWays* is not exactly coded as a library, as it is not an installation with pip or conda, with an absolute path to the code when used. 
It is mandatory that the folder PySkelWays containing all the code is located in the same folder as "extract.py" and "analyze.py". 

There are Four different location of the other files : 

- **Workfold** : location of the image
- **SavedBase/Savedfold**: location of the pickle file
- **FigureFold**: where the figures are saved
- **QGISFold** : Where the shapefile are saved


## Objects

The objects are : 

- **IMG**, it contains all the map and general information on the gorgonian. 
- **AllContours**, which contains all the different contours/holes and the envelope. All the different contours are stocked in **AllContours.list}
- **AllArcs**, with the same approach as AllContours but on Arcs
- **AllPlaces**, same approach
- **AllVoies**, same approach. We used the french name of "Voies" instead of ways.


Usually, ways are regenerated from the information on Arcs and places at the beginning of each analyze, and thus not stocked as such. 


All the parameters are stocked inside a file "parameters.p", which is generated at the beginning of a **Extraction.py} run. Most elements are accessible inside the code. 

## Generation of the objects

The code **Extraction.py** function as follow : 


- It read the image, and extract the binarisation, distance image and so on in **c.Image**
- It generate from the binarized image all the contours in **c.Contours**
- It generate the arcs in** c.Arcs**
- It generate the places in **c.Places**
- It calculate all the basic elements for place and link score in **M.Preparescore}
- It generate all the scores in** M.LinkByScore**
- It creates the ways in **c.Voies**
- It add the extremities in **AllPlaces.AddExtremities** (which impact also AllVoies).


After this is done, the interface for Ways split and Link manual corrections is created, and this can be done as many time as needed. Each time, it regenerate AllPlaces and AllArcs with the new corrections. 

When this is done, it removes unnecessary points with **M.UpgradeArcsVoies** for a lighter result. 

At this time, there is little information interaction between elements. 

Ways are reoriented, and all the connection between elements (local hierarchy, which contour is next to which arc and ways and so on) is generated by **Adp.NewAssociatedContours**

Some new properties are generated inside **Analyze.py** for the need of the analyzes.
