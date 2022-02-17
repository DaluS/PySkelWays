# -*- coding: utf-8 -*-
'''
Wraparound the hub
'''

from ._subclass.loadimage import fimage
from ._subclass.loadbinary import fbinary
from ._subclass.loadcontours import fcontours
from ._subclass.loadarcs import farcs
from ._subclass.loadplaces import fplaces
from ._subclass.loadways import fways


_SUPPORTED_IMAGES = ['jpg', 'jpeg', 'png']
_DPAR = {}


class Hub():
    """
    Most important object for the user.
    The hub is the object containing :
        * Informations on the network
        * Methods to get them
        * Flags to tell you where you are in the process
    """

    def __init__(
            self,
            **kwargs,
    ):

        # Dictionnary of flags to know what has processed
        self.flags = {'image': False,
                      'binarisation': False,
                      'contours': False,
                      'arcs': False,
                      'places': False,
                      'ways': False,
                      }

        # Default values of parameters
        self.dparam = _DPAR
        # UPDATE MORE PARAMETERS IN THE SYSTEM
        for key, value in kwargs.items():
            self.dparam[key] = value

        # Classes initialised as False
        self.Image = False
        self.Contours = False
        self.Arcs = False
        self.Places = False
        self.Ways = False

    @property
    def __repr__(self):
        print('Image_file :', self.Image)
        print('Contours   :', self.Contours)
        print('Arcs       :', self.Arcs)
        print('Places     :', self.Places)
        print('Ways       :', self.Ways)

    def Compute_all(self, img_address, verb=False, **kwargs):
        self.set_dparam(**kwargs)
        self.Load_Image(img_address, kwargs)
        self.Compute_binarisation()
        self.Compute_contours()
        self.Compute_arcs()
        self.Compute_places()
        self.Compute_ways()

    def set_dparam(self, **kwargs):
        '''
        Change the dictionnary of parameters
        '''
        for key, value in kwargs.items():
            self.dpar[key] = value

    def Load_Image(self, img_address, verb=False, **kwargs):
        '''
        Will simply load the image as a matrix

        useful parameters :
            *
        '''

        # CHECK : REQUIRE THE NAME OF A FILE
        if img_address is False:
            raise Exception("Please provide an address to find the image !")
        if img_address.split('.')[-1] not in _SUPPORTED_IMAGES:
            raise Exception(
                f"""The format must be in {_SUPPORTED_IMAGES}, you have given:
                    {img_address.split('.')[-1]}""")

        # UPDATE MORE PARAMETERS IN THE SYSTEM
        self.set_dparam(**kwargs)
        self.dparam['img_address'] = img_address

        # LOAD THE IMAGE
        fimage.load(self)

        # Waive a new flag
        self.flags['image'] = True

    # %% COMPUTE OPERATIONS ##################################################
    def Compute_binarisation(self, verb=False, **kwargs):
        """
        Transform initial image into a binarised one

        useful parameters :
            *


        """
        # Check
        if not self.flags['image']:
            raise Exception(""""Tried to compute binarisation without image.
                            Load an image first !""")

        # UPDATE MORE PARAMETERS IN THE SYSTEM
        self.set_dparam(**kwargs)

        # Compute operations
        fbinary.load(self)

        # Waive a new flag
        self.flags['binarisation'] = True

    def Compute_contours(self, verb=False, **kwargs):
        """
        Calculate all contours of the binarised image and their typical properties

        useful parameters :
            *


        """
        # Check
        if not self.flags['image']:
            raise Exception(""""Tried to compute binarisation without image.
                            Load an image first !""")

        # UPDATE MORE PARAMETERS IN THE SYSTEM
        self.set_dparam(**kwargs)

        # Compute operations
        fcontours.load(self)

        # Waive a new flag
        self.flags['contours'] = True

    def Compute_arcs(self, verb=False, **kwargs):
        """
        Calculate all arcs (skeleton) given the contours and the binarised image

        useful parameters :
            *


        """
        # Check
        if not self.flags['image']:
            raise Exception(""""Tried compute_contours without binarised image.
                            use `hub.Compute_contours' first !""")

        # UPDATE MORE PARAMETERS IN THE SYSTEM
        self.set_dparam(**kwargs)

        # Compute operations
        farcs.load(self)

        # Waive a new flag
        self.flags['arcs'] = True

    def Compute_places(self, verb=False, **kwargs):
        """
        Calculate all places (arcs intersection) given the arcs and the binarised image

        useful parameters :
            *


        """
        # Check
        if not self.flags['arcs']:
            raise Exception(""""Tried to compute places with no arcs.
                            use `hub.Compute_arcs' first !""")

        # UPDATE MORE PARAMETERS IN THE SYSTEM
        self.set_dparam(**kwargs)

        # Compute operations
        fplaces.load(self)

        # Waive a new flag
        self.flags['places'] = True

    def Compute_score(self, verb=False, **kwargs):
        """
        Calculate all scores (arcs associations) given the places and arcs

        useful parameters :
            *


        """
        # Check
        if not self.flags['places']:
            raise Exception(""""Tried to score with no places.
                            use `hub.Compute_places' first !""")

        # UPDATE MORE PARAMETERS IN THE SYSTEM
        self.set_dparam(**kwargs)

        # Compute operations
        self.Places.compute_score(**kwargs)

        # Waive a new flag
        self.flags['score'] = True

    def Compute_ways(self, verb=False, **kwargs):
        """
        Calculate all ways as given by score of arcs associations

        useful parameters
            *


        """
        # Check
        if not self.flags['score']:
            raise Exception(""""Tried to compute ways with no scores.
                            use `hub.Compute_score' first !""")

        # UPDATE MORE PARAMETERS IN THE SYSTEM
        self.set_dparam(**kwargs)

        # Compute operations
        fways.load(self)

        # Waive a new flag
        self.flags['ways'] = True
