# Matplotlib packages to import
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import ImageGrid

# Used for plotting cases only
import seaborn as sns

# Obspy librabries
import obspy
from obspy import Stream
from obspy.core import UTCDateTime

# Standard Libraries 
import numpy as np
import pandas as pd
from glob import glob
import pickle
import math
from math import *
import random
import sys
import json
import copy
from string import digits
from scipy import stats
import time
import copy
from pyproj import Proj


# Pytorch Libraires
import torch
from torch.nn import Linear
from torch import Tensor
from torch.nn import MSELoss
from torch.optim import SGD, Adam, RMSprop
from torch.autograd import Variable, grad
from torch.utils.data.sampler import SubsetRandomSampler,WeightedRandomSampler
from torch.cuda.amp import autocast


# Sklearn libraries
from sklearn.cluster import DBSCAN
# Suppressing the warning 
pd.options.mode.chained_assignment = None  # default='warn'

class RBF(torch.nn.Module):

    """Radial Basis Function (RBF)

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. Alternatively, attributes may be documented
    inline with the attribute's declaration (see __init__ method below).

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.

    Args:
        sigma (:obj:`bool`, optional): Radial Basis Function Sigma Value 

    Attributes:
        sigma       (:obj:`float`): Radial Basis Function Sigma Value
        print_sigma (:obj:`bool`, optional): Description of `attr2`.

    """


    def __init__(self, sigma=None):
        super(RBF, self).__init__()
        self.sigma = sigma
        self.print_sigma = False

    def forward(self, X, Y):
        """ Forward Pass of the Radial Basis Function



            Args
                X (:obj:`torch.tensor`) - 
                Y (:obj:`torch.tensor`) -

        """


        XX = X.matmul(X.t())
        XY = X.matmul(Y.t())
        YY = Y.matmul(Y.t())

        dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

        # Apply the median heuristic (PyTorch does not give true median)
        if self.sigma is None:
            np_dnorm2 = dnorm2.detach().cpu().numpy()
            h         = np.median(np_dnorm2) / (2 * np.log(X.size(0) + 1))
            sigma     = np.sqrt(h).item()

            if self.print_sigma:
                print(sigma)

        else:
            sigma = self.sigma

        gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
        K_XY = (-gamma * dnorm2).exp()

        return K_XY

def IO_JSON(file,Events=None,rw_type='r'):
    '''
        Reading/Writing in JSON file into location archieve
    '''
    if rw_type == 'w':
        tmpEvents = copy.deepcopy(Events)
    elif rw_type == 'a+':
        tmpEvents = copy.deepcopy(Events)
    elif rw_type == 'r':
        with open(file, 'r') as f:
            tmpEvents = json.load(f)


    for key in tmpEvents.keys():
        if rw_type=='w':
            tmpEvents[key]['Picks']       = tmpEvents[key]['Picks'].astype(str).to_dict()
        elif rw_type=='a+':
            tmpEvents[key]['Picks']       = tmpEvents[key]['Picks'].astype(str).to_dict()
        elif rw_type=='r':
            tmpEvents[key]['Picks']       = pd.DataFrame.from_dict(tmpEvents[key]['Picks'])
        else:
            print('Please specify either "read (r)", "write (w)" or "append (a)" for handelling the data')

    if rw_type == 'w':
        with open(file, rw_type) as f:
            json.dump(tmpEvents, f)
        del tmpEvents
    elif rw_type == 'a+':
        try:
            with open(file, 'r+') as f:
                d = json.load(f)
                d.update(tmpEvents)
                f.seek(0)
        except:
            d = tmpEvents
            print('Creating Appending Catalogue - {}'.format(file))      
        with open(file, 'w') as f:
            json.dump(d, f)
        del tmpEvents,d
    elif rw_type =='r':
        return tmpEvents

# =========== INPUT/OUTPUT FORMAT =======
def IO_NLLoc2JSON(file,EVT={},startEventID=1000000):
    # Reading in the lines
    f = open(file, "r")
    lines = f.readlines()
    lds = np.where(np.array(lines) == '\n')[0] - np.arange(len(np.where(np.array(lines) == '\n')[0]))
    lines_start = np.append([0],lds[:-1])
    lines_end   = lds

    # Reading in the event lines
    evt = pd.read_csv(file,sep=r'\s+',names=['Station','Network','r1','r2','PhasePick', 'r3','Date','Time','Sec','r4','PickError','r5','r6','r7'])
    evt['DT'] = pd.to_datetime(evt['Date'].astype(str).str.slice(stop=4)                         + '/' +
                               evt['Date'].astype(str).str.slice(start=4,stop=6)                 + '/' +
                               evt['Date'].astype(str).str.slice(start=6,stop=8)                 + 'T' +  
                               evt['Time'].astype(str).str.zfill(4).str.slice(stop=2)            + ':' + 
                               evt['Time'].astype(str).str.zfill(4).str.slice(start=2)           + ':' +
                               evt['Sec'].astype(str).str.split('.',expand=True)[0].str.zfill(2) + '.' +
                               evt['Sec'].astype(str).str.split('.',expand=True)[1].str.zfill(2),format='%Y/%m/%dT%H:%M:%S.%f')
    evt = evt[['Network','Station','PhasePick','DT','PickError']]

    # Turning 
    for eds in range(len(lines_start)):
        evt_tmp = evt.iloc[lines_start[eds]:lines_end[eds]]
        EVT['{}'.format(startEventID+eds)] = {}
        EVT['{}'.format(startEventID+eds)]['Picks'] = evt_tmp.reset_index(drop=True)

    return EVT

def IO_JSON2CSV(EVT,savefile=None):
    '''
        Saving Events in CSV format
    '''

    Events = EVT

    # Loading location information
    picks =(np.zeros((len(Events.keys()),8))*np.nan).astype(str)
    for indx,evtid in enumerate(Events.keys()):
        try:
            picks[indx,0]   = str(evtid)
            picks[indx,1]   = Events[evtid]['location']['OriginTime']
            picks[indx,2:5] = (np.array(Events[evtid]['location']['Hypocentre'])).astype(str)
            picks[indx,5:]  = (np.array(Events[evtid]['location']['HypocentreError'])).astype(str)
        except:
            continue
    picks_df = pd.DataFrame(picks,
                            columns=['EventID','DT','X','Y','Z','ErrX','ErrY','ErrZ'])
    picks_df['X']    = picks_df['X'].astype(float)
    picks_df['Y']    = picks_df['Y'].astype(float)
    picks_df['Z']    = picks_df['Z'].astype(float)
    picks_df['ErrX'] = picks_df['ErrX'].astype(float)
    picks_df['ErrY'] = picks_df['ErrY'].astype(float)
    picks_df['ErrZ'] = picks_df['ErrZ'].astype(float)
    picks_df         = picks_df.dropna(axis=0)
    picks_df['DT']   = pd.to_datetime(picks_df['DT'])
    picks_df         = picks_df[['EventID','DT','X','Y','Z','ErrX','ErrY','ErrZ']]

    if type(savefile) == type(None):
        return picks_df
    else:
        picks_df.to_csv(savefile,index=False)


# =========== MAIN =======
class HypoSVI(torch.nn.Module):
    """HypoSVI - Hypocentral Earthquake Location using Stein Variational Inferance and Physics Informed Neural Networks

        Earthquake location class that leverages Physics Informed Neural Networks and Stein Variational Inference to return
        non-gridded earthquake location within continious three-dimensional travel-time models defined using EikoNet.

        Args:
            EikoNet (:obj:`list`): List of trained EikoNet Classes.
            Phases (:obj:`list`, optional): List of the Phases that the EikoNets correspond to. Default ['P','S']
            device (:obj:`list`, optional): Device to run the computation on e.g. 'cpu', 'cuda:0', 'cuda:1'. Default 'cpu'

        Travel-Time and Grid Attributes:
            eikonet_phase (:obj:`list`): List of the Phases that the EikoNets correspond to. Defined in __init__ during setup.
            eikonet_models (:obj:`list`): List of the EikoNet classes. Defined in __init__ during setup.
            VelocityClass (:obj: ' '): Definition of the velocity structure
            projection (:obj: 'Pyproj.Proj'): Definition of the projection converting EikoNet and recovered locations 
                                              from a projection of the users choice to a local UTM coordinate system
            xmin (:obj: 'float'): Definition of the upper S-W point of the region of interest.
            xmax (:obj: 'float'): Definition of the lower N-E point of the region of interest.
            device (:obj: 'string'): evice to run the computation on e.g. 'cpu', 'cuda:0', 'cuda:1'. Default 'cpu'. Defined as input to HypoSVI.


        Earthquake Location Attributes:

            location_info (:obj:`dict`): Dictionary describing all the information used in the inversion for the earthquake location. These include;

                ======> ['Log-likehood'] (:obj:`string`): Default = 'LAP'
                                                    The Log-likehood function used in the inversion. 
                                                    Options: ``LAP`` = Laplacian Differential Time or ``EDT`` Equal Differential Time
                ======> ['Travel Time Uncertainty - [Gradient(km/s),Min(s),Max(s)]'] (:obj:`arr`): Default =[0.1,0.1,2.0]:
                                                    Addition of travel-time uncertainty (s) into the inversion for the earthquake location. 
                                                    The uncertainty added is between values ``Min(s)`` and ``Max(s)``, with the the value changing with travel-time by
                                                    gradient ``Gradient(km/s)``
                ======> ['Individual Event Epoch Save and Print Rate'] (:obj:`arr`): Default = [None,False]
                                                                        Defining whether to output the particle locations of individual epochs for each earthquake location 
                                                                        First entry of the array determines the rate at which to save particle location. e.g. Every 10 epochs = 10. Default = ``None``
                                                                        Second entry of array is a boolean defining if the individual epoch particle locations should be saved. Default = ``None``


                ======> ['Number of Particles'] (:obj:`int`): Default ``150``
                                                              Number of particles that are inverted to determine the posterior of the hypocentral locaition.
                                                              This value may vary depening on the usecase problem as more complex posterior spaces would require
                                                              more particles


                ======> ['Step Size'] (:obj:`float`): Default ``1``
                                                      Stepsize for the update used in the optimization of the particle locations. 


                ======> ['Save every * events'] (:obj:`int`): Default ``100``
                                                              The rate at which to append the events to the save JSON catalogue. 
                                                              If this value is too small then processing could be bottlenecked by appending to file.


                ======> ['Location Uncertainty Percentile (%)']: Default ``99.5``
                                                                 The percentile of the posterior space to be used for the location uncertainty.



            K (:obj:`class`): Kernel function to used in the Stein Variational Gradient Descent. 
                              Default is a Radial Basis Function ``RBF()`` with the variance automatically defined ``K.sigma = None`` and
                              no printing of the changing variance ``K.print_sigma = False``



        Automated Plot Attributes:

            plot_info (:obj:`dict`): Dictionary describing all the information used in the automated plotting of event and catalogue plots. These include;

                ======> ['EventPlot']: Dictionary descibing the information used for the plotting of individual events with location uncertainty and waveform data

                --------------------->  ['Domain Distance'] (:obj:`float`): Default ``None``
                                                                           Distance in km +/- relative to the earthquake location to plot. 
                                                                           If ``None`` whole region specified by the EikoNets will be plot.

                --------------------->  ['Save Type'] (:obj:`string`): Default ``png``
                                                                      Save type for the event plots. e.g. png,pdf,svg,eps

                --------------------->  ['Figure Size Scale'] (:obj:`float`): Default ``1.0``
                                                                             Scaling factor to increase on decrease the overall plot size.

                --------------------->  ['Plot kde'] (:obj:`bool`): Default ``True``
                                                                   Whether to plot the kernel density estimation of the particle location, which would 
                                                                   represent the posterior of the earthquake locations.

                --------------------->  ['Legend'] (:obj:`arr`): Default ``True``
                                                                Whether to plot the legend for the event plot

                --------------------->  ['NonClusterd SVGD'] (:obj:`arr`): Default ``[0.5,'k']``
                                                                          DEPRICATED ONLY USE IF USING DBSCAN CLUSTERING
                                                                          Defines the size and colour of the particle locations for the Non-clustered particle locations

                --------------------->  ['Clusterd SVGD'] (:obj:`arr`): Default ``[0.5,'k']``
                                                              DEPRICATED ONLY USE IF USING DBSCAN CLUSTERING
                                                              Defines the size and colour of the particle locations for the clustered particle locations


                --------------------->  ['Stations'] (:obj:`dict`): Dictionary descibing information about the plotting of the station locations

                                        ['Stations']['Plot Stations'] (:obj:`bool`): Default ``True``
                                                                                    Whether to plot the station locations on event plots

                                        ['Stations']['Station Names'] (:obj:`bool`): Default ``True``
                                                                                    Whether to plot the station names on the event plots

                                        ['Stations']['Marker Color'] (:obj:`string`): Default ``b``
                                                                                    The colour which to plot the station locations.

                                        ['Stations']['Marker Size'] (:obj:`float`): Default ``23``
                                                                                    The marker size for the station locations.


                --------------------->  ['Traces'] (:obj:`dict`): Dictionary descibing information about the plotting traces on event plots
                                       
                                        ['Traces']['Plot Traces'] (:obj:`bool`): Default ``True``
                                                                                Whether to plot the seismic traces on the event plots

                                        ['Traces']['Trace Host'] (:obj:`str`): Default ``None``
                                                                               Path to the seismic trace host e.g. /PATH/TO/FOLDER 

                                        ['Traces']['Trace Host Type'] (:obj:`str`): Default ``'/YEAR/JD/*ST*'``
                                                                                   Format of the seismic trace host. These include
                                                                                        ``'/YEAR/JD/*ST*'`` - Year, Julian Day, station nested folders
                                                                                        ``EQTransformer``   - Format of the autogenerated folders from EQTransformer

                                        ['Traces']['Channel Types'] (:obj:`arr`): Default ``['EH*','HH*']``
                                                                                Format for the components of the traces which you wish to plot


                                        ['Traces']['Filter Freq'] (:obj:`arr`): Default ``[2,16]``
                                                                               Low frequency and High frequency to which the seismic traces will be bandpassed.

                                        ['Traces']['Normalisation Factor'] (:obj:`float`): Default ``1.0``
                                                                                           Normalization scaling factor to which you want to apply to all traces.

                                        ['Traces']['Time Bounds'] (:obj:`arr`): Default ``[0,5]``
                                                                                Time before the origin time and time after the last pick respectively 
                                                                                to which the data is load.

                                        ['Traces']['Pick linewidth'] (:obj:`float`): Default ``2.0``
                                                                                     Thickness of the pick linewidth


                                        ['Traces']['Trace linewidth'] (:obj:`float`): Default ``1.0``
                                                                                     Thickness of the seismic trace linewidth



                ======> ['CataloguePlot']: Dictionary descibing the information used for the plotting of whole catalogue and location uncertainties

                --------------------->  ['Minimum Phase Picks'] (:obj:`int`): Default ``12``
                                                                             Minimum number of event picks for the event to be plot in the catalogue plot

              
                --------------------->  ['Maximum Location Uncertainty (km)'] (:obj:`float`): Default ``15``
                                                                             Maximum total location uncertainty for the event to be plot in the catalogue plot                                                              

                --------------------->  ['Event Info - [Size, Color, Marker, Alpha]'] (:obj:`arr`): Default ``[0.1,'r','*',0.8]``
                                                                                                    Information describing how to plot the event location.
                                                                                                    [Marker Size,Marker color, Marker Symbol, Opacity Alpha] resectively

                --------------------->  ['Event Errorbar - [On/Off(Bool),Linewidth,Color,Alpha]'] (:obj:`arr`): Default ``[True,0.1,'r',0.8]``
                                                                                                    Information describing how/whether to plot the event errorbars.
                                                                                                    [On/Off,Errorbar Linewidth, Errorbar Colour, Errorbar Opacity Alpha] resectively


                --------------------->  ['Station Marker - [Size,Color,Names On/Off(Bool)]'] (:obj:`arr`): Default ``[15,'b',True]``
                                                                                                    Information describing how to plot the stations.
                                                                                                    [Station Marker Size, Station Marker Colour,On/Off Station Names] resectively


    
    """

    def __init__(self, EikoNet, Phases=['P','S'], device='cpu'):
        super(HypoSVI, self).__init__()

        # -- Defining the EikoNet input formats
        self.eikonet_Phases  = Phases
        self.eikonet_models  = EikoNet
        if len(self.eikonet_Phases) != len(self.eikonet_models):
            print('Error - Number of phases not equal to number of EikoNet models')
        



        # Determining if the EikoNets are solved for the same domain
        xmin_stack = np.vstack([self.eikonet_models[x].Params['VelocityClass'].xmin for x in range(len(self.eikonet_models))])
        xmax_stack = np.vstack([self.eikonet_models[x].Params['VelocityClass'].xmax for x in range(len(self.eikonet_models))])
        if not (xmin_stack == xmin_stack[0,:]).all() or not (xmax_stack == xmax_stack[0,:]).all():
            print('Error - EikoNet Models not in the same domain\n Min Points = {}\n Max Points = {}'.format(xmin_stack,xmax_stack))
        

        self.VelocityClass = self.eikonet_models[0].Params['VelocityClass']


        # Converting to UTM projection scheme form 
        self.proj_str = copy.copy(self.eikonet_models[0].Params['VelocityClass'].projection)
        if type(self.proj_str) != type(None):
            self.projection = Proj(self.proj_str)
            self.xmin       = copy.copy(self.VelocityClass.xmin)
            self.xmax       = copy.copy(self.VelocityClass.xmax)
            self.xmin[0],self.xmin[1] = self.projection(self.xmin[0],self.xmin[1])
            self.xmax[0],self.xmax[1] = self.projection(self.xmax[0],self.xmax[1])
        else:
            self.projection = None
            self.xmin       = copy.copy(self.VelocityClass.xmin)
            self.xmax       = copy.copy(self.VelocityClass.xmax)

        # --------- Initialising Location Information ---------
        # -- Defining the device to run the location procedure on
        self.device    = torch.device(device)

        # -- Defining the parameters required in the earthquake location procedure
        self.location_info = {}
        self.location_info['Log-likehood']                         = 'LAP'      
        self.location_info['Travel Time Uncertainty - [Gradient(km/s),Min(s),Max(s)]'] = [0.1,0.1,2.0] 
        self.location_info['Individual Event Epoch Save and Print Rate'] = [None,False]
        self.location_info['Number of Particles']                  = 150 
        self.location_info['Step Size']                            = 1
        self.location_info['Save every * events']                  = 100
        self.location_info['Location Uncertainty Percentile (%)']  = 95.0

        # Depricated values
        self.location_info['Hypocenter Cluster - Seperation (km)'] = None     
        self.location_info['Hypocenter Cluster - Minimum Samples'] = None



        # --------- Initialising Plotting Information ---------
        self.plot_info={}

        # - Event Plot parameters
        # Location plotting
        self.plot_info['EventPlot']  = {}
        self.plot_info['EventPlot']['Domain Distance']     = None#10
        self.plot_info['EventPlot']['Save Type']           = 'png'
        self.plot_info['EventPlot']['Figure Size Scale']   = 1.0
        self.plot_info['EventPlot']['Plot kde']            = True
        self.plot_info['EventPlot']['NonClusterd SVGD']    = [0.5,'k']
        self.plot_info['EventPlot']['Clusterd SVGD']       = [1.2,'g']
        self.plot_info['EventPlot']['Hypocenter Location'] = [15,'k']
        self.plot_info['EventPlot']['Hypocenter Errorbar'] = [True,'k']
        self.plot_info['EventPlot']['Legend']              = True

        # Optional Station Plotting
        self.plot_info['EventPlot']['Stations'] = {}
        self.plot_info['EventPlot']['Stations']['Plot Stations']      = True
        self.plot_info['EventPlot']['Stations']['Station Names']      = True
        self.plot_info['EventPlot']['Stations']['Marker Color']       = 'b'
        self.plot_info['EventPlot']['Stations']['Marker Size']        = 25

        # Optional Trace Plotting
        self.plot_info['EventPlot']['Traces']                         = {}
        self.plot_info['EventPlot']['Traces']['Plot Traces']          = False
        self.plot_info['EventPlot']['Traces']['Trace Host']           = None
        self.plot_info['EventPlot']['Traces']['Trace Host Type']      = '/YEAR/JD/*ST*'
        self.plot_info['EventPlot']['Traces']['Channel Types']        = ['EH*','HH*']
        self.plot_info['EventPlot']['Traces']['Filter Freq']          = [2,16]
        self.plot_info['EventPlot']['Traces']['Normalisation Factor'] = 1.0
        self.plot_info['EventPlot']['Traces']['Time Bounds']          = [0,5]
        self.plot_info['EventPlot']['Traces']['Pick linewidth']       = 2.0
        self.plot_info['EventPlot']['Traces']['Trace linewidth']      = 1.0

        # - Catalogue Plot parameters
        self.plot_info['CataloguePlot'] = {}
        self.plot_info['CataloguePlot']['Minimum Phase Picks']                                           = 12
        self.plot_info['CataloguePlot']['Maximum Location Uncertainty (km)']                             = 15
        self.plot_info['CataloguePlot']['Event Info - [Size, Color, Marker, Alpha]']                     = [0.1,'r','*',0.8]
        self.plot_info['CataloguePlot']['Event Errorbar - [On/Off(Bool),Linewidth,Color,Alpha]']         = [True,0.1,'r',0.8]
        self.plot_info['CataloguePlot']['Station Marker - [Size,Color,Names On/Off(Bool)]']              = [15,'b',True]
        self.plot_info['CataloguePlot']['Fault Planes - [Size,Color,Marker,Alpha]']                      = [0.1,'gray','-',1.0]
        
        # ----- Kernel Information ----
        self.K             = RBF()
        self.K.sigma       = None
        self.K.print_sigma = False

        # --- Variables that are updated in run-time
        self._σ_T       = None
        self._optimizer = None
        self._orgTime   = None

    def locVar(self,T_obs,T_obs_err):
        '''
            Variance of the observations used for the inverison procedure
            
            Applying variance from Pick and travel-time percentage weighting to each of the observtions

            Args:
                T_obs (:obj:`torch.tensor`): The observational arrival times
                T_obs_err (:obj:`torch.tensor`): The pick standard-deviation in the observational arrival times 
        
        '''                
        # Intialising a variance of the LOCGAU2 settings
        self._σ_T  = torch.clamp(T_obs*self.location_info['Travel Time Uncertainty - [Gradient(km/s),Min(s),Max(s)]'][0],
                                       self.location_info['Travel Time Uncertainty - [Gradient(km/s),Min(s),Max(s)]'][1],
                                       self.location_info['Travel Time Uncertainty - [Gradient(km/s),Min(s),Max(s)]'][2]).to(self.device)**2
        # Adding the variance of the Station Pick Uncertainties 
        self._σ_T += (T_obs_err**2)
        # Turning back into a std
        self._σ_T  = torch.sqrt(self._σ_T)
                
    def log_L(self, T_pred, T_obs, σ_T):
        '''
            Log-Likelihood estimation from the observated travel-times, predicted travel-times
            and the variance of each of the observational data

            Args:
                T_pred (:obj:`torch.tensor`): Predicted travel-times detemined as output from the EikoNet models
                T_obs (:obj:`torch.tensor`): The observational arrival times
                T_obs_err (:obj:`torch.tensor`): The pick standard-deviation in the observational arrival times 

        '''


        if self.location_info['Log-likehood'] == 'EDT':
            from itertools import combinations
            pairs     = combinations(np.arange(T_obs.shape[1]), 2)
            pairs     = np.array(list(pairs))
            dT_obs    = T_obs[:,pairs[:,0]] - T_obs[:,pairs[:,1]]
            dT_pred   = T_pred[:,pairs[:,0]] - T_pred[:,pairs[:,1]]
            σ_T       = ((σ_T[:,pairs[:,0]])**2 + (σ_T[:,pairs[:,1]])**2)
            logL      = torch.exp((-(dT_obs-dT_pred)**2)/(σ_T))*(1/torch.sqrt(σ_T))
            logL      = torch.sum(logL,dim=1)
            logL      = torch.log(logL)
            logL      = logL.sum()

        if self.location_info['Log-likehood'] == 'LAP':
            from itertools import combinations
            pairs     = combinations(np.arange(T_obs.shape[1]), 2)
            pairs     = np.array(list(pairs))
            dT_obs    = T_obs[:,pairs[:,0]] - T_obs[:,pairs[:,1]]
            dT_pred   = T_pred[:,pairs[:,0]] - T_pred[:,pairs[:,1]]
            σ_T       = torch.sqrt((σ_T[:,pairs[:,0]])**2 + (σ_T[:,pairs[:,1]])**2)
            logL      = (-(sqrt(2.)*abs(dT_obs-dT_pred))/σ_T ) + torch.log(1/(sqrt(2.)*σ_T))
            logL      = torch.sum(logL,dim=1)
            logL      = logL.sum()




        return logL
    
    def phi(self, X_src, X_rec, t_obs,t_obs_err,t_phase):
        # Preparing EikoNet input
        n_particles = X_src.shape[0]

        # Forcing points to stay within domain 
        X_src[:,0] = torch.clamp(X_src[:,0],self.xmin[0],self.xmax[0])
        X_src[:,1] = torch.clamp(X_src[:,1],self.xmin[1],self.xmax[1])
        X_src[:,2] = torch.clamp(X_src[:,2],self.xmin[2],self.xmax[2])

        # Setting up the gradient requirements
        X_src = X_src.detach().requires_grad_(True)

        # Determining the predicted Travel-time for the different phases
        n_obs = 0
        cc=0
        
        for ind,phs in enumerate(self.eikonet_Phases):
            phase_index = np.where(t_phase==phs)[0]
            if len(phase_index) != 0:
                pha_T_obs     = t_obs[phase_index].repeat(n_particles, 1)
                pha_T_obs_err = t_obs_err[phase_index].repeat(n_particles, 1)
                pha_X_inp     = torch.cat([X_src.repeat_interleave(len(phase_index), dim=0), X_rec[phase_index,:].repeat(n_particles, 1)], dim=1)
                pha_T_pred    = self.eikonet_models[ind].TravelTimes(pha_X_inp,projection=False).reshape(n_particles,len(phase_index))

                if cc == 0:
                    n_obs     = len(phase_index)
                    T_obs     = pha_T_obs
                    T_obs_err = pha_T_obs_err
                    T_pred    = pha_T_pred
                    cc+=1
                else:
                    n_obs    += len(phase_index)
                    T_obs     = torch.cat([T_obs,pha_T_obs],dim=1)
                    T_obs_err = torch.cat([T_obs_err,pha_T_obs_err],dim=1)
                    T_pred    = torch.cat([T_pred,pha_T_pred],dim=1)
        

        
        self.locVar(T_obs,T_obs_err)
        log_prob   = self.log_L(T_pred,T_obs,self._σ_T)
        score_func = torch.autograd.grad(log_prob, X_src)[0]

        # Determining the phi
        K_XX     = self.K(X_src, X_src.detach())
        grad_K   = -torch.autograd.grad(K_XX.sum(), X_src)[0]
        phi      = (K_XX.detach().matmul(score_func) + grad_K) / (n_particles)

        # Setting Misfit to zero to restart
        self._σ_T     = None

        return phi

    def step(self, X_src, X_rec, T_obs, T_obs_err, T_phase):
        self.optim.zero_grad()
        X_src.grad = -self.phi(X_src, X_rec, T_obs, T_obs_err, T_phase)
        self.optim.step()

    def _compute_origin(self,Tobs,t_phase,X_rec,Hyp):
        '''
            Internal function to compute origin time and predicted Travel-times from Obs and Predicted Travel-times
        '''

        # Determining the predicted Travel-time for the different phases
        n_obs = 0
        cc=0
        for ind,phs in enumerate(self.eikonet_Phases):
            phase_index = np.where(t_phase==phs)[0]
            if len(phase_index) != 0:
                pha_X_inp     = torch.cat([torch.repeat_interleave(Hyp[None,:],len(phase_index),dim=0), X_rec[phase_index,:]], dim=1)
                pha_T_obs     = Tobs[phase_index]

                #pha_X_inp.requires_grad_()
                pha_T_pred    = self.eikonet_models[ind].TravelTimes(pha_X_inp,projection=False)
                
                # # -- Determinig take-off angles -- 
                # dpha_T_pred   = torch.autograd.grad(outputs=pha_T_pred,inputs=pha_X_inp)[0]
                # dst  = torch.sum(dpha_T_pred[:,:3]**2,dim=1)
                # plng = torch.arcsin(-dpha_T_pred[:,2]/dst)
                # azi  = torch.rad2deg(torch.arctan(pha_T_pred[:,0]/dpha_T_pred[:,1]))

                if cc == 0:
                    T_obs     = pha_T_obs
                    T_pred    = pha_T_pred
                    cc+=1
                else:
                    T_obs     = torch.cat([T_obs,pha_T_obs])
                    T_pred    = torch.cat([T_pred,pha_T_pred])

        OT      = np.median((T_pred - Tobs).detach().cpu().numpy())
        pick_TD = ((T_pred - OT) - T_obs).detach().cpu().numpy()
        OT_std  = np.nanmedian(abs(pick_TD))

        return OT,OT_std,pick_TD

    def SyntheticCatalogue(self,input_file,Stations,save_file=None):
        '''
            Determining synthetic Travel-times between source and reciever locations, returning a JSON pick file for each event

    
            Event_Locations - EventNum, OriginTime, PickErr, X, Y, Z 

            Stations -

            # JDS - MAKE CORRECTIONS TO PROJECTION !! 


        '''

        # Determining the predicted Travel-time to each of the stations to corresponding
        #source locations. Optional argumenent to return them as json pick
        evtdf  = pd.read_csv(input_file)
        EVT = {}
        for indx in range(len(evtdf)):
            EVT['{}'.format(evtdf['EventNum'].iloc[indx])] = {}

            OT = evtdf['OriginTime'].iloc[indx]

            # Defining the picks to append
            picks = pd.DataFrame(columns=['Network','Station','PhasePick','DT','PickError'])
            for ind,phs in enumerate(self.eikonet_Phases):
                picks_phs = Stations[['Network','Station','X','Y','Z']]
                picks_phs['PhasePick'] = phs
                picks_phs['PickError'] = evtdf['PickErr'].iloc[indx]
                Pairs       = np.zeros((int(len(Stations)),6))
                Pairs[:,:3] = np.array(evtdf[['X','Y','Z']].iloc[indx])
                Pairs[:,3:] = np.array(picks_phs[['X','Y','Z']])

                if type(self.projection) != type(None):
                    Pairs[:,0],Pairs[:,1] = self.projection(Pairs[:,0],Pairs[:,1])
                    Pairs[:,3],Pairs[:,4] = self.projection(Pairs[:,3],Pairs[:,4])

                Pairs       = Tensor(Pairs)

                Pairs       = Pairs.to(self.device)
                TT_pred     = self.eikonet_models[ind].TravelTimes(Pairs,projection=False).detach().to('cpu').numpy()
                del Pairs

                picks_phs['DT']  = TT_pred
                picks_phs['DT']  = (pd.to_datetime(OT ) + pd.to_timedelta(picks_phs['DT'],unit='S')).dt.strftime('%Y/%m/%dT%H:%M:%S.%f')

                picks = picks.append(picks_phs[['Network','Station','PhasePick','DT','PickError']])


            EVT['{}'.format(evtdf['EventNum'].iloc[indx])]['Picks'] = picks


        if type(save_file) == str:
            IO_JSON('{}.json'.format(save_file),Events=EVT,rw_type='w')
        

        return EVT

    def Events2CSV(self,EVT=None,savefile=None,projection=None):
        '''
            Saving Events in CSV format
        '''

        if type(EVT) == type(None):
            Events = self.Events
        else:
            Events = EVT

        # Loading location information
        picks =(np.zeros((len(Events.keys()),8))*np.nan).astype(str)
        for indx,evtid in enumerate(Events.keys()):
            try:
                picks[indx,0]   = str(evtid)
                picks[indx,1]   = self.Events[evtid]['location']['OriginTime']
                picks[indx,2:5] = (np.array(self.Events[evtid]['location']['Hypocentre'])).astype(str)
                picks[indx,5:]  = (np.array(self.Events[evtid]['location']['HypocentreError'])).astype(str)
            except:
                continue
        picks_df = pd.DataFrame(picks,
                                columns=['EventID','DT','X','Y','Z','ErrX','ErrY','ErrZ'])
        picks_df['X'] = picks_df['X'].astype(float)
        picks_df['Y'] = picks_df['Y'].astype(float)
        picks_df['Z']    = picks_df['Z'].astype(float)
        picks_df['ErrX'] = picks_df['ErrX'].astype(float)
        picks_df['ErrY'] = picks_df['ErrY'].astype(float)
        picks_df['ErrZ'] = picks_df['ErrZ'].astype(float)
        picks_df = picks_df.dropna(axis=0)
        picks_df['DT'] = pd.to_datetime(picks_df['DT'])
        picks_df = picks_df[['EventID','DT','X','Y','Z','ErrX','ErrY','ErrZ']]

        if type(savefile) == type(None):
            return picks_df
        else:
            picks_df.to_csv(savefile,index=False)

    def LocateEvents(self,EVTS,Stations,output_path,epochs=175,output_plots=False,timer=False,PriorCatalogue=False):
        self.Events      = EVTS

        if PriorCatalogue == False:
            try:
                os.system('rm {}/Catalogue.json'.format(output_path))
            except:
                print

        print('============================================================================================================')
        print('============================================================================================================')
        print('========================================= HYPOSVI - Earthquake Location ====================================')
        print('============================================================================================================')
        print('============================================================================================================')
        print('\n')
        print('      Procssing for {} Events  - Starting DateTime {}'.format(len(EVTS.keys()),time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
        print('      Output Folder = {}'.format(output_path))
        if PriorCatalogue == False:
            try:
                os.system('rm {}/Catalogue.json'.format(output_path))
            except:
                print('      No Prior Catalogue defined for appending')
        print('\n')
        print('======== Location Settings:')
        print(json.dumps(self.location_info, indent=2, sort_keys=True))

        print('\n')

        if output_plots:
            print('======== Plotting Settings:')
            print(json.dumps(self.plot_info['EventPlot'], indent=2, sort_keys=True))
            print('\n')        
        print('============================================================================================================')
        print('============================================================================================================')

        evtid = []
        for c,ev in enumerate(self.Events.keys()):
            # try:
                if timer == True:
                    timer_start = time.time()
                # Determining the event to look at
                evtid.append(ev)
                Ev = self.Events[ev]
                
                # Formating the pandas datatypes
                Ev['Picks']['Network']   = Ev['Picks']['Network'].astype(str)
                Ev['Picks']['Station']   = Ev['Picks']['Station'].astype(str)
                Ev['Picks']['PhasePick'] = Ev['Picks']['PhasePick'].astype(str)
                Ev['Picks']['DT']        = pd.to_datetime(Ev['Picks']['DT'])
                Ev['Picks']['PickError'] = Ev['Picks']['PickError'].astype(float)
                Ev['Picks']              = Ev['Picks'][Ev['Picks']['PhasePick'].isin(self.eikonet_Phases)].reset_index(drop=True)

                if len(Ev['Picks']) == 0:
                    print('Now phase picks for the desired phases ! Event cannot be located')
                    continue

                # printing the current event being run
                print('================= Processing Event:{} - Event {} of {} - Number of observtions={} =============='.format(ev,c+1,len(self.Events.keys()),len(Ev['Picks'])))

                # Adding the station location to the pick files
                pick_info   = pd.merge(Ev['Picks'],Stations[['Network','Station','X','Y','Z']])
                Ev['Picks'] = pick_info[['Network','Station','X','Y','Z','PhasePick','DT','PickError']]

                # Setting up the random seed locations
                X_src       = torch.zeros((int(self.location_info['Number of Particles']),3))
                X_src[:,:3] = Tensor(np.random.rand(int(self.location_info['Number of Particles']),3))*(Tensor(self.xmax)-Tensor(self.xmin))[None,:] + Tensor(self.xmin)[None,:]
                X_src       = Variable(X_src).to(self.device)
                self.optim  = torch.optim.Adam([X_src], self.location_info['Step Size'])
                
                # Defining the arrivals times in seconds
                pick_info['Seconds'] = (pick_info['DT'] - np.min(pick_info['DT'])).dt.total_seconds()

                # Applying projection
                X_rec       = np.array(pick_info[['X','Y','Z']])
                if type(self.projection) != type(None):
                    X_rec[:,0],X_rec[:,1] = self.projection(X_rec[:,0],X_rec[:,1])

                X_rec       = Tensor(X_rec).to(self.device)
                T_obs       = Tensor(np.array(pick_info['Seconds'])).to(self.device)
                T_obs_err   = Tensor(np.array(pick_info['PickError'])).to(self.device)
                T_obs_phase = np.array(pick_info['PhasePick'])
                X_rec.requires_grad_()
                l = None
                losses = []
                best_l = np.inf
                #with autocast():
                cc=0

                if self.location_info['Individual Event Epoch Save and Print Rate'][0] != None:
                    for epoch in range(epochs):
                        self.optim.zero_grad()
                        if epoch % self.location_info['Individual Event Epoch Save and Print Rate'][0] == 0:
                            with torch.no_grad():
                                if self.location_info['Individual Event Epoch Save and Print Rate'][1] == True:
                                    print("Epoch - {} ".format(epoch))
                                if cc==0:
                                    PointsSVGD = X_src[...,None]
                                    cc+=1
                                else:
                                    PointsSVGD = torch.cat((PointsSVGD, X_src[...,None]), -1)
                        self.step(X_src, X_rec, T_obs, T_obs_err, T_obs_phase)
                else:
                    for epoch in range(epochs):
                        self.optim.zero_grad()
                        self.step(X_src, X_rec, T_obs, T_obs_err, T_obs_phase)

                del cc

                # -- Drop points outside of the domain
                dmindx = [(X_src[:,2] > self.xmin[2]) & (X_src[:,2] < self.xmax[2])]
                X_src                                     = X_src[dmindx[0],:]
                Ev['location']                            = {}
                Ev['location']['SVGD_points']             = X_src.detach().cpu().numpy().tolist()
                
                if len(Ev['location']['SVGD_points']) == 0:
                     Ev['location']['Hypocentre']     = (np.ones(3)*np.nan).tolist()
                     Ev['location']['HypocentreError'] = (np.ones(3)*np.nan).tolist()
                     print('Earthquake particle locations failed to converge - all particles outside the volume of interest')
                     continue

                # -- SVGD Points in Epochs --
                if self.location_info['Individual Event Epoch Save and Print Rate'][0] != None:
                    Ev['location']['SVGD_Epochs'] = PointsSVGD.detach().cpu().numpy()
                    for ii in range(Ev['location']['SVGD_Epochs'].shape[-1]):
                        if type(self.projection) != type(None):
                            Ev['location']['SVGD_Epochs'][:,0,ii],Ev['location']['SVGD_Epochs'][:,1,ii] = self.projection(Ev['location']['SVGD_Epochs'][:,0,ii],Ev['location']['SVGD_Epochs'][:,1,ii],inverse=True)
                    Ev['location']['SVGD_Epochs'] = Ev['location']['SVGD_Epochs'].tolist()


                # === DEPRICATED CLUSTERING TECHNIQUE ===
                if (type(self.location_info['Hypocenter Cluster - Seperation (km)']) != type(None)) and (type(self.location_info['Hypocenter Cluster - Minimum Samples']) != type(None)):
                    clustering = DBSCAN(eps=self.location_info['Hypocenter Cluster - Seperation (km)'], min_samples=self.location_info['Hypocenter Cluster - Minimum Samples']).fit(X_src.detach().cpu())
                    try:
                        indx    = np.where((clustering.labels_ == (np.argmax(np.bincount(np.array(clustering.labels_[clustering.labels_ !=-1]+1)))-1)))[0]
                        print('Events Clustered - Cluster Size ={}'.format(len(indx)))
                    except:
                        indx    = np.arange(X_src.shape[0])
                        print('Warning - No particles clustering, making all particles a single cluster')
                else:
                    indx    = np.arange(X_src.shape[0])

                pts     = np.transpose(X_src[indx,:].detach().cpu().numpy())
                try:
                    print(pts.shape)
                    kde         = stats.gaussian_kde(pts,'silverman')
                    pdf         = kde(pts)
                    percentile  = abs(np.sqrt(abs(kde.covariance))*stats.norm.ppf(self.location_info['Location Uncertainty Percentile (%)']/100))
                    err         = [percentile[0,0],percentile[1,1],percentile[2,2]]
                    hyp         = pts[:,np.argmax(pdf)]


                    # import matplotlib.pyplot as plt
                    # from scipy import stats
                    # from sklearn.covariance import EmpiricalCovariance,MinCovDet
                    # robust_cov  = MinCovDet().fit(np.transpose(pts))
                    # hyp         = robust_cov.location_
                    # percentile  = abs(np.sqrt(abs(robust_cov.covariance_))*stats.norm.ppf(self.location_info['Location Uncertainty Percentile (%)']/100))
                    # err         = np.array([np.sum(np.dot(abs(percentile),[1,0,0])),
                    #                         np.sum(np.dot(abs(percentile),[0,1,0])),
                    #                         np.sum(np.dot(abs(percentile),[0,0,1]))])





                except:
                    print('Earthquake particle locations failed to converge - kernel density failure')
                    continue



                #cov = np.sqrt(abs(np.cov(pts)))
                Ev['location']['SVGD_points_clusterindx']  = indx.tolist()
                Ev['location']['Hypocentre']               = (hyp).tolist()
                Ev['location']['HypocentreError']          = np.array([err[0],err[1],err[2]]).tolist()
                Ev['location']['Covariance']               = kde.covariance.tolist()

                # -- Determining the origin time and pick times
                originOffset,originOffset_std,pick_TD        = self._compute_origin(T_obs,T_obs_phase,X_rec,Tensor(Ev['location']['Hypocentre']).to(self.device))
                Ev['location']['OriginTime_std']             = float(originOffset_std)
                Ev['location']['OriginTime']                 = str(np.min(pick_info['DT']) - pd.Timedelta(float(originOffset),unit='S'))
                Ev['Picks']['TimeDiff']                      = pick_TD 


                # -- Applying the projection from UTM to LatLong
                if type(self.projection) != type(None):
                    Ev['location']['Hypocentre'] = np.array(Ev['location']['Hypocentre'])
                    Ev['location']['Hypocentre'][0],Ev['location']['Hypocentre'][1] = self.projection(Ev['location']['Hypocentre'][0],Ev['location']['Hypocentre'][1],inverse=True)
                    Ev['location']['Hypocentre'] = Ev['location']['Hypocentre'].tolist()

                    Ev['location']['SVGD_points'] = np.array(Ev['location']['SVGD_points'])
                    Ev['location']['SVGD_points'][:,0],Ev['location']['SVGD_points'][:,1] = self.projection(Ev['location']['SVGD_points'][:,0],Ev['location']['SVGD_points'][:,1],inverse=True)
                    Ev['location']['SVGD_points'] = Ev['location']['SVGD_points'].tolist()


                print('---- OT= {} +/- {}s - Hyp=[{:.2f},{:.2f},{:.2f}] - Hyp Uncertainty (+/- km)=[{:.2f},{:.2f},{:.2f}]'.format(Ev['location']['OriginTime'],Ev['location']['OriginTime_std'],Ev['location']['Hypocentre'][0],Ev['location']['Hypocentre'][1],Ev['location']['Hypocentre'][2],
                                                                                      Ev['location']['HypocentreError'][0],Ev['location']['HypocentreError'][1],Ev['location']['HypocentreError'][2]))

                if timer == True:
                    timer_end = time.time()
                    print('Processing took {}s'.format(timer_end-timer_start))

                # Plotting Event plots
                if output_plots:
                    if timer == True:
                        timer_start = time.time()
                    print('---- Saving Event Plot ----')
                    #try:
                    self.EventPlot(output_path,Ev,EventID=ev)
                    # except:
                    #     print('----Issue with saving plot !  ----')

                    if timer == True:
                        timer_end = time.time()
                        print('Plotting took {}s'.format(timer_end-timer_start))

                # Saving Catalogue instance
                if (self.location_info['Save every * events']  != None) and ((c%self.location_info['Save every * events']) == 0):
                    if timer == True:
                        timer_start = time.time()
                    print('---- Saving Catalogue instance ----')
                    IO_JSON('{}/Catalogue.json'.format(output_path),Events={ev:self.Events[ev] for ev in evtid},rw_type='a+')
                    {ev:'Located & Saved' for ev in evtid} #Freeing up memory
                    if timer == True:
                        timer_end = time.time()
                        print('Saving took {}s'.format(timer_end-timer_start))
            # except:
            #    print('Event Location failed ! Continuing to next event')


        # Writing out final catalogue
        IO_JSON('{}/Catalogue.json'.format(output_path),Events={ev:self.Events[ev] for ev in evtid},rw_type='a+')
        {ev:'Located & Saved' for ev in evtid}

    def EventPlot(self,PATH,Event,EventID=None):
        plt.close('all')
        OT             = str(Event['location']['OriginTime'])
        OT_std         = Event['location']['OriginTime_std']
        locs           = np.array(Event['location']['SVGD_points'])
        optimalloc     = np.array(Event['location']['Hypocentre'])
        optimalloc_Err = np.array(Event['location']['HypocentreError'])
        indx_cluster   = np.array(Event['location']['SVGD_points_clusterindx'])
        Stations       = Event['Picks'][['Station','X','Y','Z']]

        if self.plot_info['EventPlot']['Traces']['Plot Traces']==True:
            fig = plt.figure(figsize=(20*self.plot_info['EventPlot']['Figure Size Scale'], 9*self.plot_info['EventPlot']['Figure Size Scale']))
            xz  = plt.subplot2grid((3, 5), (2, 0), colspan=2)
            xy  = plt.subplot2grid((3, 5), (0, 0), colspan=2, rowspan=2,sharex=xz)
            yz  = plt.subplot2grid((3, 5), (0, 2), rowspan=2, sharey=xy)
            trc = plt.subplot2grid((3, 5), (0, 3), rowspan=3, colspan=2)
        else:
            fig = plt.figure(figsize=(9*self.plot_info['EventPlot']['Figure Size Scale'], 9*self.plot_info['EventPlot']['Figure Size Scale']))
            xz  = plt.subplot2grid((3, 3), (2, 0), colspan=2)
            xy  = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2,sharex=xz)
            yz  = plt.subplot2grid((3, 3), (0, 2), rowspan=2, sharey=xy) 

        fig.patch.set_facecolor("white")

        # Specifying the label names
        xz.set_xlabel('UTM X (km)')
        xz.set_ylabel('Depth (km)')
        yz.set_ylabel('UTM Y (km)')
        yz.yaxis.tick_right()
        yz.yaxis.set_label_position("right")
        yz.set_xlabel('Depth (km)')


        if self.plot_info['EventPlot']['Domain Distance'] != None:
            if type(self.projection) != type(None):
                optimalloc_UTM = copy.copy(optimalloc)
                optimalloc_UTM[0],optimalloc_UTM[1] = self.projection(optimalloc_UTM[0],optimalloc_UTM[1])
                boundsmin = optimalloc_UTM-self.plot_info['EventPlot']['Domain Distance']/2
                boundsmax = optimalloc_UTM+self.plot_info['EventPlot']['Domain Distance']/2
                boundsmin[0],boundsmin[1] = self.projection(boundsmin[0],boundsmin[1],inverse=True)
                boundsmax[0],boundsmax[1] = self.projection(boundsmax[0],boundsmax[1],inverse=True)
            else:
                boundsmin = optimalloc-self.plot_info['EventPlot']['Domain Distance']/2
                boundsmax = optimalloc+self.plot_info['EventPlot']['Domain Distance']/2
            xy.set_xlim([boundsmin[0],boundsmax[0]])
            xy.set_ylim([boundsmin[1],boundsmax[1]])
            xz.set_xlim([boundsmin[0],boundsmax[0]])
            xz.set_ylim([boundsmin[2],boundsmax[2]])
            yz.set_xlim([boundsmin[2],boundsmax[2]])
            yz.set_ylim([boundsmin[1],boundsmax[1]])
        else:
            if type(self.projection) != type(None):
                lim_min = self.VelocityClass.xmin
                lim_max = self.VelocityClass.xmax
            else:
                lim_min = self.xmin
                lim_max = self.xmax
            xy.set_xlim([lim_min[0],lim_max[0]])
            xy.set_ylim([lim_min[1],lim_max[1]])
            xz.set_xlim([lim_min[0],lim_max[0]])
            xz.set_ylim([lim_min[2],lim_max[2]])
            yz.set_xlim([lim_min[2],lim_max[2]])
            yz.set_ylim([lim_min[1],lim_max[1]])

        # Invert yaxis
        xz.invert_yaxis()

        # Plotting the kde representation of the scatter data
        if self.plot_info['EventPlot']['Plot kde']:
            sns.kdeplot(x=locs[indx_cluster,0],y=locs[indx_cluster,1], cmap="Reds",ax=xy,zorder=-1,gridsize=500,bw_method='silverman')
            sns.kdeplot(x=locs[indx_cluster,0],y=locs[indx_cluster,2], cmap="Reds",ax=xz,zorder=-1,gridsize=500,bw_method='silverman')
            sns.kdeplot(x=locs[indx_cluster,2],y=locs[indx_cluster,1], cmap="Reds",ax=yz,zorder=-1,gridsize=500,bw_method='silverman')

        # Plotting the SVGD samples
        xy.scatter(locs[:,0],locs[:,1],float(self.plot_info['EventPlot']['NonClusterd SVGD'][0]),str(self.plot_info['EventPlot']['NonClusterd SVGD'][1]),label='SVGD Samples')
        xz.scatter(locs[:,0],locs[:,2],float(self.plot_info['EventPlot']['NonClusterd SVGD'][0]),str(self.plot_info['EventPlot']['NonClusterd SVGD'][1]))
        yz.scatter(locs[:,2],locs[:,1],float(self.plot_info['EventPlot']['NonClusterd SVGD'][0]),str(self.plot_info['EventPlot']['NonClusterd SVGD'][1]))

        # Plotting the SVGD samples after clustering
        xy.scatter(locs[indx_cluster,0],locs[indx_cluster,1],float(self.plot_info['EventPlot']['Clusterd SVGD'][0]),str(self.plot_info['EventPlot']['Clusterd SVGD'][1]),label=' Clustered SVGD Samples')
        xz.scatter(locs[indx_cluster,0],locs[indx_cluster,2],float(self.plot_info['EventPlot']['Clusterd SVGD'][0]),str(self.plot_info['EventPlot']['Clusterd SVGD'][1]))
        yz.scatter(locs[indx_cluster,2],locs[indx_cluster,1],float(self.plot_info['EventPlot']['Clusterd SVGD'][0]),str(self.plot_info['EventPlot']['Clusterd SVGD'][1]))

        # Plotting the predicted hypocentre and standard deviation location
        xy.scatter(optimalloc[0],optimalloc[1],float(self.plot_info['EventPlot']['Hypocenter Location'][0]),str(self.plot_info['EventPlot']['Hypocenter Location'][1]),label='Hypocentre')
        xz.scatter(optimalloc[0],optimalloc[2],float(self.plot_info['EventPlot']['Hypocenter Location'][0]),str(self.plot_info['EventPlot']['Hypocenter Location'][1]))
        yz.scatter(optimalloc[2],optimalloc[1],float(self.plot_info['EventPlot']['Hypocenter Location'][0]),str(self.plot_info['EventPlot']['Hypocenter Location'][1]))

        # Defining the Error bar location
        if self.plot_info['EventPlot']['Hypocenter Errorbar'][0]:

            # JDS - Currently these are rough errorbars, need to improve
            xy.errorbar(optimalloc[0],optimalloc[1],xerr=optimalloc_Err[0]/111, yerr=optimalloc_Err[1]/111,color=self.plot_info['EventPlot']['Hypocenter Errorbar'][1],label='Hyp {}% Confidence'.format(self.location_info['Location Uncertainty Percentile (%)']))
            xz.errorbar(optimalloc[0],optimalloc[2],xerr=optimalloc_Err[0]/111, yerr=optimalloc_Err[2],color=self.plot_info['EventPlot']['Hypocenter Errorbar'][1],label='Hyp {}% Confidence'.format(self.location_info['Location Uncertainty Percentile (%)']))
            yz.errorbar(optimalloc[2],optimalloc[1],xerr=optimalloc_Err[2], yerr=optimalloc_Err[1]/111,color=self.plot_info['EventPlot']['Hypocenter Errorbar'][1],label='Hyp {}% Confidence'.format(self.location_info['Location Uncertainty Percentile (%)']))


        # Optional Station Location used in inversion
        if self.plot_info['EventPlot']['Stations']['Plot Stations']:
            idxsta = Stations['Station'].drop_duplicates().index
            station_markersize  = self.plot_info['EventPlot']['Stations']['Marker Size']
            station_markercolor = self.plot_info['EventPlot']['Stations']['Marker Color']

            xy.scatter(Stations['X'].iloc[idxsta],
                       Stations['Y'].iloc[idxsta],
                       station_markersize, marker='^',color=station_markercolor,label='Stations')

            if self.plot_info['EventPlot']['Stations']['Station Names']:
                for i, txt in enumerate(Stations['Station'].iloc[idxsta]):
                    xy.annotate(txt, (np.array(Stations['X'].iloc[idxsta])[i], np.array(Stations['Y'].iloc[idxsta])[i]))

            xz.scatter(Stations['X'].iloc[idxsta],
                       Stations['Z'].iloc[idxsta],
                       station_markersize,marker='^',color=station_markercolor)

            yz.scatter(Stations['Z'].iloc[idxsta],
                       Stations['Y'].iloc[idxsta],
                       station_markersize,marker='^',color=station_markercolor)

        # Defining the legend as top lef
        if self.plot_info['EventPlot']['Legend']:
            xy.legend(loc='upper left')
        plt.suptitle(' Earthquake {} +/- {:.2f}s\n Hyp=[{:.2f},{:.2f},{:.2f}] - Hyp Uncertainty (km) +/- [{:.2f},{:.2f},{:.2f}]'.format(OT,OT_std,optimalloc[0],optimalloc[1],optimalloc[2],optimalloc_Err[0],optimalloc_Err[1],optimalloc_Err[2]))                                                                                                 

        if self.plot_info['EventPlot']['Traces']['Plot Traces']:
            evt_starttime  = UTCDateTime(OT) + self.plot_info['EventPlot']['Traces']['Time Bounds'][0]
            evt_endtime    = UTCDateTime(Event['Picks']['DT'].max()) + self.plot_info['EventPlot']['Traces']['Time Bounds'][1]
            nf             = self.plot_info['EventPlot']['Traces']['Normalisation Factor']
            Host_path      = self.plot_info['EventPlot']['Traces']['Trace Host']
            pick_linewidth = self.plot_info['EventPlot']['Traces']['Pick linewidth']
            tr_linewidth   = self.plot_info['EventPlot']['Traces']['Trace linewidth']

            # Loading the trace data
            ST = Stream()
            stations = np.array(Event['Picks']['Station'].drop_duplicates())
            network  = np.array(Event['Picks']['Network'].iloc[Event['Picks']['Station'].drop_duplicates().index])
            for indx,sta in enumerate(stations):
                try:
                    net = network[indx]

                    # Loading the data of interest
                    if self.plot_info['EventPlot']['Traces']['Trace Host Type'] == '/YEAR/JD/*ST*':
                        evt_yr         = str(pd.to_datetime(Event['Picks']['DT'].min()).year)
                        evt_jd         = str(pd.to_datetime(Event['Picks']['DT'].min()).dayofyear).zfill(3)
                        st  = obspy.read('{}/{}/{}/*{}*'.format(Host_path,evt_yr,evt_jd,sta),
                                     starttime=evt_starttime-10,endtime=evt_endtime)

                    if self.plot_info['EventPlot']['Traces']['Trace Host Type'] == 'EQTransformer':
                        stdate = (pd.to_datetime(Event['Picks']['DT'].min())).strftime('%Y%m%d')
                        endate = (pd.to_datetime(Event['Picks']['DT'].min()) + pd.Timedelta(days=1)).strftime('%Y%m%d')

                        st  = obspy.read('{}/{}/{}.{}*__{}T*__{}T*.mseed'.format(Host_path,sta,net,sta,stdate,endate), starttime=evt_starttime-10,endtime=evt_endtime)


                    # Selecting only the user specified channels
                    for ch in self.plot_info['EventPlot']['Traces']['Channel Types']:
                        ST  = ST +  st.select(channel=ch,network=net).filter('bandpass',freqmin=self.plot_info['EventPlot']['Traces']['Filter Freq'][0],freqmax=self.plot_info['EventPlot']['Traces']['Filter Freq'][1])
                except:
                    continue

            # Plotting the Trace data
            yloc = np.arange(len(stations)) + 1
            for indx,staName in enumerate(stations):
                # try:
                    net = network[indx]
                    # Plotting the Station traces
                    stm = ST.select(station=staName)
                    for tr in stm:
                        normdata = (tr.data/abs(tr.data).max())
                        normdata = normdata - np.mean(normdata)
                        if (tr.stats.channel[-1] == '1') or (tr.stats.channel[-1] == 'N'):
                            trc.plot(tr.times(reftime=evt_starttime),np.ones(tr.data.shape)*yloc[indx] + normdata*nf,'c',linewidth=tr_linewidth)
                        if (tr.stats.channel[-1] == '2') or (tr.stats.channel[-1] == 'E'):
                            trc.plot(tr.times(reftime=evt_starttime),np.ones(tr.data.shape)*yloc[indx] + normdata*nf,'g',linewidth=tr_linewidth)
                        if (tr.stats.channel[-1] == 'Z'):
                            trc.plot(tr.times(reftime=evt_starttime),np.ones(tr.data.shape)*yloc[indx] + normdata*nf,'m',linewidth=tr_linewidth)

                    # Plotting the picks
                    stadf = Event['Picks'][(Event['Picks']['Station'] == staName) & (Event['Picks']['Network'] == net)].reset_index(drop=True)
                    for indxrw in range(len(stadf)):

                        pick_time    = UTCDateTime(stadf['DT'].iloc[indxrw]) - evt_starttime
                        synpick_time = UTCDateTime(stadf['DT'].iloc[indxrw]) - evt_starttime + stadf['TimeDiff'].iloc[indxrw]
                        if stadf.iloc[indxrw]['PhasePick'] == 'P':
                            trc.plot([pick_time,pick_time],[yloc[indx]-0.6*nf,yloc[indx]+0.6*nf],linestyle='-',color='r',linewidth=pick_linewidth)
                            trc.plot([synpick_time,synpick_time],[yloc[indx]-0.6*nf,yloc[indx]+0.6*nf],linestyle='--',color='r',linewidth=pick_linewidth)


                        if stadf.iloc[indxrw]['PhasePick'] == 'S':
                            trc.plot([pick_time,pick_time],[yloc[indx]-0.6*nf,yloc[indx]+0.6*nf],linestyle='-',color='b',linewidth=pick_linewidth)
                            trc.plot([synpick_time,synpick_time],[yloc[indx]-0.6*nf,yloc[indx]+0.6*nf],linestyle='--',color='b',linewidth=pick_linewidth)
                # except:
                #     continue

            trc.yaxis.tick_right()
            trc.yaxis.set_label_position("right")
            #trc.set_xlim([0,evt_endtime - evt_starttime])
            trc.set_ylim([0,len(stations)+1])
            trc.set_yticks(np.arange(1,len(stations)+1))
            trc.set_yticklabels(stations)
            trc.set_xlabel('Seconds since earthquake origin')

        plt.savefig('{}/{}.{}'.format(PATH,EventID,self.plot_info['EventPlot']['Save Type']))
        plt.clf();plt.close('all')

    def CataloguePlot(self,filepath=None,Events=None,Stations=None,user_xmin=[None,None,None],user_xmax=[None,None,None], Faults=None):

        if type(Events) != type(None):
            self.Events = Events


        # - Catalogue Plot parameters
        min_phases            = self.plot_info['CataloguePlot']['Minimum Phase Picks']
        max_uncertainty       = self.plot_info['CataloguePlot']['Maximum Location Uncertainty (km)']
        event_marker          = self.plot_info['CataloguePlot']['Event Info - [Size, Color, Marker, Alpha]']
        event_errorbar_marker = self.plot_info['CataloguePlot']['Event Errorbar - [On/Off(Bool),Linewidth,Color,Alpha]']
        stations_plot         = self.plot_info['CataloguePlot']['Station Marker - [Size,Color,Names On/Off(Bool)]'] 
        fault_plane           = self.plot_info['CataloguePlot']['Fault Planes - [Size,Color,Marker,Alpha]']


        fig = plt.figure(figsize=(15, 15))
        xz  = plt.subplot2grid((3, 3), (2, 0), colspan=2)
        xy  = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2,sharex=xz)
        yz  = plt.subplot2grid((3, 3), (0, 2), rowspan=2, sharey=xy) 


        # Defining the limits of the domain

        if type(self.projection) != type(None):
            lim_min = self.VelocityClass.xmin
            lim_max = self.VelocityClass.xmax
        else:
            lim_min = self.xmin
            lim_max = self.xmax

        for indx,val in enumerate(user_xmin):
            if val != None:
                lim_min[indx] = val
        for indx,val in enumerate(user_xmax):
            if val != None:
                lim_max[indx] = val

        xy.set_xlim([lim_min[0],lim_max[0]])
        xy.set_ylim([lim_min[1],lim_max[1]])
        xz.set_xlim([lim_min[0],lim_max[0]])
        xz.set_ylim([lim_min[2],lim_max[2]])
        yz.set_xlim([lim_min[2],lim_max[2]])
        yz.set_ylim([lim_min[1],lim_max[1]])

        # Specifying the label names
        xz.set_xlabel('UTM X (km)')
        xz.set_ylabel('Depth (km)')
        xz.invert_yaxis()
        yz.set_ylabel('UTM Y (km)')
        yz.yaxis.tick_right()
        yz.yaxis.set_label_position("right")
        yz.set_xlabel('Depth (km)')


        # Plotting the station locations
        if type(Stations) != type(None):
            sta = Stations[['Station','X','Y','Z']].drop_duplicates()
            xy.scatter(sta['X'],sta['Y'],stations_plot[0], marker='^',color=stations_plot[1],label='Stations')

            if stations_plot[2]:
                for i, txt in enumerate(sta['Station']):
                    xy.annotate(txt, (np.array(sta['X'])[i], np.array(sta['Y'])[i]))


            xz.scatter(sta['X'],sta['Z'],stations_plot[0], marker='^',color=stations_plot[1])
            yz.scatter(sta['Z'],sta['Y'],stations_plot[0], marker='<',color=stations_plot[1])


        picks_df         = self.Events2CSV()
        picks_df         = picks_df[np.sum(picks_df[['ErrX','ErrY','ErrZ']],axis=1) <= max_uncertainty].reset_index(drop=True)

        # # Plotting Location info
        # if event_errorbar_marker[0]:
        #     xy.errorbar(picks_df['X'],picks_df['Y'],xerr=picks_df['ErrX'],yerr=picks_df['ErrY'],fmt='none',linewidth=event_errorbar_marker[1],color=event_errorbar_marker[2],alpha=event_errorbar_marker[3],label='Catalogue Errorbars')
        #     xz.errorbar(picks_df['X'],picks_df['Z'],xerr=picks_df['ErrX'],yerr=picks_df['ErrZ'],fmt='none',linewidth=event_errorbar_marker[1],color=event_errorbar_marker[2],alpha=event_errorbar_marker[3])
        #     yz.errorbar(picks_df['Z'],picks_df['Y'],xerr=picks_df['ErrZ'],yerr=picks_df['ErrY'],fmt='none',linewidth=event_errorbar_marker[1],color=event_errorbar_marker[2],alpha=event_errorbar_marker[3])


        xy.scatter(picks_df['X'],picks_df['Y'],event_marker[0],event_marker[1],marker=event_marker[2],alpha=event_marker[3],label='Catalogue Locations')
        xz.scatter(picks_df['X'],picks_df['Z'],event_marker[0],event_marker[1],marker=event_marker[2],alpha=event_marker[3])
        yz.scatter(picks_df['Z'],picks_df['Y'],event_marker[0],event_marker[1],marker=event_marker[2],alpha=event_marker[3])


        # # # Plotting Fault-planes
        # if type(Faults) == str:
        #   FAULTS = pd.read_csv(Faults,names=['X','Y'])
        #   FAULTS = FAULTS[(FAULTS['X']>=lim_min[0]) & (FAULTS['X']<=lim_max[0]) & (FAULTS['Y']>=lim_min[1]) & (FAULTS['Y']<=lim_max[1])].reset_index(drop=True)
        #   xy.scatter(FAULTS['X'],FAULTS['Y'],fault_plane[0],color=fault_plane[1],linestyle=fault_plane[2],alpha=fault_plane[3],label='Mapped Faults')

        # Plotting legend
        xy.legend(loc='upper left',  markerscale=2, scatterpoints=1, fontsize=10)

        if filepath != None:
            plt.savefig('{}'.format(filepath))
        else:
            plt.show()

        plt.close('all')
