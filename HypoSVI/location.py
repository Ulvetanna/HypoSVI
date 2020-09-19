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
import random
import sys
import json
import copy
from string import digits

# Pytorch Libraires
import torch
from torch.nn import Linear
from torch import Tensor
from torch.nn import MSELoss
from torch.optim import SGD, Adam, RMSprop
from torch.autograd import Variable, grad
from torch.utils.data.sampler import SubsetRandomSampler,WeightedRandomSampler

# Sklearn libraries
from sklearn.cluster import DBSCAN



# === Zach Suggestions ===
# you might want to run this code through an auto formatter
# to clean it up and make it look PEP8 compliant

# -- Additional things that could be added
# ---> Take-off Angles -- Further along the line.

class RBF(torch.nn.Module):
    ''' 
        Radial Basis Function (RBF) 


    '''
    def __init__(self, sigma=None):
        super(RBF, self).__init__()
        self.sigma = sigma

    def forward(self, X, Y):
        XX = X.matmul(X.t())
        XY = X.matmul(Y.t())
        YY = Y.matmul(Y.t())

        dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

        # Apply the median heuristic (PyTorch does not give true median)
        if self.sigma is None:
            np_dnorm2 = dnorm2.detach().cpu().numpy()
            h = np.median(np_dnorm2) / (2 * np.log(X.size(0) + 1))
            sigma = np.sqrt(h).item()
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
    elif rw_type == 'r':
        with open(file, 'r') as f:
            tmpEvents = json.load(f)

    for key in tmpEvents.keys():
        if rw_type=='w':
            tmpEvents[key]['Picks']       = tmpEvents[key]['Picks'].astype(str).to_dict()
        elif rw_type=='r':
            tmpEvents[key]['Picks']       = pd.DataFrame.from_dict(tmpEvents[key]['Picks'])
        else:
            print('Please specify either "read" or "write" for handelling the data')

    if rw_type == 'w':
        with open(file, rw_type) as f:
            json.dump(tmpEvents, f)
    elif rw_type =='r':
        return tmpEvents

def IO_NLLoc2JSON(file,EVT={},startEventID=1000000):
    import pandas as pd
    import numpy as np
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



class HypoSVI(torch.nn.Module):
    def __init__(self, EikoNet, Phases=['P','S'], device=torch.device('cpu')):
        super(HypoSVI, self).__init__()

        # -- Defining the EikoNet input formats
        self.eikonet_Phases  = Phases
        self.eikonet_models  = EikoNet
        if len(self.eikonet_Phases) != len(self.eikonet_models):
            print('Error - Number of phases not equal to number of EikoNet models')
        
        # Determining if the EikoNets are solved for the same domain
        xmin_stack = np.vstack([self.eikonet_models[x].VelocityClass.xmin for x in range(len(self.eikonet_models))])
        xmax_stack = np.vstack([self.eikonet_models[x].VelocityClass.xmax for x in range(len(self.eikonet_models))])
        if not (xmin_stack == xmin_stack[0,:]).all() or not (xmax_stack == xmax_stack[0,:]).all():
            print('Error - EikoNet Models not in the same domain\n Min Points = {}\n Max Points = {}'.format(xmin_stack,xmax_stack))
        
        # Defining the Velocity Class
        self.VelocityClass = self.eikonet_models[0].VelocityClass

        # -- Defining the device to run the location procedure on
        self.device    = device

        # -- Defining the parameters required in the earthquake location procedure
        self.location_info = {}
        self.location_info['Log-likehood']                         = 'EDT' 
        self.location_info['OriginTime Cluster - Seperation (s)']  = 0.3   
        self.location_info['OriginTime Cluster - Minimum Samples'] = 3     
        self.location_info['Hypocenter Cluster - Seperation (km)'] = 3     
        self.location_info['Hypocenter Cluster - Minimum Samples'] = 3     
        self.location_info['Travel Time Uncertainty - [Gradient(km/s),Min(s),Max(s)]'] = [0.1,0.1,0.5] 
        self.location_info['Individual Event Epoch Print Rate']    = None
        self.location_info['Number of Particles']                  = 250 
        self.location_info['Step Size']                            = 5e0 
        self.location_info['Save every * events']                  = 10


        # --------- Initialising Plotting Information ---------
        self.plot_info={}

        # - Event Plot parameters
        # Location plotting
        self.plot_info['EventPlot']  = {}
        self.plot_info['EventPlot']['Errbar std']          = 2.0
        self.plot_info['EventPlot']['Domain Distance']     = 10
        self.plot_info['EventPlot']['Save Type']           = 'png'
        self.plot_info['EventPlot']['Plot kde']            = True
        self.plot_info['EventPlot']['NonClusterd SVGD']    = [0.5,'k']
        self.plot_info['EventPlot']['Clusterd SVGD']       = [1.2,'g']
        self.plot_info['EventPlot']['Hypocenter Location'] = [15,'k']
        self.plot_info['EventPlot']['Hypocenter Errorbar'] = [True,'k']

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
        self.plot_info['EventPlot']['Traces']['Channel Types']        = ['EH*','HH*']
        self.plot_info['EventPlot']['Traces']['Filter Freq']          = [2,16]
        self.plot_info['EventPlot']['Traces']['Normalisation Factor'] = 1.0
        self.plot_info['EventPlot']['Traces']['Time Bounds']          = [0,5]
        self.plot_info['EventPlot']['Traces']['Pick linewidth']       = 2.0
        self.plot_info['EventPlot']['Traces']['Trace linewidth']      = 1.0


        # - Catalogue Plot parameters
        self.plot_info['CataloguePlot'] = {}
        self.plot_info['CataloguePlot']['Minimum Phase Picks']                                           = 12 #min_phases
        self.plot_info['CataloguePlot']['Maximum Location Uncertainty (km)']                             = 15 #max_uncertainty
        self.plot_info['CataloguePlot']['Num Std to define errorbar']                                    = 2  #num_std
        self.plot_info['CataloguePlot']['Event Info - [Size, Color, Marker, Alpha]']                     = [0.1,'r','*',0.8] # event_marker
        self.plot_info['CataloguePlot']['Event Errorbar - [On/Off(Bool),Linewidth,Color,Alpha]']         = [True,0.1,'r',0.8] # event_errorbar_marker
        self.plot_info['CataloguePlot']['Station Marker - [Size,Color,Names On/Off(Bool)]']              = [15,'b',True] # stations_plot
        self.plot_info['CataloguePlot']['Fault Planes - [Size,Color,Marker,Alpha]']                      = [0.1,'gray','-',1.0] # fault_plane


        faults                = '/content/Example_Cahuilla/datasets/socalfaults.llz.txt'
        user_xmin             = [None,None,0]
        user_xmax             = [None,None,20]




        # --- Defining variables and classes to be used
        self._σ_T       = None
        self._optimizer = None
        self._orgTime   = None
        self.K          = RBF()


    def locVar(self,T_obs,T_obs_err):
        '''
            Applying variance from Pick and Distance weighting to each of the observtions
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
        if self.location_info['Log-likehood'] == 'EDT':
            from itertools import combinations
            pairs     = combinations(np.arange(T_obs.shape[1]), 2)
            pairs     = np.array(list(pairs))
            dT_obs    = T_obs[:,pairs[:,0]] - T_obs[:,pairs[:,1]]
            dT_pred   = T_pred[:,pairs[:,0]] - T_pred[:,pairs[:,1]]
            σ_T       = ((σ_T[:,pairs[:,0]])**2 + (σ_T[:,pairs[:,1]])**2)
            logL      = torch.exp((-(dT_obs-dT_pred)**2)/(σ_T))
            logL      = torch.sum(logL,dim=1)

        return logL
    
    def phi(self, X_src, X_rec, t_obs,t_obs_err,t_phase):
        # Setting up the gradient requirements
        X_src = X_src.detach().requires_grad_(True)

        # Preparing EikoNet input
        n_particles = X_src.shape[0]

        # Forcing points to stay within domain 
        X_src[:,0] = torch.clamp(X_src[:,0],self.VelocityClass.xmin[0],self.VelocityClass.xmax[0])
        X_src[:,1] = torch.clamp(X_src[:,1],self.VelocityClass.xmin[1],self.VelocityClass.xmax[1])
        X_src[:,2] = torch.clamp(X_src[:,2],self.VelocityClass.xmin[2],self.VelocityClass.xmax[2])

        # Determining the predicted travel-time for the different phases
        n_obs = 0
        cc=0
        for ind,phs in enumerate(self.eikonet_Phases):
            phase_index = np.where(t_phase==phs)[0]
            if len(phase_index) != 0:
                pha_T_obs     = t_obs[phase_index].repeat(n_particles, 1)
                pha_T_obs_err = t_obs_err[phase_index].repeat(n_particles, 1)
                pha_X_inp     = torch.cat([X_src.repeat_interleave(len(phase_index), dim=0), X_rec[phase_index,:].repeat(n_particles, 1)], dim=1)
                pha_T_pred    = self.eikonet_models[ind].TravelTimes(pha_X_inp).reshape(n_particles,len(phase_index))

                
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
        
        log_L     = self.log_L(T_pred,T_obs,self._σ_T)
        log_prob  = log_L.sum()
        score_func = torch.autograd.grad(log_prob, X_src)[0]

        # Determining the phi
        K_XX     = self.K(X_src, X_src.detach())
        grad_K   = -torch.autograd.grad(K_XX.sum(), X_src)[0]
        phi      = (K_XX.detach().matmul(score_func) + grad_K) / (n_particles)

        # Setting Misfit to zero to restart
        self._σ_T     = None


        # calculating the time offset
        flt_timediff = (abs(T_pred - T_obs).flatten()).detach().cpu().numpy()
        clustering   = DBSCAN(eps=self.location_info['OriginTime Cluster - Seperation (s)'], min_samples=self.location_info['OriginTime Cluster - Minimum Samples']).fit(flt_timediff[None,:])
        indx         = np.where(clustering.labels_ == (np.argmax(np.bincount(np.array(clustering.labels_+1)))-1))[0]

        self.samples_timeDiff           = (T_obs - (T_pred-np.mean(flt_timediff[indx]))).detach().cpu().numpy()
        self.originoffset_mean          = np.mean(flt_timediff[indx])
        self.originoffset_std           = np.std(flt_timediff[indx])
        self.HypocentreSample_timediff  = (self.samples_timeDiff[np.argmin(np.sum(abs(self.samples_timeDiff),axis=1)),:])
        self.HypocentreSample_loc       = (X_src[np.argmin(np.sum(abs(self.samples_timeDiff),axis=1)),:]).detach().cpu().numpy()

        return phi

    def step(self, X_src, X_rec, T_obs, T_obs_err, T_phase):
        self.optim.zero_grad()
        X_src.grad = -self.phi(X_src, X_rec, T_obs, T_obs_err, T_phase)
        self.optim.step()


    def SyntheticCatalogue(self,input_file,Stations,save_path=None):
        '''
            Determining synthetic travel-times between source and reciever locations, returning a JSON pick file for each event

    
            Event_Locations - EventNum, OriginTime, PickErr, X, Y, Z 

            Stations -

        '''

        # Determining the predicted travel-time to each of the stations to corresponding
        #source locations. Optional argumenent to return them as json pick
        evtdf  = pd.read_csv(input_file)
        EVT = {}
        for indx in range(len(evtdf)):
            EVT['{}'.format(evtdf['EventNum'].iloc[indx])] = {}

            # Defining the picks to append
            picks = pd.DataFrame(columns=['Network','Station','PhasePick','DT','PickError'])
            for ind,phs in enumerate(self.eikonet_Phases):
                picks_phs = Stations[['Network','Station','X','Y','Z']]
                picks_phs['PhasePick'] = phs
                picks_phs['PickError'] = evtdf['PickErr'].iloc[indx]
                Pairs       = torch.zeros((int(len(Stations)),6))
                Pairs[:,:3] = Tensor(np.array(evtdf[['X','Y','Z']].iloc[indx]))
                Pairs[:,3:] = Tensor(np.array(picks_phs[['X','Y','Z']]))
                Pairs       = Pairs.to(self.device)
                TT_pred     = self.eikonet_models[ind].TravelTimes(Pairs).detach().to('cpu').numpy()
                del Pairs

                picks_phs['DT']  = (pd.to_datetime(evtdf['OriginTime'].iloc[indx]) + pd.to_timedelta(TT_pred,unit='S')).strftime('%Y/%m/%dT%H:%M:%S.%f')

                picks = picks.append(picks_phs[['Network','Station','PhasePick','DT','PickError']])


            EVT['{}'.format(evtdf['EventNum'].iloc[indx])]['Picks'] = picks


        if type(save_file) == str:
            IO_JSON('{}.json'.format(save_file),Events=EVT,rw_type='w')
        

        return EVT




    def LocateEvents(self,EVTS,Stations,epochs=200,output_plots=True,output_path=None):
        self.Events      = EVTS

        for c,ev in enumerate(self.Events.keys()):

            # Determining the event to look at
            Ev = self.Events[ev]
            
            # Formating the pandas datatypes
            Ev['Picks']['Network']   = Ev['Picks']['Network'].astype(str)
            Ev['Picks']['Station']   = Ev['Picks']['Station'].astype(str)
            Ev['Picks']['PhasePick'] = Ev['Picks']['PhasePick'].astype(str)
            Ev['Picks']['DT']        = pd.to_datetime(Ev['Picks']['DT'])
            Ev['Picks']['PickError'] = Ev['Picks']['PickError'].astype(float)

            # printing the current event being run
            print('Processing Event:{} - Event {} of {} - Number of observtions={}'.format(ev,c,len(self.Events.keys()),len(Ev['Picks'])))

            # Adding the station location to the pick files
            pick_info = pd.merge(Ev['Picks'],Stations[['Network','Station','X','Y','Z']])
            Ev['Picks'] = pick_info[['Network','Station','X','Y','Z','PhasePick','DT','PickError']]

            # Setting up the random seed locations
            X_src       = torch.zeros((int(self.location_info['Number of Particles']),3))
            X_src[:,:3] = Tensor(np.random.rand(int(self.location_info['Number of Particles']),3))*(Tensor(self.VelocityClass.xmax)-Tensor(self.VelocityClass.xmin))[None,:] + Tensor(self.VelocityClass.xmin)[None,:]
            X_src       = Variable(X_src).to(self.device)
            self.optim  = torch.optim.Adam([X_src], self.location_info['Step Size'])
            
            # Defining the arrivals times in seconds
            pick_info['Seconds'] = (pick_info['DT'] - np.min(pick_info['DT'])).dt.total_seconds()

            X_rec       = Tensor(np.array(pick_info[['X','Y','Z']])).to(self.device)
            T_obs       = Tensor(np.array(pick_info['Seconds'])).to(self.device)
            T_obs_err   = Tensor(np.array(pick_info['PickError'])).to(self.device)
            T_obs_phase = np.array(pick_info['PhasePick'])

            X_rec.requires_grad_()
            l = None
            losses = []
            best_l = np.inf
            for epoch in range(epochs):
                self.optim.zero_grad()

                if self.location_info['Individual Event Epoch Print Rate'] != None:
                    if epoch % self.location_info['Individual Event Epoch Print Rate'] == 0:
                        with torch.no_grad():
                            print("Epoch:", epoch, torch.mean(X_src, dim=0), torch.std(X_src, dim=0))

                self.step(X_src, X_rec, T_obs, T_obs_err, T_obs_phase)

            # -- Drop points outside of the domain
            dmindx = [(X_src[:,2] > self.VelocityClass.xmin[2]) & (X_src[:,2] < self.VelocityClass.xmax[2])]
            X_src   = X_src[dmindx[0],:]
            Ev['location']                            = {}
            Ev['location']['SVGD_points']             = X_src.detach().cpu().numpy().tolist()
            
            if len(Ev['location']['SVGD_points']) == 0:
                 Ev['location']['Hypocentre']     = (np.ones(3)*np.nan).tolist()
                 Ev['location']['Hypocentre_std'] = (np.ones(3)*np.nan).tolist()
                 continue

            # -- Determining the dominant cluster of points and estimating hypocentre 
            clustering = DBSCAN(eps=self.location_info['Hypocenter Cluster - Seperation (km)'], min_samples=self.location_info['Hypocenter Cluster - Minimum Samples']).fit(X_src.detach().cpu())
            indx = np.where(clustering.labels_ == (np.argmax(np.bincount(np.array(clustering.labels_+1)))-1))[0]
            optHyp              = torch.mean(X_src[indx,:], dim=0)
            optHyp_std          = torch.std(X_src[indx,:], dim=0)
            Ev['location']['SVGD_points_clusterindx']    = indx.tolist()
            Ev['location']['SVGD_SampleTimeDifferences'] = (self.samples_timeDiff).tolist()
            Ev['location']['Hypocentre']                 = optHyp.detach().cpu().numpy().tolist()
            Ev['location']['Hypocentre_std']             = optHyp_std.detach().cpu().numpy().tolist()
            Ev['location']['Hypocentre_optimalsample']   = (self.HypocentreSample_loc).tolist()
            Ev['location']['OriginTime_std']             = float(self.originoffset_std)
            Ev['location']['OriginTime']                 = str(np.min(pick_info['DT']) - pd.Timedelta(float(self.originoffset_mean),unit='S'))
            Ev['Picks']['TimeDiff']                      = self.HypocentreSample_timediff 

            print('-------- OT= {} - Hyp=[{:.2f},{:.2f},{:.2f}] - Hyp/Std=[{:.2f},{:.2f},{:.2f}]'.format(Ev['location']['OriginTime'],Ev['location']['Hypocentre'][0],Ev['location']['Hypocentre'][1],Ev['location']['Hypocentre'][2],
                                                                                  Ev['location']['Hypocentre_std'][0],Ev['location']['Hypocentre_std'][1],Ev['location']['Hypocentre_std'][2]))

            if output_plots:
                print('-------- Saving Event Plot --------')
                try:
                    self.EventPlot(output_path,Ev,EventID=ev)
                except:
                    print('-------- Issue with saving plot !  --------')

            if self.location_info['Save every * events']  != None:
                if (c%self.location_info['Save every * events']) == 0:
                    IO_JSON('{}/Catalogue.json'.format(output_path),Events=self.Events,rw_type='w')

        # Writing out final catalogue
        IO_JSON('{}/Catalogue.json'.format(output_path),Events=self.Events,rw_type='w')

    def EventPlot(self,PATH,Event,EventID=None):
        plt.close('all')
        OT             = str(Event['location']['OriginTime'])
        OT_std         = Event['location']['OriginTime_std']
        locs           = np.array(Event['location']['SVGD_points'])
        optimalloc     = np.array(Event['location']['Hypocentre'])
        optimalloc_std = np.array(Event['location']['Hypocentre_std'])*self.plot_info['EventPlot']['Errbar std']
        indx_cluster   = np.array(Event['location']['SVGD_points_clusterindx'])
        Stations       = Event['Picks'][['Station','X','Y','Z']]

        if self.plot_info['EventPlot']['Traces']['Plot Traces']==True:
            fig = plt.figure(figsize=(20, 9))
            xz  = plt.subplot2grid((3, 5), (2, 0), colspan=2)
            xy  = plt.subplot2grid((3, 5), (0, 0), colspan=2, rowspan=2,sharex=xz)
            yz  = plt.subplot2grid((3, 5), (0, 2), rowspan=2, sharey=xy)
            trc = plt.subplot2grid((3, 5), (0, 3), rowspan=3, colspan=2)
        else:
            fig = plt.figure(figsize=(9, 9))
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
            xy.set_xlim([optimalloc[0]-self.plot_info['EventPlot']['Domain Distance']/2,optimalloc[0]+self.plot_info['EventPlot']['Domain Distance']/2])
            xy.set_ylim([optimalloc[1]-self.plot_info['EventPlot']['Domain Distance']/2,optimalloc[1]+self.plot_info['EventPlot']['Domain Distance']/2])
            xz.set_xlim([optimalloc[0]-self.plot_info['EventPlot']['Domain Distance']/2,optimalloc[0]+self.plot_info['EventPlot']['Domain Distance']/2])
            xz.set_ylim([optimalloc[2]-self.plot_info['EventPlot']['Domain Distance']/2,optimalloc[2]+self.plot_info['EventPlot']['Domain Distance']/2])
            yz.set_xlim([optimalloc[2]-self.plot_info['EventPlot']['Domain Distance']/2,optimalloc[2]+self.plot_info['EventPlot']['Domain Distance']/2])
            yz.set_ylim([optimalloc[1]-self.plot_info['EventPlot']['Domain Distance']/2,optimalloc[1]+self.plot_info['EventPlot']['Domain Distance']/2])
        else:
            lim_min = self.VelocityClass.xmin
            lim_max = self.VelocityClass.xmax
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
            sns.kdeplot(locs[indx_cluster,0],locs[indx_cluster,1], cmap="Reds",ax=xy,zorder=-1)
            sns.kdeplot(locs[indx_cluster,0],locs[indx_cluster,2], cmap="Reds",ax=xz,zorder=-1)
            sns.kdeplot(locs[indx_cluster,2],locs[indx_cluster,1], cmap="Reds",ax=yz,zorder=-1)

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
            xy.errorbar(optimalloc[0], optimalloc[1],xerr=optimalloc_std[0], yerr=optimalloc_std[1],color=self.plot_info['EventPlot']['Hypocenter Errorbar'][1],label='Hyp {}-stds'.format(self.plot_info['EventPlot']['Errbar std']))
            xz.errorbar(optimalloc[0],optimalloc[2],xerr=optimalloc_std[0], yerr=optimalloc_std[2],color=self.plot_info['EventPlot']['Hypocenter Errorbar'][1],label='Hyp {}stds'.format(self.plot_info['EventPlot']['Errbar std']))
            yz.errorbar(optimalloc[2],optimalloc[1],xerr=optimalloc_std[2], yerr=optimalloc_std[1],color=self.plot_info['EventPlot']['Hypocenter Errorbar'][1],label='Hyp {}stds'.format(self.plot_info['EventPlot']['Errbar std']))


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
        xy.legend(loc='upper left')
        plt.suptitle(' Earthquake {} +/- {:.2f}s\n Hyp=[{:.2f},{:.2f},{:.2f}] - Hyp Uncertainty +/- [{:.2f},{:.2f},{:.2f}]'.format(OT,OT_std,optimalloc[0],optimalloc[1],optimalloc[2],optimalloc_std[0],optimalloc_std[1],optimalloc_std[2]))                                                                                                 

        if self.plot_info['EventPlot']['Traces']['Plot Traces']:
            # Determining Event data information
            evt_yr         = str(pd.to_datetime(Event['Picks']['DT'].min()).year)
            evt_jd         = str(pd.to_datetime(Event['Picks']['DT'].min()).dayofyear).zfill(3)
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
                    st  = obspy.read('{}/{}/{}/*{}*'.format(Host_path,evt_yr,evt_jd,sta),
                                 starttime=evt_starttime-10,endtime=evt_endtime)

                    # Selecting only the user specified channels
                    for ch in self.plot_info['EventPlot']['Traces']['Channel Types']:
                        ST  = ST +  st.select(channel=ch,network=net).filter('bandpass',freqmin=self.plot_info['EventPlot']['Traces']['Filter Freq'][0],freqmax=self.plot_info['EventPlot']['Traces']['Filter Freq'][1])
                except:
                    continue

            # Plotting the Trace data
            yloc = np.arange(len(stations)) + 1
            for indx,staName in enumerate(stations):
                try:
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
                except:
                    continue

            trc.yaxis.tick_right()
            trc.yaxis.set_label_position("right")
            trc.set_xlim([0,evt_endtime - evt_starttime])
            trc.set_ylim([0,len(stations)+1])
            trc.set_yticks(np.arange(1,len(stations)+1))
            trc.set_yticklabels(stations)
            trc.set_xlabel('Seconds since earthquake origin')

        plt.savefig('{}/{}.{}'.format(PATH,EventID,self.plot_info['EventPlot']['Save Type']))


    def CataloguePlot(self,filepath=None,Events=None,Stations=None,user_xmin=[None,None,None],user_xmax=[None,None,None], Faults=None):

        if type(Events) != type(None):
            self.Events = Events


        # - Catalogue Plot parameters
        min_phases            = self.plot_info['CataloguePlot']['Minimum Phase Picks']
        max_uncertainty       = self.plot_info['CataloguePlot']['Maximum Location Uncertainty (km)']
        num_std               =  self.plot_info['CataloguePlot']['Num Std to define errorbar']
        event_marker          = self.plot_info['CataloguePlot']['Event Info - [Size, Color, Marker, Alpha]']
        event_errorbar_marker = self.plot_info['CataloguePlot']['Event Errorbar - [On/Off(Bool),Linewidth,Color,Alpha]']
        stations_plot         = self.plot_info['CataloguePlot']['Station Marker - [Size,Color,Names On/Off(Bool)]'] 
        fault_plane           = self.plot_info['CataloguePlot']['Fault Planes - [Size,Color,Marker,Alpha]']


        fig = plt.figure(figsize=(15, 15))
        xz  = plt.subplot2grid((3, 3), (2, 0), colspan=2)
        xy  = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2,sharex=xz)
        yz  = plt.subplot2grid((3, 3), (0, 2), rowspan=2, sharey=xy) 


        # Defining the limits of the domain
        lim_min = self.VelocityClass.xmin
        lim_max = self.VelocityClass.xmax

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


        # Loading location information
        picks =(np.zeros((len(self.Events.keys()),8))*np.nan).astype(str)
        for indx,evtid in enumerate(self.Events.keys()):
            try:
                picks[indx,0]   = str(evtid)
                picks[indx,1]   = self.Events[evtid]['location']['OriginTime']
                picks[indx,2:5] = (np.array(self.Events[evtid]['location']['Hypocentre'])).astype(str)
                picks[indx,5:]  = (np.array(self.Events[evtid]['location']['Hypocentre_std'])*num_std).astype(str)
            except:
                continue
        picks_df = pd.DataFrame(picks,
                                columns=['EventID','DT','X','Y','Z','ErrX','ErrY','ErrZ'])
        picks_df['X'] = picks_df['X'].astype(float)
        picks_df['Y'] = picks_df['Y'].astype(float)
        picks_df['Z'] = picks_df['Z'].astype(float)
        picks_df['ErrX'] = picks_df['ErrX'].astype(float)
        picks_df['ErrY'] = picks_df['ErrY'].astype(float)
        picks_df['ErrZ'] = picks_df['ErrY'].astype(float)
        picks_df = picks_df.dropna(axis=0)
        picks_df = picks_df[np.sum(picks_df[['ErrX','ErrY','ErrZ']],axis=1) <= max_uncertainty].reset_index(drop=True)
        picks_df['DT'] = pd.to_datetime(picks_df['DT'])

        # Plotting Location info
        if event_errorbar_marker[0]:
            xy.errorbar(picks_df['X'],picks_df['Y'],xerr=picks_df['ErrX'],yerr=picks_df['ErrY'],fmt='none',linewidth=event_errorbar_marker[1],color=event_errorbar_marker[2],alpha=event_errorbar_marker[3],label='Catalogue Errorbars')
            xz.errorbar(picks_df['X'],picks_df['Z'],xerr=picks_df['ErrX'],yerr=picks_df['ErrZ'],fmt='none',linewidth=event_errorbar_marker[1],color=event_errorbar_marker[2],alpha=event_errorbar_marker[3])
            yz.errorbar(picks_df['Z'],picks_df['Y'],xerr=picks_df['ErrZ'],yerr=picks_df['ErrY'],fmt='none',linewidth=event_errorbar_marker[1],color=event_errorbar_marker[2],alpha=event_errorbar_marker[3])


        xy.scatter(picks_df['X'],picks_df['Y'],event_marker[0],event_marker[1],marker=event_marker[2],alpha=event_marker[3],label='Catalogue Locations')
        xz.scatter(picks_df['X'],picks_df['Z'],event_marker[0],event_marker[1],marker=event_marker[2],alpha=event_marker[3])
        yz.scatter(picks_df['Z'],picks_df['Y'],event_marker[0],event_marker[1],marker=event_marker[2],alpha=event_marker[3])


        # # Plotting Fault-planes
        if type(Faults) == str:
          FAULTS = pd.read_csv(Faults,sep=r'\s+',names=['Long','Lat','Z','iD'])
          FAULTS = FAULTS.dropna(axis=0).reset_index(drop=True)
          FAULTS['X'],FAULTS['Y'] = projection(np.array(FAULTS['Long']),np.array(FAULTS['Lat']))
          xy.scatter(FAULTS['X'],FAULTS['Y'],fault_plane[0],color=fault_plane[1],linestyle=fault_plane[2],alpha=fault_plane[3],label='Mapped Faults')

        # Plotting legend
        xy.legend(loc='upper left',  markerscale=2, scatterpoints=1, fontsize=10)

        if filepath != None:
            plt.savefig('{}'.format(filepath))
        else:
            plt.show()

        plt.close('all')