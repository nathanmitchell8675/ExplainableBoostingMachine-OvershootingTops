import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import axes
import pandas as pd
from sklearn import metrics
import xarray as xr
import pickle as pkl
import matplotlib.ticker as tkr

from interpret import set_visualize_provider
from interpret.provider import InlineProvider
set_visualize_provider(InlineProvider())
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show

from matplotlib import ticker
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

from scipy.ndimage import zoom

import math
import warnings
warnings.filterwarnings('ignore')
#####################################################

tile_size = 4

#Load in the data
#data = xr.open_dataset('/home/nmitchell/GLCM/training1.nc')
#data = xr.open_dataset('/home/nmitchell/GLCM/validation1.nc')
data = xr.open_dataset('/home/nmitchell/GLCM/test1.nc')

min_ir = 190.65048
max_ir = 321.95
max_value = 7.052267824115893

### Model Loading ###
filepath = r'/home/nmitchell/GLCM/models/13BM'

pred_threshold = 0.50

with open(filepath, 'rb') as file:
    model = pkl.load(file)
ebm = model["Model"][0]

### Feature Renaming ###
names = ebm.term_names_
names = '  '.join(names)

feature_names = ['Brightness', 'Cool Contrast Tiles', 'Infrared']
for i in range(len(feature_names)):
    names = names.replace('feature_000' + str(i), feature_names[i])
names = names.split('  ')

#Sort the scores in descending order and sort the names along with them
all_scores, all_names = zip(*sorted(zip(ebm.explain_global().data()['scores'], names)))

### Shape Function Placeholders ###
global edited_shape_fxs
edited_shape_fxs = np.array([])

global edit_range
edit_range = np.array([])

### Function to Alter the EBM Model ###
def alter_model():

    global edited_shape_fxs
    edited_shape_fxs = np.array([])

    global edit_range
    edit_range = np.array([])

    ### Brightness (0) ###
    xvals = np.array(ebm.explain_global().data(0)['names'][:])
    yvals = np.array(ebm.explain_global().data(0)['scores'][:])

    yvals[0:523] = np.zeros((523)) - 1.9423130631477825
    ebm.explain_global().data(0)['scores'][:] = yvals

    ### Cool Contrast Tiles (1) ###
    xvals = np.array(ebm.explain_global().data(1)['names'][:])
    yvals = np.array(ebm.explain_global().data(1)['scores'][:])

    ebm.scale(1,2.15)
    ebm.explain_global().data(1)['scores'][:] = ebm.explain_global().data(1)['scores'][:] - 5
    ebm.explain_global().data(1)['scores'][0] = ebm.explain_global().data(1)['scores'][1]

#alter_model()

###########################

def get_statistics():
    brightness = data.Brightness.values.reshape(-1,1).flatten()
    infrared   = data.Infrared_Image.values.reshape(-1,1).flatten()
    cool_glcm  = data.Cool_Contrast_Tiles.values.reshape(-1,1).flatten()

    X_val = np.transpose(np.array([brightness, cool_glcm, infrared]))
    y_val = data.Masked_Truth.values.flatten()

    pred_percentage = np.array(ebm.predict_proba(X_val)[:,1]).flatten()
    y_pred = np.where(pred_percentage > pred_threshold, 1, 0)

    tn, fp, fn, tp = metrics.confusion_matrix(y_val, y_pred).ravel()

    print("Edited Model:")
    print("True Positives:  ", tp)
    print("True Negatives:  ", tn)
    print("False Positives: ", fp)
    print("False Negatives: ", fn)
    print("SR:  ", (tp/(tp+fp)))
    print("POD: ", (tp/(tp+fn)))

#get_statistics()

#Create useful colormaps to be used while plotting
ones   = np.ones((int(256/tile_size), int(254/tile_size)))
red_cm = mpl.colors.LinearSegmentedColormap.from_list(" ", ["crimson", "black"])
gld_cm = mpl.colors.LinearSegmentedColormap.from_list(" ", ["gold", "black"])
blu_cm = mpl.colors.LinearSegmentedColormap.from_list(" ", ["mediumblue", "black"])
h_cm   = mpl.colors.LinearSegmentedColormap.from_list(" ", ["red", "black"])

#Case Studies, in order
#examples = [766]
#examples = [1042]
#examples = [437]
examples = [637]
#examples = [726]

for isamp in examples:
    print(isamp)
    #Pulls data needed to create 'X'
    #Data Pulled: Convoled Image, Cool Contrast Tiles, Infrared Image
    brightness          = data.Brightness.sel(Sample = isamp).values.reshape(-1,1).flatten()
    infrared            = data.Infrared_Image.sel(Sample = isamp).values.reshape(-1,1).flatten()
    cool_contrast_tiles = data.Cool_Contrast_Tiles.sel(Sample = isamp).values.reshape(-1,1).flatten()

    #Combine features into one variable for prediction
    X_val = np.transpose(np.array([brightness, cool_contrast_tiles, infrared]))
    #y_val = data.Masked_Truth.values.flatten()

    #Pull Remaining Data / Reshape
    full_mrms      = data.Ground_Truth.sel(Sample = isamp).values.reshape(int(256/tile_size), -1)
    masked_mrms    = data.Masked_Truth.sel(Sample = isamp).values.reshape(int(256/tile_size), -1).astype(np.uint8)
    original_image = data.Original_Image.sel(Sample = isamp).values.reshape(256,256)

    brightness          = brightness.reshape(64,64)
    infrared            = infrared.reshape(64,64)
    cool_contrast_tiles = cool_contrast_tiles.reshape(64,64)

    #Latitude / Longitude for 2 km resolution
    lat_hi = data.Latitude_High.sel(Sample = isamp).values
    lon_hi = data.Longitude_High.sel(Sample = isamp).values

    #Latitude / Longitude for 0.5 km resolution
    lat_lo = data.Latitude_Low.sel(Sample = isamp).values
    lon_lo = data.Longitude_Low.sel(Sample = isamp).values

    #Date & Time when the image was captured
    date = data.Date.sel(Sample = isamp).values

    #########################################

    ### Functions for Generating Satellite-Imagery-Based Figures, In Order ###

    #Function to remove ticks from large images
    def remove_ticks(ax, title):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, y = 0.99)

    def fig_one():
        fig, ax = plt.subplots(2,2)

        for i in range(2):
            for j in range(2):
                for spine in ['top', 'bottom', 'left', 'right']:
                    ax[i,j].spines[spine].set_visible(False)

        ticks = [0.0,255.0]
        positions = [ax[0,0], ax[1,0], ax[1,1]]
        for pos in positions:
            pos.set_xticks(ticks)
            pos.set_yticks(ticks)

        dic1 = { 0: str(round(min(lon_hi.flatten()),2)) + r"$^\circ$W", 255.0 : str(round(max(lon_hi.flatten()),2)) + r"$^\circ$W"}
        dic2 = { 0: str(round(min(lat_hi.flatten()),2)) + r"$^\circ$N", 255.0 : str(round(max(lat_hi.flatten()),2)) + r"$^\circ$N"}

        labels1 = [ticks[i] if t not in dic1.keys() else dic1[t] for i,t in enumerate(ticks)]
        labels2 = [ticks[i] if t not in dic2.keys() else dic2[t] for i,t in enumerate(ticks)]

        for pos in positions:
            pos.set_xticklabels(labels1)
            pos.set_yticklabels(labels2)

        #For small scene
        ticks = [0.0,63.0]
        ax[0,1].set_xticks(ticks)
        ax[0,1].set_yticks(ticks)

        dic1 = { 0: str(round(min(lon_lo.flatten()),2)) + r"$^\circ$W", 63.0 : str(round(max(lon_lo.flatten()),2)) + r"$^\circ$W"}
        dic2 = { 0: str(round(min(lat_lo.flatten()),2)) + r"$^\circ$N", 63.0 : str(round(max(lat_lo.flatten()),2)) + r"$^\circ$N"}

        labels1 = [ticks[i] if t not in dic1.keys() else dic1[t] for i,t in enumerate(ticks)]
        labels2 = [ticks[i] if t not in dic2.keys() else dic2[t] for i,t in enumerate(ticks)]

        ax[0,1].set_xticklabels(labels1)
        ax[0,1].set_yticklabels(labels2)

        reflectance_ax = ax[0,0].imshow(original_image, cmap = 'gray', extent = (0,256,0,256), clim = (0,225))
        cbar = plt.colorbar(reflectance_ax, ax = ax[0,0], location = 'top', fraction = 0.04675, pad = 0.01)
        cbar.set_ticks(np.linspace(0, 225, 5))

        cbar.ax.spines['outline'].set_linewidth(0.5)

        infrared_ax = ax[0,1].imshow(infrared, cmap = 'gray_r', extent = (0,64,0,64), clim = (185,325))
        cbar = plt.colorbar(infrared_ax, ax = ax[0,1], location = 'top', fraction = 0.04675, pad = 0.01)
        cbar.set_ticks(np.linspace(185, 325, 5))

        cbar.ax.spines['outline'].set_linewidth(0.5)

        ir = zoom(infrared, 4.0, order=3, mode = 'nearest')
        ir = np.where(ir > 250, np.nan, ir)
        ax[1,0].imshow(original_image, cmap = 'gray', extent = (0,256,0,256))
        sandwich = ax[1,0].imshow(ir, cmap = 'turbo_r', extent = (0,256,0,256), clim = (min_ir,250), alpha = 0.65)
        cbar = plt.colorbar(sandwich, ax = ax[1,0], location = 'top', fraction=0.046, pad=0.01)
        cbar.set_ticks(np.linspace(min_ir, 250, 5))

        cbar.ax.spines['outline'].set_linewidth(0.5)

        labels = ax[1,1].imshow(original_image, cmap = 'gray', extent = (0,256,0,256), clim = (0,225))
        ax[1,1].imshow(ones, alpha = full_mrms*0.25, cmap = red_cm, extent = (0,256,0,256)) #Full Ground Truth (low opacity)
        ax[1,1].imshow(ones, alpha = masked_mrms,    cmap = red_cm, extent = (0,256,0,256)) #IR Masked Ground Truth (full opacity)
        cbar = plt.colorbar(reflectance_ax, ax = ax[1,1], location = 'top', fraction = 0.04675, pad = 0.01)
        cbar.set_ticks(np.linspace(0, 225, 5))

        cbar.ax.spines['outline'].set_linewidth(0.5)

        all_pos = [ax[0,0], ax[0,1], ax[1,0], ax[1,1]]
        for pos in all_pos:
            pos.tick_params(axis='both', which='major', direction='out', length=7.5)

        fig = plt.gcf()
        fig.set_size_inches(18,7.75)
        plt.subplots_adjust(left = 0.05, bottom = 0.038, right = 0.971, top = .964, wspace = 0.34, hspace = 0.5)
        #plt.savefig(r'/home/nmitchell/GLCM/useful-images/fig_one.png', format="png", dpi=300)

        plt.show()

    #fig_one()
    #continue

    def fig_four():
        fig, ax = plt.subplots()

        ticks = [0.0,63.0]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        dic1 = { 0: str(round(min(lon_lo.flatten()),1)) + r"$^\circ$W", 63.0 : str(round(max(lon_lo.flatten()),1)) + r"$^\circ$W"}
        dic2 = { 0: str(round(min(lat_lo.flatten()),1)) + r"$^\circ$N", 63.0 : str(round(max(lat_lo.flatten()),1)) + r"$^\circ$N"}

        labels1 = [ticks[i] if t not in dic1.keys() else dic1[t] for i,t in enumerate(ticks)]
        labels2 = [ticks[i] if t not in dic2.keys() else dic2[t] for i,t in enumerate(ticks)]

        ax.set_xticklabels(labels1)
        ax.set_yticklabels(labels2)
        ax.tick_params(labelsize=20)

        bright = ax.imshow(brightness, cmap = 'gray', extent = (0,64,0,64), clim = (0,175))
        cbar = plt.colorbar(bright, ax = ax, location = 'right', fraction = 0.04675, pad = 0.025)
        cbar.set_ticks(np.linspace(0, 175, 5))
        cbar.ax.tick_params(labelsize=20)

        for spine in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine].set_visible(False)
        cbar.ax.spines['outline'].set_linewidth(0.5)

        ax.tick_params(axis='both', which='major', direction='out', length=15)

        fig = plt.gcf()
        fig.set_size_inches(18,7.75)
        plt.subplots_adjust(left = 0, bottom = 0.08, right = 0.795, top = 0.986, wspace = 0.2, hspace = 0.5)
        plt.savefig(r'/home/nmitchell/GLCM/useful-images/fig_four.png', format="png", dpi=300)

        plt.show()

    #fig_four()
    #continue

    def fig_five():
        fig, ax = plt.subplots()

        ticks = [0.0,63.0]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        dic1 = { 0: str(round(min(lon_lo.flatten()),1)) + r"$^\circ$W", 63.0 : str(round(max(lon_lo.flatten()),1)) + r"$^\circ$W"}
        dic2 = { 0: str(round(min(lat_lo.flatten()),1)) + r"$^\circ$N", 63.0 : str(round(max(lat_lo.flatten()),1)) + r"$^\circ$N"}

        labels1 = [ticks[i] if t not in dic1.keys() else dic1[t] for i,t in enumerate(ticks)]
        labels2 = [ticks[i] if t not in dic2.keys() else dic2[t] for i,t in enumerate(ticks)]

        ax.set_xticklabels(labels1)
        ax.set_yticklabels(labels2)
        ax.tick_params(labelsize=20)

        glcm = ax.imshow(cool_contrast_tiles, cmap = 'gray_r', clim = (0,1))
        cbar = plt.colorbar(glcm, ax = ax, location = 'right', fraction = 0.04675, pad = 0.025)
        cbar.set_ticks(np.linspace(0, 1, 5))
        cbar.ax.tick_params(labelsize=20)

        cbar.ax.spines['outline'].set_linewidth(0.5)

        ax.tick_params(axis='both', which='major', direction='out', length=15)

        fig = plt.gcf()
        fig.set_size_inches(18,7.75)
        plt.subplots_adjust(left = 0, bottom = 0.08, right = 0.795, top = 0.986, wspace = 0.2, hspace = 0.5)
        #plt.savefig(r'/home/nmitchell/GLCM/useful-images/fig_five.png', format="png", dpi=300)

        plt.show()

    #fig_five()
    #continue

    def fig_six():
        fig, ax = plt.subplots()

        ticks = [0.0,63.0]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        dic1 = { 0: str(round(min(lon_lo.flatten()),1)) + r"$^\circ$W", 63.0 : str(round(max(lon_lo.flatten()),1)) + r"$^\circ$W"}
        dic2 = { 0: str(round(min(lat_lo.flatten()),1)) + r"$^\circ$N", 63.0 : str(round(max(lat_lo.flatten()),1)) + r"$^\circ$N"}

        labels1 = [ticks[i] if t not in dic1.keys() else dic1[t] for i,t in enumerate(ticks)]
        labels2 = [ticks[i] if t not in dic2.keys() else dic2[t] for i,t in enumerate(ticks)]

        ax.set_xticklabels(labels1)
        ax.set_yticklabels(labels2)
        ax.tick_params(labelsize=20)

        inf = ax.imshow(infrared, cmap = 'gray_r', extent = (0,64,0,64), clim = (185,325))
        cbar = plt.colorbar(inf, ax = ax, location = 'right', fraction = 0.04675, pad = 0.025)
        cbar.set_ticks(np.linspace(185, 325, 5))
        cbar.ax.tick_params(labelsize=20)

        for spine in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine].set_visible(False)
        cbar.ax.spines['outline'].set_linewidth(0.5)

        ax.tick_params(axis='both', which='major', direction='out', length=15)

        fig = plt.gcf()
        fig.set_size_inches(18,7.75)
        plt.subplots_adjust(left = 0, bottom = 0.08, right = 0.795, top = 0.986, wspace = 0.2, hspace = 0.5)
        #plt.savefig(r'/home/nmitchell/GLCM/useful-images/fig_six.png', format="png", dpi=300)

        plt.show()

    #fig_six()
    #continue

    def fig_seven():
        fig, ax = plt.subplots()

        ticks = [0.0,255.0]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        dic1 = { 0: str(round(min(lon_hi.flatten()),2)) + r"$^\circ$W", 255.0 : str(round(max(lon_hi.flatten()),2)) + r"$^\circ$W"}
        dic2 = { 0: str(round(min(lat_hi.flatten()),2)) + r"$^\circ$N", 255.0 : str(round(max(lat_hi.flatten()),2)) + r"$^\circ$N"}

        labels1 = [ticks[i] if t not in dic1.keys() else dic1[t] for i,t in enumerate(ticks)]
        labels2 = [ticks[i] if t not in dic2.keys() else dic2[t] for i,t in enumerate(ticks)]

        ax.set_xticklabels(labels1)
        ax.set_yticklabels(labels2)
        ax.tick_params(labelsize=20)

        reflectance = ax.imshow(original_image, cmap = 'gray', extent = (0,256,0,256), clim = (0,225))
        cbar = plt.colorbar(reflectance, ax = ax, location = 'right', fraction = 0.04675, pad = 0.025)
        cbar.set_ticks(np.linspace(0,225,5))
        cbar.ax.tick_params(labelsize=20)

        ax.imshow(ones, alpha = full_mrms*0.25, cmap = red_cm, extent = (0,256,0,256)) #Full Ground Truth (low opacity)
        ax.imshow(ones, alpha = masked_mrms,    cmap = red_cm, extent = (0,256,0,256)) #IR Masked Ground Truth (full opacity)
        ax.tick_params(axis='both', which='major', direction='out', length=15)

        for spine in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine].set_visible(False)
        cbar.ax.spines['outline'].set_linewidth(0.5)

        fig = plt.gcf()
        fig.set_size_inches(18,7.75)
        plt.subplots_adjust(left = 0, bottom = 0.08, right = 0.795, top = 0.986, wspace = 0.2, hspace = 0.5)
        plt.savefig(r'/home/nmitchell/GLCM/useful-images/fig_seven.png', format="png", dpi=300)

        plt.show()

    #fig_seven()
    #continue

    def fig_eight(a):
        #alter_model()

        name = names[a].lower().replace(' ', '_')
        fig, ax = plt.subplots(1,4)
        for i in range(4):
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        gs = ax[0].get_gridspec()

        shape_fx = fig.add_subplot(gs[2:4])
        shape_fx.grid(True, linestyle = ':')
        shape_fx.set_axisbelow(True)

        max_cm = max(ebm.eval_terms(X_val).flatten())
        min_cm = min(ebm.eval_terms(X_val).flatten())
        plot_max = max(abs(max_cm), abs(min_cm))

        for i in range(2):
            for spine in ['top', 'bottom', 'left', 'right']:
                ax[i].spines[spine].set_visible(False)

        IM = np.array(ebm.eval_terms(X_val)[:,a])
        importance = ax[1].imshow(IM.reshape(64,64), cmap = 'seismic_r', extent = (0,64,0,64), clim = (-plot_max, plot_max))
        cbar = plt.colorbar(importance, ax = ax[1], location = 'bottom', fraction=0.046, pad=0.04)
        cbar.set_ticks(np.linspace(-plot_max, plot_max, 3))
        cbar.ax.yaxis.set_major_formatter(tkr.FormatStrFormatter('%.1f'))
        cbar.ax.spines['outline'].set_linewidth(0.5)
        cbar.ax.tick_params(labelsize=15)

        og_image = ax[0].imshow(brightness, cmap = 'gray', extent = (0,64,0,64), clim = (0,175))
        cbar = plt.colorbar(og_image, ax = ax[0], location = 'bottom', fraction = 0.046, pad = 0.04)
        cbar.set_ticks(np.linspace(0, 175, 3))
        cbar.ax.spines['outline'].set_linewidth(0.5)
        cbar.ax.tick_params(labelsize=15)

        xvals = ebm.explain_global().data(a)['names'][1:len(ebm.explain_global().data(a)['names'])]
        yvals = ebm.explain_global().data(a)['scores']

        shape_fx.plot(xvals, yvals, color = 'dimgray')

        lower_bounds = ebm.explain_global().data(a)['lower_bounds']
        upper_bounds = ebm.explain_global().data(a)['upper_bounds']

        shape_fx.fill_between(xvals, lower_bounds, upper_bounds, alpha = 0.25, color = 'dimgray')
        shape_fx.tick_params(labelsize=15)

        fig = plt.gcf()
        fig.set_size_inches(18,7.75)
        plt.subplots_adjust(left = 0.005, bottom = 0.335, right = 0.8, top = .69, wspace = 0.175, hspace = 0.347)
        plt.savefig(r'/home/nmitchell/GLCM/useful-images/fig_eight.png', format="png", dpi=600)

        plt.show()

    #fig_eight(0)
    #continue

    def fig_nine(a):
        #alter_model()

        name = names[a].lower().replace(' ', '_')
        fig, ax = plt.subplots(1,4)
        for i in range(4):
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        gs = ax[0].get_gridspec()

        shape_fx = fig.add_subplot(gs[2:4])
        shape_fx.grid(True, linestyle = ':')
        shape_fx.set_axisbelow(True)

        max_cm = max(ebm.eval_terms(X_val).flatten())
        min_cm = min(ebm.eval_terms(X_val).flatten())
        plot_max = max(abs(max_cm), abs(min_cm))

        for i in range(2):
            for spine in ['top', 'bottom', 'left', 'right']:
                ax[i].spines[spine].set_visible(False)

        og_image = ax[0].imshow(brightness, cmap = 'gray', extent = (0,64,0,64), clim = (0,175))
        cbar = plt.colorbar(og_image, ax = ax[0], location = 'bottom', fraction = 0.046, pad = 0.04)
        cbar.set_ticks(np.linspace(0, 175, 3))
        cbar.ax.spines['outline'].set_linewidth(0.5)
        cbar.ax.tick_params(labelsize=15)

        og_image = ax[1].imshow(brightness, cmap = 'gray', extent = (0,64,0,64), clim = (0,175))
        cbar = plt.colorbar(og_image, ax = ax[1], location = 'bottom', fraction = 0.046, pad = 0.04)
        cbar.set_ticks(np.linspace(0, 175, 3))
        cbar.ax.spines['outline'].set_linewidth(0.5)
        cbar.ax.tick_params(labelsize=15)

        mask_1 = np.where(((brightness >= 11.027594245690477) & (brightness <= 24.38345612209219)), 1.0, np.nan)
        mask_2 = np.where(((brightness >= 0) & (brightness <= 32)), 1.0, np.nan)
        ax[1].imshow(mask_2, cmap = gld_cm, extent = (0,64,0,64))
        ax[1].imshow(mask_1, cmap = red_cm, extent = (0,64,0,64))

        x_1 = ebm.explain_global().data(a)['names'][1:118]
        y_1 = ebm.explain_global().data(a)['scores'][1:118]

        x_2 = ebm.explain_global().data(a)['names'][119:440]
        y_2 = ebm.explain_global().data(a)['scores'][119:440]

        x_3 = ebm.explain_global().data(a)['names'][440:523]
        y_3 = ebm.explain_global().data(a)['scores'][440:523]

        x_4 = ebm.explain_global().data(a)['names'][523:len(ebm.explain_global().data(a)['names'])]
        y_4 = ebm.explain_global().data(a)['scores'][522:len(ebm.explain_global().data(a)['scores'])]

        shape_fx.plot(x_1, y_1, color = 'gold')
        shape_fx.plot(x_2, y_2, color = 'red')
        shape_fx.plot(x_3, y_3, color = 'gold')
        shape_fx.plot(x_4, y_4, color = 'dimgray')

        lower_bounds_1 = ebm.explain_global().data(a)['lower_bounds'][1:118]
        lower_bounds_2 = ebm.explain_global().data(a)['lower_bounds'][119:440]
        lower_bounds_3 = ebm.explain_global().data(a)['lower_bounds'][440:523]
        lower_bounds_4 = ebm.explain_global().data(a)['lower_bounds'][522:len(ebm.explain_global().data(a)['names'])]

        upper_bounds_1 = ebm.explain_global().data(a)['upper_bounds'][1:118]
        upper_bounds_2 = ebm.explain_global().data(a)['upper_bounds'][119:440]
        upper_bounds_3 = ebm.explain_global().data(a)['upper_bounds'][440:523]
        upper_bounds_4 = ebm.explain_global().data(a)['upper_bounds'][522:len(ebm.explain_global().data(a)['names'])]

        shape_fx.fill_between(x_1, lower_bounds_1, upper_bounds_1, alpha = 0.25, color = 'gold')
        shape_fx.fill_between(x_2, lower_bounds_2, upper_bounds_2, alpha = 0.25, color = 'red')
        shape_fx.fill_between(x_3, lower_bounds_3, upper_bounds_3, alpha = 0.25, color = 'gold')
        shape_fx.fill_between(x_4, lower_bounds_4, upper_bounds_4, alpha = 0.25, color = 'dimgray')

        shape_fx.tick_params(labelsize=15)

        fig = plt.gcf()
        fig.set_size_inches(18,7.75)
        plt.subplots_adjust(left = 0.005, bottom = 0.335, right = 0.8, top = .69, wspace = 0.175, hspace = 0.347)
        #plt.savefig(r'/home/nmitchell/GLCM/useful-images/fig_nine.png', format="png", dpi=600)

        plt.show()

    #fig_nine(0)
    #continue

    def fig_ten(a):
        alter_model()

        name = names[a].lower().replace(' ', '_')
        fig, ax = plt.subplots(1,4)
        for i in range(4):
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        gs = ax[0].get_gridspec()

        shape_fx = fig.add_subplot(gs[2:4])
        shape_fx.grid(True, linestyle = ':')
        shape_fx.set_axisbelow(True)

        max_cm = max(ebm.eval_terms(X_val).flatten())
        min_cm = min(ebm.eval_terms(X_val).flatten())
        plot_max = max(abs(max_cm), abs(min_cm))

        for i in range(2):
            for spine in ['top', 'bottom', 'left', 'right']:
                ax[i].spines[spine].set_visible(False)

        IM = np.array(ebm.eval_terms(X_val)[:,a])
        importance = ax[1].imshow(IM.reshape(64,64), cmap = 'seismic_r', extent = (0,64,0,64), clim = (-plot_max, plot_max))
        cbar = plt.colorbar(importance, ax = ax[1], location = 'bottom', fraction=0.046, pad=0.04)
        cbar.set_ticks(np.linspace(-plot_max, plot_max, 3))
        cbar.ax.yaxis.set_major_formatter(tkr.FormatStrFormatter('%.1f'))
        cbar.ax.spines['outline'].set_linewidth(0.5)
        cbar.ax.tick_params(labelsize=15)

        og_image = ax[0].imshow(brightness, cmap = 'gray', extent = (0,64,0,64), clim = (0,175))
        cbar = plt.colorbar(og_image, ax = ax[0], location = 'bottom', fraction = 0.046, pad = 0.04)
        cbar.set_ticks(np.linspace(0, 175, 3))
        cbar.ax.spines['outline'].set_linewidth(0.5)
        cbar.ax.tick_params(labelsize=15)

        xvals = ebm.explain_global().data(a)['names'][1:len(ebm.explain_global().data(a)['names'])]
        yvals = ebm.explain_global().data(a)['scores']

        shape_fx.plot(xvals, yvals, color = 'dimgray')

        lower_bounds = ebm.explain_global().data(a)['lower_bounds']
        upper_bounds = ebm.explain_global().data(a)['upper_bounds']

        lower_bounds[0:523] = np.full(523,np.nan)
        upper_bounds[0:523] = np.full(523,np.nan)

        shape_fx.fill_between(xvals, lower_bounds, upper_bounds, alpha = 0.25, color = 'dimgray')

        shape_fx.tick_params(labelsize=15)

        fig = plt.gcf()
        fig.set_size_inches(18,7.75)
        plt.subplots_adjust(left = 0.005, bottom = 0.335, right = 0.8, top = .69, wspace = 0.175, hspace = 0.347)
        plt.savefig(r'/home/nmitchell/GLCM/useful-images/fig_ten.png', format="png", dpi=600)

        plt.show()

    #fig_ten(0)
    #continue

    def fig_eleven():
        fig,ax = plt.subplots(2,6)
        gs = ax[0,0].get_gridspec()

        for i in range(2):
            for j in range(6):
                ax[i,j].set_xticks([])
                ax[i,j].set_yticks([])

        for i in range(0,6,2):
            before = fig.add_subplot(gs[0,i:(i+2)])
            before.grid(True, linestyle = ':')
            before.set_axisbelow(True)

            xvals = np.array(ebm.explain_global().data(i//2)['names'][1:1023])
            yvals = np.array(ebm.explain_global().data(i//2)['scores'])

            before.step(xvals,yvals,color='dimgray')
            before.fill_between(xvals, ebm.explain_global().data(i//2)['lower_bounds'], ebm.explain_global().data(i//2)['upper_bounds'], alpha = 0.25, color = 'dimgray')

            before.set_title(names[i//2] + " \nFeature Function", fontsize = 15)
            if(i//2 == 0):
                before.set_ylabel("Score", labelpad = -1)
#            before.set_xlabel(names[i//2] + " Values")
            before.set_ylim(-4.75, 7.5)

            #before.set_ticks(np.linspace(0, 1, 5))
            before.tick_params(labelsize=15)


        alter_model()

        for i in range(0,6,2):
            after = fig.add_subplot(gs[1,i:(i+2)])
            after.grid(True, linestyle = ':')
            after.set_axisbelow(True)

            xvals = np.array(ebm.explain_global().data(i//2)['names'][1:1023])
            yvals = np.array(ebm.explain_global().data(i//2)['scores'])

            after.step(xvals,yvals,color='dimgray')

            if(i//2 != 0):
                after.fill_between(xvals, ebm.explain_global().data(i//2)['lower_bounds'], ebm.explain_global().data(i//2)['upper_bounds'], alpha = 0.25, color = 'dimgray')
                if(i//2 == 1):
                    after.set_title("Altered Version", fontsize = 15)
            else:
                after.fill_between(np.where(xvals > 32, xvals, np.nan), ebm.explain_global().data(i//2)['lower_bounds'], ebm.explain_global().data(i//2)['upper_bounds'], alpha = 0.25, color = 'dimgray')
                after.set_title("Altered Version", fontsize = 15)

            if(i//2 == 0):
                after.set_ylabel("Score", labelpad = -1)
            after.set_xlabel(names[i//2] + " Values", fontsize = 15)
            after.set_ylim(-4.75, 7.5)

            after.tick_params(labelsize=15)

        fig = plt.gcf()
        fig.set_size_inches(18,7.75)
        plt.subplots_adjust(left = 0.033, bottom = 0.102, right = 0.99, top = .893, wspace = 0.26, hspace = 0.338)
        plt.savefig(r'/home/nmitchell/GLCM/useful-images/fig_eleven.png', format="png", dpi=300)
        plt.show()

    #fig_eleven()
    #continue

    def fig_twelve(a):
        fig, ax = plt.subplots(1,3)

        for i in range(3):
            for spine in ['top', 'bottom', 'left', 'right']:
                ax[i].spines[spine].set_visible(False)
            ax[i].set_xticks([])
            ax[i].set_yticks([])

        max_cm = max(ebm.eval_terms(X_val).flatten())
        min_cm = min(ebm.eval_terms(X_val).flatten())
        plot_max = max(abs(max_cm), abs(min_cm))

        og_image = ax[0].imshow(original_image, cmap = 'gray', extent = (0,256,0,256), clim = (0,225))
        cbar = plt.colorbar(og_image, ax = ax[0], location = 'bottom', fraction = 0.046, pad = 0.04)
        cbar.set_ticks(np.linspace(0, 225, 3))
        cbar.ax.spines['outline'].set_linewidth(0.5)
        cbar.ax.tick_params(labelsize=15)
        cbar.ax.set_xlabel('Reflectance',fontsize=15)
        IM = np.array(ebm.eval_terms(X_val)[:,a])
        importance = ax[1].imshow(IM.reshape(64,64), cmap = 'seismic', extent = (0,64,0,64), clim = (-plot_max, plot_max))
        cbar = plt.colorbar(importance, ax = ax[1], location = 'bottom', fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=20)
        cbar.ax.set_xlabel('Score',fontsize=15)
        #ax[0].set_title("Cool Contrast Tiles\nOriginal Feature Importance")

        alter_model()

        IM = np.array(ebm.eval_terms(X_val)[:,a]).reshape(64,64)
        importance = ax[2].imshow(IM, cmap = 'seismic', extent = (0,64,0,64), clim = (-plot_max, plot_max))
        cbar = plt.colorbar(importance, ax = ax[2], location = 'bottom', fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=20)
        cbar.ax.set_xlabel('Score',fontsize=15)
        #ax[1].set_title("Cool Contrast Tiles\nAltered Model Feature Importance")

        fig = plt.gcf()
        fig.set_size_inches(18,7.75)
        plt.savefig(r'/home/nmitchell/GLCM/useful-images/fig_twelve.png', format="png", dpi=300)

        plt.show()

    #fig_twelve(1)
    #continue

    def fig_thirteen():
        max_cm = max(np.array(ebm.term_scores_[4:8]).flatten())
        min_cm = min(np.array(ebm.term_scores_[4:8]).flatten())
        int_max = max(abs(max_cm), abs(min_cm))

        fig,ax = plt.subplots(1,3)

        def get_int(b):
            x = np.array(ebm.explain_global().data(b)['left_names'])
            y = np.array(ebm.explain_global().data(b)['right_names'])
            z = np.array(ebm.explain_global().data(b)['scores'].T)
            return(x,y,z)

        for i in range(3,6):
            x,y,z = get_int(i)
            if((i-3)!=2):
                ax[(i-3)].pcolormesh(x,y,z, cmap = 'seismic_r', vmin = -int_max, vmax = int_max)
            else:
                interaction = ax[(i-3)].pcolormesh(x,y,z, cmap = 'seismic_r', vmin = -int_max, vmax = int_max)
                cbar = plt.colorbar(interaction, ax = ax[(i-3)], location = 'right', fraction = 0.046, pad = 0.04)
                cbar.ax.tick_params(labelsize=15)

            y_labels    = np.zeros(len(ax[(i-3)].yaxis.set_ticklabels([])))
            y_positions = np.linspace(min(ebm.explain_global().data(i)['right_names']), max(ebm.explain_global().data(i)['right_names']), len(y_labels))
            y_labels[0] = np.round(y_positions[0],0)
            y_labels[len(y_labels) - 1] = np.round(y_positions[len(y_labels) - 1],2)
            ax[(i-3)].yaxis.set_major_locator(ticker.FixedLocator(y_positions))
            ax[(i-3)].yaxis.set_major_formatter(ticker.FixedFormatter(y_labels))
            [label.set_visible(False) for label in ax[(i-3)].yaxis.get_major_ticks()[1:len(y_labels) - 1]]

            x_labels    = np.zeros(len(ax[(i-3)].xaxis.set_ticklabels([])))
            x_positions = np.linspace(min(ebm.explain_global().data(i)['left_names']), max(ebm.explain_global().data(i)['left_names']), len(x_labels))
            x_labels[0] = np.round(x_positions[0],2)
            x_labels[len(x_labels) - 1] = np.round(x_positions[len(x_labels) - 1],2)
            ax[(i-3)].xaxis.set_major_locator(ticker.FixedLocator(x_positions))
            ax[(i-3)].xaxis.set_major_formatter(ticker.FixedFormatter(x_labels))
            [label.set_visible(False) for label in ax[(i-3)].xaxis.get_major_ticks()[1:len(x_labels) - 1]]

            ax[(i-3)].tick_params(labelsize=15)
            ax[(i-3)].tick_params(axis='x', which='major', direction='out', length=15)


        ax[0].set_xlabel('Brightness',labelpad=-4,fontsize = 15)
        ax[0].set_ylabel('Cool Contrast Tiles',labelpad=-25, fontsize = 15)

        ax[1].set_xlabel('Brightness',labelpad=-4, fontsize = 15)
        ax[1].set_ylabel('Infrared',labelpad=-35, fontsize = 15)

        ax[2].set_xlabel('Cool Contrast Tiles',labelpad=-4, fontsize = 15)
        ax[2].set_ylabel('Infrared',labelpad=-35, fontsize = 15)

        fig = plt.gcf()
        fig.set_size_inches(18,7.75)
        plt.subplots_adjust(left = 0.05, bottom = 0.312, right = 0.9, top = .9, wspace = 0.2, hspace = 0.2)
        plt.savefig(r'/home/nmitchell/GLCM/useful-images/fig_thirteen.png', format="png", dpi=300)

        plt.show()

    #fig_thirteen()
    #continue

    def fig_fourteen():
        fig, ax = plt.subplots(1,3)

        max_cm = max(ebm.eval_terms(X_val).flatten())
        min_cm = min(ebm.eval_terms(X_val).flatten())
        plot_max = max(abs(max_cm), abs(min_cm))

        for i in range(3):
            ax[i].set_xticks([])
            ax[i].set_yticks([])

        for i in range(2):
             ax[i].imshow(ebm.eval_terms(X_val)[:,i+3].reshape(64,64), cmap = 'seismic_r', extent = (0,64,0,64), clim = (-plot_max, plot_max))

        importance = ax[2].imshow(ebm.eval_terms(X_val)[:,5].reshape(64,64), cmap = 'seismic_r', extent = (0,64,0,64), clim = (-plot_max, plot_max))
        cbar = plt.colorbar(importance, ax = ax[2], location = 'right', fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=15)

        fig = plt.gcf()
        fig.set_size_inches(18,7.75)
        plt.subplots_adjust(left = 0.143, bottom = 0.17, right = 0.895, top = .626, wspace = 0.0, hspace = 0.2)
        #plt.savefig(r'/home/nmitchell/GLCM/useful-images/fig_fourteen.png', format="png", dpi=300)

        plt.show()

    #fig_fourteen()
    #continue

    def fig_fifteen():
        fig, ax = plt.subplots(4,10)
        gs = ax[0,0].get_gridspec()

        for i in range(4):
            for j in range(10):
                ax[i,j].remove()

        #Original Features:
        reflec = fig.add_subplot(gs[0:2, 0:2])
        ir = fig.add_subplot(gs[2:4, 0:2])

        re = reflec.imshow(original_image, cmap = 'gray', extent = (0,256,0,256), clim = (0,225))
        cbar = plt.colorbar(re, ax = reflec, location = 'bottom', fraction=0.046, pad=0.04)
        cbar.set_ticks(np.linspace(0, 225, 5))
        remove_ticks(reflec, "")

        i = ir.imshow(infrared, cmap = 'gray_r', extent = (0,256,0,256), clim = (185,325))
        cbar = plt.colorbar(i, ax = ir, location = 'bottom', fraction=0.046, pad=0.04)
        cbar.set_ticks(np.linspace(185, 325, 5))
        remove_ticks(ir, "")

        #Inputs:
        bright = fig.add_subplot(gs[0:2, 2:4])
        cool_glcm = fig.add_subplot(gs[2:4, 2:4])
        ir_2 = fig.add_subplot(gs[1:3, 4:6])

        bri = bright.imshow(brightness, cmap = 'gray', extent = (0,256,0,256), clim = (0,185))
        cbar = plt.colorbar(bri, ax = bright, location = 'bottom', fraction=0.046, pad=0.04)
        cbar.set_ticks(np.linspace(0, 185, 5))
        remove_ticks(bright, "")

        cool = cool_glcm.imshow(cool_contrast_tiles, cmap = 'gray_r', clim = (0,1))
        cbar = plt.colorbar(cool, ax = cool_glcm, location = 'bottom', fraction=0.046, pad=0.04)
        cbar.set_ticks(np.linspace(0, 1, 5))
        remove_ticks(cool_glcm, "")

        i_2 = ir_2.imshow(infrared, cmap = 'gray_r', extent = (0,256,0,256), clim = (185,325))
        cbar = plt.colorbar(i_2, ax = ir_2, location = 'bottom', fraction=0.046, pad=0.04)
        cbar.set_ticks(np.linspace(185, 325, 5))
        remove_ticks(ir_2, "")

        #Ground Truth:
        mrms = fig.add_subplot(gs[0:2, 6:8])
        sandwich = fig.add_subplot(gs[2:4, 6:8])

        m = mrms.imshow(original_image, cmap = 'gray', extent = (0,256,0,256), clim = (0,225))
        mrms.imshow(ones, alpha = full_mrms*0.25, cmap = red_cm, extent = (0,256,0,256))
        mrms.imshow(ones, alpha = masked_mrms,    cmap = red_cm, extent = (0,256,0,256))
        cbar = plt.colorbar(m, ax = mrms, location = 'bottom', fraction=0.046, pad=0.04)
        cbar.set_ticks(np.linspace(0, 225, 5))
        remove_ticks(mrms, "")

        ir = zoom(infrared, 4.0, order = 3, mode = 'nearest')
        ir = np.where(ir > 250, np.nan, ir)
        sandwich.imshow(original_image, cmap = 'gray', extent = (0,256,0,256), clim = (0,225))
        sand = sandwich.imshow(ir, cmap = 'turbo_r', extent = (0,256,0,256), clim = (min_ir,250), alpha = 0.65)
        cbar = plt.colorbar(sand, ax = sandwich, location = 'bottom', fraction=0.046, pad=0.04)
        cbar.set_ticks(np.linspace(min_ir, 250, 5))
        remove_ticks(sandwich, "")

        #Predictions:
        before = fig.add_subplot(gs[0:2, 8:10])
        after = fig.add_subplot(gs[2:4, 8:10])

        pred_percentage = np.array(ebm.predict_proba(X_val)[:,1]).flatten()
        y_pred = np.where(pred_percentage > pred_threshold, pred_percentage, 0).reshape(64,64)
        b = before.imshow(original_image, cmap = 'gray', extent = (0,256,0,256), clim = (0,225))
        cbar = plt.colorbar(b, ax = before, location = 'bottom', fraction=0.046, pad=0.04)
        cbar.set_ticks(np.linspace(0, 225, 5))
        before.imshow(ones, alpha = y_pred, cmap = red_cm, extent = (0,256,0,256))
        remove_ticks(before, "")

        alter_model()

        pred_percentage = np.array(ebm.predict_proba(X_val)[:,1]).flatten()
        y_pred = np.where(pred_percentage > pred_threshold, pred_percentage, 0).reshape(64,64)
        a = after.imshow(original_image, cmap = 'gray', extent = (0,256,0,256), clim = (0,225))
        cbar = plt.colorbar(a, ax = after, location = 'bottom', fraction=0.046, pad=0.04)
        cbar.set_ticks(np.linspace(0, 225, 5))
        after.imshow(ones, alpha = y_pred, cmap = red_cm, extent = (0,256,0,256))
        remove_ticks(after, "")

        fig = plt.gcf()
        fig.set_size_inches(18,7.75)
        plt.subplots_adjust(left = 0.01, bottom = 0.038, right = 1, top = .964, wspace = 0.2, hspace = 0.6)
        plt.savefig(r'/home/nmitchell/GLCM/useful-images/fig_fifteen.png', format="png", dpi=300)

        plt.show()

    #fig_fifteen()
    #continue

    def fig_sixteen():
        fig,ax = plt.subplots(1,4)

        ticks = [0.0,255.0]
        ax[0].set_xticks(ticks)
        ax[0].set_yticks(ticks)

        dic1 = { 0: str(round(min(lon_hi.flatten()),2)) + r"$^\circ$W", 255.0 : str(round(max(lon_hi.flatten()),2)) + r"$^\circ$W"}
        dic2 = { 0: str(round(min(lat_hi.flatten()),2)) + r"$^\circ$N", 255.0 : str(round(max(lat_hi.flatten()),2)) + r"$^\circ$N"}

        labels1 = [ticks[i] if t not in dic1.keys() else dic1[t] for i,t in enumerate(ticks)]
        labels2 = [ticks[i] if t not in dic2.keys() else dic2[t] for i,t in enumerate(ticks)]

        ax[0].set_xticklabels(labels1)
        ax[0].set_yticklabels(labels2)
        ax[0].tick_params(labelsize=15)
        ax[0].tick_params(axis='both', which='major', direction='out', length=10)

        for i in [1,2,3]:
            ax[i].set_xticks([])
            ax[i].set_yticks([])

        importance = ax[0].imshow(original_image, cmap = 'gray', extent = (0,256,0,256), clim = (0,225))
        cbar = plt.colorbar(importance, ax = ax[0], location = 'bottom', fraction=0.046, pad=0.07)
        cbar.set_ticks(np.linspace(0,225,3))
        cbar.ax.tick_params(labelsize=15)

        ir = zoom(infrared, 4.0, order=3, mode = 'nearest')
        ir = np.where(ir > 250, np.nan, ir)
        vis = ax[2].imshow(original_image, cmap = 'gray', extent = (0,256,0,256), clim = (0,225))
        importance = ax[2].imshow(ir, cmap = 'turbo_r', extent = (0,256,0,256), clim = (min_ir,250), alpha = 0.65)
        cbar = plt.colorbar(importance, ax = ax[2], location = 'bottom', fraction=0.046, pad=0.07)
        cbar.set_ticks(np.linspace(min_ir,250,3))
        cbar.ax.tick_params(labelsize=15)

        importance = ax[1].imshow(ebm.eval_terms(X_val)[:,4].reshape(64,64), cmap = 'seismic_r', clim = (-6,6))
        cbar = plt.colorbar(importance, ax = ax[1], location = 'bottom', fraction = 0.046, pad = 0.07)
        cbar.set_ticks(np.linspace(-6,6,3))
        cbar.ax.tick_params(labelsize=15)

        pred_percentage = np.array(ebm.predict_proba(X_val)[:,1]).flatten()
        y_pred = np.where(pred_percentage > pred_threshold, pred_percentage, 0).reshape(64,64)
        importance = ax[3].imshow(original_image, cmap = 'gray', extent = (0,256,0,256), clim = (0,225))
        ax[3].imshow(ones, alpha = y_pred, cmap = red_cm, extent = (0,256,0,256))
        cbar = plt.colorbar(importance, ax = ax[3], location = 'bottom', fraction=0.046, pad=0.07)
        cbar.set_ticks(np.linspace(0,225,3))
        cbar.ax.tick_params(labelsize=15)

        fig = plt.gcf()
        fig.set_size_inches(18,7.75)
        plt.subplots_adjust(left = 0.098, bottom = 0.11, right = 0.902, top = 1, wspace = 0.338, hspace = 0.2)

        plt.savefig(r'/home/nmitchell/GLCM/useful-images/fig_sixteen.png', format="png", dpi=300)

        plt.show()

    #fig_sixteen()
    #continue

    def fig_seven_nine_teen():

        fig, ax = plt.subplots(4,6)
        gs = ax[0,0].get_gridspec()

        for i in range(4):
            for j in range(6):
                ax[i,j].remove()

        ticks = [0.0,255.0]

        visir = fig.add_subplot(gs[1:3,0:2])
        visir.set_xticks(ticks)
        visir.set_yticks(ticks)

        dic1 = { 0: str(round(min(lon_hi.flatten()),2)) + r"$^\circ$W", 255.0 : str(round(max(lon_hi.flatten()),2)) + r"$^\circ$W"}
        dic2 = { 0: str(round(min(lat_hi.flatten()),2)) + r"$^\circ$N", 255.0 : str(round(max(lat_hi.flatten()),2)) + r"$^\circ$N"}

        labels1 = [ticks[i] if t not in dic1.keys() else dic1[t] for i,t in enumerate(ticks)]
        labels2 = [ticks[i] if t not in dic2.keys() else dic2[t] for i,t in enumerate(ticks)]

        visir.set_xticklabels(labels1)
        visir.set_yticklabels(labels2)
        visir.tick_params(labelsize=15)
        visir.tick_params(axis='both', which='major', direction='out', length=10)

        ir = zoom(infrared, 4.0, order=3, mode = 'nearest')
        ir = np.where(ir > 250, np.nan, ir)
        visir.imshow(original_image, cmap = 'gray', extent = (0,256,0,256))
        importance = visir.imshow(ir, cmap = 'turbo_r', extent = (0,256,0,256), clim = (min_ir,250), alpha = 0.65)
        cbar = plt.colorbar(importance, ax = visir, location = 'bottom', fraction=0.041, pad=0.15)
        cbar.set_ticks(np.linspace(min_ir,250,3))
        cbar.ax.tick_params(labelsize=15)

        before = fig.add_subplot(gs[0:2,2:4])
        after = fig.add_subplot(gs[2:4,2:4])

        before.set_xticks([])
        after.set_xticks([])
        before.set_yticks([])
        after.set_yticks([])

        importance = before.imshow(ebm.eval_terms(X_val)[:,1].reshape(64,64), cmap = 'seismic_r', clim = (-6,6))
        cbar = plt.colorbar(importance, ax = before, location = 'bottom', fraction = 0.046, pad = 0.07)
        cbar.set_ticks(np.linspace(-6,6,3))
        cbar.ax.tick_params(labelsize=15)
        cbar.ax.set_visible(False)

        alter_model()

        importance = after.imshow(ebm.eval_terms(X_val)[:,1].reshape(64,64), cmap = 'seismic_r', clim = (-6,6))
        cbar = plt.colorbar(importance, ax = after, location = 'bottom', fraction = 0.0445, pad = 0.07)
        cbar.set_ticks(np.linspace(-6,6,3))
        cbar.ax.tick_params(labelsize=15)

        pred = fig.add_subplot(gs[1:3,4:6])

        pred.set_xticks(ticks)
        pred.set_yticks(ticks)

        dic1 = { 0: str(round(min(lon_hi.flatten()),2)) + r"$^\circ$W", 255.0 : str(round(max(lon_hi.flatten()),2)) + r"$^\circ$W"}
        dic2 = { 0: str(round(min(lat_hi.flatten()),2)) + r"$^\circ$N", 255.0 : str(round(max(lat_hi.flatten()),2)) + r"$^\circ$N"}

        labels1 = [ticks[i] if t not in dic1.keys() else dic1[t] for i,t in enumerate(ticks)]
        labels2 = [ticks[i] if t not in dic2.keys() else dic2[t] for i,t in enumerate(ticks)]

        pred.set_xticklabels(labels1)
        pred.set_yticklabels(labels2)
        pred.tick_params(labelsize=15)
        pred.tick_params(axis='both', which='major', direction='out', length=10)

        pred_percentage = np.array(ebm.predict_proba(X_val)[:,1]).flatten()
        y_pred = np.where(pred_percentage > pred_threshold, pred_percentage, 0).reshape(64,64)
        importance = pred.imshow(original_image, cmap = 'gray', extent = (0,256,0,256), clim = (0,225))
        pred.imshow(ones, alpha = y_pred, cmap = red_cm, extent = (0,256,0,256))
        cbar = plt.colorbar(importance, ax = pred, location = 'bottom', fraction=0.041, pad=0.15)
        cbar.set_ticks(np.linspace(0,225,3))
        cbar.ax.tick_params(labelsize=15)

        fig = plt.gcf()
        fig.set_size_inches(18,7.75)
        plt.subplots_adjust(left = 0.10, bottom = 0.043, right = 0.90, top = 0.988, wspace = 0, hspace = 0)

        plt.savefig(r'/home/nmitchell/GLCM/useful-images/fig_nineteen.png', format="png", dpi=300)

        plt.show()

    #fig_seven_nine_teen()
    #continue

    def fig_eighteen():
        fig, ax = plt.subplots(1,3)

        ticks = [0.0,255.0]
        for i in range(3):
            ax[i].set_xticks(ticks)
            ax[i].set_yticks(ticks)

        dic1 = { 0: str(round(min(lon_hi.flatten()),2)) + r"$^\circ$W", 255.0 : str(round(max(lon_hi.flatten()),2)) + r"$^\circ$W"}
        dic2 = { 0: str(round(min(lat_hi.flatten()),2)) + r"$^\circ$N", 255.0 : str(round(max(lat_hi.flatten()),2)) + r"$^\circ$N"}

        labels1 = [ticks[i] if t not in dic1.keys() else dic1[t] for i,t in enumerate(ticks)]
        labels2 = [ticks[i] if t not in dic2.keys() else dic2[t] for i,t in enumerate(ticks)]

        for i in range(3):
            ax[i].set_xticklabels(labels1)
            ax[i].set_yticklabels(labels2)
            ax[i].tick_params(labelsize=15)
            ax[i].tick_params(axis='both', which='major', direction='out', length=10)

        importance = ax[0].imshow(original_image, cmap = 'gray', extent = (0,256,0,256), clim = (0,225))
        ax[0].imshow(ones, alpha = masked_mrms, cmap = red_cm, extent = (0,256,0,256))
        cbar = plt.colorbar(importance, ax = ax[0], location = 'bottom', fraction=0.046, pad=0.07)
        cbar.set_ticks(np.linspace(0,225,3))
        cbar.ax.tick_params(labelsize=15)

        ir = zoom(infrared, 4.0, order=3, mode = 'nearest')
        ir = np.where(ir > 250, np.nan, ir)
        ax[1].imshow(original_image, cmap = 'gray', extent = (0,256,0,256))
        importance = ax[1].imshow(ir, cmap = 'turbo_r', extent = (0,256,0,256), clim = (min_ir,250), alpha = 0.65)
        cbar = plt.colorbar(importance, ax = ax[1], location = 'bottom', fraction=0.046, pad=0.07)
        cbar.set_ticks(np.linspace(min_ir,250,3))
        cbar.ax.tick_params(labelsize=15)

        pred_percentage = np.array(ebm.predict_proba(X_val)[:,1]).flatten()
        y_pred = np.where(pred_percentage > pred_threshold, pred_percentage, 0).reshape(64,64)
        importance = ax[2].imshow(original_image, cmap = 'gray', extent = (0,256,0,256), clim = (0,225))
        ax[2].imshow(ones, alpha = y_pred, cmap = red_cm, extent = (0,256,0,256))
        cbar = plt.colorbar(importance, ax = ax[2], location = 'bottom', fraction=0.046, pad=0.07)
        cbar.set_ticks(np.linspace(0,225,3))
        cbar.ax.tick_params(labelsize=15)

        fig = plt.gcf()
        fig.set_size_inches(18,7.75)
        #plt.subplots_adjust(left = 0.15, bottom = 0.11, right = 0.85, top = 1, wspace = 0.338, hspace = 0.2)
        plt.subplots_adjust(left = 0.10, bottom = 0.043, right = 0.90, top = 0.988, wspace = 0.615, hspace = 0)

        plt.savefig(r'/home/nmitchell/GLCM/useful-images/fig_eighteen.png', format="png", dpi=300)

        plt.show()

    #fig_eighteen()
    #continue

    #INPUTS:
    # (1) Feature (0: brightness, 1: cool contrast tiles, 2: infrared)
    # (2) Lower Bound
    # (3) Upper Bound
    def highlight_single(feat, lb, ub):
        fig, ax = plt.subplots(2,3)
        gs = ax[0,0].get_gridspec()

        for i in range(2):
            for j in range(3):
                ax[i,j].set_xticks([])
                ax[i,j].set_yticks([])

        feature = eval(names[feat].lower().replace(' ', '_'))
        mask = np.where(((feature >= lb) & (feature <= ub)),1,np.nan)

        #mask = np.where(((infrared_image >= 180) & (infrared_image <= 200)), 1, np.nan)

        xvals = np.array(ebm.explain_global().data(feat)['names'][0:-1])
        yvals = np.array(ebm.explain_global().data(feat)['scores'][:])

        #First Row: original image, feature, feature with highlight
        ax[0,0].imshow(original_image, cmap = 'gray', clim = (0,175))
        ax[0,0].set_title("Reflectance")

        if(feat == 0):
            ax[0,1].imshow(brightness, cmap = 'gray', clim = (0,175))
            ax[0,2].imshow(brightness, cmap = 'gray', clim = (0,175))
        if(feat == 1):
            ax[0,1].imshow(np.ones((64,64)), alpha = cool_contrast_tiles, cmap = 'gray')
            ax[0,2].imshow(np.ones((64,64)), alpha = cool_contrast_tiles, cmap = 'gray')
        if(feat == 2):
            ax[0,1].imshow(infrared, cmap = 'gray_r')
            ax[0,2].imshow(infrared, cmap = 'gray_r')

        ax[0,1].set_title(names[feat])

        ax[0,2].imshow(mask, cmap = h_cm)
        ax[0,2].set_title("Values of Interest")

        #Second Row: feature importance, feature function
        importance = ax[1,0].imshow(ebm.eval_terms(X_val)[:,feat].reshape(64,64), cmap = 'seismic_r', clim = (-6,6))
        plt.colorbar(importance, ax = ax[1,0], location = 'left', fraction = 0.04675, pad = 0.025)
        ax[1,0].set_title("Feature Importance")

        #For re-coloring the feature function in the specified region
        region_x = np.where(((xvals >= lb) & (xvals <= ub)),xvals,np.nan)
        region_y = np.where(((xvals >= lb) & (xvals <= ub)),yvals,np.nan)

        ff = fig.add_subplot(gs[1,1:3])
        ff.hlines(y = 0, xmin = np.min(xvals), xmax  = np.max(xvals), linestyles = "dashed", colors = "black", alpha = 0.25)
        ff.plot(xvals, yvals, color = 'black')
        ff.plot(region_x, region_y, color = 'red', linewidth = 1.75)
        ff.grid(True, linestyle = ':')

        ff.set_title("Feature Function")

        plt.subplots_adjust(left = 0.129, bottom = 0.048, right = 0.886, top = 0.945, wspace = 0.126, hspace = 0.112)

        plt.show()

    #highlight_single(0,11,32)
    #continue

    def highlight_double(feat, a_lb, a_ub, b_lb, b_ub):
        fig, ax = plt.subplots(3,2)

        for i in range(3):
            for j in range(2):
                if((i == 2) & (j == 1)):
                    continue
                else:
                    ax[i,j].set_xticks([])
                    ax[i,j].set_yticks([])

        #Lowercase names
        lower = names[feat].lower().replace(' ', '_').split('_&_')

        #Uppercase names
        upper = names[feat].replace('&', '').split('  ')

        feature_a = eval(lower[0])
        feature_b = eval(lower[1])

        mask_a = np.where(((feature_a >= a_lb) & (feature_a <= a_ub)),1,np.nan)
        mask_b = np.where(((feature_b >= b_lb) & (feature_b <= b_ub)),1,np.nan)

        if('brightness' in lower[0]):
            ax[0,0].imshow(brightness, cmap = 'gray')
            ax[1,0].imshow(brightness, cmap = 'gray')
        else:
            ax[0,0].imshow(np.ones((64,64)), alpha = cool_contrast_tiles, cmap = 'gray')
            ax[1,0].imshow(np.ones((64,64)), alpha = cool_contrast_tiles, cmap = 'gray')
        if('cool' in lower[1]):
            ax[0,1].imshow(np.ones((64,64)), alpha = cool_contrast_tiles, cmap = 'gray')
            ax[1,1].imshow(np.ones((64,64)), alpha = cool_contrast_tiles, cmap = 'gray')
        else:
            ax[0,1].imshow(infrared, cmap = 'gray_r')
            ax[1,1].imshow(infrared, cmap = 'gray_r')

        ax[0,0].set_title(upper[0])
        ax[0,1].set_title(upper[1])

        #Second row: add the highlight
        ax[1,0].imshow(mask_a, cmap = h_cm)
        ax[1,1].imshow(mask_b, cmap = h_cm)

        #Third row: feature importance / feature function
        importance = ax[2,0].imshow(ebm.eval_terms(X_val)[:,feat].reshape(64,64), cmap = 'seismic_r', clim = (-6,6))
        plt.colorbar(importance, ax = ax[2,0], location = 'left', fraction = 0.04675, pad = 0.025)
        ax[2,0].set_title("Feature Importance")

        x = np.array(ebm.explain_global().data(feat)['left_names'])
        y = np.array(ebm.explain_global().data(feat)['right_names'])
        z = np.array(ebm.explain_global().data(feat)['scores'].T)

        #mask = np.where(((x > 5) & (x < 20)), z, np.nan)

        x_lower = np.max(np.where(x <= a_lb))
        x_upper = np.min(np.where(x >= a_ub))

        y_lower = np.max(np.where(y <= b_lb))
        y_upper = np.min(np.where(y >= b_ub))

        #z[x_lower:x_upper,y_lower:y_upper] = np.zeros((x_upper - x_lower, y_upper - y_lower)) - 10
        z[y_lower:y_upper,x_lower:x_upper] = np.zeros((y_upper - y_lower, x_upper - x_lower)) - 10

        interaction = ax[2,1].pcolormesh(x,y,z, cmap = 'seismic_r', vmin = -6, vmax = 6)
        plt.colorbar(interaction, ax = ax[2,1], location = 'right', fraction = 0.046, pad = 0.04)
        ax[2,1].set_xlabel(upper[0])
        ax[2,1].set_ylabel(upper[1])
        ax[2,1].set_title("Feature Function")

        plt.subplots_adjust(left = 0.25, bottom = 0.036, right = 0.664, top = 0.95, wspace = 0.08, hspace = 0.162)

        plt.show()

    #highlight_double(3, 3,25, 0.6,0.96)
    #continue

    ### Plot Everything ###

    #Create the plotting interface
    fig, ax = plt.subplots(4,10)
    gs = ax[0,0].get_gridspec()

    #Function to remove ticks & axes from small images
    def remove_ax_labels(b):
        for i in range(4):
            for a in ax[0:4,i]:
                a.remove()
        for i in range(0,4):
            for j in range(0,b):
                ax[i,j].set_xticks([])
                ax[i,j].set_yticks([])

    #Creates the "Large Images," i.e. the Original Image, Ground Truth, and Predicted Convection
    def big_images(original_features, interactions):

        ### Original Image ###
        og_im_ax = fig.add_subplot(gs[0:2, 0:2])
        og_im_ax.imshow(original_image, cmap = 'gray', extent = (0,256,0,256), clim = (0,186))
        og_im_ax.yaxis.set_label_position("right")
        #intercept  = "Intercept: " + str(np.round(ebm.intercept_[0], 3)) + "\n\n\n\n\nSample Number: " + str(isamp)
        #intercept2 = "\nOriginal: " + str(np.round(ebm.intercept_[0], 3) - intercept_addition)
        #og_im_ax.set_ylabel(intercept, rotation = 0, fontsize = 15)
        #og_im_ax.set_xlabel(intercept2, fontsize = 10)
        #og_im_ax.xaxis.set_label_coords(1.65, 0.725)
        #og_im_ax.yaxis.set_label_coords(1.65, 0.75)
        remove_ticks(og_im_ax, "Channel 2: Visible Imagery")
        ###----------------###

        ### Ground Truth ###
        ones2 = np.ones((256,256))
        mrms_ax = fig.add_subplot(gs[2:4, 0:2])
        mrms_ax.imshow(original_image, alpha = 0.5,  cmap = 'gray', extent = (0,256,0,256), clim = (0,186))
        mrms_ax.imshow(ones, alpha = full_mrms*0.25, cmap = red_cm, extent = (0,256,0,256)) #Full Ground Truth (low opacity)
        mrms_ax.imshow(ones, alpha = masked_mrms,    cmap = red_cm, extent = (0,256,0,256)) #IR Masked Ground Truth (full opacity)
        remove_ticks(mrms_ax, "Multi-Radar Multi-Sensor (MRMS)")
        ###--------------###

        ### Predicted Convection ###
        pred_percentage = np.array(ebm.predict_proba(X_val)[:,1]).flatten()
        y_pred = np.where(pred_percentage > pred_threshold, pred_percentage, 0).reshape(64,64)

        pred_convection_ax = fig.add_subplot(gs[2:4, 2:4])
        pred_convection_ax.imshow(original_image, cmap = 'gray', extent = (0,256,0,256), clim = (0,186))
        pred_convection_ax.imshow(ones, alpha = y_pred, cmap = red_cm, extent = (0,256,0,256))
        remove_ticks(pred_convection_ax, "Predicted Convection")
        ### ---------------------###


    ### Feature Imporance ###
    max_cm = max(ebm.eval_terms(X_val).flatten())
    min_cm = min(ebm.eval_terms(X_val).flatten())
    plot_max = max(abs(max_cm), abs(min_cm))

    def importance(a,b,c, title):
        ax[a,b].set_title(title)
        importance = ax[a,b].imshow(ebm.eval_terms(X_val)[:,c].reshape(64,64), cmap = 'seismic_r', extent = (0,64,0,64), clim = (-plot_max, plot_max))
        plt.colorbar(importance, ax = ax[a,b], location = 'bottom', fraction=0.046, pad=0.04)
    ###-------------------###

    def shape_function(a, name, first_plot, last_plot):
        #Set the data based on the name passed
        data = eval(name).flatten()

        #Create the plotting interface
        shape_fx = fig.add_subplot(gs[a, 6:8])
        shape_fx.grid(True, linestyle = ':')
        shape_fx.set_axisbelow(True)

        #Set the x- and y- values
        xvals = ebm.explain_global().data(a)['names'][1:len(ebm.explain_global().data(a)['names'])]
        yvals = ebm.explain_global().data(a)['scores']

        #Sets the upper and lower values for the error bounds
        lower_error = ebm.explain_global().data(a)['lower_bounds']
        upper_error = ebm.explain_global().data(a)['upper_bounds']

        #Plot a dotted line at y = 0 for easy comparison across the full plot
        shape_fx.plot(xvals, np.zeros(len(xvals)), ':', color = 'black')

        #Data min & max
        data_min = np.min(data)
        data_max = np.max(data)

        #Find the sum of the data -- 0 is a special case that happens when we have no pixels above or below 250 K
        data_sum = np.sum(data)

        #Check to see if there is convection (per feature!)--if there is, the non-convective section needs to be split up
        x = np.sum(masked_mrms.flatten()*data) == 0
        convec_flag = {x == 0: False, x != 0: True}.get(False,True)

        if(convec_flag):
            #Sets the minimum and maximum values of convcetion found within the scene (actual *data* values)
            convec_lower = np.nanmin(np.where(masked_mrms.flatten() == 1, data, np.nan))
            convec_upper = np.nanmax(np.where(masked_mrms.flatten() == 1, data, np.nan))

            #Translates the min/max from the DATA to the min/max found within the FUNCTION (*function* values)
            convec_bounds = np.where((xvals <= convec_upper) & (xvals >= convec_lower))[0]
            convec_new_lower = xvals[convec_bounds[0]]
            convec_new_upper = xvals[convec_bounds[len(convec_bounds)-1]]

            #If we know where the convective chunk is, we can find the places where there *isn't* convection
            non_convec_bounds = np.where(((xvals >= data_min) & (xvals <= convec_lower)) | ((xvals >= convec_upper) & (xvals <= data_max)))

            #Convection occurs where the x values are greater than the lower bound and less than the upper bound
            convection = np.where(((xvals >= convec_new_lower) & (xvals <= convec_new_upper)), xvals, np.nan)
            convec_lower_error = np.where(((xvals >= convec_new_lower) & (xvals <= convec_new_upper)), lower_error, np.nan)
            convec_upper_error = np.where(((xvals >= convec_new_lower) & (xvals <= convec_new_upper)), upper_error, np.nan)
        else:
            #If there's no convection within the scene, set the line to be plotted as NaN values
            convection = np.ones(len(xvals))*np.nan
            convec_lower_error = np.ones(len(xvals))*np.nan
            convec_upper_error = np.ones(len(xvals))*np.nan

        if(convec_flag):
            #Checks to see if the non-convective bounds are in one "section" or two
            if(len(non_convec_bounds) == 1):
                lower_l = convec_new_lower
                upper_u = convec_new_upper
                non_convection = np.ones(len(xvals))*np.nan
                non_convec_lower_error = np.ones(len(xvals))*np.nan
                non_convec_upper_error = np.ones(len(xvals))*np.nan

            elif(max(np.ediff1d(non_convec_bounds)) != 1):
                lower_l = non_convec_bounds[0][0]
                lower_u = non_convec_bounds[0][np.argmax(np.ediff1d(non_convec_bounds))]

                if(lower_u == lower_l):
                    lower_u = lower_u + 1

                lower_l = xvals[lower_l]
                lower_u = convec_new_lower

                upper_l = non_convec_bounds[0][np.argmax(np.ediff1d(non_convec_bounds))+1]
                upper_u = non_convec_bounds[0][len(non_convec_bounds[0])-1]

                if(upper_l == upper_u):
                    upper_l = upper_l - 1

                upper_l = convec_new_upper
                upper_u = xvals[upper_u]

                non_convection = np.where((((xvals >= lower_l) & (xvals <= lower_u)) | ((xvals >= upper_l) & (xvals <= upper_u))), xvals, np.nan)
                non_convec_lower_error = np.where((((xvals >= lower_l) & (xvals <= lower_u)) | ((xvals >= upper_l) & (xvals <= upper_u))), lower_error, np.nan)
                non_convec_upper_error = np.where((((xvals >= lower_l) & (xvals <= lower_u)) | ((xvals >= upper_l) & (xvals <= upper_u))), upper_error, np.nan)

            else:
                #If in one section, plotting non-convection is straightforward
                lower_l = non_convec_bounds[0][0]
                upper_u = non_convec_bounds[0][len(non_convec_bounds[0])-1]

                lower_l = xvals[lower_l]
                upper_u = xvals[upper_u]

                non_convection = np.where(((xvals >= lower_l) & (xvals <= upper_u)), xvals, np.nan)
                non_convec_lower_error = np.where(((xvals >= lower_l) & (xvals <= upper_u)), lower_error, np.nan)
                non_convec_upper_error = np.where(((xvals >= lower_l) & (xvals <= upper_u)), upper_error, np.nan)

                #Check to see if it's a lower section or an upper section (useful for no_data_bounds)
                if(lower_l >= convec_new_upper): #if we have an upper section
                    lower_l = convec_new_lower
                elif(upper_u <= convec_new_lower): #if we have a lower section
                    upper_u = convec_new_upper
        elif(data_sum != 0):
            non_convec_bounds = np.where(((xvals >= data_min) & (xvals <= data_max)))

            #If there's no convection, plotting non-convection is straightforward
            lower_l = non_convec_bounds[0][0]
            upper_u = non_convec_bounds[0][len(non_convec_bounds[0])-1]

            non_convection = np.where(((xvals >= lower_l) & (xvals <= upper_u)), xvals, np.nan)
            non_convec_lower_error = np.where(((xvals >= lower_l) & (xvals <= upper_u)), lower_error, np.nan)
            non_convec_upper_error = np.where(((xvals >= lower_l) & (xvals <= upper_u)), upper_error, np.nan)

            no_data_bounds = np.where(((xvals <= lower_l) | (xvals >= upper_u)), xvals, np.nan)

        #Check to see
        if(data_sum != 0):
            if(lower_l != data_min):
                no_data_bounds_l = np.where((xvals <= lower_l), xvals, np.nan) #[0]
            else:
                no_data_bounds_l = np.ones(len(xvals))*np.nan
            if(upper_u != data_max):
                no_data_bounds_u = np.where((xvals >= upper_u), xvals, np.nan)
            else:
                no_data_bounds_u = np.ones(len(xvals))*np.nan
        else:
            no_data_bounds_l = xvals
            no_data_bounds_u = np.ones(len(xvals))*np.nan
            non_convection = np.ones(len(xvals))*np.nan
            non_convec_lower_error = np.ones(len(xvals))*np.nan
            non_convec_upper_error = np.ones(len(xvals))*np.nan

        no_data_bounds = np.nansum([no_data_bounds_l, no_data_bounds_u], axis = 0)

        no_data_lower_error = np.where(no_data_bounds != 0, lower_error, np.nan)
        no_data_upper_error = np.where(no_data_bounds != 0, upper_error, np.nan)

        no_data_bounds = np.where(no_data_bounds != 0, no_data_bounds, np.nan)

        if name in edited_shape_fxs:
            position = np.where(edited_shape_fxs == name)[0][0]
            bound_range = edit_range[position]

            arr = np.full(int(bound_range[1] - bound_range[0]), np.nan)
            convec_lower_error[int(bound_range[0]):int(bound_range[1])] = arr
            convec_upper_error[int(bound_range[0]):int(bound_range[1])] = arr

            non_convec_lower_error[int(bound_range[0]):int(bound_range[1])] = arr
            non_convec_upper_error[int(bound_range[0]):int(bound_range[1])] = arr

            no_data_lower_error[int(bound_range[0]):int(bound_range[1])] = arr
            no_data_upper_error[int(bound_range[0]):int(bound_range[1])] = arr


        convec_color = '#DC267F'
        non_convec_color = '#785EF0'
        no_data_color = '#648FFF'

        #Plots the convective region
        shape_fx.step(convection, yvals, color = convec_color, label = "Convective")
        shape_fx.fill_between(xvals, convec_lower_error, convec_upper_error, color = convec_color, alpha = 0.25)

        #Plots the non-convective region(s)
        shape_fx.step(non_convection, yvals, color = non_convec_color, label = "No Data")
        shape_fx.fill_between(xvals, non_convec_lower_error, non_convec_upper_error, color = non_convec_color, alpha = 0.25)

        #Plots the regions of no data -- if they exist
        shape_fx.step(no_data_bounds, yvals, color = no_data_color, label = "Non-Convective")
        shape_fx.fill_between(xvals, no_data_lower_error, no_data_upper_error, color = no_data_color, alpha = 0.25)

        if(first_plot):
            shape_fx.set_title("Shape Function:", y = 1.15)
            shape_fx.legend(bbox_to_anchor=(1.03, 1.175),fontsize=7,ncol = 3, columnspacing=0.8)

        return shape_fx, min(xvals), max(xvals)

    def plot_limits(ax, lower_x, upper_x, step_x, lab_type, round_by):
        ax.xaxis.set_major_locator(ticker.FixedLocator(np.round(np.arange(lower_x, upper_x, step_x).astype(lab_type), round_by)))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(np.round(np.arange(lower_x, upper_x, step_x).astype(lab_type), round_by)))
        [label.set_visible(False) for label in ax.xaxis.get_ticklabels()[1::2]]
    ###-----------------###

    ### Feature Histograms ###
    def feat_hist(a, name, first_plot, last_plot):
        data = eval(name).flatten()
        hist = fig.add_subplot(gs[a,8:10])
        hist.grid(True, linestyle = ':')
        hist.set_axisbelow(True)
        bins = ebm.explain_global().data(a)['density']['names']

        if "cool" in name:
            n_mrms = masked_mrms.flatten()[infrared.flatten() <= 250]
            data = data.flatten()[infrared.flatten() <= 250]
        elif "warm" in name:
            n_mrms = masked_mrms.flatten()[infrared.flatten() > 250]
            data = data.flatten()[infrared.flatten() > 250]
        else:
            n_mrms = masked_mrms.flatten()

        hist.hist([np.where(n_mrms==0.0,data,np.nan), np.where(n_mrms==1.0,data,np.nan)],
                  bins = bins, histtype='bar',
                  stacked=True, label=['Non-Convective','Convective'], color=['#58944e','#405072'])

        hist.tick_params(bottom=True, top=False, left=False, right=True)
        hist.tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=True)

        if(first_plot):
            hist.set_title("Feature Distribution:", y = 1.15)
            hist.legend(bbox_to_anchor=(0.925, 1.175),fontsize=7,ncol = 2)
        if(last_plot):
            hist.set_xlabel("Histogram: Local\nColors: Local", fontsize=7.5)

        return hist, min(bins), max(bins)

    ### ORIGINAL FEATURES ###

    #Convolved Image
    ax[0,4].set_ylabel(names[0])
    ax[0,4].set_title("Feature:")
    ax[0,4].imshow(brightness, cmap = 'gray', extent = (0,64,0,64))

    #Cool GLCM
    ax[1,4].set_ylabel(names[1])
    ax[1,4].imshow(ones, alpha = cool_contrast_tiles, cmap = blu_cm, extent = (0,256,0,256))

    #Infrared
    ax[2,4].set_ylabel(names[2])
    ax[2,4].imshow(infrared, cmap = 'gray_r', extent = (0,64,0,64))

    ### FUNCTION CALLS ###
    big_images(True, False)

    #Plots the shape functions
    axis, min_x, max_x = shape_function(0, names[0].lower().replace(' ', '_'), True, False)
    plot_limits(axis, math.floor(min_x/10)*10, (math.ceil(max_x/10)*10) + 1, (math.ceil(max_x/10)*10)/10, int, 0)

    axis, min_x, max_x = shape_function(1, names[1].lower().replace(' ', '_'), False, False)
    plot_limits(axis, math.floor(min_x/10)*10, (math.ceil(max_x/10)*1) + 0.1, (math.ceil(max_x/10)*1)/10, float, 2)

    axis, min_x, max_x = shape_function(2, names[2].lower().replace(' ', '_'), False, True)
    plot_limits(axis, math.floor(min_x/10)*10, (math.ceil(max_x/10)*10) + 1, (math.ceil(max_x/5)*5 - math.floor(min_x/10)*10)/10, int, 0)

    #Plots the feature imporance
    importance(0,5,0, "Local\nImportance:")
    importance(1,5,1, "")
    importance(2,5,2, "")

    #Plots the density histograms
    axis, min_x, max_x = feat_hist(0, names[0].lower().replace(' ', '_'), True,False)
    plot_limits(axis, math.floor(min_x/10)*10, (math.ceil(max_x/10)*10) + 1,(math.ceil(max_x/10)*10)/10, int, 0)

    axis, min_x, max_x = feat_hist(1, names[1].lower().replace(' ', '_'), False,False)
    plot_limits(axis, min_x, max_x + 0.1, (math.ceil(max_x/10)*1)/10, float, 2)

    axis, min_x, max_x = feat_hist(2, names[2].lower().replace(' ', '_'), False,True)
    plot_limits(axis, math.floor(min_x/10)*10 - 10, (math.ceil(max_x/10)*10) + 1, (math.ceil(max_x/5)*5 - math.floor(min_x/10)*10)/10, int, 0)

    remove_ax_labels(10)

    #Set the size of the figure
    fig.set_size_inches((8.5, 15), forward=False)
    plt.subplots_adjust(left = 0.012, bottom = 0.070, right = 0.967, top = 0.938, wspace = 0.275, hspace = 0.330)

    #plt.show()
    #continue

    ### SECOND SET OF PLOTS -- INTERACTIONS ###
    fig, ax = plt.subplots(4,8)
    remove_ax_labels(7)

    big_images(False,True)

    ### USEFUL FUNCTIONS ###
    max_cm = max(np.array(ebm.term_scores_[3:6]).flatten())
    min_cm = min(np.array(ebm.term_scores_[3:6]).flatten())
    int_max = max(abs(max_cm), abs(min_cm))

    ### Interactions ###
    def int_image_plots(a, images, title, b):
        if ('infrared' in images[0]):
            ax[a,4].imshow(eval(images[0]), cmap = 'gray_r')
        else:
            ax[a,4].imshow(eval(images[0]), cmap = 'gray')

        if ('infrared' in images[1]):
            ax[a,5].imshow(eval(images[1]), cmap = 'gray_r')
        else:
            ax[a,5].imshow(eval(images[1]), cmap = 'gray')
        ax[a,4].set_xlabel(title[0])
        ax[a,5].set_xlabel(title[1])
        ax[a,4].set_title(b, x = 1.15)

    ### Interaction Shape Functions ###
    def shape_fx_ints(a,b,labels,title):
        x = np.array(ebm.explain_global().data(b)['left_names'])
        y = np.array(ebm.explain_global().data(b)['right_names'])
        z = np.array(ebm.explain_global().data(b)['scores'].T)

        interaction = ax[a,7].pcolormesh(x,y,z, cmap = 'seismic_r', vmin = -plot_max, vmax = plot_max)
        plt.colorbar(interaction, ax = ax[a,7], location = 'right', fraction = 0.046, pad = 0.04)

        #Manages y-ticks
        y_labels    = np.zeros(len(ax[a,7].yaxis.set_ticklabels([])))
        y_positions = np.linspace(min(ebm.explain_global().data(b)['right_names']), max(ebm.explain_global().data(b)['right_names']), len(y_labels))

        y_labels[0] = np.round(y_positions[0],2)
        y_labels[len(y_labels) - 1] = np.round(y_positions[len(y_labels) - 1],2)

        ax[a,7].yaxis.set_major_locator(ticker.FixedLocator(y_positions))
        ax[a,7].yaxis.set_major_formatter(ticker.FixedFormatter(y_labels))

        [label.set_visible(False) for label in ax[a,7].yaxis.get_ticklabels()[1:len(y_labels) - 1]]

        #Manages x-ticks
        x_labels    = np.zeros(len(ax[a,7].xaxis.set_ticklabels([])))
        x_positions = np.linspace(min(ebm.explain_global().data(b)['left_names']), max(ebm.explain_global().data(b)['left_names']), len(x_labels))

        x_labels[0] = np.round(x_positions[0],2)
        x_labels[len(x_labels) - 1] = np.round(x_positions[len(x_labels) - 1],2)

        ax[a,7].xaxis.set_major_locator(ticker.FixedLocator(x_positions))
        ax[a,7].xaxis.set_major_formatter(ticker.FixedFormatter(x_labels))

        [label.set_visible(False) for label in ax[a,7].xaxis.get_ticklabels()[1:len(x_labels) - 1]]

        #Sets axis labels and title
        ax[a,7].set_xlabel(labels[0],size=8)
        ax[a,7].set_ylabel(labels[1],size=8)
        if 'Tiles' in labels[1]:
            ax[a,7].yaxis.labelpad = -15
        else:
            ax[a,7].yaxis.labelpad = -29
        ax[a,7].xaxis.labelpad = -7
        ax[a,7].set_title(title)
        ax[a,7].tick_params(axis='both', which='major', labelsize=7)

    #Plots the interactions themselves
    int_image_plots(0, names[3].lower().replace(' ', '_').split('_&_'), names[3].replace('&', '').split('  '), "Features:")
    int_image_plots(1, names[4].lower().replace(' ', '_').split('_&_'), names[4].replace('&', '').split('  '), "")
    int_image_plots(2, names[5].lower().replace(' ', '_').split('_&_'), names[5].replace('&', '').split('  '), "")

    #6,5,4,7
    #Plots the feature importance of the interactions
    importance(0,6,3, "Local\n Importance:")
    importance(1,6,4, "")
    importance(2,6,5, "")

    #Plots the shape functions of the interactions
    shape_fx_ints(0,3, names[3].replace('&', '').split('  '), "Shape Function:")
    shape_fx_ints(1,4, names[4].replace('&', '').split('  '), "")
    shape_fx_ints(2,5, names[5].replace('&', '').split('  '), "")

    #Sets the size of the figure & plots
    fig.set_size_inches((8.5, 11), forward=False)
    plt.subplots_adjust(left = 0.012, bottom = 0.045, right = 0.971, top = 0.938, wspace = 0.2, hspace = 0.245)
    plt.show()

#    print(ebm.term_names_)

#    filepath = r'/home/nmitchell/GLCM/EBM-no-interaction/'
#    filepath += 'EBM_no_interaction_' + str(isamp) + ".png"
#    fig.savefig(filepath)
#    print(filepath)
#    plt.close()

