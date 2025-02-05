# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:00:20 2024

@author: aneta.kartali

This script is used for calculating
hand-crafted features of eye-tracking data.

"""

from matplotlib import pyplot as plt
import numpy as np
import os
import porespy as ps
from PIL import Image
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

def fixation_saccade_data(x, y, t):
    fxs, fixations = fixation_detection(x, y, t, missing=-1, maxdist=0.02, mindur=48)    

    FIC, FIV = fixation_intersection_coefficient(x, y, t, fixations)
    FFD = fixation_fractal_dimension(x, y, t, fixations)
    
    sdurs, sspeeds, fdurs = [], [], []
    for ix in range(1, len(fixations)):
        fstart0, fend0, dur0, fx0, fy0 = fixations[ix-1]
        fstart1, fend1, dur1, fx1, fy1 = fixations[ix]

        dx = fx1 - fx0
        dy = abs(fy1 - fy0)
        dt = fstart1 - fend0

        sdurs.append(dt)
        sspeeds.append(dx / dt)
        fdurs.append(dur1)

    avgSaccDur = np.mean(sdurs)  # ms
    stdSaccDur = np.std(sdurs)  # ms
    avgSaccSpeed = 1000*np.mean(sspeeds)  # %screen / us
    avgFixDur = np.mean(fdurs)  # ms

    return sdurs, sspeeds, fdurs, avgSaccDur, stdSaccDur, avgSaccSpeed, avgFixDur, FIC, FIV, FFD

def intersection(x1,x2,x3,x4,y1,y2,y3,y4):
    d = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if d:
        xs = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / d
        ys = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)) / d
        if (xs >= min(x1,x2) and xs <= max(x1,x2) and
            xs >= min(x3,x4) and xs <= max(x3,x4)):
            return xs, ys
            
def fixation_intersection_coefficient(x, y, t, fixations):
    nfx = len(fixations)
    FI = []
    for ix in range(len(fixations)):
        fstart, fend, dur, endx, endy = fixations[ix]
        idx_start = list(t).index(fstart)
        idx_end = list(t).index(fend)
        fx = x[idx_start:idx_end]
        fy = y[idx_start:idx_end]

        # xs, ys = [], []
        n_intersect = 0
        for i in range(len(fx)-1):
            for j in range(i-1):
                if xs_ys := intersection(fx[i],fx[i+1],fx[j],fx[j+1],fy[i],fy[i+1],fy[j],fy[j+1]):
                    # xs.append(xs_ys[0])
                    # ys.append(xs_ys[1])
                    n_intersect += 1
        FI.append(n_intersect)
    FIC = np.sum(FI) / nfx
    FIV = np.std(FI)
    return FIC, FIV

def fixation_fractal_dimension(x, y, t, fixations):
    nfx = len(fixations)
    FD = []
    for ix in range(len(fixations)):
        fstart, fend, dur, endx, endy = fixations[ix]
        idx_start = list(t).index(fstart)
        idx_end = list(t).index(fend)
        fx = x[idx_start:idx_end]
        fy = y[idx_start:idx_end]

        plt.figure(frameon=False)
        plt.plot(fx, fy)
        plt.xlim([np.min(fx), np.max(fx)])
        plt.ylim([np.max(fy), np.min(fy)])
        plt.axis('off')
        plt.savefig("tmp.png", dpi=300)
        plt.close()
        with Image.open("tmp.png").convert("L") as img:
            data = ps.metrics.boxcount(np.asarray(img)<255)
        # Fit the successive log(sizes) with log (counts)
        FD.append(data.slope[0])

    FFD = np.sum(FD) / nfx
    return FFD

def fixation_detection(x, y, time, missing=0.0, maxdist=0.02, mindur=50):

    """
    Modified version of PyGaze fixation_detection function. Original version
    can be found here: 
        https://github.com/esdalmaijer/PyGazeAnalyser/blob/master/pygazeanalyser/detectors.py
    
    Detects fixations, defined as consecutive samples with an inter-sample
	distance of less than a set amount of pixels (disregarding missing data)
	

	arguments

	x		-	numpy array of x positions
	y		-	numpy array of y positions
	time	-	numpy array of EyeTribe miliseconds


	keyword arguments

	missing	-	value to be used for missing data (default = 0.0)
	maxdist	-	maximal inter sample distance in relative screen position (default = 0.02)
	mindur	-	minimal duration of a fixation in milliseconds; detected
				fixation cadidates will be disregarded if they are below
				this duration (default = 100)


	returns

	Sfix, Efix
		Sfix -	list of lists, each containing [starttime]
		Efix -	list of lists, each containing [starttime, endtime, duration, endx, endy]
	"""
    
    # empty list to contain data
    Sfix = []
    Efix = []
    
    x_diff = np.abs(np.diff(x, prepend=x[0]))

    # loop through all coordinates
    si = 0
    fixstart = False

    for i in range(1, len(x)):
        if x[i] == missing or y[i] == missing: continue
    
        # calculate distance along x-axis from the current fixation coordinate
        # to the next coordinate
        dist = x_diff[i]

        # check if the next coordinate is below maximal distance
        if dist <= maxdist and not fixstart:
            # start a new fixation
            si = 0 + i
            fixstart = True
            Sfix.append([time[i]])

        elif dist > maxdist and fixstart:
            # end the current fixation
            fixstart = False
            # only store the fixation if the duration is ok
            if time[i-1]-Sfix[-1][0] >= mindur:
                Efix.append([Sfix[-1][0], time[i-1], time[i-1]-Sfix[-1][0], x[si], y[si]])
            
            # delete the last fixation start if it was too short
            else:
                Sfix.pop(-1)
            si = 0 + i

        elif not fixstart:
            si += 1

    return Sfix, Efix

def groznik_features(x, y, Fs, maxdist=0.02, mindur=48):
    readTime = len(x) / Fs
    t = np.linspace(0, readTime*1000, len(x), 1/Fs*1000)  # Time in milliseconds
    
    xs = np.array(x)
    ys = np.array(y)
    ts = np.array(t)

    fxs, fixations = fixation_detection(xs, ys, ts, missing=-1, maxdist=maxdist, mindur=mindur)

    nfx = len(fixations)   # number of fixations
    nfxs = nfx / readTime   # number of fixations per second

    forw, forwSpeed, backw, backwSpeed, fdurs = [], [], [], [], []
    for ix in range(1, len(fixations)):
        fstart0, fend0, dur0, fx0, fy0 = fixations[ix-1]
        fstart1, fend1, dur1, fx1, fy1 = fixations[ix]

        dx = fx1 - fx0
        dy = abs(fy1 - fy0)
        dt = fstart1 - fend0

        # forward saccades
        if dx > 0 and 2*dx > dy:
            forw.append(dx)
            forwSpeed.append(dx / dt)
        # backward saccades
        if dx < 0 and 1*abs(dx) > dy:
            backw.append(dx)
            backwSpeed.append(dx / dt)
        # fixation durations
        if ix == 1: fdurs.append(dur0)
        fdurs.append(dur1)

    nForw = len(forw)
    nBackw = len(backw)

    lenForw = 100*np.median(forw)    # measured as percentage of screen
    medForwSpeed = 1000*np.median(forwSpeed)  # %screen / us
    medFixDur = np.median(fdurs)  # ms
    stdFixDur = np.std(fdurs)  # ms
    
    if backw:
        lenBackw = 100*np.median(backw)  # measured as percentage of screen
        stdlenBackw = 100*np.std(backw)  # %screen
        medBackwSpeed = 1000*np.median(backwSpeed)  # %screen / us
    else:
        lenBackw = 0
        stdlenBackw = 0
        medBackwSpeed = 0

    return readTime, nfx, lenForw, nForw, lenBackw, nBackw, stdlenBackw, medForwSpeed, medBackwSpeed, medFixDur, stdFixDur


def feature_extraction(t, x, y, Fs, thrBackSacc=0, verbose=False):
    # thrStart and thrEnd are the amount (given as percentage) of data to cut off from the start and end of data
    # thrBackSacc: if not zero this function will be used to calculate and will return thrEnd needed
    #              to reach a given number of backward saccades
    readTime, nfx, lenForw, nForw, lenBackw, nBackw, stdlenBackw, medForwSpeed, medBackwSpeed, medFixDur, stdFixDur = groznik_features(x, y, Fs)
    if not nForw and not nBackw:
        return False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0   # signify that calculation wasn't possible
    
    sdurs, speeds, fdurs, avgSaccDur, stdSaccDur, avgSaccSpeed, avgFixDur, FIC, FIV, FFD = fixation_saccade_data(x, y, t)

    fixation_intersection_coeff = FIC
    saccade_variability = stdFixDur
    fixation_intersection_variability = FIV
    fixation_fractal_dimension = FFD
    fixation_count = nfx
    fixation_total_dur = np.sum(fdurs)
    fixation_freq = nfx/readTime
    fixation_avg_dur =  avgFixDur
    saccade_count = nForw + nBackw
    saccade_total_dur = np.sum(sdurs)
    saccade_freq = (nForw + nBackw)/readTime
    saccade_avg_dur = avgSaccDur
    
    active_read_time = (fixation_total_dur + saccade_total_dur) / 1000
    total_read_time = readTime

    if verbose:
        # Proposed
        print('Active reading time: %.1f s' %active_read_time)
        print('Fixation intersection coefficient: %.2f' %FIC)
        print('Saccade variability: %.1f  msec' %stdFixDur)
        print('Fixation intersection variability: %.2f' %FIV)
        print('Fixation fractal dimension: %.2f' %FFD)
        # Conventional
        print('Fixation count: %d' %nfx)
        print('Fixation total duration: %.1f ms' %np.sum(fdurs))
        print('Fixation frequency: %.1f Hz' %(nfx/readTime))
        print('Fixation average duration: %.1f ms' %avgFixDur)
        print('Saccade count: %d' %(nForw + nBackw))
        print('Saccade total duration: %.1f ms' %np.sum(sdurs))
        print('Saccade frequency: %.1f Hz' %((nForw + nBackw)/readTime))
        print('Saccade average duration: %.1f ms' %avgSaccDur)
        print('Total reading time: %.1f s' %total_read_time)
        print()

    return True, active_read_time, fixation_intersection_coeff, saccade_variability, fixation_intersection_variability, fixation_fractal_dimension, fixation_count, fixation_total_dur, fixation_freq, fixation_avg_dur, saccade_count, saccade_total_dur, saccade_freq, saccade_avg_dur, total_read_time
