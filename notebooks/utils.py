from pickletools import float8
from obspy.core import *
from disba import PhaseDispersion
from scipy.signal import filtfilt
import os
import obspy
import scipy
import scipy.fftpack
from scipy import interpolate, ndimage, misc
import segyio
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import colorcet as cc
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')


def readsu(path: str, filename=None, endian='little', wholepath=False) -> obspy.Stream:
    """Reads seismic unix files

    Args:
        path (str): Path of the file or folder, depending on the wholepath parameter
        filename (str, optional): Name of the file to read. Defaults to None.
        endian (str, optional): Parameter for reading in obspy. Defaults to 'little'.
        wholepath (bool, optional): If the path is whole or splitted in path/filename. Defaults to False.

    Returns:
        stream: stream of data readed
    """

    if wholepath:
        fname = path + '/' + filename
    else:
        fname = path

    if endian == 'big':
        stream = obspy.read(fname,
                            format='SU',
                            byteorder='>')
    else:
        stream = obspy.read(fname,
                            format='SU',
                            byteorder='<')
    return stream


def get_fft(traces, dt, nt):
    # Get temporal Fourier transform for each of the traces
    # f = np.linspace(0.0, 1.0/(2.0*dt), nt//2)
    f = scipy.fftpack.fftfreq(nt, dt)
    U = scipy.fftpack.fft(traces)
    if np.size(U.shape) > 1:
        return U[:, 0:nt//2], f[0:nt//2]
    else:
        return U[0:nt//2], f[0:nt//2]


def get_dispersion(traces, dx, cmin, cmax, dc, fmax):
    """ calculate dispersion curves after Park et al. 1998
    INPUTS
    traces: SU traces
    dx: distance between stations (m)
    cmax: upper velocity limit (m/s)
    fmax: upper frequency limit (Hz)
    OUTPUTS
    f: 1d array frequency vector
    c: 1d array phase velocity vector
    img: 2d array (c x f) dispersion image
    fmax_idx: integer index corresponding to the given fmax
    U: 2d array (nr x npts//2) Fourier transform of traces
    t: 1d array time vector
    """
    nr = len(traces)
    dt = traces[0].stats.delta
    print('dt: ', dt)
    nt = traces[0].stats.npts
    print('nt: ', nt)
    t = np.linspace(0.0, nt*dt, nt)
    traces.detrend()
    traces.taper(0.05, type='hann')
    U, f = get_fft(traces, dt, nt)
    # dc = 10.0 # phase velocity increment
    c = np.arange(cmin, cmax, dc)  # set phase velocity range
    df = f[1] - f[0]
    fmax_idx = int(fmax//df)
    print('Frequency resolution up to %5.2f kHz: %i bins' % (fmax, fmax_idx))
    print('Phase velocity resolution up to %5.2f m/s: %i bins' % (cmax, len(c)))
    img = np.zeros((len(c), fmax_idx))
    x = np.linspace(0.0, (nr-1)*dx, nr)
    for fi in range(fmax_idx):  # loop over frequency range
        for ci in range(len(c)):  # loop over phase velocity range
            k = 2.0*np.pi*f[fi]/(c[ci])
            img[ci, fi] = np.abs(
                np.dot(dx * np.exp(1.0j*k*x), U[:, fi]/np.abs(U[:, fi])))

    return f, c, img, fmax_idx, U, t


def cn(x, snr):
    """Create colored noise """
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    n = filtfilt(np.ones(10)/10, 1,
                 filtfilt(np.ones(5)/5, 1, (np.random.randn(len(x))
                          * np.sqrt(npower)).T, method='gust').T,
                 method='gust')
    noise_signal = x + n
    return noise_signal


def plotdata(dx: float, st=None, nx=100, cmin=10, cmax=2000, dc=1.0, fmin=5, fmax=70, clip=100, cmap='jet', tx=2000, cpr=None):
    """Function to generate a plot of the seismic data + dispersion panel. Requires Disba.
    Default values are taken from the general analysis.

    Args:
        st (obspy.Stream, optional): Stream of traces to be plotted. Defaults to None.
        nx (int, optional): Max number to plot for trace axis. Defaults to 100.
        tx (int, optional): Max number to plot for time axis. Defaults to 2000.
        dx (float, optional): Receiver separation of survey. Required from the user.
        cmin (int, optional): Minimum velocity value. Defaults to 10.
        cmax (int, optional): Maximum velocity value. Defaults to 2000.
        dc (float, optional): Step value for velocity ranges. Defaults to 1.0.
        fmin (int, optional): Minimum value for frequency range. Defaults to 5.
        fmax (int, optional): Maximum value for frequency range. Defaults to 100.
        clip (_type_, optional): Maximum value for colormap to be clipped with. Defaults to 1e0.
        cmap (str, optional): colormap to be used, accepts colorcet cmaps. Defaults to 'jet'.

    """
    data_su = np.array([st[i].data for i in range(len(st))]).T
    data_su = data_su[:tx, :nx]

    maxval = np.max(data_su)
    clip = maxval*clip/100

    # phase shift method
    f, c, img, fmax_idx, U, t = get_dispersion(
        st, dx, cmin, cmax, dc, fmax)

    fig, (ax1, ax) = plt.subplots(1, 2, figsize=(12, 8))
    ax1.imshow(data_su, cmap='seismic', vmin=-clip,
               vmax=clip, interpolation='bicubic')
    ax1.axis('tight')
    ax1.set_xlabel('Receiver [m]', fontsize=14)
    ax1.set_ylabel('Time [s]', fontsize=14)

    ax.imshow(img[:, :], aspect='auto', origin='lower', cmap=cmap, extent=(
        f[0], f[fmax_idx], c[0], c[-1]), interpolation='bicubic')
    ax.grid(linestyle='--', linewidth=1, alpha=0.5)

    if cpr != None:
        ax.plot(1/cpr[0], cpr[1], linewidth=2, color='red')

    ax.set_xlabel('Frequency [Hz]', fontsize=14)
    ax.set_ylabel('Phase velocity [m/s]', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    #plt.plot(freq_fm, outputs_nparray[n], '--r')
    plt.xlim((fmin, f[fmax_idx]))
    plt.ylim((cmin, cmax))
    plt.tight_layout()

    return fig, ax1, ax


def plot_models(mods: list, u, dx: float, fmin=5, fmax=70, cmin=0, cmax=2500, dc=1):

    fig = plt.figure(figsize=[10, 10])
    ax1 = fig.add_subplot(231)
    ax1.invert_yaxis()
    ax2 = fig.add_subplot(232)
    ax2.invert_yaxis()
    ax3 = fig.add_subplot(233)
    ax3.invert_yaxis()
    ax4 = fig.add_subplot(212)

    ax1.set_xlim([0, 1500])
    ax1.set_ylim([mods[0].maxZ, 0])
    ax2.set_xlim([0, 3000])
    ax2.set_ylim([mods[0].maxZ, 0])
    ax3.set_xlim([0, 2500])
    ax3.set_ylim([mods[0].maxZ, 0])
    ax4.set_xlim([fmin, fmax])
    ax4.set_ylim([cmin, cmax])

    ax1.set_xlabel('Vs [m/s]')
    ax1.set_ylabel('Thinkness [m]')
    ax2.set_xlabel('Vp [m/s]')
    ax2.set_ylabel('Thinkness [m]')
    ax3.set_xlabel('Rho [g/cm^3]')
    ax3.set_ylabel('Thinkness [m]')
    ax4.set_xlabel('Frequency [Hz]')
    ax4.set_ylabel('Phase velocity [m/s]')

    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()

    f, c, img, fmax_idx, U, t = get_dispersion(
        u, dx, cmin, cmax, dc, fmax)

    colors = cc.cm.glasbey(np.linspace(0, 1, len(mods)))
    colors2 = cc.cm.glasbey_dark(np.linspace(0, 1, len(mods)))

    sumpv = 0

    for model in tqdm(mods):

        sum = 0
        vpplot = []
        vsplot = []
        rhoplot = []
        y = [0]

        for l in range(len(model.velocity_model)):
            sum = sum + model.velocity_model[l][0]
            y.append(sum)
            vpplot.append(model.velocity_model[l][1])
            vsplot.append(model.velocity_model[l][2])
            rhoplot.append(model.velocity_model[l][3])

        vpplot.append(vpplot[-1])
        rhoplot.append(rhoplot[-1])
        vsplot.append(vsplot[-1])

        plt.subplot(231)
        ax1.step(vsplot, y, color=colors[model.counter], alpha=0.5)

        plt.subplot(232)
        ax2.step(vpplot, y, color=colors[model.counter], alpha=0.5)

        plt.subplot(233)
        ax3.step(rhoplot, y, color=colors[model.counter], alpha=0.5)

        plt.subplot(212)
        ax4.plot(1/model.cpr[0], model.cpr[1],
                 color=colors2[model.counter], alpha=0.1)

        sumpv = sumpv + model.cpr[1]

    fig.subplots_adjust(wspace=0.5, hspace=0.2)

    differpv = 0

    meanpv = sumpv / len(mods)

    for mod in mods:
        diffpv = (mod.cpr[1] - meanpv)**2
        differpv = differpv + diffpv
    anovpv = differpv / len(mods)
    stdpv = np.sqrt(anovpv)

    ax4.plot(1/mods[0].cpr[0], meanpv, 'w--', linewidth=1)
    ax4.plot(1/mods[0].cpr[0], meanpv - stdpv, 'w--', linewidth=1)
    ax4.plot(1/mods[0].cpr[0], meanpv + stdpv, 'w--', linewidth=1)

    #plt.fill_between(1/mods[0].cpr[0], meanpv - stdpv, meanpv + stdpv, color='green', alpha=0.5, zorder=100)

    cax = ax4.imshow(img[:, :], aspect='auto', origin='lower', cmap='YlGnBu', extent=(
        f[0], f[fmax_idx], c[0], c[-1]), interpolation='bicubic', zorder=3, alpha=0.4)

    ax4.contour(np.where(img > 9.8, img, np.nan), aspect='auto', origin='lower', extent=(
        f[0], f[fmax_idx], c[0], c[-1]), interpolation='bicubic', zorder=3, alpha=0.05, linewidth=1, levels=10, vmin=18, colors='black')

    ax4.set_xlabel('Frequency [Hz]', fontsize=14)
    ax4.set_ylabel('Phase velocity [m/s]', fontsize=14)
    ax4.tick_params(axis='both', which='major', labelsize=14)
    ax4.tick_params(axis='both', which='minor', labelsize=14)
    plt.xlim((fmin, f[fmax_idx]))
    plt.ylim((cmin, cmax))
    plt.tight_layout()


class VelModel:

    def __init__(self, maxZ: int, maxL: int, minVs: int, maxVs: int, sizex: int, path=str, ranvar=False, minran1=1, maxran1=10, maxranf=20, get_plot=False, maxfreq=40, save_data=False, random_range=50, counter=1, dx=0.5, first_mode=False):
        """1D velocity model object.

        Args:
            maxZ (int): Maximum depth to reach
            maxL (int): Number of layers in the model
            minVs (int): Min value of Vs used for the model
            maxVs (int): Max value of Vs used for the model
            maxran1 (int, optional): Max value for random thickness of first layer. Defaults to 10.
            maxranf (int, optional): Max value for random thickness of following layers. Defaults to 20.
            counter (int, optional): Parameter used for counter when creating batch of models. Defaults to 1. 
            sizex (int, optional): Total offset used for fdelmodc calculation.
            dx, dz (float): Values used for dx and dz parameters in fdelmodc.
            path (str): Path where to save the .npy data and labels
            maxfreq (int): max value of frequency for dispersion curve of fundamental mode

        Methods:
            get_params(plot=False): Creates 1D vel. models by random initialization of given parameters.
            get_dispercurves(period, plot=False): Generates the dispersion curve of the model using Disba.
            get_message(): Creates the message that needs to be passed to the fdelmodc script.
        """
        self.maxZ = maxZ
        self.maxL = maxL
        self.minVs = minVs
        self.maxVs = maxVs
        self.aVs = (maxZ - 0) / (maxVs - minVs)
        self.bVs = -self.aVs * minVs
        self.minran1 = minran1
        self.maxran1 = maxran1
        self.maxranf = maxranf
        self.random_range = random_range
        self.counter = counter
        self.sizex = sizex
        self.dx = dx
        self.maxfreq = maxfreq
        self.message = None
        self.ranvar = ranvar
        self.first_mode = first_mode

        self.get_params(plot=get_plot)
        self.get_dispercurves(plot=get_plot)
        self.get_message()

        if save_data:
            self.path = path
            self.save_data()

    def get_params(self, plot=False):
        """Generates the parameters of the model (Vp, Vs, Rho)

        Args:
            plot (bool, optional): Creates a simple plot of the variables for inspection. Defaults to False.
        """

        n = 0
        self.Thi = [[]] * (self.maxL+1)
        remThi = [[]] * (self.maxL+1)
        self.Vs = [[]] * (self.maxL+1)
        self.Vp = [[]] * (self.maxL+1)
        self.Rho = [[]] * (self.maxL+1)

        while n < self.maxL:
            i = 0
            self.Thi[0] = 0
            self.Vs[0] = 0
            self.Vp[0] = 0
            self.Rho[0] = 0
            remThi[0] = self.maxZ
            thi = 0  # Zero inicialization
            rand_left1 = self.minran1  # Lesser value for thickness of first layer
            rand_right1 = self.maxran1  # Upper value for thickness of first layer
            rand_left = 1  # Lesser value for thickness of following layers
            rand_right = self.maxranf  # Upper value for thickness of following layers

            while True:
                i = i + 1

                if i == 1:  # Layer generation based on order and defined parameters
                    self.Thi[i] = random.randint(rand_left1, rand_right1)
                elif i == self.maxL:
                    self.Thi[i] = self.maxZ - thi
                else:
                    self.Thi[i] = random.randint(rand_left, rand_right)
                remThi[i] = remThi[i-1] - self.Thi[i]

                # If the last layer goes beyond maxZ it recalculates the thickness of the layer
                if remThi[i] <= 0:
                    self.Thi[i] = self.Thi[i] - abs(remThi[i])
                    # Vs
                    if self.ranvar:
                        self.Vs[i] = (((self.Thi[i] / 2 + thi) - self.bVs) / self.aVs) + \
                            np.random.uniform(
                                low=-1*self.random_range, high=self.random_range, size=1)[0]

                    else:
                        modran = (100 - 40*(self.Thi[i] / 2 + thi)/50)

                        self.Vs[i] = (((self.Thi[i] / 2 + thi) - self.bVs) / self.aVs) + \
                            np.random.uniform(
                                low=-1*modran, high=modran, size=1)[0]
                    # Vp
                    self.Vp[i] = 2.5 * self.Vs[i]
                    # Density
                    # Gardner's equation
                    self.Rho[i] = 310 * self.Vp[i] ** 0.25
                    break

                else:  # Model definition of rho, Vs, Vp
                    # Vs
                    if self.ranvar:
                        self.Vs[i] = (((self.Thi[i] / 2 + thi) - self.bVs) / self.aVs) + \
                            np.random.uniform(
                                low=-1*self.random_range, high=self.random_range, size=1)[0]

                    else:

                        modran = (100 - 40*(self.Thi[i] / 2 + thi)/50)

                        self.Vs[i] = (((self.Thi[i] / 2 + thi) - self.bVs) / self.aVs) + \
                            np.random.uniform(
                                low=-1*modran, high=modran, size=1)[0]
                    # Vp
                    self.Vp[i] = 2.5 * self.Vs[i]
                    # Density
                    # Gardner's equation
                    self.Rho[i] = 310 * self.Vp[i] ** 0.25

                thi = thi + self.Thi[i]

            self.velocity_model = np.zeros((i, 4), dtype='float')

            for j in range(i):
                self.velocity_model[j, 0] = self.Thi[j+1]
                self.velocity_model[j, 1] = self.Vp[j+1]
                self.velocity_model[j, 2] = self.Vs[j+1]
                self.velocity_model[j, 3] = self.Rho[j+1]

            n = n + 1

        if plot:

            sum = 0
            vpplot = []
            vsplot = []
            rhoplot = []
            y = [0]

            for l in range(i):
                sum = sum + self.Thi[l]
                y.append(sum)
                vpplot.append(self.Vp[l])
                vsplot.append(self.Vs[l])
                rhoplot.append(self.Rho[l])

            vpplot.append(vpplot[-1])
            rhoplot.append(rhoplot[-1])
            vsplot.append(vsplot[-1])

            fig, axs = plt.subplots(1, 3, figsize=(10, 5), sharey=True)

            axs[0].step(vsplot, y)
            axs[1].step(vpplot, y, color='red')
            axs[2].step(rhoplot, y, color='green')

            axs[0].set(title='Vs', xlabel='Wave Velocity [m/s]',
                       ylabel='Depth [m]')
            axs[1].set(title='Vp', xlabel='Wave Velocity [m/s]',
                       ylabel='Depth [m]')
            axs[2].set(title='Density', xlabel='Rho', ylabel='Depth [m]')

            plt.gca().invert_yaxis()

    def get_dispercurves(self, plot=False):
        period = np.linspace(1/self.maxfreq, 0.2, 500)
        period_first = np.linspace(1/self.maxfreq, 1/15, 500)

        pd = PhaseDispersion(*self.velocity_model.T)

        self.cpr = pd(period, mode=0, wave='rayleigh')
        
        if self.first_mode:
            
            _first = pd(period_first, mode=1, wave='rayleigh')
            # f_first = interpolate.interp1d(self._first[0], self._first[1], bounds_error=False)
            
            self.first = _first[1]

            if max(np.gradient(self.first)) > 10:
                self.first = ndimage.median_filter(self.first, size=17, mode='reflect')

        if plot:
            plt.plot(1/self.cpr[0], self.cpr[1])

    def save_data(self):

        np.savetxt(self.path + '/model/' + str(self.counter) +
                   '_velocity_model.txt', self.velocity_model, fmt='%10.5f')
                   
        np.save(self.path + '/label/' +
                str(self.counter) + '_label.npy', self.cpr[1])
        
        if self.first_mode:
            np.save(self.path + '/label/' + str(self.counter) + '_label.npy', np.append(self.cpr[1], self.first))

    def get_message(self):

        fthi = 0
        for k in range(len(self.velocity_model)):
            if k == 0:
                self.message = 'makemod file_base=' + str(self.counter) + '.su cp0=' + str(self.Vp[k+1]) + ' cs0=' + str(self.Vs[k+1]) + ' ro0=' + str(self.Rho[k+1]) \
                    + ' sizex=' + str(self.sizex) + ' sizez=' + str(self.maxZ) + \
                    ' dx=' + str(self.dx) + ' dz=' + str(self.dx) + ' orig=0,0'
            else:
                fthi = fthi + self.Thi[k]
                self.message = self.message + ' intt=def poly=0 cp=' + str(self.Vp[k+1]) + ' cs=' + str(self.Vs[k+1]) + ' ro=' + str(self.Rho[k+1]) \
                    + ' x=' + str(0) + ',' + str(self.sizex) + ' z=' + \
                    str(fthi) + ',' + str(fthi)


def checkdx(mods, dt=0.0001, Fmax=57.539680, disp=5):
    """Checks if the values of dx and dt set for the modeling allow for stability in fdelmodc

    Args:
        mods (list): List of models generated previously
        dt (float, optional): value for dt of wavelet. Defaults to 0.0001.
        Fmax (float, optional): Max frequency in the wavelet. Defaults to 57.539680.
        disp (int, optional): Reference value from the manual. Defaults to 5.
    """

    Vsmin = [[]] * len(mods)
    Vpmax = [[]] * len(mods)

    dx = mods[0].dx

    n = 0
    for model in mods:

        Vpmax[n] = max(model.velocity_model[:, 1])
        Vsmin[n] = min(model.velocity_model[:, 2])
        n = n+1

    Cmax = max(Vpmax)
    Cmin = min(Vsmin)

    need_change = False

    if dt < ((0.606*dx)/Cmax):
        print('Passed the STABILITY test required for convergence, the models can be computed :)')

        if Cmin < Fmax*dx*disp:

            if (dt*Cmax)/0.606 < Cmin/(Fmax*disp):
                print('There is dispersion in the parameters, be cautious, the critical value for dx is less than {:.3f}'.format(
                    Cmin/(Fmax*disp)))

                need_change = True

            else:
                print('Under the current parameters there is no real value for dx that can fix the dispersion, change the parameters of Vs or decrease dt')
                need_change = True

        else:
            print('Everything seems to be fine \U0001F600')

    else:
        print('Failed to pass the stability test, please check the values for dx (and dz as well), the critical value for dx is greater than {:.3f}'.format(
            dt*Cmax/0.606))
        need_change = True

    print('\n The ranges for dx are between {:.3f} and {:.3f}'.format(
        dt*Cmax/0.606, Cmin/(Fmax*disp)))

    return need_change


def getnorm(mat, scale=1) -> colors.TwoSlopeNorm:
    """Generates the color norm for plotting matrixes 

    Args:
        mat (numpy array): _description_
        scale (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    return colors.TwoSlopeNorm(vmin=np.min(mat)/scale, vcenter=0, vmax=np.max(mat)/scale)
