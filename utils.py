import matplotlib.pyplot as plt
import seaborn as sns
import pywt
import numpy as np
def mdwtdec(dirdec, signal, lev, wavelet):
    """
    Multisignal 1D Discrete Wavelet decomposition.
    [cA_n, cD_n, cD_n-1, …, cD2, cD1]: list of approximation at level n and details at levels n, n-1,  ..., 1, per electrode

    Parameters
    ----------
    dirdec  : char
        Direction indicator: 'r' (row) or 'c' (column)
    signal       : matrix
        Input matrix
    wavelet : Wavelet object or name string
        Wavelet to use
    lev     : int
        Decomposition level (must be >= 0). If level is None then it
        will be calculated using the ``dwt_max_level`` function.
    """
    waveDec = []
    if dirdec == 'r':
        assert(pywt.dwt_max_level(len(signal[0,:]), wavelet)<=lev)
        for i in range(len(signal[:,0])):
            waveDec.append(pywt.wavedec(signal[i,:], wavelet, level=lev))
    elif dirdec == 'c':
        assert(pywt.dwt_max_level(len(signal[:,0]), wavelet)<=lev)
        for i in range(len(signal[0,:])):
            waveDec.append(pywt.wavedec(signal[:,i], wavelet, level=lev))
    return waveDec

def mdwtrec(coeffs, wavelet, coeffsToUse=[]):
    reconstructedSignal = []
    numElectrodes = len(coeffs)
    numLevels = len(coeffs[0])
    for i in range(numElectrodes):
        coeffsAux = [coeffs[i][j] if j in coeffsToUse else np.zeros((1, len(coeffs[i][j]))).flatten() for j in range(numLevels)]
        reconstructedSignal.append(pywt.waverec(coeffsAux, wavelet))
    return reconstructedSignal

def printLevelsFrequencies(SamplingRate, lev):
    levelsFreq = []
    freq = SamplingRate/2.
    for i in range(lev):
        levelsFreq.append([freq/2., freq])
        freq /= 2.
    levelsFreq.append([0, freq])
    [print("Approximate frequency range for detail at level %d: %s" % (i+1, levelsFreq[i])) for i in range(lev)]
    print("Approximate frequency range for approximation at level %d: %s" % (lev, levelsFreq[-1]))
    return levelsFreq

def loadFile(file_path='ARAAVERINT.dat', delimiter='\t', headerlines=20, electrodeNumbers=33, samplingRate=500.):
    import pandas
    electrodeNames = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
                      'CZ', 'FZ', 'PZ', 'FCZ', 'CPZ', 'CP3', 'CP4', 'FC3', 'FC4', 'TP7', 'TP8', 'FPZ', 'OZ', 'FT7',
                      'FT8', 'A1', 'BP1']
    df = pandas.read_csv(file_path, names=electrodeNames, delimiter=delimiter, header=None, skiprows=headerlines, usecols=range(electrodeNumbers))
    lineSeparator = int(np.argwhere((df == '[Standard Deviation Data]').any(axis=1)).flatten()[0])
    GrandAverage = df.iloc[0:lineSeparator].astype(float)
    StandardDev = df.iloc[lineSeparator+1:].astype(float)
    GrandAverage['time'] = np.arange(0, len(GrandAverage)/samplingRate, 1/samplingRate)
    StandardDev['time'] = np.arange(0, len(StandardDev)/samplingRate, 1/samplingRate)
    GrandAverage.set_index('time', inplace=True)
    StandardDev.set_index('time', inplace=True)
    return GrandAverage, StandardDev, electrodeNames

def CalculatePowerSpectrum(waveDec, wavelet_name, levels, samplingRate):
    partialReconstruction = [mdwtrec(waveDec, wavelet_name, coeffsToUse=[i]) for i in range(levels + 1)]
    powerPartial = []
    for electrodeNumber in range(len(waveDec)):  # To use as an example
        powerPartial.append([np.abs(np.fft.fft(partialReconstruction[i][electrodeNumber])) ** 2 for i in range(levels + 1)])
    time_step = 1. / samplingRate
    freqs = np.fft.fftfreq(partialReconstruction[0][0].size, time_step)
    idx = np.argsort(freqs)

    powerPartial = np.asarray(powerPartial)

    return powerPartial, freqs, idx

def CalculateWaveletEnergy(waveDec, coeffsToUse=[]):
    numElectrodes = len(waveDec)
    totalWavEnergyPerLevel = []
    relativeWavEnergyPerLevel = []
    for i in range(numElectrodes):
        totalWavEnergyPerLevel.append([sum(waveDec[i][j]**2) for j in coeffsToUse])
        relativeWavEnergyPerLevel.append([sum(waveDec[i][j]**2) for j in coeffsToUse]/sum([sum(waveDec[i][j]**2) for j in coeffsToUse]))

    totalWavEnergyPerLevel, relativeWavEnergyPerLevel = np.asarray(totalWavEnergyPerLevel), np.asarray(relativeWavEnergyPerLevel)
    return totalWavEnergyPerLevel, relativeWavEnergyPerLevel

def plotSignalAndReconstruction(GrandAverage, waveRec, electrodeNames, electrodeNumber, windowLength, samplingRate):
    plt.figure()
    plt.title("Electrode: %s" % electrodeNames[electrodeNumber])
    sns.lineplot(
        x=np.arange(0, len(GrandAverage[electrodeNames[electrodeNumber]].iloc[0:windowLength]) * 1000. / samplingRate,
                    1000. / samplingRate), y=GrandAverage['FP1'].iloc[0:windowLength].values, marker='o', label='Signal')
    sns.lineplot(
        x=np.arange(0, len(GrandAverage[electrodeNames[electrodeNumber]].iloc[0:windowLength]) * 1000. / samplingRate,
                    1000. / samplingRate), y=waveRec[0], label='Wavelet reconstruction')
    plt.legend(loc='best')
    plt.show()
    plt.close()
    return None

def plotPowerSpectra(powerPartial, freqs, idx, freqRanges, levels, electrodeNames, electrodeNumber, samplingRate):
    plt.figure()
    plt.title("Electrode: %s" % electrodeNames[electrodeNumber])
    [plt.plot(freqs[idx], powerPartial[electrodeNumber][i][idx], label='Detail_%d %s Hz, f_max=%d Hz' % (
    levels - i + 1, freqRanges[levels - i], abs(freqs[np.argmax(powerPartial[electrodeNumber][i])]))) for i in
     reversed(range(1, levels + 1))]
    plt.plot(freqs[idx], powerPartial[electrodeNumber][0][idx], label='Approximation_%d %s Hz, f_max=%d Hz' % (
    levels, freqRanges[-1], abs(freqs[np.argmax(powerPartial[electrodeNumber][0])])))
    plt.legend(loc='best')
    plt.xlim([0, samplingRate / 2.])
    plt.show()
    plt.close()
    return None