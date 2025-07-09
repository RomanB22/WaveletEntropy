from utils import *

###################
# Data information
samplingRate = 500.
levels = 6
windowLength = 2**(levels+2)
electrodeNumbers=33
wavelet_name = 'db2'
dirdec = 'c'
# To see list of wavelets
# print(pywt.wavelist(kind='discrete'))
# To print the approximate freqeuncy range for each level.
# See: https://dsp.stackexchange.com/questions/82146/discrete-wavelet-decomposition-over-detail-coefficients
freqRanges = printLevelsFrequencies(samplingRate, levels)
print("Window length is %.2f ms" % (windowLength*1000./samplingRate))
###################
# Loading file
GrandAverage, StandardDev, electrodeNames = loadFile(file_path='ARAAVERNOINT.dat', delimiter='\t', headerlines=20, electrodeNumbers=33, samplingRate=samplingRate)
###################
# Calculate Wavelet Decomposition
waveDec = mdwtdec(dirdec, GrandAverage.iloc[0:windowLength].values, levels, wavelet_name)
waveRec = mdwtrec(waveDec, wavelet_name, coeffsToUse=[0,1,2,3,4,5,6])
###################
# Plot signal and reconstruction
electrodeNumber = 0
plotSignalAndReconstruction(GrandAverage, waveRec, electrodeNames, electrodeNumber, windowLength, samplingRate)
###################
# Plot power spectra for each decomposition level
powerPartial, freqs, idx = CalculatePowerSpectrum(waveDec, wavelet_name, levels, samplingRate)
plotPowerSpectra(powerPartial, freqs, idx, freqRanges, levels, electrodeNames, electrodeNumber, samplingRate)
###################
# Calculate Wavelet Energy
totalWavEnergyPerLevel, relativeWavEnergyPerLevel = CalculateWaveletEnergy(waveDec, coeffsToUse=[0,1,2,3,4])
meanWavEnergy = np.mean(totalWavEnergyPerLevel,axis=0)
stdWavEnergy = np.std(totalWavEnergyPerLevel,axis=0)
meanRelWavEnergy = np.mean(relativeWavEnergyPerLevel,axis=0)
stdRelWavEnergy = np.std(relativeWavEnergyPerLevel,axis=0)
print(meanRelWavEnergy, stdRelWavEnergy)