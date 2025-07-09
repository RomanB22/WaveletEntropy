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
# Calculate Wavelet Decomposition and Reconstruct the signal
waveDec = mdwtdec(dirdec, GrandAverage.iloc[0:windowLength].values, levels, wavelet_name)
waveRec = mdwtrec(waveDec, wavelet_name, coeffsToUse=[0,1,2,3,4,5,6])
###################
# Plot signal and reconstruction
# electrodeNumber = 0
# plotSignalAndReconstruction(GrandAverage, waveRec, electrodeNames, electrodeNumber, windowLength, samplingRate)
###################
# Plot power spectra for each decomposition level
# powerPartial, freqs, idx = CalculatePowerSpectrum(waveDec, wavelet_name, levels, samplingRate)
# plotPowerSpectra(powerPartial, freqs, idx, freqRanges, levels, electrodeNames, electrodeNumber, samplingRate)
###################
# Calculate Wavelet Energy

###################
# Compare interference and no interference
GrandAverageNoInt, StandardDevNoInt, electrodeNamesNoInt = loadFile(file_path='ARAAVERNOINT.dat', delimiter='\t', headerlines=20, electrodeNumbers=33, samplingRate=samplingRate)
waveDecNoInt = mdwtdec(dirdec, GrandAverageNoInt.iloc[0:windowLength].values, levels, wavelet_name)
GrandAverageInt, StandardDevInt, electrodeNamesInt = loadFile(file_path='ARAAVERINT.dat', delimiter='\t', headerlines=20, electrodeNumbers=33, samplingRate=samplingRate)
waveDecInt = mdwtdec(dirdec, GrandAverageInt.iloc[0:windowLength].values, levels, wavelet_name)

coeffsToUse = [0,1,2,3,4]

totalWavEnergyPerLevelInt, relativeWavEnergyPerLevelInt = CalculateWaveletEnergy(waveDecInt, coeffsToUse=coeffsToUse)
meanWavEnergyInt = np.mean(totalWavEnergyPerLevelInt,axis=0)
stdWavEnergyInt = np.std(totalWavEnergyPerLevelInt,axis=0)
meanRelWavEnergyInt = np.mean(relativeWavEnergyPerLevelInt,axis=0)
stdRelWavEnergyInt = np.std(relativeWavEnergyPerLevelInt,axis=0)

totalWavEnergyPerLevelNoInt, relativeWavEnergyPerLevelNoInt = CalculateWaveletEnergy(waveDecNoInt, coeffsToUse=coeffsToUse)
meanWavEnergyNoInt = np.mean(totalWavEnergyPerLevelNoInt,axis=0)
stdWavEnergyNoInt = np.std(totalWavEnergyPerLevelNoInt,axis=0)
meanRelWavEnergyNoInt = np.mean(relativeWavEnergyPerLevelNoInt,axis=0)
stdRelWavEnergyNoInt = np.std(relativeWavEnergyPerLevelNoInt,axis=0)
plt.errorbar(coeffsToUse, meanRelWavEnergyNoInt, yerr=stdRelWavEnergyNoInt, linestyle='None', marker='^', color='blue', label='No Int', capsize=5.0)
plt.errorbar(coeffsToUse, meanRelWavEnergyInt, yerr=stdRelWavEnergyInt, linestyle='None', marker='^', color='red', label='Int', capsize=5.0)
plt.legend(loc='best')
plt.show()
plt.errorbar(coeffsToUse, meanWavEnergyNoInt, yerr=stdWavEnergyNoInt, linestyle='None', marker='^', color='blue', label='No Int', capsize=5.0)
plt.errorbar(coeffsToUse, meanWavEnergyInt, yerr=stdWavEnergyInt, linestyle='None', marker='^', color='red', label='Int', capsize=5.0)
plt.legend(loc='best')
plt.show()