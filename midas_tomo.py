import numpy as np
from math import pow
import collections.abc
import subprocess
import os
import time


def run_tomo(data,dark,whites,workingdir,thetas,shifts,filterNr=2,doLog=1,extraPad=0,autoCentering=1,numCPUs=40,doCleanup=1,ringRemoval=0):
	# Return format: [nrShifts, nrSlices, xDimNew, xDimNew]
	# data (one dark, 2 whites and data floats, tilt corrected projections) [shape: nrThetas,nrSlices,xDim]
	# workingdir
	# thetas (array)
	# filterNr: [2-default] 0 (nothing),1(shepp/logan),2(hann),3(hamming),4(ramp)
	# shiftValue (pixels) # single or start end interval array
	# doLog (1,0)
	# extraPad (1,0)
	# autocentering (1,0)
	# numCPUs [int]
	# doCleanup (1,0) want to remove temporary storage files
	# ringRemoval (default 0)
	start_time = time.time()
	nrThetas,nrSlices,xDim = data.shape
	infn = workingdir+'/input.bin'
	data = data.astype(np.float32)
	inF = open(infn,'w')
	dark.astype(np.float32).tofile(inF)
	inF.close()
	inF = open(infn,'a')
	whites.astype(np.float32).tofile(inF)
	data.astype(np.uint16).tofile(inF)
	inF.close()
	# We have tilt corrected projections, one dark and two whites in the beginning.
	nrThetas -= 2
	outfnstr = workingdir+'/output'
	still_smaller = True
	power = 0
	while (still_smaller):
		if (xDim > pow (2, power)):
			power+= 1
			still_smaller = True
		else:
			still_smaller = False
	if (xDim == pow (2, power)):
		xDimNew = int(xDim)
	else:
		xDimNew = int(pow(2,power))
	if (extraPad==1):
		power+=1
		xDimNew = int(pow(2,power))
	thetasFile = open(workingdir+'/midastomo_thetas.txt','w')
	for theta in thetas: thetasFile.write(str(theta)+'\n')
	thetasFile.close()
	# Write the config to a config file
	configFile = open(workingdir+'/midastomo.par','w')
	configFile.write('saveReconSeparate 0\n')
	configFile.write('dataFileName '+infn+'\n')
	configFile.write('reconFileName '+outfnstr+'\n')
	configFile.write('areSinos 0\n')
	configFile.write('detXdim '+str(xDim)+'\n')
	configFile.write('detYdim '+str(nrSlices)+'\n')
	configFile.write('thetaFileName '+workingdir+'/midastomo_thetas.txt\n')
	if not isinstance(shifts,collections.abc.Sequence):
		configFile.write('shiftValues '+str(shifts)+' '+str(shifts)+' 1\n')
		nrShifts = 1
	else:
		nrShifts = round(abs((shifts[1]-shifts[0]))/shifts[2])+1
		configFile.write('shiftValues '+str(shifts[0])+' '+str(shifts[1])+' '+str(shifts[2])+'\n')
	configFile.write('ringRemovalCoefficient '+str(ringRemoval)+'\n')
	configFile.write('doLog '+str(doLog)+'\n')
	configFile.write('slicesToProcess -1\n')
	configFile.write('ExtraPad '+str(extraPad)+'\n')
	configFile.write('AutoCentering '+str(autoCentering)+'\n')
	configFile.close()
	print('Time elapsed in preprocessing: '+str(time.time()-start_time)+'s.')
	# Run tomo
	subprocess.call(os.path.expanduser("~/opt/MIDAS/TOMO/bin/MIDAS_TOMO")+" "+workingdir+'/midastomo.par '+str(numCPUs),
									   cwd=workingdir,
				 					   shell=True)
	# Read result
	start_time = time.time()
	outfn = outfnstr+'_NrShifts_'+str(nrShifts).zfill(3)+'_NrSlices_'+str(nrSlices).zfill(5)+'_XDim_'+str(xDimNew).zfill(6)+'_YDim_'+str(xDimNew).zfill(6)+'_float32.bin'
	recon = np.fromfile(outfn,dtype=np.float32,count=(nrSlices*nrShifts*xDimNew*xDimNew)).reshape((nrShifts,nrSlices,xDimNew,xDimNew))
	if doCleanup:
		os.remove(outfn)
		os.remove(os.path.join(workingdir,'midastomo.par'))
		os.remove(os.path.join(workingdir,'midastomo_thetas.txt'))
		os.remove(infn)
		os.remove(os.path.join(workingdir,'fftwf_wisdom_1d_'+str(int(2*xDimNew))+'.txt'))
		os.remove(os.path.join(workingdir,'fftwf_wisdom_2d_'+str(int(2*xDimNew))+'.txt'))
	print("Time elapsed in postprocessing: "+str(time.time()-start_time)+'s.')
	return recon

