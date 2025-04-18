#!/bin/bash

echo "Using Python version:"
python3 --version
echo "PYTHONPATH is: $PYTHONPATH"
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el9_amd64_gcc12
cd /cvmfs/cms.cern.ch/el9_amd64_gcc12/cms/cmssw/CMSSW_14_1_0/src
eval `scramv1 runtime -sh`

export PYTHONNOUSERSITE=1
cd /afs/cern.ch/user/a/avendras/work/Underlying-Event
echo "starting postProcess python file"
python3 /afs/cern.ch/user/a/avendras/work/Underlying-Event/Underlying-Event/main_scripts/underlying_event_main_postProcess.py --csv_path /eos/user/a/avendras/Underlying-Event/Underlying-Event/filtered_Z_events_30.csv| tee postProcess30.txt










