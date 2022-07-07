#!/usr/bin/env python3

import sys
import os

import numpy as np
import math

from icecube import LeptonInjector,dataclasses,recclasses,tableio,hdfwriter
from I3Tray import *

import LeptonWeighter as LW

config      = sys.argv[1]
gcdfile     = sys.argv[2]
infile      = sys.argv[3]
outdir      = sys.argv[4]

outname = os.path.basename(infile)

dsdxdy_nu_CC    = "/n/holylfs05/LABS/arguelles_delgado_lab/Lab/meows_dnn/CrossSections/dsdxdy_nu_CC_iso.fits"
dsdxdy_nubar_CC = "/n/holylfs05/LABS/arguelles_delgado_lab/Lab/meows_dnn/CrossSections/dsdxdy_nubar_CC_iso.fits"
dsdxdy_nu_NC    = "/n/holylfs05/LABS/arguelles_delgado_lab/Lab/meows_dnn/CrossSections/dsdxdy_nu_NC_iso.fits"
dsdxdy_nubar_NC = "/n/holylfs05/LABS/arguelles_delgado_lab/Lab/meows_dnn/CrossSections/dsdxdy_nubar_NC_iso.fits"
Flux_conv             = "/n/holylfs05/LABS/arguelles_delgado_lab/Lab/meows_dnn/Fluxes/Flux_AIRS_sib_HG_th24_dm2/atmospheric_0_0.000000_0.000000_0.000000_0.000000_0.000000_0.000000.hdf5"
Flux_astro            = "/n/holylfs05/LABS/arguelles_delgado_lab/Lab/meows_dnn/Fluxes/Flux_AIRS_sib_HG_th24_dm2/astro_0_0.000000_0.000000_0.000000_0.000000_0.000000_0.000000.hdf5"
Flux_prompt            = "/n/holylfs05/LABS/arguelles_delgado_lab/Lab/meows_dnn/Fluxes/Flux_AIRS_sib_HG_th24_dm2/prompt_atmospheric_0_0.000000_0.000000_0.000000_0.000000_0.000000_0.000000.hdf5"
LIConfiguration       = config
simulation_generation = LW.MakeGeneratorsFromLICFile(LIConfiguration)
nusquids_flux_conv    = LW.nuSQUIDSAtmFlux(Flux_conv)
nusquids_flux_astro   = LW.nuSQUIDSAtmFlux(Flux_astro)
nusquids_flux_prompt  = LW.nuSQUIDSAtmFlux(Flux_prompt)
xs                    = LW.CrossSectionFromSpline(dsdxdy_nu_CC, dsdxdy_nubar_CC, dsdxdy_nu_NC, dsdxdy_nubar_NC)
weighter_conv         = LW.Weighter(nusquids_flux_conv, xs, simulation_generation)
weighter_astro        = LW.Weighter(nusquids_flux_astro, xs, simulation_generation)
weighter_prompt       = LW.Weighter(nusquids_flux_prompt, xs, simulation_generation)


def get_variables(frame):

    LWevent                         = LW.Event()
    EventProperties                 = frame['EventProperties']
    LeptonInjectorProperties        = frame['LeptonInjectorProperties']
    LWevent.primary_type            = LW.ParticleType(EventProperties.initialType)
    LWevent.final_state_particle_0  = LW.ParticleType(EventProperties.finalType1)
    LWevent.final_state_particle_1  = LW.ParticleType(EventProperties.finalType2)
    LWevent.zenith                  = EventProperties.zenith
    LWevent.energy                  = EventProperties.totalEnergy
    LWevent.azimuth                 = EventProperties.azimuth
    LWevent.interaction_x           = EventProperties.finalStateX
    LWevent.interaction_y           = EventProperties.finalStateY
    LWevent.total_column_depth      = EventProperties.totalColumnDepth
    LWevent.radius                  = EventProperties.impactParameter
    LWevent.x                       = 0. 
    LWevent.y                       = 0.
    LWevent.z                       = 0.
    wconv                           = weighter_conv.weight(LWevent)
    wastro                          = weighter_astro.weight(LWevent)
    wprompt                         = weighter_prompt.weight(LWevent)
    
    frame.Put("PrimaryType"      ,dataclasses.I3Double(LW.ParticleType(EventProperties.initialType)))
    frame.Put("NuEnergy"         ,dataclasses.I3Double(EventProperties.totalEnergy))
    frame.Put("NuZenith"         ,dataclasses.I3Double(EventProperties.zenith))
    frame.Put("wconv"            ,dataclasses.I3Double(wconv))
    frame.Put("wastro"           ,dataclasses.I3Double(wastro))
    frame.Put("wprompt"          ,dataclasses.I3Double(wprompt))

    pulse_series = dataclasses.I3RecoPulseSeriesMap.from_frame(frame,'TTPulses_NoDC')
    NChan_NoDC = len(pulse_series)
    Qtot_NoDC  = 0
    for om, pulses in pulse_series:
        Qom = 0
        for pulse in pulses: Qom += pulse.charge
        Qtot_NoDC += Qom


    frame.Put("MuExZenith"  ,dataclasses.I3Double(frame['MuEx'].dir.zenith))
    frame.Put("MuExEnergy"  ,dataclasses.I3Double(frame['MuEx'].energy))
    frame.Put("DnnEnergy"   ,dataclasses.I3Double(frame["energy_reco_new"]["EnergyVisible"]))
    frame.Put("DeepStart"   ,dataclasses.I3Double(frame["classification"]["Starting_Track"]))
    frame.Put("DeepSkim"    ,dataclasses.I3Double(frame["classification"]["Skimming"]))
    frame.Put("DeepThrough" ,dataclasses.I3Double(frame["classification"]["Through_Going_Track"]))
    frame.Put("DeepStop"    ,dataclasses.I3Double(frame["classification"]["Stopping_Track"]))
    frame.Put("AvgDistQ"    ,dataclasses.I3Double(frame["TrackFit_AvgDistQ"]))
    frame.Put("MuExZ"       ,dataclasses.I3Double(frame["MuEx"].pos.z))
    frame.Put("MuExR"       ,dataclasses.I3Double(frame["MuEx"].pos.r))
    frame.Put("DirNDoms"    ,dataclasses.I3Double(frame["TrackFit_dh"].n_dir_doms))
    frame.Put("DirS"        ,dataclasses.I3Double(frame["TrackFit_dh"].dir_track_hit_distribution_smoothness))
    frame.Put("DirL"        ,dataclasses.I3Double(frame["TrackFit_dh"].dir_track_length))
    frame.Put("Qtot_NoDC"   ,dataclasses.I3Double(Qtot_NoDC))
    frame.Put("NChan_NoDC"  ,dataclasses.I3Double(NChan_NoDC))
    return True

outputKeys = ['RLogL','Overburden','BayesLLHR','CorrectedParaboloidSigma','DirL','DirS','DirNDoms','MuExR','MuExZ','AvgDistQ',
              'DeepSkim','DeepStop','DeepThrough','DeepStart','DnnEnergy','MuExEnergy','MuExZenith','wconv','wastro',
              'wprompt','NuEnergy','NuZenith','PrimaryType']

tray = I3Tray()
tray.Add("I3Reader",FilenameList = [gcdfile,infile])
tray.Add(get_variables)
tray.AddModule(tableio.I3TableWriter, "hdfwriter")(
        ("tableservice",hdfwriter.I3HDFTableService(outdir+"/"+outname[:-7]+"_lite_platinum.h5")),
        ("SubEventStreams",["TTrigger"]),
        ("keys",outputKeys)
        )
tray.Execute()
tray.Finish()
