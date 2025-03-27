"""Synthetic data generation

This is a copy of the original authors' `notebooks/fig3_architectures.ipynb
notebook. We will use this script as an initial exploration and to generate 
synthetic data.

"""

import os
import numpy as np
import pandas as pd
from Bio import SeqIO
import matplotlib.pyplot as plt

import tregs
from tregs.mpl_pboc import plotting_style
plt.rcParams.update({"font.size": 12})
plotting_style()


DATDIR = "data"
OUTDIR = "out/lacZ_v1"

NUM_MUTS = 2*10**4
SUFFIX = "_20k"

os.makedirs(OUTDIR, exist_ok=True)

#~~~  Load input data  ~~~#

_genome = []
for record in SeqIO.parse(f"{DATDIR}/mg1655_genome.fasta", "fasta"):
    _genome.append(str(record.seq))
genome = _genome[0]


df = pd.read_csv(f"{DATDIR}/ecoli_gene_list.csv")
df.head()


lacZYA_TSS = int(
    df[(df.gene == "lacZ") & (df.promoter == "lacZp1")]["tss"].iloc[0]
)

promoter = tregs.seq_utils.get_regulatory_region(
    genome, lacZYA_TSS, reverse=True
)

np.savetxt(f"{OUTDIR}/gene_name.txt", ["lacZ"], fmt="%s")
np.savetxt(f"{OUTDIR}/wtseq.txt", [promoter], fmt="%s")


rnap_site = "CAGGCTTTACACTTTATGCTTCCGGCTCGTATGTTGTGTGG"
rep_site = "AATTGTGAGCGGATAACAATT"
crp_site = "ATTAATGTGAGTTAGCTCACTCATTA"


rnap_emat_raw = np.loadtxt(f"{DATDIR}/energy_matrices/RNAP_energy_matrix.txt")
rnap_emat = tregs.simulate.fix_wt(rnap_emat_raw, rnap_site)


O1_1027_raw = np.load(f"{DATDIR}/energy_matrices/lacThermoforTO1.npy")

O1_1027_fixed = np.zeros(np.shape(O1_1027_raw))
for i in range(3):
    O1_1027_fixed[i] = tregs.simulate.fix_wt(O1_1027_raw[i], rep_site)

rep_emat = np.mean(O1_1027_fixed, axis=0)


crp_emat_raw = pd.read_table(
    f"{DATDIR}/energy_matrices/crp_tau_final_all_26.txt", delim_whitespace=True
)
crp_emat = crp_emat_raw[["par_A", "par_C", "par_G", "par_T"]].to_numpy() * 1.62
crp_emat = tregs.simulate.fix_wt(crp_emat, crp_site).T


#~~~  Contitutive promoter  ~~~#
def sim_constitutive_promoter():
    n_NS = len(genome)
    n_p = 5000
    ep_wt = -5

    df = tregs.simulate.sim(
        promoter, tregs.simulate.constitutive_pbound, [rnap_site],
        *[n_NS, n_p, rnap_emat], scaling_factor=10**6,
        num_mutants=NUM_MUTS
    )
    df.to_csv(f"{OUTDIR}/constitutive{SUFFIX}.csv")

    region_params = [
        (-38, -30, "P", "RNAP"), 
        (-15, -5, "P", "RNAP"),
    ]
    tregs.footprint.plot_footprint(
        promoter, df, region_params,
        x_lims=(-115, 45),
        outfile=f"{OUTDIR}/constitutive.pdf"
    )
    tregs.footprint.plot_exshift(
        promoter, df,
        outfile=f"{OUTDIR}/constitutive_exshift_matrix.pdf"
    )


#~~~  Simple repression  ~~~#
def sim_simple_repression():
    n_NS = len(genome)
    n_p = 5000
    n_r = 10
    ep_wt = -5
    er_wt = -15

    df = tregs.simulate.sim(
        promoter, tregs.simulate.simrep_pbound, [rnap_site, rep_site], 
        *[n_NS, n_p, n_r, rnap_emat, rep_emat, ep_wt, er_wt],
        num_mutants=NUM_MUTS
    )
    df.to_csv(f"{OUTDIR}/simrep{SUFFIX}.csv")

    region_params = [
        (-38, -30, "P", "RNAP"), 
        (-15, -5, "P", "RNAP"), 
        (2, 17, "R", "Repressor"),
    ]
    tregs.footprint.plot_footprint(
        promoter, df, region_params,
        outfile=f"{OUTDIR}/simrep.pdf"
    )
    tregs.footprint.plot_exshift(
        promoter, df,
        outfile=f"{OUTDIR}/simrep_exshift_matrix.pdf"
    )


#~~~  Simple activation  ~~~#
def sim_simple_activation():
    n_NS = len(genome)
    n_p = 5000
    n_a = 50
    ep_wt = -3
    ea_wt = -13
    e_ap = -4

    df = tregs.simulate.sim(
        promoter, tregs.simulate.simact_pbound, [rnap_site, crp_site], 
        *[n_NS, n_p, n_a, rnap_emat, crp_emat, ep_wt, ea_wt, e_ap],
        num_mutants=NUM_MUTS
    )
    df.to_csv(f"{OUTDIR}/simact{SUFFIX}.csv")

    region_params = [
        (-38, -30, "P", "RNAP"), 
        (-15, -5, "P", "RNAP"), 
        (-70, -52, "A", "Activator"),
    ]
    tregs.footprint.plot_footprint(
        promoter, df, region_params,
        outfile=f"{OUTDIR}/simact.pdf"
    )
    tregs.footprint.plot_exshift(
        promoter, df,
        outfile=f"{OUTDIR}/simact_exshift_matrix.pdf",
        vmax_scaling=0.5
    )


#~~~  Repression-activation  ~~~#
def sim_repression_activation():
    act_site = promoter[(115 - 110 + 40):(115 - 110 + 55)]
    rep_site = promoter[(115+5):(115+20)]

    act_emat = tregs.simulate.generate_emap(act_site, max_mut_energy=1)
    rep_emat = tregs.simulate.generate_emap(rep_site, max_mut_energy=1)

    n_NS = len(genome)
    n_p, n_r, n_a = 5000, 50, 50
    ep_wt, er_wt, ea_wt = -2, -15, -12
    e_int = -8

    df = tregs.simulate.sim(
        promoter, tregs.simulate.repact_pbound, [rnap_site, rep_site, act_site],
        *[n_NS, n_p, n_r, n_a, rnap_emat, rep_emat, 
        act_emat, ep_wt, er_wt, ea_wt, e_int], 
        scaling_factor=10*6,
        num_mutants=NUM_MUTS
    )
    df.to_csv(f"{OUTDIR}/repact{SUFFIX}.csv")

    region_params = [
        (-38, -30, "P", "RNAP"), 
        (-15, -5, "P", "RNAP"),             
        (-70, -55, "A", "Activator"), 
        (5, 20, "R", "Repressor"),
    ]
    tregs.footprint.plot_footprint(
        promoter, df, region_params,
        outfile=f"{OUTDIR}/repact.pdf"
    )
    tregs.footprint.plot_exshift(
        promoter, df,
        outfile=f"{OUTDIR}/repact_exshift_matrix.pdf"
    )


#~~~  Double repression  ~~~#
def sim_double_repression():
    rep1_site = promoter[(115-50):(115-40)]
    rep2_site = promoter[(115+15):(115+25)]
    r1_emat = tregs.simulate.generate_emap(rep1_site, max_mut_energy=1)
    r2_emat = tregs.simulate.generate_emap(rep2_site, max_mut_energy=1)

    n_NS = len(genome)
    n_p, n_r1, n_r2 = 4600, 15, 15
    ep_wt, er1_wt, er2_wt = -5, -12, -12
    e_int = -5

    df = tregs.simulate.sim(
        promoter, tregs.simulate.doublerep_pbound, 
        [rnap_site, rep1_site, rep2_site],
        *[n_NS, n_p, n_r1, n_r2, rnap_emat, r1_emat, 
        r2_emat, ep_wt, er1_wt, er2_wt, e_int, "OR"],
        scaling_factor=10**6, num_mutants=NUM_MUTS
    )
    df.to_csv(f"{OUTDIR}/doubrep{SUFFIX}.csv")

    region_params = [
        (-38, -30, "P", "RNAP"), 
        (-15, -5, "P", "RNAP"), 
        (-50, -40, "R", "R1"), 
        (15, 25, "R", "R2"),
    ]
    tregs.footprint.plot_footprint(
        promoter, df, region_params,
        outfile=f"{OUTDIR}/double_rep.pdf"
    )
    tregs.footprint.plot_exshift(
        promoter, df,
        outfile=f"{OUTDIR}/doublerep_exshift_matrix.pdf"
    )


#~~~  Double activation  ~~~#
def sim_double_activation():
    act1_site = promoter[(115 - 110 + 15):(115 - 110 + 25)]
    act2_site = promoter[(115 - 110 + 35):(115 - 110 + 45)]
    a1_emat = tregs.simulate.generate_emap(act1_site, fixed=True, fixed_value=1)
    a2_emat = tregs.simulate.generate_emap(act2_site, fixed=True, fixed_value=1)

    n_NS = len(genome)
    n_p, n_a1, n_a2 = 4600, 50, 50
    ep_wt, ea1_wt, ea2_wt = -2, -7, -7
    e_int_pa1, e_int_pa2, e_int_a1a2 = -7, -7, -7

    df = tregs.simulate.sim(
        promoter, tregs.simulate.doubleact_pbound, 
        [rnap_site, act1_site, act2_site],
        *[n_NS, n_p, n_a1, n_a2, rnap_emat, a1_emat, a2_emat, 
        ep_wt, ea1_wt, ea2_wt, e_int_pa1, e_int_pa2, e_int_a1a2, "OR"], 
        scaling_factor=100,
        num_mutants=NUM_MUTS
    )
    df.to_csv(f"{OUTDIR}/doubact{SUFFIX}.csv")

    region_params = [
        (-38, -30, "P", "RNAP"), 
        (-15, -5, "P", "RNAP"), 
        (-95, -85, "A", "A1"), 
        (-75, -65, "A", "A2"),
    ]
    tregs.footprint.plot_footprint(
        promoter, df, region_params,
        outfile=f"{OUTDIR}/double_act.pdf"
    )
    tregs.footprint.plot_exshift(
        promoter, df,
        outfile=f"{OUTDIR}/doubleact_exshift_matrix.pdf"
    )


def main():
    sim_constitutive_promoter()
    sim_simple_repression()
    sim_simple_activation()
    sim_repression_activation()
    sim_double_repression()
    sim_double_activation()

if __name__ == "__main__":
    main()
