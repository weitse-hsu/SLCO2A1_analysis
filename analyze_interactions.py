import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
import MDAnalysis as mda
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import prolif as plf
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from prolif import fingerprint
from MDAnalysis.analysis import contacts
from matplotlib import rc


class Logger:
    """
    A logger class that redirects the STDOUT and STDERR to a specified output file while
    preserving the output on screen. This is useful for logging terminal output to a file
    for later analysis while still seeing the output in real-time during execution.

    Parameters
    ----------
    logfile : str
        The file path of which the standard output and standard error should be logged.

    Attributes
    ----------
    terminal : :code:`io.TextIOWrapper` object
        The original standard output object, typically :code:`sys.stdout`.
    log : :code:`io.TextIOWrapper` object
        File object used to log the output in append mode.
    """

    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        """
        Writes a message to the terminal and to the log file.

        Parameters
        ----------
        message : str
            The message to be written to STDOUT and the log file.
        """
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure the message is written immediately

    def flush(self):
        """
        This method is needed for Python 3 compatibility. This handles the flush command by doing nothing.
        Some extra behaviors may be specified here.
        """
        # self.terminal.log()
        pass

def format_time(t):
    """
    Converts time in seconds to a more readable format.

    Parameters
    ----------
    t : float
        The time in seconds.

    Returns
    -------
    t_str : str
        A string representing the time duration in a format of "X hour(s) Y minute(s) Z second(s)", adjusting the units
        as necessary based on the input duration, e.g., 1 hour(s) 0 minute(s) 0 second(s) for 3600 seconds and
        15 minute(s) 30 second(s) for 930 seconds.
    """
    hh_mm_ss = str(datetime.timedelta(seconds=t)).split(":")

    if "day" in hh_mm_ss[0]:
        # hh_mm_ss[0] will contain "day" and cannot be converted to float
        hh, mm, ss = hh_mm_ss[0], float(hh_mm_ss[1]), float(hh_mm_ss[2])
        t_str = f"{hh} hour(s) {mm:.0f} minute(s) {ss:.0f} second(s)"
    else:
        hh, mm, ss = float(hh_mm_ss[0]), float(hh_mm_ss[1]), float(hh_mm_ss[2])
        if hh == 0:
            if mm == 0:
                t_str = f"{ss:.1f} second(s)"
            else:
                t_str = f"{mm:.0f} minute(s) {ss:.0f} second(s)"
        else:
            t_str = f"{hh:.0f} hour(s) {mm:.0f} minute(s) {ss:.0f} second(s)"

    return t_str

def summarize_interactions(df, freq_threshold=0.5):
    """
    Given a DataFrame converted from interaction fingerprints, this function calculate the frequency of interactions
    for those that occur in more than a specified percentage of the frames, and returns a summary DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        A MultiIndex DataFrame converted from interactoin fingerprints.
    freq_threshold : float, optional
        The minimum frequency threshold for interactions to be included in the summary. Default is 0.5.
    
    Returns
    -------
    final_df : pd.DataFrame
        A DataFrame summarizing the interaction frequencies for each ligand-residue pair.
    """
    # 1. Identify interactions that occur in more than freq_threshold fraction of frames
    assert 0 <= freq_threshold <= 1, "Frequency threshold must be between 0 and 1."
    freq_interactions = (
        df.T.groupby(level=["ligand", "protein"])
        .sum()
        .T.astype(bool)
        .mean()
        .pipe(lambda s: s[s > freq_threshold].sort_values(ascending=False) * 100)
        .to_frame(name="%")
        .T
        .squeeze()
    )

    # 2. Create a dictionary of interaction frequencies for each ligand-residue pair and convert it to a DataFrame
    interaction_dict = {}
    for ligand, residue in freq_interactions.index:
        subset = df.xs(residue, level="protein", axis=1)
        interaction_freq = subset.mean() * 100
        interaction_dict[(ligand, residue)] = interaction_freq

    summary_df = pd.DataFrame(interaction_dict).T
    summary_df.index.names = ["Ligand", "Residue"]
    summary_df = summary_df.fillna(0)

    # 3. Add "Any interaction" column
    ligands = freq_interactions.index.get_level_values("ligand").unique()  # should be only one ligand
    any_interaction = freq_interactions.to_frame(name=(ligands[0], "Any interaction"))
    any_interaction.columns = pd.MultiIndex.from_tuples(any_interaction.columns, names=["ligand", "interaction"])
    full_df = pd.concat([summary_df, any_interaction], axis=1)

    interaction_types = ["Any interaction"] + fingerprint.Fingerprint.list_available()
    interaction_cols = pd.MultiIndex.from_product(
        [ligands, interaction_types], names=["ligand", "interaction"]
    )
    interaction_cols = interaction_cols.intersection(full_df.columns)  # Get interactions occurring in the full_df
    full_df = full_df[interaction_cols]

    # 4. Format the DataFrame for easier readability
    final_df = full_df.copy()
    final_df.columns = final_df.columns.droplevel("ligand")
    final_df.index = final_df.index.get_level_values(1)
    final_df.index = final_df.index.str.title()
    final_df.index = final_df.index.str.replace("^Hs[dep]", "His", regex=True)  # Replace Hsd, Hse, Hsp with His

    rename_dict = {
        "VdWContact": "VdW contact",
        "HBAcceptor": "HB acceptor",
        "HBDonor": "HB donor",
        "CationPi": "Cation-Pi",
        "PiCation": "Pi-cation",
        "PiStacking": "Pi-stacking",
        "EdgeToFace": "Edge-to-face",
        "FaceToFace": "Face-to-face",
        "MetalAcceptor": "Metal receptor",
        "MetalDonor": "Metal donor",
        "XBAcceptor": "XB acceptor",
        "XBDonor": "XB donor",
    }
    final_df = final_df.rename(columns=rename_dict)

    # Reorder the columns
    interaction_order = [
        "Any interaction", "Hydrophobic", "VdW contact", "HB acceptor", "HB donor", 
        "Cation-Pi", "Pi-cation", "Pi-stacking", "Edge-to-face", "Face-to-face",
        "Metal receptor", "Metal donor", "XB acceptor", "XB donor"
    ]
    available_columns = [col for col in interaction_order if col in final_df.columns]
    final_df = final_df[available_columns]

    return final_df


def plot_interaction_frequencies(df, output_file):
    """
    Plots the interaction frequencies for a given DataFrame and saves the figure to the specified output file.
    """
    # Assign colors to different interaction types
    interaction_types = [
        "Any interaction", "Hydrophobic", "VdW contact", "HB acceptor", "HB donor", 
        "Cation-Pi", "Pi-cation", "Pi-stacking", "Edge-to-face", "Face-to-face",
        "Metal receptor", "Metal donor", "XB acceptor", "XB donor"
    ]
    colors = {interaction: plt.cm.tab20(i) for i, interaction in enumerate(interaction_types)}
    colors["Any interaction"] = "#0B84A5"
    
    # Overwrite a few with my favorites :)
    colors.update({
            "Hydrophobic": "lightblue",
            "VdW contact": "#F6C85F",
            "HB acceptor": "#CA472F",
            "HB donor": "lightpink",
        }
    )

    # Plot the grouped bar chart with custom colors
    ax = df.plot(kind="bar", figsize=(14, 6), width=0.8, color=[colors[col] for col in df.columns])

    # Labels
    for label in ax.get_yticklabels():
        label.set_fontproperties(fontprop)
    for label in ax.get_xticklabels():
        label.set_fontproperties(fontprop)

    ax.set_ylabel("Frequency (%)", fontproperties=fontprop)
    ax.tick_params(axis="y")
    plt.xticks(rotation=45, ha="center", fontproperties=fontprop)
    plt.legend(title="Interaction Type", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=18, title_fontsize=18, prop=fontprop)
    plt.tight_layout()
    ax.grid(axis='y')
    plt.savefig(output_file, dpi=600, bbox_inches="tight")

if __name__ == "__main__":
    t0 = time.time()

    sys.stdout = Logger("results/interaction_analysis.log")
    sys.stderr = Logger("results/interaction_analysis.log")

    rc('font', **{
        'family': 'sans-serif',
        'sans-serif': ['DejaVu Sans'],
        'size': 10
    })
    rc('mathtext', **{'default': 'regular'})
    plt.rcParams['font.family'] = 'DejaVu Sans'
    font_path = '/home/bioc1870/Software/mambaforge/envs/md_env/fonts/Arial.ttf'
    fontprop = fm.FontProperties(fname=font_path, size=12)

    systems = [
        "RatSLCO2A1_P2E",
        "RatSLCO2A1_ZLK",
        "RatSLCO2A1_LSN",
        "RatSLCO2A1_FEN",
        "RatSLCO2A1_TCW",
        "RatSLCO2A1_PGF",
        "RatSLCO2A1_FEN_flipped",
        "RatSLCO2A1_P2E_flipped"
    ]

    ligand_names = [
        'P2E',
        'ZLK',
        'LSN',
        'FEN',
        'TCW',
        'PGF',
        'FEN_flipped',
        'P2E_flipped'
    ]

    ligand_resnames = [
        'P2E',
        'ZLK',
        'LSN',
        'FEN',
        'TCW',
        'UGU',
        'FEN',
        'P2E'
    ]

    # We need to GRO files just to get the correct residue numbering
    simulation_dir = "/home/bioc1870/SLCO2A1_simulations/"
    gro_files = [f"{simulation_dir}{system}/production/rep_1/sys.gro" for system in systems]
    tpr_files = [f"{simulation_dir}{system}/production/rep_1/md_system.tpr" for system in systems]
    xtc_files = [f"{simulation_dir}{system}/analysis/md_all_center.xtc" for system in systems]

    salt_bridge_percentage_avg, salt_bridge_percentage_std = [], []

    for gro_file, tpr_file, xtc_file, system, ligand_name, ligand_resname in zip(gro_files, tpr_files, xtc_files, systems, ligand_names, ligand_resnames):
        print(f"\nProcessing {system} ...")
        assert os.path.exists(gro_file), f"File {gro_file} does not exist."
        assert os.path.exists(tpr_file), f"File {tpr_file} does not exist."
        assert os.path.exists(xtc_file), f"File {xtc_file} does not exist."

        os.makedirs(f"results/{ligand_name}", exist_ok=True)

        # Step 1. Load the MD trajectory into an MDAnalysis universe
        u = mda.Universe(tpr_file, xtc_file)
        u_ref = mda.Universe(gro_file)
        u.residues.resids = u_ref.residues.resids[:len(u.residues.resids)]  # Fix the residue numbering in u
        
        # Step 2. Generate the IFP
        if os.path.exists(f"results/{ligand_name}/{ligand_name}_ifp_all.pkl"):
            print(f"IFP for {system} already exists, loading {ligand_name}_ifp_all.pkl ...")
            fp = fingerprint.Fingerprint.from_pickle(f"results/{ligand_name}/{ligand_name}_ifp_all.pkl")
            df = fp.to_dataframe()
        else:
            print(f"Starting IFP calculation for {system}...")
            if ligand_resname not in ['FEN', 'LSN']:
                ligand_sel = u.select_atoms(f"resname {ligand_resname}")
            else:
                ligand_sel = u.select_atoms(f"resname {ligand_resname} and not name LP*")
            protein_sel = u.select_atoms("protein and byres around 20.0 group ligand", ligand=ligand_sel)
            ligand_sel.chainIDs = np.array(['' for i in range(len(ligand_sel.chainIDs))])
            protein_sel.chainIDs = np.array(['' for i in range(len(protein_sel.chainIDs))])

            fp = fingerprint.Fingerprint()
            fp.run(u.trajectory[::5], ligand_sel, protein_sel)  # 1 ns interval

            print(f'Saving {ligand_name}_ifp_all.pkl and {ligand_name}_ifp_all.tsv...')
            fp.to_pickle(f"results/{ligand_name}/{ligand_name}_ifp_all.pkl")

            df = fp.to_dataframe()
            df.to_csv(f"results/{ligand_name}/{ligand_name}_ifp_all.tsv", sep="\t", index=False)

        # Step 3. Simplify ligand and protein names by removing possible chain IDs, e.g. "LIG.A" -> "LIG"
        new_ligand_levels = df.columns.levels[df.columns.names.index('ligand')].map(lambda x: x.split('.')[0])
        new_protein_levels = df.columns.levels[df.columns.names.index('protein')].map(lambda x: x.split('.')[0])
        df.columns = df.columns.set_levels(new_ligand_levels, level='ligand')
        df.columns = df.columns.set_levels(new_protein_levels, level='protein')

        # Step 4. Some specific interactions
        hydrophobic_band = ['ASN371', 'MET379', 'HSD533', 'ARG561', 'PHE557', 'TRP565']
        ligand_level = df.columns.names.index("ligand")
        protein_level = df.columns.names.index("protein")
        interaction_level = df.columns.names.index("interaction")
        new_ligand_levels = df.columns.levels[ligand_level].map(lambda x: x.split('.')[0])
        new_protein_levels = df.columns.levels[protein_level].map(lambda x: x.split('.')[0])
        df.columns = df.columns.set_levels(new_ligand_levels, level='ligand')
        df.columns = df.columns.set_levels(new_protein_levels, level='protein')
        
        any_interaction_cols = [col for col in df.columns if col[protein_level].upper() in hydrophobic_band]
        p_any = df[any_interaction_cols].any(axis=1).mean() * 100
        print(f"The ligand interacted with the hydrophobic band {p_any:.2f}% of the frames.")

        hydrophobic_cols = [col for col in df.columns if col[protein_level].upper() in hydrophobic_band and col[interaction_level] == "Hydrophobic"]
        p_hydrophobic = df[hydrophobic_cols].any(axis=1).mean() * 100
        print(f"The ligand had hydrophobic interactions with the hydrophobic band {p_hydrophobic:.2f}% of the frames.")

        final_df = summarize_interactions(df, freq_threshold=0)
        interaction_565_dict = final_df[final_df.index == "Trp565"].to_dict()
        print('Interaction with Trp565:')
        for key in interaction_565_dict:
            print(f"  - {key}: {interaction_565_dict[key]['Trp565']:.2f}%")

        # hydrophobic_band = ['Asn371', 'Met379', 'His533', 'Arg561', 'Phe557', 'Trp565']
        # hydrophobic_band_df = final_df[final_df.index.isin(hydrophobic_band)]
        # print()
        # print(hydrophobic_band_df)

        # Step 3. Plot the frequencies of the interactions that occur in more than 50% of the frames
        print("Plotting interaction frequencies...")
        final_df = summarize_interactions(df, freq_threshold=0.5)
        output_file = f"results/{ligand_name}/{ligand_name}_interaction_frequencies.png"
        plot_interaction_frequencies(final_df, output_file)

        # Step 4. Plot the interaction timeseries
        print("Plotting interaction time series...")
        fp.plot_barcode()
        output_file = f"results/{ligand_name}/{ligand_name}_barcode.png"
        plt.savefig(output_file, dpi=600, bbox_inches="tight")

        # Step 5. Arg561-Glu78 Salt-bridge analysis
        basic_residue = u.select_atoms("resname ARG and resid 561 and (name NH* NZ)")
        acidic_residue = u.select_atoms("resname GLU and resid 78 and (name OE* OD*)")
        percentages = []  # This should contain 3 values for each of the 3 replicates
        for i in range(3):
            contact_list = []
            for ts in u.trajectory[i*5000:(i+1)*5000]:
                dist = contacts.distance_array(basic_residue.positions, acidic_residue.positions)
                contact_list.append(contacts.contact_matrix(dist, radius=4.5).sum())
            contact_list = np.array(contact_list)
            percentages.append(np.sum(contact_list > 0) / len(contact_list) * 100)
        salt_bridge_percentage_avg.append(np.mean(percentages))
        salt_bridge_percentage_std.append(np.std(percentages))
        print(f"Average percentage of frames with Arg561-Glu78 salt-bridge across replicates: {np.mean(percentages):.2f}% ± {np.std(percentages):.2f}%")

    # Analysis across systems
    plt.figure()
    salt_bridge_percentage_avg = np.array(salt_bridge_percentage_avg)
    salt_bridge_percentage_std = np.array(salt_bridge_percentage_std)
    plt.bar(systems, salt_bridge_percentage_avg, yerr=salt_bridge_percentage_std, capsize=5)
    plt.ylim(0, 100)
    plt.ylabel("Occurrence of Arg561-Glu78 salt-bridge (%)")
    plt.xticks(rotation=45, ha="right")
    plt.grid()
    plt.tight_layout()
    plt.savefig("results/salt_bridge_percentage.png", dpi=600, bbox_inches="tight")

    print(f"\nTime elapsed: {format_time(time.time() - t0)}.")
