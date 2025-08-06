import sys
import argparse
import MDAnalysis as mda

def initialize(args):
    parser = argparse.ArgumentParser(
        description="Given a GRO file, identify the binding pocket residues around a ligand."
    )
    parser.add_argument(
        "-i",
        "--input_gro",
        type=str,
        required=True,
        help="Input GRO file containing the protein-ligand binding complex.",
    )
    parser.add_argument(
        "-l",
        "--ligand_resname",
        type=str,
        default="LIG",
        help="Residue name of the ligand to identify the binding pocket. Default is 'LIG'.",
    )
    parser.add_argument(
        "-c",
        "--cutoff",
        type=float,
        default=6.0,
        help="Cutoff distance (in Å) for defining the binding pocket. Default is 6.0 Å.",
    )

    args_parse = parser.parse_args(args)

    return args_parse

if __name__ == "__main__":
    args = initialize(sys.argv[1:])

    u = mda.Universe(args.input_gro)
    ligand = u.select_atoms(f"resname {args.ligand_resname}")
    pocket = u.select_atoms(f"protein and around {args.cutoff} group ligand", ligand=ligand)
    
    print(f"Identified {len(pocket.residues)} residues in the binding pocket around the ligand '{args.ligand_resname}' within {args.cutoff} Å:")
    res_list = [f"{res.resname}{res.resid}" for res in pocket.residues]
    print(", ".join(res_list))

    pymol_selection = f"select pocket, resi {'+'.join([str(i.resid) for i in pocket.residues])}"    
    print(f"\nPyMOL command binding pocket residues:\n{pymol_selection}")

    residues = f" ".join([str(i.resid) for i in pocket.residues])
    ndx_selection = f'a N CA C O & r {residues}'
    print(f"\nNDX command to select the pocket backbone:\n{ndx_selection}")
