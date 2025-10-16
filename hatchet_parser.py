import argparse

def parse_arguments_build_haplotype_blocks(args=None):
    parser = argparse.ArgumentParser(
        prog="HATCHet build_haplotype_blocks",
        description="build haplotype blocks",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ##################################################
    # general inputs for bulk data
    parser.add_argument(
        "--work_dir",
        required=True,
        type=str,
        help="working directory, <work_dir>/allele, <work_dir>/phase",
    )
    parser.add_argument(
        "--read_type",
        required=True,
        type=str,
        default="NGS",
        choices=["NGS", "TGS"],
        help="NGS, TGS",
    )

    parser.add_argument(
        "--out_dir",
        required=True,
        type=str,
        help="<work_dir>/<out_dir>",
    )

    ##################################################
    # Aux files
    parser.add_argument(
        "--reference",
        required=False,
        type=str,
        help="reference genome FASTA file",
    )
    parser.add_argument(
        "--genetic_map",
        required=False,
        type=str,
        help="genetic map file",
    )

    ##################################################
    # Parameters
    parser.add_argument(
        "--mspb",
        required=False,
        default=10,
        type=int,
        help="max-snps-per-block",
    )
    parser.add_argument(
        "--mserr",
        required=False,
        default=0.1,
        type=float,
        help="max-switch-error-rate",
    )
    parser.add_argument(
        "--no_gc_correct",
        action="store_true",
        default=False,
        help="don't perform GC correction",
    )
    args = parser.parse_args()
    return vars(args)
