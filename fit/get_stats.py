#!/usr/bin/env python3
import os
import argparse
from semiphore_public.utils import stats

parser = argparse.ArgumentParser(description="""
Run photo-z estimation.
""", formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('-i', '--input', type=str, default=None,
                    help='Input filename / folder')
parser.add_argument('-o', '--output', type=str, default=None,
                    help='Output filename')
parser.add_argument('-V', '--verbose', action='store_true',
                    default=False,
                    help='Be verbose')

args = parser.parse_args()

if os.path.isdir(args.input):
    result = stats.get_stats_for_folder(args.input)
else:
    result = stats.get_stats_for_file(args.input)

if args.output is not None:
    result.write(args.output)
elif args.verbose:
    print(result.to_pandas().to_latex(float_format='%.2f', index=False))
else:
    result.remove_columns(['Count'])
    print(result.to_pandas().to_latex(float_format='%.2f', index=False))



