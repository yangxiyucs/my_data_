#! /usr/bin/env python
import argparse

parser = argparse.ArgumentParser(description='your script description')

parser.add_argument('--verbose', '-v', action='store_true', help='select pkg')
parser.add_argument('--file_name', '-f', default=1, type=int, help='select pkg')
parser.add_argument('--parses', '-p', default=2, type=int, help='select pkg')

args = parser.parse_args()

print('###args ', args.verbose)
# print('###filename ', args.filename)
if args.verbose:
    print('Verbose mode on')
else:
    print('Verbose mode off')

if args.file_name:
    print(args.file_name)
else:
    print('Verbose mode off')

if args.parses:
    print(args.parses)
else:
    print('Verbose mode off')
