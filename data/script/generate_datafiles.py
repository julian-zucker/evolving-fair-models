#!/usr/bin/env python3
"""This script reads all the datafiles from `./in/`, parses them into a consistent data format,
and writes the results to `./out`.

Invoke with `./data/script/generate_datafiles.py` from the top level."""

# The idea is that we can write each data-file processing script separately, and import
# and call them here to make re-generation easy.
import create_german

if __name__ == '__main__':
    create_german.main()

