#!/usr/bin/env python
# -*- coding: utf-8 -*-

import fnmatch
import os
import re
import sys
from subprocess import Popen, PIPE, STDOUT
import time
import pexpect

NO_RESULT_TAG = "no result"
OUTPUT_FN = "german_ted.txt"
LEN_CUTOFF = 8 # only pass words past this number through the morphological parser

# Set up morphology parser as a subprocess
child = pexpect.spawn('./fst-infl2 zmorge-20150315-smor_newlemma.ca')
child.expect('reading.*')
child.expect('finished.*')

def get_morphology_parse(word):
    # Send the word query to the prompt
    child.sendline(word)

    # Skip two lines of junk, before getting the parse
    child.readline(); child.readline()
    parse = child.readline()

    # May want this guy here
    #     child.expect(".*")

    # Wait for prompt to reappear
    gotprompt = 0
    while not gotprompt:
        # child.expect("", timeout=None)
        gotprompt = child.expect([".", pexpect.TIMEOUT], timeout=0.0001)

    # Check for "no result" and just return the word
    if len(parse) == 0 or parse[0:len(NO_RESULT_TAG)] == NO_RESULT_TAG:
        return [word]

    # Process parse
    parse = parse.replace("\n","").replace("<~>","").replace("<->s<","<").replace("<->","").replace("<CAP>","")
    parse = re.sub(r"<\+.*", '', unicode(parse,"utf-8"), flags=re.UNICODE)

    # TODO: deal with dangling s-characters
    return parse.split("<#>")

# Collect files in the TED corpus
matches = []
for root, dirnames, filenames in os.walk('data/ted-cldc/de-en'):
    for filename in fnmatch.filter(filenames, '*.ted'):
        matches.append(os.path.join(root, filename))

# Process files
with open(OUTPUT_FN,"w+") as outfile:
    for i, fn in enumerate(matches):
        with open(fn, 'r') as f:
            for line in f:
                for delim in ["._de", ":_de", ",_de", ";_de"]:
                    line = line.replace(delim, "s ")
                line = re.sub(r'[^\w ]', '', unicode(line,"utf-8"), flags=re.UNICODE).replace("_UNK_","").replace("_de","").replace("  ", " ")
                
                # Split the words
                words = line.split(" ")

                # Edit them
                for j, w in enumerate(words):
                    if (len(w) >= LEN_CUTOFF): # and w[0].isupper(): # sadly no upper case here :(  
                        # Get morphology parse
                        split_word = " ".join(get_morphology_parse(w))
                        words[j] = split_word
                        # print w+" : "+split_word

                # Join them back together, making them all lowercase
                line = " ".join(words).lower()

                # Write to the giant target file
                outfile.write(line.encode("utf-8"))
        # print("Processed: %d of %d files.   \r" % (i+1, len(matches)))
        sys.stdout.write("  Processed: %d of %d files.   \r" % (i+1, len(matches)) )
        sys.stdout.flush()

