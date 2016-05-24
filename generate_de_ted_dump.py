#!/usr/bin/env python
# -*- coding: utf-8 -*-

import fnmatch
import os
import re
import sys
from subprocess import Popen, PIPE, STDOUT
import time
import pexpect
from nltk.stem.snowball import SnowballStemmer

NO_RESULT_TAG = "no result"
OUTPUT_FN = "german_ted.txt"
LEN_CUTOFF = 20 #15 #8 # only pass words past this number through the morphological parser
ENABLE_STEMMER = True
ENABLE_MORPH_SPLIT = True

# Set up morphology parser as a subprocess
child = pexpect.spawn('./fst-infl2 zmorge-20150315-smor_newlemma.ca')
child.expect('reading.*')
child.expect('finished.*')

# TODO: figure out a way to get this guy to work with umlaut unicode
stemmer2 = SnowballStemmer("german", ignore_stopwords=True)

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
        gotprompt = child.expect([".", pexpect.TIMEOUT], timeout=0) #0.0001

    # Check for "no result" and just return the word
    if len(parse) == 0 or parse[0:len(NO_RESULT_TAG)] == NO_RESULT_TAG:
        return [word]

    # Process parse
    parse = parse.replace("\n","").replace("<~>","").replace("<->s<","<").replace("<->","")
    parse = re.sub(r"<.{0,5}>", '', unicode(parse,"utf-8"), flags=re.UNICODE) # remove any dangling <.....> tags
    parse = re.sub(r"<\+.*", '', unicode(parse,"utf-8"), flags=re.UNICODE)
    # TODO: check this and cill
    print parse

    # TODO: deal with more dangling s-characters? Remove any gunk under a few characters we get back? Let stemmer and glove take care of that?
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

                if ENABLE_MORPH_SPLIT:
                    # Morphological splitting
                    for j, w in enumerate(words):
                        if (len(w) >= LEN_CUTOFF): # and w[0].isupper(): # sadly no upper case here :(  
                            # Get morphology parse
                            capitalized = w[:1].upper() + w[1:]
                            morph_splits = get_morphology_parse(capitalized)
                            if ENABLE_STEMMER:
                                morph_splits = [stemmer2.stem(s) for s in morph_splits]
                            split_word = " ".join(morph_splits)
                            words[j] = split_word
                            # print w+" : "+split_word

                # Stemming
                if ENABLE_STEMMER:
                    words = [stemmer2.stem(s) for s in words]

                # Join them back together, making them all lowercase
                line = " ".join(words).lower()

                # Write to the giant target file
                outfile.write(line.encode("utf-8"))
        # print("Processed: %d of %d files.   \r" % (i+1, len(matches)))
        sys.stdout.write("  Processed: %d of %d files.   \r" % (i+1, len(matches)) )
        sys.stdout.flush()

# TODO: pick last rather than first?
# Materialwissenschaften --> last better than first
# Eigentumsverhältnissen --> last better than first
# Infektionskrankheit --> last better than first
# Truppenstationierung--> ok
# Produkt ion landschaft --> mistake

# Schlaf über wachungs system
# Schlafüberwachungssystem  --> last better than first


# Wirtschaftswissenschaftler