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
ENABLE_STEMMER = True
ENABLE_MORPH_SPLIT = True
ENABLE_STEMMER = True
LEN_CUTOFF = 10 #20 #15 #8 # only pass words past this number through the morphological parser

# Set up morphology parser as a subprocess
child = pexpect.spawn('./fst-infl2 zmorge-20150315-smor_newlemma.ca')
child.expect('reading.*')
child.expect('finished.*')

# TODO: figure out a way to get this guy to work with umlaut unicode
stemmer2 = SnowballStemmer("german", ignore_stopwords=True)

# TODO:
# - figure out a way to get the last parse rather than a first for a lot of these words
# - use another corpus with capitalized nouns

# TODO: pick last rather than first?
# Materialwissenschaften --> last better than first
# Eigentumsverhältnissen --> last better than first
# Infektionskrankheit --> last better than first
# Truppenstationierung--> ok
# Produkt ion landschaft --> mistake
# Schlaf über wachungs system
# Schlafüberwachungssystem  --> last better than first
# Wirtschaftswissenschaftler

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
    parse = re.sub(r"<[A-z]{0,6}>", '', unicode(parse,"utf-8"), flags=re.UNICODE) # remove any dangling <.....> tags
    parse = re.sub(r"<\+.*", '', parse)

    # TODO: deal with more dangling s-characters? Remove any gunk under a few characters we get back? Let stemmer and glove take care of that?
    morph_splits = parse.split("<#>")
    if ENABLE_STEMMER:
        morph_splits = [stemmer2.stem(s) for s in morph_splits]

    return morph_splits

def stem(s):
    return stemmer2.stem(s)

def split_sentence(line):
    words = line.split(" ")

    if ENABLE_MORPH_SPLIT:
        # Morphological splitting
        for j, w in enumerate(words):
            if (len(w) >= LEN_CUTOFF): # and w[0].isupper(): # sadly no upper case here :(  
                # Get morphology parse
                # HACK! For now, just capitalize th word always. Pretty good chance it's a noun if it's this long!
                capitalized = w[:1].upper() + w[1:]

                morph_splits = get_morphology_parse(capitalized)

                split_word = " ".join(morph_splits)
                words[j] = split_word

    # Stemming
    if ENABLE_STEMMER:
        words = [stem(s) for s in words]

    return words
