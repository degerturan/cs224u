#!/usr/bin/env python
# -*- coding: utf-8 -*-
import fnmatch
import os
import morph_parse_german
import re

OUTPUT_FN = "german_ted_big_test.txt"

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
                words = morph_parse_german.split_sentence(line)

                # Join them back together, making them all lowercase
                line = " ".join(words).lower()

                # Write to the giant target file
                outfile.write(line.encode("utf-8"))
        print("Processed: %d of %d files.   \r" % (i+1, len(matches)))
        # sys.stdout.write("  Processed: %d of %d files.   \r" % (i+1, len(matches)) )
        # sys.stdout.flush()

