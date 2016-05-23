#!/usr/bin/env python
# -*- coding: utf-8 -*-

import fnmatch
import os
import re
import sys
from subprocess import Popen, PIPE, STDOUT
import time
import pexpect

OUTPUT_FN = "german_ted.txt"

matches = []
for root, dirnames, filenames in os.walk('data/ted-cldc/de-en'):
    for filename in fnmatch.filter(filenames, '*.ted'):
        matches.append(os.path.join(root, filename))

# with open(OUTPUT_FN,"w+") as outfile:
#     for i, fn in enumerate(matches):
#         with open(fn, 'r') as f:
#             for line in f:
#                 for delim in ["._de", ":_de", ",_de", ";_de"]:
#                     line = line.replace(delim, "s ")
#                 line = re.sub(r'[^\w ]', '', unicode(line,"utf-8"), flags=re.UNICODE).replace("_UNK_","").replace("_de","").replace("  ", " ")
#                 outfile.write(line.encode("utf-8"))
#         sys.stdout.write("  Processed: %d of %d files.   \r" % (i+1, len(matches)) )
#         sys.stdout.flush()


def experiment():
    # analyzer = Popen('./fst-infl2 zmorge-20150315-smor_newlemma.ca', stdout=PIPE, stdin=PIPE, stderr=PIPE, shell=True)
    analyzer = Popen(['./fst-infl2', 'zmorge-20150315-smor_newlemma.ca'], stdout=PIPE, stdin=PIPE, stderr=PIPE)
    # print analyzer.communicate(input='Rechtswiederspruch\nAutobahn')[0]
    # print analyzer.communicate(input='Autobahn')[0]
    # time.sleep(10) # delays for 10 seconds
    print analyzer.stderr.readline()
    print analyzer.stderr.readline()
    analyzer.stdin.write('Rechtswiederspruch\r\n')
    analyzer.stdin.flush()
    # print analyzer.stderr.read(1)
    print analyzer.stdout.read(1)
    print "here"
    # communicate returns a tuple (stdoutdata, stderrdata)
    # http://stackoverflow.com/questions/6346650/keeping-a-pipe-to-a-process-open

import pexpect

# Set up morphology parser as a subprocess
child = pexpect.spawn('./fst-infl2 zmorge-20150315-smor_newlemma.ca')
child.expect('reading.*')
child.expect('finished.*')

NO_RESULT_TAG = "no result"
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

    # Check for "no result"
    if len(parse) == 0 or parse[0:len(NO_RESULT_TAG)] == NO_RESULT_TAG:
        return None

    # Process parse
    parse = parse.replace("\n","").replace("<~>","")

    return parse


print get_morphology_parse('Rindfleischetikettierungsüberwachungsaufgabenübertragungsgesetz')
print get_morphology_parse('Wiederstand')
print get_morphology_parse('Ausrufezeichen')
print get_morphology_parse('kfdhskfd')

# child.sendline('Rechtswiederspruch')
# child.expect("")
# child.readline(); child.readline()
# parse = child.readline()
# print(child.before)
# print child.read()


# to wait for child to finish: p.expect (pexpect.EOF)

#child.interact()