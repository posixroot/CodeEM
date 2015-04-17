#!/usr/bin/python

__author__ = "posixroot"

import sys
import os
import random
import math
import numpy
from sklearn import mixture


def guassmix_sklearn(argv):
  """This function computes the EM algorithm."""

  print "AI HW1\n"

  if len(argv) != 4:
    print "Usage: python guassmix.py <#clusters> <data-file> <model-file>"
    sys.exit()

  os.path.dirname(os.path.abspath(__file__))
  clusters = int(argv[1])
  data = argv[2]  #'wine.train'
  model = argv[3]  #'wineout'

  random.seed(clusters*random.random())

  f = open(data, 'r')

  testdata = []

  testrows, testfeatures = f.readline().split()
  testrows = int(testrows)
  testfeatures = int(testfeatures)

  for line in f:
    testdata.append([float(x) for x in line.split()])

  g = mixture.GMM(n_components=clusters)
  g.fit(testdata)
  print "means: "+str(g.means_)
  print "covars: "+str(g.covars_)

  test_file_name = "wine.train"
  outf = open(model, 'w')

  with open(test_file_name, 'r') as test_file:
    test_file.readline()
    for line in test_file:
      label = g.predict([float(x) for x in line.split()])
      outf.write(str(label)+' '+line)



if __name__== '__main__':
    guassmix_sklearn(sys.argv)
