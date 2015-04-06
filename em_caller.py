#!/usr/bin/python

__autor__= "posixroot"


import os
import sys
import json
from pprint import pprint
import random

def em_loader(argv):
  """This function is responsible for preprocessing the input json."""

  filename = "training_input.json"
  # if len(argv) != 2:
  #   print "Usage: python guassmix.py <input-json-file>"
  #   sys.exit()

  os.path.dirname(os.path.abspath(__file__))
  # filename = int(argv[1])

  training_data = []
  with open(filename) as training_json:
    training_data = json.load(training_json)
    training_data = training_data["training-data"]

  #extract the distinct labels from the training data
  labels_list = [x["label"] for x in training_data]
  labels = set(labels_list)

  labels_count = len(labels)

  em(labels_count, training_data)


if __name__ == '__main__':
  em_loader(sys.argv)
