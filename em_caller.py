#!/usr/bin/python

__autor__= "posixroot"


import os
import sys
import json
from compute_parameters import compute_parameters
import numpy as np

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

  labels = set([key for item in training_data for key in item.keys()])

  attr_length = max([len(vector) for item in training_data for vector in item.values()])

  mu, variance = compute_parameters(labels, attr_length, training_data)

  #Inference
  test_data = [1,2,3]
  ret = compute_label(test_data, mu, variance, labels)

  out = [(key,ret[key]) for key in ret.keys()]
  out = sorted(out, key=getKey, reverse=True)

  print "Output Label:", out[0][0]


def getKey(item):
  return item[1]


def compute_label(data_vector, mu, variance, labels):
  out = {}
  for label in labels:
    probability = 0
    const = -0.5*(np.log(2*np.pi*np.sqrt(np.array(variance[label]))))
    for num in const:
      probability += num
    temp = data_vector[:]
    temp = -0.5*(((np.array(temp) - np.array(mu[label]))**2)/np.array(variance[label]))
    for num in temp:
      probability+=num
    probability = np.exp(probability)
    out[label] = probability
  return out


if __name__ == '__main__':
  em_loader(sys.argv)
