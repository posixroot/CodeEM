#!/usr/bin/python

__author__ = "posixroot"

import sys
import os
import random
import math
import numpy


def initialize_prior(prior, clusters):
  for i in range(clusters):
    prior.append(1/float(clusters))


def initialize_mu_method_1(clusters, testdata, mu):
  for i in range(clusters):
    mu.append(random.choice(testdata))


def initialize_mu_method_2(clusters, minarr, rangearr, mu):
  for i in range(clusters):
    mu.append([a+(b*i/clusters) for a,b in zip(minarr,rangearr)])


def initialize_sd(clusters, rangearr, sd):
  for i in range(clusters):
    frac = random.random()
    sd.append([a*frac for a in rangearr])


def log_prior_probability(clusters, prior, logprior):
  for i in range(clusters):
    logprior.append(math.log(prior[i]))


def calculate_estep_loglikelihood(clusters, testrows, testfeatures, testdata, mu, sd):
  ell = []
  for i in range(clusters):
    temp = []
    for j in range(testrows):
      temp.append([(-0.5)*((a-b)**2)/(c**2) for a,b,c in zip(testdata[j],mu[i],sd[i])])
    ell.append(temp)

  temp2 = []
  for i in range(clusters):
    temp = []
    for j in range(testrows):
      temp.append([a+(-0.5*math.log(2*math.pi*b)) for a,b in zip(ell[i][j],sd[i])])
    temp2.append(temp)
  ell = []
  ell = temp2[:]

  temp2 = []
  for i in range(clusters):
    temp = []
    for j in range(testrows):
      sumx = 0.0
      for k in range(testfeatures):
        sumx += ell[i][j][k]
      temp.append(sumx)
    temp2.append(temp)
  ell = []
  ell = temp2[:]
  return ell


def calculate_estep_loglikelihood_with_prior(clusters, ell, logprior, ellprior):
  for i in range(clusters):
    ellprior.append([logprior[i]+x for x in ell[i]])


def calculate_max_lsx_by_rows(testrows, clusters, ellprior, lsxmax):
  for i in range(testrows):
    temp = []
    for j in range(clusters):
      temp.append(ellprior[j][i])
    lsxmax.append(max(temp))


def guassmix(argv):
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

  #Initialize prior values for all the clusters
  prior = []
  initialize_prior(prior, clusters)

  mu = []
  maxarr = []
  minarr = []
  rangearr = []

  #Find the min, max and range of each of the features
  for i in range(testfeatures):
    temp = []
    for j in range(testrows):
      temp.append(testdata[j][i])
    maxarr.append(max(temp))
    minarr.append(min(temp))
  rangearr = [abs(a-b) for a,b in zip(maxarr, minarr)]

  #Initialize mu for all clusters. Method 1 (random datapoints as means)
  initialize_mu_method_1(clusters, testdata, mu)

  #Method 2 to Initialize the mean(uniform dist. over range). To enable, uncomment the below line.
  # initialize_mu_method_2(clusters, minarr, rangearr, mu)

  #Initialize Standard Deviation values for all the clusters
  sd = []
  initialize_sd(clusters, rangearr, sd)

  loopvar = 1
  #Iteration 1
  while(loopvar>0):
    logprior = []
    log_prior_probability(clusters, prior, logprior)

    #Calculate loglikelihood estimate (E-Step)
    ell = calculate_estep_loglikelihood(clusters, testrows, testfeatures, testdata, mu, sd)

    #loop break condition
    if(loopvar>1):
      ellmat = numpy.array(ell)
      oldellmat = numpy.array(oldell)

      loopbreaker = 1
      for i in range(clusters):
        for j in range(testrows):
          ellmat[i,j] = math.fabs((ellmat[i,j] - oldellmat[i,j])/oldellmat[i,j])
          if(ellmat[i,j]>=0.001):
            loopbreaker = 0
        if(loopbreaker == 0):
          break

      if(loopbreaker==1):
        break

    ellprior = []
    calculate_estep_loglikelihood_with_prior(clusters, ell, logprior, ellprior)

    lsxmax = []
    calculate_max_lsx_by_rows(testrows, clusters, ellprior, lsxmax)





    temp = []
    for i in range(clusters):
      temp.append([a-b for a,b in zip(ellprior[i],lsxmax)])
    ellprior = []
    ellprior = temp[:]

    epost = []
    #for i in range(clusters):
      #epost.append(ellprior[i])
    epost = ellprior[:]


    temp = []
    for i in range(clusters):
      temp.append([math.exp(a) for a in ellprior[i]])
    ellprior = []
    ellprior = temp[:]

    sumarr = []
    for i in range(testrows):
      sumrow = 0.0
      for j in range(clusters):
        sumrow += ellprior[j][i]
      sumarr.append((-1)*math.log(sumrow))

    #print '\nThe sumarr test: '
    #print sumarr[12]

    temp = []
    for i in range(clusters):
      temp.append([a+b for a,b in zip(epost[i],sumarr)])
    epost = []
    epost = temp[:]

    ############print '\n The posterior check: '
    ############for i in range(clusters):
      ############print epost[i][12]

    temp = []
    for i in range(clusters):
      temp.append([math.exp(a) for a in epost[i]])
    epost = []
    epost = temp[:]

    ############print '\n The posterior check (normal space): '
    ############for i in range(clusters):
      ############print epost[i][12]

    ##End of E step

    #M Step
    priorval = []
    for i in range(clusters):
      sumcol = 0.0
      for j in range(testrows):
        sumcol += epost[i][j]
      #smoothening incorporated so that prior is never 0
      priorval.append((sumcol+0.00001)/testrows)

    ############print '\n PriorVal check: '
    ############for i in range(clusters):
      ############print priorval[i]

    muval = []
    for i in range(clusters):
      temp = []
      for j in range(testfeatures):
        sumval = 0.0
        for k in range(testrows):
          sumval += (testdata[k][j]*epost[i][k])
        temp.append(sumval)
      muval.append(temp)

    ############print '\n Muval check (before division): '
    ############for i in range(clusters):
      ############print muval[i]

    temp = []
    for i in range(clusters):
      temp.append([a/(priorval[i]*testrows) for a in muval[i]])
    muval = temp[:]

    ############print '\n Muval check (after div): '
    ############for i in range(clusters):
      ############print muval[i]

    sdval = []
    for i in range(clusters):
      temp = []
      for j in range(testfeatures):
        sumval = 0.0
        for k in range(testrows):
          sumval += (((testdata[k][j]-muval[i][j])**2)*epost[i][k])
        temp.append(sumval)
      sdval.append(temp)

    ############print '\n sdval check (before division): '
    ############for i in range(clusters):
      ############print sdval[i]

    temp = []
    for i in range(clusters):
      #smoothening incorporated so that sd can never be 0
      temp.append([math.sqrt((a+0.00001)/(priorval[i]*testrows)) for a in sdval[i]])
    sdval = []
    sdval = temp[:]

    ############print '\n sdval check (after division): '
    ############for i in range(clusters):
      ############print sdval[i]

    oldell = ell[:]
    prior = priorval[:]
    mu = muval[:]
    sd = sdval[:]

    loopvar +=1

  print "\nThreshold met. Loop Exited!\n\nFor Reference: "
  print "\nPrior val: ", prior
  print "\nMu val: ", mu
  print "\nSD val: ", sd
  for i in range(clusters):
    print "\nOLDELL row vals: ", oldell[i]

  print '\n\nNumber of loops: ', loopvar

  outf = open(model, 'w')
  outf.write(str(clusters)+' '+str(testfeatures)+'\n')
  for i in range(clusters):
    outf.write(str(prior[i])+' ')
    for j in range(testfeatures):
      outf.write(str(mu[i][j])+' ')
    for j in range(testfeatures):
      outf.write(str(sd[i][j])+' ')
    outf.write('\n')

  outf.close()



if __name__ == '__main__':
  guassmix(sys.argv)
