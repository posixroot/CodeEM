#!/usr/bin/python

""" Author: Kiran Maddipati OSUID: 200405214 """

import sys
import os
import random
import math
import numpy


def guassmix(argv):
  """This function computes the EM algorithm."""

  print "AI HW1\n"

  if len(argv) != 4:
    print "Usage: python guassmix.py <#clusters> <data-file> <model-file>"
    sys.exit()

  os.chdir('/home/kiran/Desktop/AI/hw/hw1')
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

  #################print testdata[12]

  #Initialize prior values for all the clusters
  prior = []
  for i in range(clusters):
    prior.append(1/float(clusters))

  ############print '\nprior check: '
  ############print prior

  #Find the min, max and range of each of the features
  maxarr = []
  minarr = []
  for i in range(testfeatures):
    temp = []
    for j in range(testrows):
      temp.append(testdata[j][i])
    maxarr.append(max(temp))
    minarr.append(min(temp))

  rangearr = [abs(a-b) for a,b in zip(maxarr, minarr)]

  #Initialize mu for all clusters. Method 1 (random datapoints as means)
  mu = []
  for i in range(clusters):
    mu.append(random.choice(testdata))

  #Method 2 to Initialize the mean(uniform dist. over range). To enable, uncomment the below lines.
  #mu = []
  #for i in range(clusters):
    #mu.append([a+(b*i/clusters) for a,b in zip(minarr,rangearr)])

  #mu.append(testdata[1])
  #mu.append(testdata[60])
  #mu.append(testdata[140])


  #Initialize Standard Deviation values for all the clusters
  sd = []
  for i in range(clusters):
    frac = random.random()
    sd.append([a*frac for a in rangearr])

  ########print "sd values:"
  ########for i in range(clusters):
    ########print 'sd[0] is ', sd[i]
  ########print '\n'

  loopvar = 1
  #Iteration 1
  while(loopvar>0):
    logprior = []
    for i in range(clusters):
      #ell.append(testdata)
      logprior.append(math.log(prior[i]))

    ############print 'logprior: ', logprior

    ell = []
    for i in range(clusters):
      temp = []
      for j in range(testrows):
        temp.append([(-1)*((a-b)**2)/(2*c*c) for a,b,c in zip(testdata[j],mu[i],sd[i])])
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

    #print '\n\nEll\n',loopvar
    #for i in range(clusters):
      #print ell[i][0]

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
    for i in range(clusters):
      ellprior.append([logprior[i]+x for x in ell[i]])

    ########print '\n\nElls are: '
    ########for i in range(clusters):
      ########print ell[i][12]

    ########print '\n\nEllpriors are: '
    ########for i in range(clusters):
      ########print ellprior[i][12]

    lsxmax = []
    for i in range(testrows):
      temp = []
      for j in range(clusters):
        temp.append(ellprior[j][i])
      lsxmax.append(max(temp))

    ########print 'lsxmax is: ', lsxmax[12]


    temp = []
    for i in range(clusters):
      temp.append([a-b for a,b in zip(ellprior[i],lsxmax)])
    ellprior = []
    ellprior = temp[:]

    epost = []
    #for i in range(clusters):
      #epost.append(ellprior[i])
    epost = ellprior[:]


    #print '\n'
    #for i in range(clusters):
      #print 'epost[0][12] is: ', epost[i][12]



    ############print '\nELLPRIOR CHECK (with lsxmax subtracted): '
    ############for i in range(clusters):
      ############print ellprior[i][12]

    temp = []
    for i in range(clusters):
      temp.append([math.exp(a) for a in ellprior[i]])
    ellprior = []
    ellprior = temp[:]

    ############print '\nELLPRIOR CHECK (E power): '
    ############for i in range(clusters):
      ############print ellprior[i][12]

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
