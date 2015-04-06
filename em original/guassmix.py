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

def posterior_after_lsx_factorization(clusters, testrows, ellprior, lsxmax):
  ellprior_with_lsx_max_constant = []
  for i in range(clusters):
    ellprior_with_lsx_max_constant.append([a-b for a,b in zip(ellprior[i],lsxmax)])
  ellprior = ellprior_with_lsx_max_constant

  #save the intermediate result in epost -> E-step Posterior Probability
  epost = ellprior_with_lsx_max_constant[:]

  lsx_exp_term = []
  for i in range(clusters):
    lsx_exp_term.append([math.exp(a) for a in ellprior_with_lsx_max_constant[i]])
  ellprior = lsx_exp_term

  lsx_exp_terms_sum = []#sumarr
  for i in range(testrows):
    row_sum = 0.0
    for j in range(clusters):
      row_sum += lsx_exp_term[j][i]
    lsx_exp_terms_sum.append((-1)*math.log(row_sum))

  epost_log_space = []
  for i in range(clusters):
    epost_log_space.append([a+b for a,b in zip(ellprior_with_lsx_max_constant[i], lsx_exp_terms_sum)])

  return epost_log_space


def convert_epost_to_decimal_space(clusters, epost_log_space):
  epost_decimal_space = []
  for i in range(clusters):
    epost_decimal_space.append([math.exp(a) for a in epost_log_space[i]])

  return epost_decimal_space


def calculate_cluster_priors(clusters, testrows, epost):
  cluster_priors = []
  for i in range(clusters):
    posterior_sum = 0.0
    for j in range(testrows):
      posterior_sum += epost[i][j]
    #smoothening incorporated so that prior is never 0
    if posterior_sum==0.0:
      cluster_priors.append((posterior_sum+0.00000001)/testrows)
    else:
      cluster_priors.append(posterior_sum/testrows)
  return cluster_priors


def calculate_cluster_mus(clusters, testfeatures, testrows, testdata, cluster_priors, epost):
  mu_numerator_all_clusters = []
  for i in range(clusters):
    mu_numerator_each_cluster = []
    for j in range(testfeatures):
      feature_sum = 0.0
      for k in range(testrows):
        feature_sum += (testdata[k][j]*epost[i][k])
      mu_numerator_each_cluster.append(feature_sum)
    mu_numerator_all_clusters.append(mu_numerator_each_cluster)

  mu_all_clusters = []
  for i in range(clusters):
    mu_all_clusters.append([numerator/(cluster_priors[i]*testrows) for numerator in mu_numerator_all_clusters[i]])
  return mu_all_clusters


def calculate_cluster_sds(clusters, testfeatures, testrows, testdata, mu_all_clusters, cluster_priors, epost):
  sd_numerator_all_clusters = []
  for i in range(clusters):
    sd_numerator_each_cluster = []
    for j in range(testfeatures):
      feature_sum = 0.0
      for k in range(testrows):
        feature_sum += (((testdata[k][j]-mu_all_clusters[i][j])**2)*epost[i][k])
      sd_numerator_each_cluster.append(feature_sum)
    sd_numerator_all_clusters.append(sd_numerator_each_cluster)

  sd_all_clusters = []
  for i in range(clusters):
    #smoothening incorporated so that sd can never be 0
    sd_all_clusters.append([math.sqrt((numerator+0.00000001)/(cluster_priors[i]*testrows)) if numerator==0.0 else math.sqrt(numerator/(cluster_priors[i]*testrows)) for numerator in sd_numerator_all_clusters[i]])
  return sd_all_clusters


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

    epost_log_space = posterior_after_lsx_factorization(clusters, testrows, ellprior, lsxmax)

    epost = convert_epost_to_decimal_space(clusters, epost_log_space)
    ##End of E step

    #M Step
    cluster_priors = calculate_cluster_priors(clusters, testrows, epost)

    mu_all_clusters = calculate_cluster_mus(clusters, testfeatures, testrows, testdata, cluster_priors, epost)

    sd_all_clusters = calculate_cluster_sds(clusters, testfeatures, testrows, testdata, mu_all_clusters, cluster_priors, epost)
    #End of M step

    oldell = ell[:]
    prior = cluster_priors[:]
    mu = mu_all_clusters[:]
    sd = sd_all_clusters[:]
    loopvar +=1

  #out of while loop (when threshold is met)
  print "\nThreshold met. Loop Exited!\n\nFor Reference: "
  print "\nPrior val: ", prior
  print "\nMu val: ", mu
  print "\nSD val: ", sd
  for i in range(clusters):
    print "\nOLDELL row vals: ", oldell[i]

  print '\n\nNumber of loops: ', loopvar

  #write output to out-file
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
