# -*- coding:utf-8 -*-
import numpy as np

def Jaccard(a, b): #�Զ���ܿ�������ϵ������������0-1������Ч
  return 1.0*(a*b).sum()/(a+b-a*b).sum()

class Recommender():
  
  sim = None #���ƶȾ���
  
  def similarity(self, x, distance): #�������ƶȾ���ĺ���
    y = np.ones((len(x), len(x)))
    for i in range(len(x)):
      for j in range(len(x)):
        y[i,j] = distance(x[i], x[j])
    return y
  
  def fit(self, x, distance = Jaccard): #ѵ������
    self.sim = self.similarity(x, distance)
  
  def recommend(self, a): #�Ƽ�����
    return np.dot(self.sim, a)*(1-a)