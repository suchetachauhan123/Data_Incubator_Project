# Data preparation for Gender Classification in Male and Female Category.

import os
import gzip
import shutil
import wget
import nilearn
from nilearn.image import resample_img

import tarfile
#fname = 'COBRE_scan_data.tgz'
'''tarball = tarfile.open('COBRE_scan_data.tgz')
tarball.extractall('COBRE')
tarball.close()
'''

import pandas as pd
import csv
from datetime import date, timedelta
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
anno_filename = '../COBRE_phenotypic_data.csv'
data = pd.read_csv(anno_filename, sep=',')
print (data.head())
print (list(data))

data1 = data[['Unnamed: 0', 'Gender', 'Subject Type']]
print (data1.head())
print (list(data1))
print data1.shape
data1 = data1[data1['Subject Type']!='Disenrolled']

print data1.shape
data_female = data1[data1['Gender']== 'Female']
print (data_female.head())
print data_female.shape
data_male = data1[data1['Gender']== 'Male']
print (data_male.head())
print data_male.shape

class_female = np.array(data_female.iloc[:,0])
class_male = np.array(data_male.iloc[:,0])

print class_female.shape , class_male.shape
import nibabel as nib
female_inp=[]
female_label=[]
j=0
for i in class_female:
        q = class_female[j]
        print q
        example_filename = os.path.join("../COBRE/COBRE/00{0}/session_1/anat_1/".format(q), 'mprage.nii.gz')
        img = nib.load(example_filename)
        print (img)

        target_affine = np.array([[-3,-0,-0,90], [-0,3,-0,-126], [0,0,3,-72], [0,0,0,1]])
        print (target_affine)
        new_img = nilearn.image.resample_img(img,target_affine=target_affine, target_shape=(61,73,61))
        print (new_img, new_img.shape)
        a = np.array(new_img.dataobj)
        y = np.swapaxes(a,0,2)
        z = np.swapaxes(y,1,2)
        zz = z.reshape(61,4453)
        female_inp.append(zz.reshape(1,271633))
        female_label.append(np.array([0]))
        j+=1

female_inp = np.reshape(np.array(female_inp),(len(class_female),271633))
print female_inp.shape
female_label = np.array(female_label)
print female_label.shape

male_inp=[]
male_label=[]
j=0
for i in class_male:
        q = class_male[j]
        print q
        example_filename = os.path.join("../COBRE/COBRE/00{0}/session_1/anat_1/".format(q), 'mprage.nii.gz')
        img = nib.load(example_filename)
        print (img)

        target_affine = np.array([[-3,-0,-0,90], [-0,3,-0,-126], [0,0,3,-72], [0,0,0,1]])
        print (target_affine)
        new_img = nilearn.image.resample_img(img,target_affine=target_affine, target_shape=(61,73,61))
        print (new_img, new_img.shape)
        a = np.array(new_img.dataobj)
        y = np.swapaxes(a,0,2)
        z = np.swapaxes(y,1,2)
        zz = z.reshape(61,4453)
        male_inp.append(zz.reshape(1,271633))
        male_label.append(np.array([1]))
        j+=1

male_inp = np.reshape(np.array(male_inp),(len(class_male),271633))
print male_inp.shape
male_label = np.array(male_label)
print male_label.shape

inp = np.concatenate((female_inp, male_inp), axis=0)
tar = np.concatenate((female_label, male_label), axis=0)

idx = np.random.permutation(inp.shape[0])
inp = inp[idx]
tar = tar[idx]

np.savez_compressed('train_gender', input=inp, labels=tar)

