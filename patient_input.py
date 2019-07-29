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
anno_filename = 'COBRE_phenotypic_data.csv'
data = pd.read_csv(anno_filename, sep=',')
data1 = data[['Unnamed: 0', 'Subject Type', 'Diagnosis']]
data1 = data1[data1['Subject Type']!='Disenrolled']
data_control = data1[data1['Subject Type']== 'Control']
data_control = data_control[data_control['Diagnosis']=='None']
data_patient = data1[data1['Subject Type']== 'Patient']

class_control = np.array(data_control.iloc[:,0])
class_patient = np.array(data_patient.iloc[:,0])

import nibabel as nib
control_inp=[]
control_label=[]
j=0
for i in class_control:
	q = class_control[j]
	example_filename = os.path.join("COBRE/COBRE/00{0}/session_1/anat_1/".format(q), 'mprage.nii.gz')
	img = nib.load(example_filename)

  	target_affine = np.array([[-3,-0,-0,90], [-0,3,-0,-126], [0,0,3,-72], [0,0,0,1]])
  	new_img = nilearn.image.resample_img(img,target_affine=target_affine, target_shape=(61,73,61))
  	print (new_img, new_img.shape)
  	a = np.array(new_img.dataobj)
  	y = np.swapaxes(a,0,2)
  	z = np.swapaxes(y,1,2)
  	zz = z.reshape(61,4453)
	control_inp.append(zz.reshape(1,271633))
	control_label.append(np.array([0]))
	j+=1

control_inp = np.reshape(np.array(control_inp),(len(class_control),271633))
control_label = np.array(control_label)

patient_inp=[]
patient_label=[]
j=0
for i in class_patient:
        q = class_patient[j]
        print q
        example_filename = os.path.join("COBRE/COBRE/00{0}/session_1/anat_1/".format(q), 'mprage.nii.gz')
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
        patient_inp.append(zz.reshape(1,271633))
        patient_label.append(np.array([1]))
        j+=1

patient_inp = np.reshape(np.array(patient_inp),(len(class_patient),271633))
print patient_inp.shape
patient_label = np.array(patient_label)
print patient_label.shape

inp = np.concatenate((control_inp, patient_inp), axis=0)
tar = np.concatenate((control_label, patient_label), axis=0)

idx = np.random.permutation(inp.shape[0])
inp = inp[idx]
tar = tar[idx]

np.savez_compressed('patient_train', input=inp, labels=tar)
