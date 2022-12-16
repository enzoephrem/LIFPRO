import sys
sys.path.append('../src/')
import display
import data_process as dp


patient = "00000"


patient_path = "/Users/enzo/Desktop/LIFPRO/Rapport/Segmentation/_patients/BraTS2021_{}/BraTS2021_{}_".format(patient, patient)

# without mask
display.display3DCuts(dp.load(patient_path+"t1ce.nii.gz"))
# with mask
display.display3DCuts(dp.load(patient_path+"t1ce.nii.gz"), dp.load(patient_path+"seg.nii.gz"))