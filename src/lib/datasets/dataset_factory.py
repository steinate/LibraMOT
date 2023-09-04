from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .dataset.jde import JointDataset
from .dataset.siamjde import Siam_Dataset


def get_dataset(dataset, task, siam):
  if task == 'mot':
    if siam == 'none':
      print('JointDataset','=================================dataset===================================')
      return JointDataset
    else:
      print('Siam_Dataset','=================================dataset===================================')
      return Siam_Dataset
  else:
    return None
  
