'''
WIDER FACE : http://shuoyang1213.me/WIDERFACE/

## Annotation

The format of txt ground truth.
File name
Number of bounding box
x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose

## Description

blur:
  clear->0
  normal blur->1
  heavy blur->2

expression:
  typical expression->0
  exaggerate expression->1

illumination:
  normal illumination->0
  extreme illumination->1

occlusion:
  no occlusion->0
  partial occlusion->1
  heavy occlusion->2

pose:
  typical pose->0
  atypical pose->1

invalid:
  false->0(valid image)
  true->1(invalid image)

## Example 

0--Parade/0_Parade_Parade_0_904.jpg
1
361 98 263 339 0 0 0 0 0 0 
0--Parade/0_Parade_marchingband_1_117.jpg
9
69 359 50 36 1 0 0 0 0 1 
227 382 56 43 1 0 1 0 0 1 
296 305 44 26 1 0 0 0 0 1 
353 280 40 36 2 0 0 0 2 1 
885 377 63 41 1 0 0 0 0 1 
819 391 34 43 2 0 0 0 1 0 
727 342 37 31 2 0 0 0 0 1 
598 246 33 29 2 0 0 0 0 1 
740 308 45 33 1 0 0 0 2 1 

'''

import json

def convert(path):
  # 0 : to read path
  # 1 : to read count
  # 2 : to read annotation
  state_read = 0
  with open(path, 'rb') as f:
    for line in f:
      if state_read == 0 : 
        
      elif state_read == 1 :

      elif state_read == 2 and num2read > 1 :
      
      elif state_read == 2 and num2read == 1:

      else :
        raise Exception('Unknown state_read : ' + str(state_read))
      


if __name__ == '__main__':
  convert()
  convert()
  convert()