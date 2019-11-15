import json
import scipy
import numpy as np

from PIL import Image
from scipy.optimize import minimize, rosen, rosen_der

def xy_convert_paranoma(xs, rs, FOV = 240):
  us = xs * (FOV / 180) * np.pi
  coorsx = np.multiply(np.cos(us), rs)
  coorsy = np.multiply(np.sin(us), rs)
  return coorsx, coorsy

# estimate r0 r1 r2 r3, force r0 == 1
def x_residual_func(rs, xs):
  us = xs * ((240. / 180.) * np.pi)
  coorsx = np.multiply(-np.cos(us), rs)
  coorsy = np.multiply(np.sin(us), rs)
  len_r = rs.shape[0]
  residual = np.zeros(len_r + 1);
  residual[0] = (rs[0] - 1.)**2
  for i in range(1, 1 + len_r):
    cur_indx = i-1;
    vec_cur_last_x = coorsx[cur_indx-1] - coorsx[cur_indx]
    vec_cur_last_y = coorsy[cur_indx-1] - coorsy[cur_indx]
    vec_cur_next_x = coorsx[(cur_indx+1)%len_r] - coorsx[cur_indx]
    vec_cur_next_y = coorsy[(cur_indx+1)%len_r] - coorsy[cur_indx]
    norm_vec_cur_last = vec_cur_last_x**2 + vec_cur_last_y**2
    norm_vec_cur_next = vec_cur_next_x**2 + vec_cur_next_y**2
    residual[i] = ((vec_cur_last_x * vec_cur_next_x + vec_cur_next_y * vec_cur_next_y) / (norm_vec_cur_last * norm_vec_cur_next))**2
  print(residual)
  return np.sum(residual)

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  #parser.add_argument('--layout', required=True,
  #                    help='Txt file containing layout corners (cor_id)')
  args = parser.parse_args()

  #with open(args.layout) as f:
  #  inferenced_result = json.load(f)
  r = [1, 1, 1, 1]
  res = minimize(x_residual_func, r, args=(np.array([0.1, 0.4, 0.7, 0.9])), tol=1e-6)
  print(res.x)
  coorsx, coorsy = xy_convert_paranoma(np.array([0.1, 0.2, 0.3, 0.4]), res.x)
  print(coorsx)
  print(coorsy)







