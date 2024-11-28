#!/usr/bin/env python
from __future__ import annotations
import sys

sys.path.insert(0, '/home/benoit/opt/lib/python3.12/site-packages')
from chdrft.config.env import init_jupyter

init_jupyter()

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
from chdrft.utils.misc import Attributize as A
import glog
import numpy as np
from pydantic import Field
import pandas as pd
import polars as pl
import inspect
import chdrft.projects.gpx as ogpx
import chdrft.utils.K as K
import chdrft.utils.colors as ocolors
import pipe as p
from chdrft.dsp.datafile import Dataset
from chdrft.display.ui import OpaPlot
import tqdm
from tqdm.contrib.concurrent import process_map
import functools
import re
import cv2 as cv
import chdrft.utils.Z as Z
import chdrft.utils.K as K
import chdrft.dsp.image as oimg
import chdrft.display.utils as dsp_utils
import chdrft.display.video_helper as vh
from chdrft.sim.rb.base import Transform, AABB, TriangleActorBase
import chdrft.utils.omath as omath
import shutil

global flags, cache
flags = None
cache = None

pl.config.Config.set_tbl_rows(20)
pl.config.Config.set_tbl_cols(20)

import astropy.units as u


def remap_scene(pattern, outfile):
  for i, x in enumerate(cmisc.get_input(pattern)):
    shutil.move(x, outfile % (i + 1))




def read_img(fname):
  img  =cv.imread(fname)
  img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
  if img.shape[0] > img.shape[1]:
    img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
    pass
  return oimg.ImageData.Make(img)


@cmisc.yield_wrapper
def get_normalized_image_sequence(fnames) -> list[oimg.ImageData]:
  for fname in fnames:
    yield read_img(fname)


def images2video(images: list[oimg.ImageData], fname=None, fps=10.0):
  from moviepy.editor import VideoClip, ImageClip, concatenate_videoclips
  video = concatenate_videoclips(list([ImageClip(x.u8.img, duration=1 / fps) for x in images]))
  video.fps = fps
  if fname is not None:
    video.write_videofile(fname, fps=fps)
  else:
    return video


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)
  parser.add_argument('--input-dir')


def test_annotate(ctx):
  return
  pass
  #%%

  images = get_normalized_image_sequence(list(cmisc.get_input('../../uav/data/scene1*.jpg')))
  images = get_normalized_image_sequence(list(cmisc.get_input('../../uav/data/try3/scene1_*.jpg')))
  gwh = K.oplt.create_window()
  s0 = vh.sink_graph()
  s1 = vh.sink_graph()
  gwh.gw.add(s0.internal)
  gwh.gw.add(s1.internal)
  s0.push(images[0].flip_y)
  s1.push(images[2].flip_y)

  for i, sx in enumerate([s0, s1]):
    texts = []
    for j, pt in enumerate(mpx[i]):
      sx.internal.dcs['PointContext'].handle_click(pt)
      texts.append(A(text=f'{j}', pos=pt))
    sx.internal.vctx.plot_meshes(A(text=texts))

  mp = mpx
  cmisc.return_to_ipython()
  #%%
  mp = []
  for sx in [s0, s1]:
    pts = [x.obj for x in sx.internal.dcs['PointContext'].data]
    mp.append(pts)
  mp = np.array(mp)

  #%%
  conf_file = '../../slam/bindings_vslam/resources/phone.yaml'
  conf_file = '../data/cam_params.yaml'
  conf = A.FromYaml(open(conf_file, 'r')).Camera
  kx = np.array([conf.k1, conf.k2, conf.p1, conf.p2, conf.k3])
  cam_f = [conf.fx, conf.fy]
  cam_c = [conf.cx, conf.cy][::-1]
  cam_mat = np.array([
      [cam_f[0], 0, cam_c[0]],
      [0, cam_f[1], cam_c[1]],
      [0, 0, 1],
  ])
  pix2cam = np.linalg.inv(cam_mat)

  mp3= np.array([omath.as_pts3(mp[f]) for f in range(2)])
  mp_cam= np.array([omath.MatHelper.mat_apply_nd(pix2cam, mp3[f].T, point=True).T for f in range(2)])
  mpa = np.array([cv.undistortPoints(mp[i], cam_mat, np.array(kx))[:,0,:] for i in range(2)])
  mat_f, _ = cv.findFundamentalMat(mp[0], mp[1], cv.FM_RANSAC)
  mat_fe, _ = cv.findFundamentalMat(mp_cam[0], mp_cam[1], cv.FM_RANSAC)
  mat_e, mask = cv.findEssentialMat(mp[0], mp[1], cameraMatrix1=cam_mat, cameraMatrix2=cam_mat, dist_coeff1=kx, dist_coeff2=kx, params=None)
  mat_e, mask = cv.findEssentialMat(mp[0], mp[1], cam_mat[0,0], cam_mat[:2,2], cv.LMEDS)
  _a, R, t, _b = cv.recoverPose(mat_e, mp[0], mp[1], cam_mat)
  print(_a,_b)

  np.sum(omath.as_pts3(mp_cam[1]).T * (mat_fe @ omath.as_pts3(mp_cam[0]).T), axis=0)
  if 0:
    cam_id = np.identity(3)
    mat_e, mask = cv.findEssentialMat(mpa[0], mpa[1], cam_id)
    _, R, t, _ = cv.recoverPose(mat_e, mpa[0], mpa[1], cam_id, mask)
  t = t[:, 0]
  tx = np.array([
      [0, -t[2], t[1]],
      [t[2], 0, -t[0]],
      [-t[1], t[0], 0],
  ])
  tx @ R
  mat_e

  tcheck = tx @ R
  tmp = np.sum(
      (pix2cam @ omath.as_pts3(mp[1]).T) * (mat_e @ pix2cam @ omath.as_pts3(mp[0]).T), axis=0
  )
  print(tmp)

  T1 = Transform.From(data=np.identity(4))
  T2 = Transform.From(rot=R, pos=t)
  m0 = omath.MatHelper.mat_apply_nd(pix2cam, mp[0].T, point=1).T
  m1 = omath.MatHelper.mat_apply_nd(pix2cam, mp[1].T, point=1).T
  res = cv.triangulatePoints(T1.data[:-1], T2.data[:-1], m0.T, m1.T)

  r1 = omath.as_pts3(res.T)
  import sklearn
  v = sklearn.metrics.pairwise_distances(r1)
  n= len(v)
  v = np.append((np.arange(0,n),), v, axis=0)
  v = np.append(np.array((np.arange(-1,n),)).T, v, axis=1)
  #%%



  #%%

  #%%
  dim= images[0].img.shape[0:2][::-1]
  pix2cam = np.linalg.inv(cam_mat)
  box = Z.Box.FromSize(dim)
  ff = 1
  nf = 0.01

  pix_points = omath.as_pts3(np.array(box.corners))


  aabb_src = AABB.FromPoints(np.array([[0,0,nf], [dim[0], dim[1], ff]]))
  mesh= aabb_src.surface_mesh
  def tsf_points(pts):
    ptsT = pts.T
    zp = ptsT * ptsT[2]
    return omath.as_pts3((pix2cam @ zp).T)
  mesh.map_points(tsf_points)


  #%%

  map_points = (pix2cam @ pix_points.T).T
  mmin = np.abs(np.min(map_points, axis=0))
  mmax = np.max(map_points, axis=0)
  fov_x = np.arctan2(mmin[0], mmax[2]) + np.arctan2(mmax[0], mmax[2])
  fov_y = np.arctan2(mmin[1], mmax[2]) + np.arctan2(mmax[1], mmax[2])
  fov_x, fov_y
  np.rad2deg(fov_x), np.rad2deg(fov_y)

  #%%

  colors = [ocolors.ColorConv.to_rgb(x.color, as_float=1) for x in s0.internal.dcs['PointContext'].data]

  m2 = TriangleActorBase.BuildFrom(mesh)
  m2.map_points(lambda pts: omath.as_pts3(T2.inv.map(omath.as_pts4(pts))))

  u = res.T
  u = u[:, :3] / u[:, [3]]
  a = u[1:4] - u[[0]]
  np.linalg.det(a)
  K.oplt.plot([A(points=u, points_color=colors, mode='3d'), mesh.vispy_data, m2.vispy_data], typ='vispy')


  #%%
  outfile = '/tmp/test.mp4'
  vs = vh.VideoSource(outfile, set_frame_db=True)
  vs.show()
  #os = vh.OverlaySink(vs.graphic, sw)
  cmisc.return_to_ipython()
  return
  #%%
  images = get_normalized_image_sequence(list(cmisc.get_input('../../uav/data/scene1*.jpg')))
  outfile = '/tmp/test.mp4'
  images2video(images, fname=outfile, fps=1)
  cmisc.return_to_ipython()
  #%%




#%%

def test(ctx):
  pass

  #%%

  remap_scene('../data/try3/PXL_20240814*', '../data/try3/board5_%03d.jpg')
  #%%
  images = ['./data/board1.jpg', './data/board2.jpg']
  images = cmisc.get_input('../data/try3/board2_*.jpg')
  pattern = (7, 7)

  criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
  objpoints = []
  imgpoints = []
  objp = np.zeros((np.prod(pattern), 3), np.float32)
  objp[:, :2] = list(cmisc.itertools.product(range(pattern[0]), range(pattern[1])))
  tshape = None

  for fname in images:
    img = read_img(fname)

    gray = cv.cvtColor(img.img, cv.COLOR_BGR2GRAY)
    gray = np.where(gray > 120, 255, 0).astype(np.uint8)
    gray_base = gray
    kernel = np.ones((31, 31))
    gray = cv.medianBlur(gray, 11)
    gray = cv.dilate(gray, kernel)
    gray = cv.erode(gray, kernel)
    gray = cv.erode(gray, kernel)
    gray = cv.dilate(gray, kernel)
    gray = cv.dilate(gray, kernel)
    #K.oplt.plot(A(images=[gray]))

    #ret, corners = cv.findChessboardCorners(gray, (7,7),  flags=cv.CALIB_CB_PLAIN)

    #cmisc.return_to_ipython()

    ret, corners = cv.findChessboardCorners(gray, pattern, flags=cv.CALIB_CB_PLAIN)
    if ret:
      refineCornerWsize = (11, 11)
      corners2 = cv.cornerSubPix(gray_base, corners, refineCornerWsize, (-1, -1), criteria)
      objpoints.append(objp.reshape((-1, 3)))
      imgpoints.append(corners2.reshape((-1, 2)))
      if 0:
        cv.drawChessboardCorners(img.img, pattern, corners2, ret)
        K.oplt.plot(A(images=[img]))
    print(ret, fname)
  #%%

  ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
      objpoints, imgpoints, gray.shape[::-1], None, None
  )
  k1, k2, p1, p2, k3 = dist[0]

  cam_params = dict(
      name='phone',
      setup='monocular',
      model='perspective',
      color_order='Gray',
      fx=mtx[0, 0],
      fy=mtx[1, 1],
      cx=mtx[0, 2],
      cy=mtx[1, 2],
      k1=k1,
      k2=k2,
      p1=p1,
      p2=p2,
      k3=k3,
      fps=1,
      rows=img.dim[1],
      cols=img.dim[0],
  )
  print(cmisc.yaml_dump_custom(dict(Camera=cam_params)))
  Z.FileFormatHelper.Write('../data/cam_params.yaml', dict(Camera=cam_params))

  #%% pass

  mean_error = 0
  errs = []
  for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    errs.extend(np.linalg.norm(imgpoints[i] - imgpoints2.reshape((-1, 2)), axis=1))
  print(K.stats.describe(errs))

  #%%
  # undistort
  img = cv.imread(images[0])
  h, w = img.shape[:2]
  newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
  dst = cv.undistort(img, mtx, dist, None, newcameramtx)

  # crop the image
  x, y, w, h = roi
  dst = dst[y:y + h, x:x + w]
  K.oplt.plot(A(images=[dst]))
  K.oplt.plot(A(images=[img]))

  #%%


def test_orb(ctx):
  #%%
  a = cv.imread('../data/scene1_001.jpg')[::-1]
  b = cv.imread('../data/scene1_002.jpg')[::-1]
  dsize = (np.array(a.shape)[:2] / 1).astype(int)
  a = cv.resize(a, dsize[::-1])
  b = cv.resize(b, dsize[::-1])
  matcher = cv.DescriptorMatcher.create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

  matcher = cv.BFMatcher()

  orb = cv.ORB.create(1000, nlevels=10, firstLevel=3, patchSize=64, WTA_K=4)
  orb = cv.SIFT.create()
  ka, fa = orb.detectAndCompute(a, None)
  kb, fb = orb.detectAndCompute(b, None)
  ka = np.array(ka)
  kb = np.array(kb)

  #%%
  #matches = matcher.match(fa, fb)
  matches = matcher.knnMatch(fa, fb, k=1)
  matches = np.array(matches)[:, 0]

  dists = np.array([x.distance for x in matches])
  min_dist = np.min(dists)
  thresh = 1.5 * min_dist
  mx = np.column_stack((np.arange(0, len(fa)), np.array([x.trainIdx for x in matches])))

  keep = dists <= thresh
  mx = mx[keep]
  print(len(mx))
  #%%

  res = cv.drawMatches(
      a,
      ka,
      b,
      kb,
      matches[keep],
      None,
      flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS | cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
  )
  K.oplt.plot(A(images=[oimg.from_cv_norm_img(res)]))
  kps = cv.drawKeypoints(a, ka, None)
  K.oplt.plot(A(images=[oimg.from_cv_norm_img(kps)]))

  #%%
  cmisc.return_to_ipython()


#%%

#%%


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


#%%
mpx = np.array(
    [
        [
            [1694.5755766, 708.8380249], [1473.3315808, 757.9553721], [1442.6004038, 499.3440532],
            [1663.8487228, 440.5444687], [3055.8040912, 1733.7890966], [1361.3001527, 281.1019954],
            [1411.6951315, 20.5118985], [306.5914016, 826.2577576], [1536.4738804, 1746.2412542],
            [2229.7977362, 2142.7837463], [3466.5562848, 58.3456547], [4022.0637683, 1254.0880457],
            [906.4688042, 672.3743040], [1924.0260631, 968.4008842], [1330.3282837, 2179.3651144],
            [2278.9149463, 551.3621435], [3200.5959114, 687.3603208], [2025.6024427, 203.3620574],
            [3862.8553369, 1671.1826032], [3252.1048614, 2172.3460394]
        ],
        [
            [1433.0536812, 1280.8055408], [1310.4408438,
                                           1399.1245208], [1182.0551235, 1268.5561598],
            [1305.0127455, 1147.4055513], [2519.1482021, 1285.0822007],
            [1107.1866255, 1189.3576143], [1114.0098493, 1046.6591146], [973.4281900, 1873.8916945],
            [1728.4375009, 1931.4004906], [2306.2390010, 1786.8205460], [2133.2778455, 363.1786556],
            [2769.8059592, 737.8521472], [1388.1841858, 1547.4941677], [1660.2170170, 1326.1630876],
            [1799.8889578, 2264.2895754], [1913.5131884, 1063.6951025], [2290.9957609, 837.9636300],
            [1627.4275913, 966.0081637], [2861.9731616, 938.3592905], [2780.5293376, 1434.0410822]
        ]
    ]
)
#

app()
