from __future__ import annotations
from chdrft.config.env import init_jupyter

init_jupyter()
from chdrft.projects.uav_py.main import *
from chdrft.utils.omath import make_norm
import polars as pl
import astropy
import tqdm.contrib.itertools
import time

import chdrft.utils.K as K
import numpy as np
import chdrft.struct.base as opa_struct

from chdrft.display.base import TriangleActorBase
from chdrft.sim.base import SimpleVisitor, TMSQuad, TMSQuadParams, TileGetter, VisitorBase
import chdrft.display.vtk as opa_vtk
from chdrft.sim.vtk import VTKHelper
from chdrft.sim.rb.base import Transform, Vec3, make_rot, as_pts3
import chdrft.utils.omath as omath
from chdrft.dsp.image import ImageData
import shapely
import chdrft.display.control_center as cc
import chdrft.projects.gpx as  ogpx

sample_file = cmisc.path_here('/tmp/test2.json')
sample_file = cmisc.path_here('../data.json')
vname = '/home/benoit/Downloads/DuskRider-Hero10-GX010231.MP4'
kParisLonLat = omath.deg2rad(np.array([2.3200410217200766, 48.8588897]))

arrow = 'arrow-up-long'
ica = cc.IconAdder(reqs=[arrow])


sx = cc.RxServer()
sx.start()

def ll2lla(ll):
  return np.append(ll, np.zeros((ll.shape[0], 1)), axis=1)


def build_orient(d):

  p = shapely.Point(dx.lla[d.idx, [0, 1]])
  s = dx.speed_dir[d.idx, :2]
  angle = 90 - omath.rad2deg(np.arctan2(s[1], s[0]))
  return dict(feature=p, rot=angle, icon=arrow)



content = Z.FileFormatHelper.Read(sample_file)
cv = Converter.Make(content)
dx = get_data(cv, tfreq=10)
cori0 = compute_cori0(cv, dx)
rot0 = xyzw2rot(cori0.output.q_pose)
gpx = dx.gpx_data

assert 0




def compute_fov(dx, idx):
  wl = Transform.From(
      pos=dx.pos[idx] * (1,1,0) + (0, 0, 15),
      #rot=rot0.inv() * dx.CORI[idx].inv() * R.from_matrix(make_rot(z=Vec3.Y(), y=Vec3.Z()))
      rot=R.from_matrix(make_rot(z=-(dx.speed_dir[idx] * (1,1,0) + (0, 0, -1)), y=vz))
  )
  ch = omath.CameraHelper(persp=omath.perspective(50, 4 / 3, 0.1, 1e2))
  res =  ch.find_proj(wl.data, 0)
  return shapely.ops.transform(lambda x,y: dx.gpx_data.xyz2lla(np.stack((x,y,np.zeros_like(x)), axis=-1))[:,:2].T, res)

compute_fov(dx, 0)




wl = Transform.From(
    pos=(0, 0, 10),
    #rot=rot0.inv() * dx.CORI[idx].inv() * R.from_matrix(make_rot(z=Vec3.Y(), y=Vec3.Z()))
    rot=R.from_matrix(make_rot(z=-(np.array([1,0,0]) * (1,1,0) + (0, 0, 0))))
)



svg_icon = Z.FileFormatHelper.Read('./resources/arrow-right-up-svgrepo-com.svg', mode='txt')
cpos = omath.rad2deg(kParisLonLat)
cpos = dx.lla[0,:2]
fh = cc.FoliumHelper()
fh.create_folium(cpos)
#folium.Marker(cpos[::-1], icon= folium.DivIcon(
#  html=svg_icon,
#  icon_size=(200, 200),
#  icon_anchor=(200, 400),
#)).add_to(fh.m)
#folium.Marker(cpos[::-1]).add_to(fh.m)

fh.setup(sx)
idx = 34000


def cb(du):
  idx = du.idx
  fh.data['abc'] = shapely.Point(dx.lla[idx,:2])
  fh.ev.on_next(compute_fov(dx, idx))


K.oplt.create_window()
gw = K.oplt.find_window().gw
sh = cc.Synchronizer.Make(gw, sx, vname, dx.gpx_data.data)
sh.at_t.subscribe_safe(cb)

e_t = sh.at_t.map(lambda x: x.t)
def link_line_obs(e_t, line):
  e_t.subscribe_safe(line.setPos, qt_sig=True)
sh.opp.add_plot(Dataset(x=dx.gpx_data.data.t, y=dx.pos, name='pos'))
link_line_obs(e_t, sh.opp.sampler.marks.add_line(0))

gw.add(fh.generate_widget())

assert 0

m = create_folium(cpos)

assert 0



sh.oe.obs.f = sh.at_t.map_safe(build_orient).map_safe(ica.tsf_feature).listen_value()

assert 0



sh.m.show_in_browser()

assert 0

cpos = omath.rad2deg(kParisLonLat)
sx = RxServer()
f = rx_helpers.WrapRX(rx.subject.BehaviorSubject(None))
oe = sx.add(ObsEntry(name='abc', f=f.map(shapely_as_feature)))
oe.obs.f = sx.at_t.map_safe(build_orient).map_safe(ica.tsf_feature).listen_value()

assert 0

m = create_folium(cpos, oe, map_cb=ica.postproc)

m.show_in_browser()

assert 0

tg = TileGetter()
#box = Z.Box.FromPoints(dx.lla[:,:2]).expand(1.1)
tidx = idx
box = Z.Box.FromPoints(omath.deg2rad(dx.lla))
box = Z.Box(center=omath.deg2rad(dx.lla[tidx, :2]), size=box.size / 4)
u = SimpleVisitor(
    tg,
    actor_builder=opa_vtk.TriangleActorVTK,
    max_depth=TMSQuad.MAX_DEPTH - 1,
    ll_box=box,
    coord_map=lambda quad: cv.lla2xyz(ll2lla(quad.box_lonlat.expand(1).poly())),
)
m2u = 1
u.run(TMSQuad.Root(TMSQuadParams(m2u=m2u)))

main = opa_vtk.vtk_offscreen_obj(width=1000, height=1000)
#main = opa_vtk.vtk_main_obj(width=1000, height=1000)

VTKHelper.SimpleVisitor2Assembly(u, main.ren)
center = dx.pos[-1]
dist = 200 * m2u
pa = center + [0, 0, dist]
print(center)
print(pa)
w2l = Transform.From(pos=pa, rot=make_rot(z=(center - pa)))
print(pa, center, w2l @ Vec3(center, vec=False))

(rot0.inv() * dx.CORI[idx].inv() * dx.IORI[idx].inv()).apply(-Vec3.Y().vdata)


def tsf_at(idx):
  return Transform.From(
      pos=cv.lla2xyz(dx.lla[idx] * (1, 1, 0)) + (0, 0, 10),
      rot=rot0.inv() * dx.CORI[idx].inv() * R.from_matrix(make_rot(z=-Vec3.Y(), y=-Vec3.Z()))
  )
  return Transform.From(
      pos=cv.lla2xyz(dx.lla[idx] * (1, 1, 0)) + (0, 0, 100),
      rot=R.from_matrix(make_rot(z=-Vec3.Z().data, y=Vec3.Y()))
  )
  return Transform.From(
      pos=cv.lla2xyz(dx.lla[idx] * (1, 1, 0)) + (0, 0, 10),
      rot=R.from_matrix(make_rot(z=-Vec3.Z().data - 3 * Vec3.X().data, y=Vec3.Y()))
  )


tsf = tsf_at(tidx)
mat = omath.MatHelper.mat_apply_nd(
    tsf.data, omath.MatHelper.simple_mat(rot=np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).T)
)
VTKHelper.set_mat(main.cam, mat, is_cam=True)

fname = '/tmp/test1.png'
main.ren_win.SetSize(3000, 3000)
main.cam.SetViewAngle(50)
main.cam.SetClippingRange(1e-3 * m2u, 1e6 * m2u)
#main.cam.SetPosition(8, 0, 8)
#main.cam.SetFocalPoint(0, 0, 0)
#main.cam.SetViewAngle(50)
#main.cam.SetClippingRange(1, 1e20)
main.render(fname)
K.oplt.plot(A(images=[ImageData.Make(fname)]))
#main.run()

assert 0

assert 0

pl = MPVPlayer()
K.oplt.find_window().gw.add(pl)
pl.player.play()

#             perm              sgn  converged  cost_final  cost_initial                    fv              fv_world                                             q_pose
#55   (0, 2, 3, 1)    (-1, 1, 1, 1)       True   31.513774   3455.893026  [0.0, 0.0, 0.0, 1.0]  [0.0, 0.0, 0.0, 1.0]  [0.9253155641600923, -0.2490187812257894, -0.1...
#56   (0, 2, 3, 1)  (1, -1, -1, -1)       True   31.513774   3455.893026  [0.0, 0.0, 0.0, 1.0]  [0.0, 0.0, 0.0, 1.0]  [0.9253155641600923, -0.2490187812257894, -0.1...
#178  (1, 3, 2, 0)  (-1, -1, 1, -1)       True   24.669458   1774.349806  [0.0, 0.0, 0.0, 1.0]  [0.0, 0.0, 0.0, 1.0]  [0.1615456021597579, -0.2246196099524083, -0.9...
#189  (1, 3, 2, 0)    (1, 1, -1, 1)       True   24.669458   1774.349806  [0.0, 0.0, 0.0, 1.0]  [0.0, 0.0, 0.0, 1.0]  [0.1615456021597579, -0.2246196099524083, -0.9...
#193  (2, 0, 1, 3)  (-1, -1, -1, 1)       True   34.509073   3466.607288  [0.0, 0.0, 0.0, 1.0]  [0.0, 0.0, 0.0, 1.0]  [-0.9278888546517222, 0.2511888195690446, -0.1...
#206  (2, 0, 1, 3)    (1, 1, 1, -1)       True   34.509073   3466.607288  [0.0, 0.0, 0.0, 1.0]  [0.0, 0.0, 0.0, 1.0]  [-0.9278888546517222, 0.2511888195690446, -0.1...
#324  (3, 1, 0, 2)  (-1, 1, -1, -1)       True   22.440345   1791.401400  [0.0, 0.0, 0.0, 1.0]  [0.0, 0.0, 0.0, 1.0]  [-0.16263563043022608, 0.2299893930815069, -0....
#331  (3, 1, 0, 2)    (1, -1, 1, 1)       True   22.440345   1791.401400  [0.0, 0.0, 0.0, 1.0]  [0.0, 0.0, 0.0, 1.0]  [-0.16263563043022608, 0.2299893930815069, -0....
#341  (3, 1, 2, 0)   (-1, 1, -1, 1)       True   34.310799   1806.088291  [0.0, 0.0, 0.0, 1.0]  [0.0, 0.0, 0.0, 1.0]  [0.21766911837996017, 0.1570816730111174, -0.9...
#346  (3, 1, 2, 0)   (1, -1, 1, -1)       True   34.310799   1806.088291  [0.0, 0.0, 0.0, 1.0]  [0.0, 0.0, 0.0, 1.0]  [0.21766911837996017, 0.1570816730111174, -0.9...

if 0:
  res = {}
  for perm, sgn in tqdm.contrib.itertools.product(
      list(Z.itertools.permutations(range(4))), list(Z.itertools.product([-1, 1], repeat=4))
  ):
    v = eval_quat_ord(sample_file, perm, sgn)
    print(perm, sgn, v)
    res[(perm, sgn)] = v
  dfx = pd.DataFrame([dict(perm=k[0], sgn=k[1], **data.output) for k, data in res.items()])
  print(dfx[dfx.cost_final < dfx.cost_final.min() * 2])
  assert 0

content = Z.FileFormatHelper.Read(sample_file)
cv = Converter.Make(content)

dx = get_data(cv)
cori0 = compute_cori0(cv, dx)
print(cori0)
rot0 = xyzw2rot(cori0.output.q_pose)

t1 = dx.speed > np.max(dx.speed) / 2
max(dx.speed)
# guessed transforms:
# image_space = IORI  * CORI * ORIG_CORI * world_space
fv_world = (
    0.509815,
    -0.315438,
    0.104368,
    0.793533,
)
fv = (
    0.72877,
    0.0695893,
    -0.0552155,
    0.678972,
)
gworld = (rot0.inv() * dx.CORI.inv()).apply(dx.GRAV)
est_front = (rot0.inv() * dx.CORI.inv() * dx.IORI.inv()).apply(-vy)
ev = rot0
err = np.linalg.norm(est_front - dx.speed_dir, axis=1)
K.oplt.plot(K.Dataset(x=dx.t, y=gworld, name='grav'), typ='graph', label='1', legend=1)
K.oplt.plot(K.Dataset(x=dx.t, y=err, name='err'), typ='graph', label='1', legend=1)
K.oplt.plot(K.Dataset(x=dx.t, y=t1, name='err'), typ='graph', label='1', legend=1)
K.oplt.plot(K.Dataset(x=dx.t, y=est_front, name='front'), typ='graph', label='1', legend=1)
K.oplt.plot(K.Dataset(x=dx.t, y=dx.pos, name='pos'), typ='graph', label='1', legend=1)
K.oplt.plot(K.Dataset(x=dx.t, y=dx.speed_dir, name='speed'), typ='graph', label='2', legend=1)

opp = K.oplt.windows[-1].gw.gp.find_by_label('1').w

#K.oplt.plot(K.Dataset(x=dx.t, y=est_front[:, 2], name='z'), typ='graph', label='1')
assert 0

fv_world = xyzw2rot(fv_world).apply(vz)
fv = xyzw2rot(fv).apply(vz)

tsf = (dx.IORI * dx.CORI).inv().apply(fv)
diff = tsf - fv_world
K.oplt.plot(K.Dataset(x=dx.t, y=diff[:, 0], name='x'), typ='graph', label='1', legend=1)
K.oplt.plot(K.Dataset(x=dx.t, y=diff[:, 1], name='y'), typ='graph', label='1')
K.oplt.plot(K.Dataset(x=dx.t, y=diff[:, 2], name='z'), typ='graph', label='1')
print(np.linalg.norm(diff, axis=0))
print(dx.CORI.inv().apply(dx.GRAV))
print(fv_world)

K.oplt.plot(K.Dataset(x=dx.t, y=gworld[:, 1], name='y'), typ='graph', label='1')
K.oplt.plot(K.Dataset(x=dx.t, y=gworld[:, 2], name='z'), typ='graph', label='1')

r0 = np.array([0.524676, 0.629478, 0.554069, -0.146564])
ev = xyzw2rot(r0)
front_dir = np.array([0, 0, 1])

tt = dx.t[t1]
err = np.linalg.norm(meas - ev.apply(real), axis=1)
print(meas)
print(ev.apply(real))
print(np.sum(err))

K.oplt.plot(K.Dataset(x=tt, y=err), typ='graph', label='1', legend=1)
sx = dx.speed_dir[t1]
K.oplt.plot(K.Dataset(x=dx.t[t1], y=sx[:, 0], name='x'), typ='graph', label='1', legend=1)
K.oplt.plot(K.Dataset(x=dx.t[t1], y=sx[:, 1], name='y'), typ='graph', label='1')
K.oplt.plot(K.Dataset(x=dx.t[t1], y=sx[:, 2], name='z'), typ='graph', label='1')

K.oplt.plot(K.Dataset(x=dx.t, y=dx.speed), typ='graph', label='1')
K.oplt.plot(K.Dataset(x=dx.t, y=dx.smooth_speed[:, 2]), typ='graph', label='1')
K.oplt.plot(K.Dataset(x=dx.t, y=bad * 1.0), typ='graph', label='1')
K.oplt.plot(K.Dataset(x=dx.t, y=dx.pos[:, 2]), typ='graph', label='1')
K.oplt.plot(K.Dataset(x=dx.t, y=dx.pos[:, 0]), typ='graph', label='1')
K.oplt.plot(K.Dataset(x=dx.t, y=dx.pos[:, 1]), typ='graph', label='1')
