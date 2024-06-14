from chdrft.config.env import init_jupyter

init_jupyter()
from chdrft.projects.uav_py.main import *
from chdrft.utils.math import make_norm
import polars as pl
import astropy
from g2o import g2opy as g2o
import tqdm.contrib.itertools

import chdrft.utils.K as K
sample_file = cmisc.path_here('/tmp/test2.json')
sample_file = cmisc.path_here('../samples.json')


def pl_to_numpy(df, **tsf_funcs):
  return A({k: tsf_funcs.get(k, cmisc.identity)(np.array(df[k].to_list())) for k in df.columns})


def cv_to_numpy(df):
  return pl_to_numpy(df, CORI=R.from_quat, IORI=R.from_quat)


def norm_vecs(vecs):
  return vecs / np.linalg.norm(vecs, axis=1).reshape((-1, 1))

vx, vy, vz = np.identity(3)

class OptRunner:

  def run(self, content):
    with Z.tempfile.TemporaryDirectory() as tempdir:
      infile = f'{tempdir}/infile.json'
      outfile = f'{tempdir}/outfile.json'
      Z.FileFormatHelper.Write(infile, content)
      self.call(action='do_opt1', infile=infile, outfile=outfile, opt_type='cori')
      return A(input=content, output=Z.FileFormatHelper.Read(outfile))

  def call(self, **kwargs):

    def norm_args(d):
      return [f'--{k}={v}' for k, v in d.items()]

    Z.sp.check_call(
        ['./build/projects/uav/projects_uav_sample_tools.cpp'] + norm_args(kwargs),
        cwd='/home/benoit/programmation'
    )

runner = OptRunner()

def get_data(cv):
  res = cv.process()


  speed = np.array(res['GPS5_smooth_speed'].to_list())
  dx = cv_to_numpy(res[['IORI', 'CORI', 'GPS5_smooth_speed', 'GPS5_pos', 'GPS5_lla', 't', 'GRAV']])
  dx.speed_dir = norm_vecs(dx.GPS5_smooth_speed)
  dx.speed = np.linalg.norm(dx.GPS5_smooth_speed, axis=1)
  return dx


def compute_cori0(cv, dx):
  t1 = dx.speed > np.max(dx.speed) / 2
  # guessed transforms:
  # image_space = IORI  * CORI * ORIG_CORI * world_space
  vx, vy, vz = np.identity(3)
  vfront = -vy
  meas_front = (dx.CORI.inv() * dx.IORI.inv()).apply(vfront)
  real_front = dx.speed_dir
  meas_grav = dx.CORI.inv().apply(norm_vecs(dx.GRAV))
  real_grav = np.repeat(-vz[np.newaxis, :], repeats=len(dx.IORI), axis=0)


  downsample_gps = cv.compute_downsample('GPS5')
  downsample_grav = cv.compute_downsample('GRAV')

  data = A(
      meas=np.concatenate((meas_front[t1][downsample_gps], meas_grav[t1][downsample_grav])),
      real=np.concatenate((real_front[t1][downsample_gps], real_grav[t1][downsample_grav])),
      #meas=meas_front[t1] ,
      #real=real_front[t1],
      cori=dx.CORI[t1].as_quat(),
  )

  res = runner.run(data)
  if res.output.converged:
    res.score = res.output.cost_final / len(data.meas)
    res.nmeas = len(data.meas)
  return res

def eval_quat_ord(ord=None, sgn=None):
  content = Z.FileFormatHelper.Read(sample_file)
  cv = Converter.Make(content)
  if ord is not None: cv.quat_ord = list(ord)
  if sgn is not None: cv.quat_sgn = list(sgn)

  dx = get_data(cv)
  return compute_cori0(cv, dx)



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
  for perm, sgn in tqdm.contrib.itertools.product(list(Z.itertools.permutations(range(4))), list(Z.itertools.product([-1, 1], repeat=4))):
    v = eval_quat_ord(perm, sgn)
    print(perm, sgn, v)
    res[(perm, sgn)] = v
  dfx = pd.DataFrame([dict(perm=k[0], sgn=k[1], **data.output) for k,data in res.items()])
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
err = np.linalg.norm(est_front -dx.speed_dir, axis=1)
K.oplt.plot(K.Dataset(x=dx.t, y=err, name='err'), typ='graph', label='1', legend=1)
K.oplt.plot(K.Dataset(x=dx.t, y=t1, name='err'), typ='graph', label='1', legend=1)
K.oplt.plot(K.Dataset(x=dx.t, y=est_front, name='front'), typ='graph', label='1', legend=1)
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



K.oplt.plot(K.Dataset(x=dx.t, y=gworld[:, 0], name='x'), typ='graph', label='1', legend=1)
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
K.oplt.plot(K.Dataset(x=dx.t, y=dx.GPS5_smooth_speed[:, 2]), typ='graph', label='1')
K.oplt.plot(K.Dataset(x=dx.t, y=bad * 1.0), typ='graph', label='1')
K.oplt.plot(K.Dataset(x=dx.t, y=dx.GPS5_pos[:, 2]), typ='graph', label='1')
K.oplt.plot(K.Dataset(x=dx.t, y=dx.GPS5_pos[:, 0]), typ='graph', label='1')
K.oplt.plot(K.Dataset(x=dx.t, y=dx.GPS5_pos[:, 1]), typ='graph', label='1')
