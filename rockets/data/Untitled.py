#!/usr/bin/env python
# coding: utf-8

# In[1]:


init_jupyter()


# In[40]:


from astropy import units as U
hd5_file = '../data/data.nc'


# In[80]:


data = '''1 6 12 18 24 30 36 42 48 54 60 66 72 78 84 90 96 102 108 110 112 114 116 118 120 122 124
1108355 1051639 981275 909343 840987 781382 726201 673831 624044 575407 526297 476384 424727 370760 314736 258294 200243 143571 89025 71584 54717 39054 24366 13310 5975 2383 2246'''

df = Z.pd.read_csv(Z.io.StringIO(data), index_col=None, header=None, sep=' ').T
df.rename(inplace=True, columns={0: 't', 1:'w'})
df.set_index('t', inplace=True)
df = df * U.imperial.lb.to(U.kg)

# CSV generated with https://apps.automeris.io/wpd/ from Srbthrust2.svg.png
df_thrust = Z.pd.read_csv('../data/ds_thrust.csv', index_col=0, header=None, names=['t', 'thrust']) * 1000 *U.imperial.lbf.to(U.N)
inter = K.interpolate.interp1d(df_thrust.index, df_thrust.thrust, fill_value='extrapolate')
df['thrust'] = inter(df.index)


# In[ ]:





# In[101]:


get_ipython().run_line_magic('matplotlib', 'notebook')
df.to_csv('../data/res.csv')


# In[99]:


import h5netcdf
with h5netcdf.File(hdf5_file, 'w') as f:
    v = f.create_variable('/srb/consts', ('tb',), data=[])
    v.attrs['abc'] = 123


# In[81]:


df.w.plot()


# In[48]:


df_thrust.plot()

