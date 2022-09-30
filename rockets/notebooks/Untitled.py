#!/usr/bin/env python
# coding: utf-8

# In[1]:


init_jupyter()
from astropy import units as U


# In[40]:


hd5_file = '../data/data.nc'


# In[12]:


data = '''1 6 12 18 24 30 36 42 48 54 60 66 72 78 84 90 96 102 108 110 112 114 116 118 120 122 124
1108355 1051639 981275 909343 840987 781382 726201 673831 624044 575407 526297 476384 424727 370760 314736 258294 200243 143571 89025 71584 54717 39054 24366 13310 5975 2383 2246'''

df = Z.pd.read_csv(Z.io.StringIO(data), index_col=None, header=None, sep=' ').T
df.rename(inplace=True, columns={0: 't', 1:'w'})
df.set_index('t', inplace=True)
df = df * U.imperial.lb.to(U.kg)

# CSV generated with https://apps.automeris.io/wpd/ from Srbthrust2.svg.png
df_thrust = Z.pd.read_csv('../data/ds_thrust.csv', index_col=0, header=None, names=['t', 'thrust']) *U.imperial.lbf.to(U.N)
inter = K.interpolate.interp1d(df_thrust.index, df_thrust.thrust, fill_value='extrapolate')
df['thrust'] = inter(df.index)

df.to_csv('../data/ds_thrust_and_weight.csv')
    
mass0 = 200000 * U.imperial.lb.to(U.kg)
diameter = 3.71 * U.m
consts = [dict(mass0=mass0, diameter=diameter, name='ss_srb')]
df_consts = pd.DataFrame.from_records(consts, index='name')
df_consts.to_csv('../data/consts.csv')


# In[10]:





# In[7]:


get_ipython().run_line_magic('matplotlib', 'notebook')


# In[8]:


U.imperial.lbf.to(U.N)


# In[81]:


df.w.plot()


# In[48]:


df_thrust.plot()

