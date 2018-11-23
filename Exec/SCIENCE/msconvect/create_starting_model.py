from __future__ import division

import numpy as np
import scipy.interpolate
import sys
import argparse
import StarKillerMicrophysics as skm

#constants in cgs
GravG = 6.67428e-8

class inputmodel(object):
    def __init__(self, r,eos_state,eos_module,eos_type):
        """r: radial coordinates, r[0] is initial value
        You have to set initial values, integration mode and g afterwards."""
        self.r = r
        self.dr = (r[-1]-r[0])/(1.0*len(r))
        self.p = np.zeros_like(self.r)
        self.temp = np.zeros_like(self.r)
        self.g = None
        self.rho = None
        self.m = None
        self.eos_mode = None
        self.nabla_mode = True
        self.self_gravity = True
        self.xn = None
        
        self.eos_type_module = eos_type
        self.eos_state = eos_state
        self.eos_module = eos_module
        
    def set_rhogiven(self, g, rho, xn, r=None, interp=scipy.interpolate.InterpolatedUnivariateSpline, xnsmoothed=True, self_gravity=True,**kwargs):
        """integrate at given temperature, all arguments can be scalar or arrays
           if xn has been smoothed to the radial coordinates of the inputmodel already, then xnsmoothed has to be True
        **kwargs are passed to interp"""
        if r is None:
            r = self.r
        if np.isscalar(g):
            self.g = np.vectorize(lambda r: g)
        else:
            if self_gravity:
                interpolator = interp(r, g, **kwargs)
                self.g = interpolator(self.r)
                self.m = self.g * self.r**2
                self.self_gravity = True
            else:
                self.g = interp(r,g,**kwargs)
                
        rho_interp = interp(r, rho, **kwargs)

        def rho_fun(r, p):
            return rho_interp(r)
        self.rho = rho_fun

        def temp_fun(r,rho,p):
            self.eos_state.rho = rho
            self.eos_state.p = p
            self.eos_state.xn = self.xn(r)
            self.eos_module.eos(self.eos_type_module.eos_input_rp,self.eos_state)
            return self.eos_state.t
        self.temp = temp_fun
        
        if xnsmoothed:
            self.xn = interp(self.r,xn,**kwargs)
        else:
            self.xn = interp(r,xn,**kwargs)
        self.nabla_mode = False

    def set_tgiven(self, g, temp, xn, r=None, interp=scipy.interpolate.InterpolatedUnivariateSpline, xnsmoothed=True, self_gravity=True,**kwargs):
        """integrate at given temperature, all arguments can be scalar or arrays
           if xn has been smoothed to the radial coordinates of the inputmodel already, then xnsmoothed has to be True
        **kwargs are passed to interp"""
        if r is None:
            r = self.r
        if np.isscalar(g):
            self.g = np.vectorize(lambda r: g)
        else:
            if self_gravity:
                interpolator = interp(r, g, **kwargs)
                self.g = interpolator(self.r)
                self.m = self.g * self.r**2
                self.self_gravity = True
            else:
                self.g = interp(r,g,**kwargs)
                
        if np.isscalar(temp):
            self.temp = np.vectorize(lambda r: temp)
        else:
            self.temp = interp(r, temp, **kwargs)

        def rho(r, p):
            self.eos_state.t = self.temp(r)
            self.eos_state.p = p
            self.eos_state.xn = self.xn(r)
            self.eos_module.eos(self.eos_type_module.eos_input_pt,self.eos_state)
            return self.eos_state.rho
        self.rho = rho
        if xnsmoothed:
            self.xn = interp(self.r,xn,**kwargs)
        else:
            self.xn = interp(r,xn,**kwargs)
        self.nabla_mode = False

    def set_sgiven(self, g, s, xn, r=None, temp=None, rho_guess=None, interp=scipy.interpolate.InterpolatedUnivariateSpline, xnsmoothed=True, self_gravity=True, **kwargs):
        """integrate at given entropy, all arguments can be scalar or arrays
           if xn has been smoothed to the radial coordinates of the inputmodel already, then xnsmoothed has to be True
        **kwargs are passed to interp"""
        if r is None:
            r = self.r
        if temp is not None:
            if np.isscalar(temp):
                self.temp = np.vectorize(lambda r: temp)
            else:
                self.temp = interp(r, temp, **kwargs)
        if rho_guess is not None:
            if np.isscalar(rho_guess):
                self.rho_guess = np.vectorize(lambda r: rho_guess)
            else:
                self.rho_guess = interp(r, rho_guess, **kwargs)
        if np.isscalar(g):
            self.g = np.vectorize(lambda r: g)
        else:
            if self_gravity:
                interpolator = interp(r, g, **kwargs)
                self.g = interpolator(self.r)
                self.m = self.g * self.r**2
                self.self_gravity = True
            else:
                self.g = interp(r, g, **kwargs)
                
        if np.isscalar(s):
            self.s = np.vectorize(lambda r: s)
        else:
            self.s = interp(r, s, **kwargs)
        
        def rho(r, p):
            self.eos_state.s = self.s(r)
            self.eos_state.p = p
            self.eos_state.xn = self.xn(r)
            self.eos_module.eos(self.eos_type_module.eos_input_ps,self.eos_state)
            return self.eos_state.rho
        self.rho = rho
        if xnsmoothed:
            self.xn = interp(self.r,xn,**kwargs)
        else:
            self.xn = interp(r,xn,**kwargs)
        self.nabla_mode = False

    def set_nablagiven(self, g, nabla, xn, r=None, temp=None, rho_guess=None, eos=None, interp=scipy.interpolate.InterpolatedUnivariateSpline, xnsmoothed=True, self_gravity=True, **kwargs):
        """ 
        if xn has been smoothed to the radial coordinates of the inputmodel already, then xnsmoothed has to be True
        """

        if r is None:
            r = self.r
        if np.isscalar(g):
            self.g = np.vectorize(lambda r: g)
        else:
            if self_gravity:
                interpolator = interp(r, g, **kwargs)
                self.g = interpolator(self.r)
                self.m = self.g * self.r**2
                self.self_gravity = True
            else:
                self.g = interp(r, g, **kwargs)
            
        if np.isscalar(nabla):
            self.nabla = np.vectorize(lambda r: nabla)
        else:
            self.nabla = interp(r, nabla, **kwargs)
            
        def rho_nabad(r, p, T):
            self.eos_state.t = T
            self.eos_state.p = p
            self.eos_state.xn = self.xn(r)
            self.eos_module.eos(self.eos_type_module.eos_input_tp,self.eos_state)
            
            chi_rho = self.eos_state.rho * self.eos_state.dpdr / self.eos_state.p
            chi_t = self.eos_state.t * self.eos_state.dpdt / self.eos_state.p
            nabad = (self.eos_state.gam1 - chi_rho) / (chi_t * self.eos_state.gam1)
            return self.eos_state.rho, nabad
        
        self.nabla_mode = True
        if xnsmoothed:
            self.xn = interp(self.r,xn,**kwargs)
        else:
            self.xn = interp(r,xn,**kwargs)
        self.rho = rho_nabad


    def set_initial_value(self, start_index=0, **kwargs):
        """specify initial value using keyword arguments (rho, temp, pres)
        example:
            Hystat.set_initial_value(12,6,rho=1e7,temp=1e8)"""
        self.start_index = start_index
        if any([k not in ('rho', 'temp', 'pres') for k in kwargs.keys()]):
            raise ValueError("unknown variable name. Please provide rho, temp, or pres in addition to abar, zbar")
        if 'pres' in kwargs:
            self.p[start_index] = kwargs['pres']
        elif 'rho' in kwargs and 'temp' in kwargs:
            r = self.r[start_index]
            self.eos_state.t = kwargs['temp']
            self.eos_state.rho = kwargs['rho']
            self.eos_state.xn = self.xn(r)
            self.eos_module.eos(self.eos_type_module.eos_input_rt,self.eos_state)  
            self.p[start_index] = self.eos_state.p
        else:
            raise NotImplementedError("Please implement this mode. ;-)")

    def integrate_gravity(self,r,rho,i,cell_centered=False):
        """
            calculates the gravity and mass of index i and returns the gravity acting on the current cell.
            
            The cell_centered mode calculates the gravity and mass at the radius of the current cell_center, 
            while the edge_centered mode ( = not cell_centered) calculates it at the lower edge of cell i
        
        """
        if not self.self_gravity:
            raise NotImplementedError("integrate_gravity without self_gravity is not implemented")
        
        
        # if we are at the starting point, we assume that the given gravity is correct and return it
        # note that this is only needed, because the integrator also evaluates the starting point 
        # when integrating the point i = start_index + 1
        if(r == self.r[self.start_index]):
            if cell_centered:
                return self.g[i]
            else:
                return self.g[i-1]
            
        if cell_centered:
            if self.nabla_mode:
                rho_old, nabla_old = self.rho(self.r[i-1],self.p[i-1],self.temp[i-1])
            else:
                rho_old = self.rho(self.r[i-1],self.p[i-1])
            
            dr = r - self.r[i-1]
            # we have to consider the upper part of the lower cell, 
            # as well as the lower part of the current cell
            self.m[i] = self.m[i-1] +  4.0/3.0 * np.pi * (
                                        rho * (dr/2.0) * ((r-dr/2.0)**2 + (r)**2 + (r-dr/2.0)*(r)) +
                                    rho_old * (dr/2.0) * ((self.r[i-1]+dr/2.0)**2 + 
                                                          (self.r[i-1])**2 + (self.r[i-1]+dr/2.0)*(r))
                                    )
            self.g[i] = -GravG*self.m[i]/r**2
            return self.g[i]
        else:
            dr = self.dr
            
            #we update the gravity at the upper edge of the current cell
            #and return the gravity at the lower edge (= upper edge of inner cell)
            self.m[i] = self.m[i-1] + rho * 4.0/3.0 * np.pi * (
                            (dr) * ((r-dr/2.0)**2 + (r+dr/2.0)**2 + (r-dr/2.0)*(r+dr/2.0))
                        )
            self.g[i] = - GravG * self.m[i]/(r+dr/2.0)**2
            return self.g[i-1]
        
        
    def integrate(self, stop_index=False, verbose=False, cell_centered=False):
        
        if self.nabla_mode:
            def hystat(r, pT, i):
                p, T = pT
                rho, nabla_ad = self.rho(r, p, T)
                
                if self.self_gravity:
                    g = self.integrate_gravity(r,rho,i,cell_centered=cell_centered)
                else:
                    g = self.g(r)
                geff = g
                
                return [geff * rho, T * rho * geff * (self.nabla(r)+nabla_ad) / p]
            
        else:
            def hystat(r, p, i):
                rho = self.rho(r, p[0])
                
                if self.self_gravity:
                    g = self.integrate_gravity(r,rho,i,cell_centered=cell_centered)
                else:
                    g = self.g(r)
                geff = g
                
                return [geff * rho]
                
        ode = scipy.integrate.ode(hystat, None)
        ode.set_integrator('dopri5',rtol=1e-13,verbosity=0,atol=1e-16,nsteps=5000)

        if not stop_index:
            stop_index = len(self.r)
        
        if self.nabla_mode and verbose:
            self.nabdiff = np.zeros(len(self.r))
            
        start_index = self.start_index
        for updown in (range(start_index+1,stop_index), range(start_index-1,-1,-1)):
            if self.nabla_mode:
                ode.set_initial_value([self.p[start_index], self.temp[start_index]], t=self.r[start_index])
            else:
                ode.set_initial_value([self.p[start_index]], t=self.r[start_index])
            for i in updown:
                try:
                    ode.set_f_params(i)
                    ode.integrate(self.r[i])  
                except:
                    raise RuntimeError('hystat: could not integrate hydrostatic equilibrium.\nError in cell %s' % i)
                self.p[i] = ode.y[0]
                if self.nabla_mode:
                    self.temp[i] = ode.y[1]
                
                if verbose:
                    if self.nabla_mode:
                        rho, nabla_ad = self.rho(self.r[i], self.p[i], self.temp[i])
                        nabla = (self.p[i] * (self.temp[i] - self.temp[i-1])) / (self.temp[i] * (self.p[i] - self.p[i-1]))
                        self.nabdiff[i] = nabla-nabla_ad
                        print(i, ode.t, ode.y, hystat(ode.t,ode.y,i),self.nabla(self.r[i]),nabla-nabla_ad)
                    else:
                        print(i, ode.t, ode.y, hystat(ode.t,ode.y,i))  


        # integration for fluff cells
        if self.nabla_mode:
            ode.set_initial_value([self.p[stop_index-1], self.temp[stop_index-1]], t=self.r[stop_index-1])
        else:
            ode.set_initial_value([self.p[stop_index-1]], t=self.r[stop_index-1])
            
        for i in range(stop_index,len(self.r)):
            if verbose:
                print(ode.t, ode.y, hystat(ode.t,ode.y,i))
            ode.set_f_params(i)
            ode.integrate(self.r[i])
            if not ode.successful():
                raise RuntimeError("could not integrate hydrostatic equilibrium")
            self.p[i] = ode.y[0]
            if verbose:
                print(self.p[i])
            if self.nabla_mode:
                self.temp[i] = ode.y[1]  




def create_model(rmin,rmax,nbins,smooth=True,nsmooth=5,mode='nablagiven',cell_centered=False,filename='maestro_out.txt',
                 eos_type=None,eos_state=None,eos_module=None,shiftnabla=0):
    """
        input parameters:
            rmin : inner radius of grid
            rmax : outer radius of grid
            rcut : radius with the highest resolutin
            nbins: total number of gridcells in the r-dimension
            ncut : gridpoint where rcut should be reached
            nref : refinement factor at ncut/rcut (if nref = 5 there will be 5 times finer refinement then with a aequidistant grid)
            varspacing: use varspacing or aequidistant grid
            nghost: number of ghostcells (defaults to 2)
    """


    n_var = 0
    var_dic = {'density':0,'pressure':0,'temperature':0,'hydrogen-1':0,'helium-4':0,'cno':0,'n_rad':0,'n_ad':0,'radius':0,'nabla':-1}
    
    with open(filename) as f:
        l = f.readline()
        i = None
        while l.startswith('#'):
            s = l[1:].strip()
            if 'num of variables =' in l:
                n_var = int(s[18:])
                i = 1
            elif i != None:
                var_dic[s]=i
                i = i+1
            l = f.readline()
    
        
    a = np.loadtxt(filename,dtype=float)
    rho = a[:,var_dic['density']]
    temp = a[:,var_dic['temperature']]
    r = a[:,var_dic['radius']]
    pres = a[:,var_dic['pressure']]
    xh = a[:,var_dic['hydrogen-1']]
    xhe = a[:,var_dic['helium-4']]
    xcno = a[:,var_dic['cno']]
    rad = a[:,var_dic['n_rad']]
    ad = a[:,var_dic['n_ad']]

    if var_dic['nabla'] != -1:
        true_nabla=True
    else:
        true_nabla=False

    nabla = a[:,var_dic['nabla']]
    nabla = nabla+shiftnabla
    
    interp = scipy.interpolate.PchipInterpolator
    
    #sanitize xnucs
    xsum = xh + xhe + xcno
    xh = xh / xsum 
    xhe = xhe / xsum
    xcno = xcno / xsum
    
    
    grav = np.empty(len(rho))
    
    if (cell_centered):
        #gravity is defined at the centre of zones
        M_enclosed = 4./3.*np.pi *  r[0]**3 * rho[0]
        grav[0] = -GravG*M_enclosed/r[0]**2
        for e,i in enumerate(rho[1:]):
            M_shell = rho[e+1] * (4./3.*np.pi * (r[e+1]-r[e]) * (r[e]**2 + r[e+1]**2 + r[e]*r[e+1]) )
            M_enclosed = M_enclosed + M_shell

            g_zone = -GravG*M_enclosed/r[e+1]**2

            grav[e+1] = g_zone
    else:
            #gravity should be defined on the lower edges of zones 
            grav[0] = 0
            r[0] = 0 #ensures proper interpolation
            M_enclosed = 4./3.*np.pi *  ((r[1]+r[0])/2)**3 * rho[0]
            grav[1] = -GravG*M_enclosed/((r[1]+r[0])/2)**2
            rold = (r[1]+r[0])/2
            for e in range(len(rho[1:])): 
                #we use rho[1:] -> everything is shifted by one index
                rnew = (r[e+1]+r[e])/2
                M_shell = rho[e] * (4./3.*np.pi * (rnew-rold) * (rold**2 + rnew**2 + rold*rnew) )
                M_enclosed = M_enclosed + M_shell

                g_zone = -GravG*M_enclosed/rnew**2
                grav[e+1] = g_zone

                rold = rnew

    #start_index = int(nbins/5)
    start_index = 0
    
    r_grid = np.linspace(rmin,rmax,nbins)

    if smooth:
        wnorm = 2.0*np.arange(nsmooth)/(nsmooth*(nsmooth+1.0))
        wnorm = wnorm.tolist() + wnorm.tolist()[-2::-1]
        wnorm = np.array(wnorm)
    
    xnuc = np.zeros([len(r_grid),3])
    for e,n in enumerate([xh,xhe,xcno]):
        ip = interp(r, n)
        xnuc[:,e] = ip(r_grid)
        if smooth:
            xnuc[:,e][nsmooth:-nsmooth] = [np.sum(xnuc[:,e][i-nsmooth+1:i+nsmooth]*wnorm) for i in range(nsmooth,len(xnuc[:,e])-nsmooth)] 
    
    #sanitize xnuc
    xsum = np.zeros(len(r_grid))
    for n in range(3):
        xsum = np.add(xsum,xnuc[:,n])
    for n in range(3):
        xnuc[:,n] = xnuc[:,n]/xsum

    
    #start_index = np.argmin(np.abs(r/mr.rsol - start_radius))
    print("start_index / radius:", start_index, r_grid[start_index])
    print("Mode:", mode)

    h = inputmodel(r_grid,eos_type=eos_type,eos_state=eos_state,eos_module=eos_module)
    
    if (mode =='rhogiven'):
        h.set_rhogiven(grav, rho, xnuc, r=r,interp=interp,self_gravity=True)
    elif (mode == 'tgiven'):
        h.set_tgiven(grav, temp, xnuc,r=r,interp=interp,self_gravity=True)
    elif (mode == 'nablagiven'):
        if (true_nabla):
            #this way we fix the overadiabaticity
            nabla = nabla
            h.set_nablagiven(grav, nabla-ad, xnuc,r=r,interp=interp,self_gravity=True)
        else:
            h.set_nablagiven(grav, rad-ad, xnuc,r=r,interp=interp,self_gravity=True)
    else:
        return ('This mode is not defined')
    
    pres_interp = interp(r,pres)
    temp_interp = interp(r,temp)
    rho_interp = interp(r,rho)

    if (mode =='rhogiven'):
        h.set_initial_value(pres=pres_interp(r[start_index]), start_index=start_index)
    elif (mode =='tgiven'):
        h.set_initial_value(rho=rho_interp(r[start_index]), temp=temp_interp(r[start_index]), start_index=start_index)
    elif (mode =='nablagiven'):
        h.set_initial_value(rho=rho_interp(r_grid[start_index]), temp=temp_interp(r_grid[start_index]), start_index=start_index)
        h.temp[start_index] = temp_interp(r_grid[start_index])
    else:
        return ('This mode is not defined')
    h.integrate(verbose=True,cell_centered=cell_centered)


    return [rho,temp,r,pres,xh,xhe,xcno,grav], h

def write_starting_model(m, h, filename='input.dat',r=None,mode='nablagiven',maestro=True):
    if h.self_gravity:
        if mode == 'tgiven':
            out = np.asarray([[r,h.rho(r,p),p,g,h.temp(r)] for r,p,g in zip(h.r,h.p,h.g)],dtype='f8')
        elif mode == 'rhogiven':
            out = np.asarray([[r,h.rho(r,p),p,g,h.temp(r,h.rho(r,p),p)] for r,p,g in zip(h.r,h.p,h.g)],dtype='f8')
        elif mode == 'nablagiven':
            out = np.asarray([[r,h.rho(r,p,t)[0],p,g,t] for r,p,t,g in zip(h.r,h.p,h.temp,h.g)],dtype='f8')
        else:
            return ('This mode is not defined')
    else:
        if mode == 'tgiven':
            out = np.asarray([[r,h.rho(r,p),p,h.g(r),h.temp(r)] for r,p in zip(h.r,h.p)],dtype='f8')
        elif mode == 'rhogiven':
            out = np.asarray([[r,h.rho(r,p),p,h.g(r),h.temp(r,h.rho(r,p),p)] for r,p in zip(h.r,h.p)],dtype='f8')
        elif mode == 'nablagiven':
            out = np.asarray([[r,h.rho(r,p,t)[0],p,h.g(r),t] for r,p,t in zip(h.r,h.p,h.temp)],dtype='f8')
        else:
            return ('This mode is not defined')
    
    xnuc = h.xn(h.r)
    gpot = np.asarray([gpotfun(h)], dtype='f8').T
    out = np.concatenate([out,gpot,xnuc], axis=1)
    with open(filename,'wb') as f:
        if maestro:
            f.write(('# npts = %d\n' % out.shape[0]).encode('ascii'))
            f.write(('# num of variables = %d\n' %(out.shape[1]-1)).encode('ascii'))
            f.write(('# density\n').encode('ascii'))
            f.write(('# pressure\n').encode('ascii'))
            f.write(('# gravity\n').encode('ascii'))
            f.write(('# temperature\n').encode('ascii'))
            f.write(('# gpot\n').encode('ascii'))
            f.write(('# hydrogen-1\n').encode('ascii'))
            f.write(('# helium-4\n').encode('ascii'))
            f.write(('# cno\n').encode('ascii'))            
        else:
            f.write(('%d\n' % out.shape[0]).encode('ascii'))
        np.savetxt(f, out)


def create_fake_model(rhomax,pmax,nturns=5,rlen=2e10,amp=1,const_grav=None):
    """
        input parameters:
            rmin : inner radius of grid
            rmax : outer radius of grid
            rcut : radius with the highest resolutin
            nbins: total number of gridcells in the r-dimension
            ncut : gridpoint where rcut should be reached
            nref : refinement factor at ncut/rcut (if nref = 5 there will be 5 times finer refinement then with a aequidistant grid)
            varspacing: use varspacing or aequidistant grid
            nghost: number of ghostcells (defaults to 2)
    """
    r = np.linspace(0,rhomax,1000)
    rho = [np.sin(i/rhomax*nturns*2*np.pi)*amp-i+rhomax+0.1 for i in r]
    r = np.linspace(0,rlen,1000)   
    abar = np.ones(len(r))
    zbar = np.ones(len(r))
    
    if const_grav == None:
        grav = np.empty(len(rho))
        #gravity is defined at the centre of zones
        M_enclosed = 4./3.*np.pi *  r[0]**3 * rho[0]
        grav[0] = 0.0
        for e,i in enumerate(rho[1:]):
            M_shell = rho[e+1] * (4./3.*np.pi * (r[e+1]-r[e]) * (r[e]**2 + r[e+1]**2 + r[e]*r[e+1]) )
            M_enclosed = M_enclosed + M_shell

            g_zone = -mr.G*M_enclosed/r[e+1]**2

            grav[e+1] = g_zone
    else:
        grav = np.ones(len(rho))
        grav = grav * const_grav
    
    nuc = [(1.0,1.0,'h1'),
           (4.0,2.0,'he4'),
           (14.87,7.43,'cno')]
    
    nucdata = np.recarray(3,dtype=[('A', float), ('Z', float), ('names', 'a4')])
    for i,d in enumerate(nuc):
        nucdata[i]=d
    nuc_z = [1.0,2.0,7.43]
    
    xnuc = np.recarray(len(r),dtype=[('h1',float),('he4',float),('cno',float)])
    xnuc['h1'] = np.ones(len(r))
    xnuc['he4'] = np.zeros(len(r))
    xnuc['cno'] = np.zeros(len(r))
    
    
    interp = scipy.interpolate.PchipInterpolator
    #start_index = int(nbins/5)
    start_index = 0
    

    h = hystat.Hystat(r)
    h.r_start = 0
    h.eos_mode = ['gas', 'ionized', 'radiation', 'degenerate', 'coulomb']
    
    if const_grav == None:
        h.set_rhogiven(grav, rho, r=r,interp=interp,self_gravity=True)
    else:
        h.set_rhogiven(grav, rho, r=r,interp=interp,self_gravity=False)
        
    
    h.set_species(nucdata['A'], nucdata['Z'], xnuc=xnuc,r=r,interp=interp)

    h.set_initial_value(h.abar(r[start_index]),h.zbar(r[start_index]),pres=pmax, start_index=start_index)

    h.integrate(verbose=True,cell_centered=True)


    return [rho,abar,zbar], h


def gpotfun(h):                                 # gravitational potential for a hystat model
    """ Integrate gravitational acceleration to get the potential.
    h -- a hystat.Hystat object
    r -- the points at which the function shall be evaluated"""
    if h.self_gravity:
        grav = h.g
    else:
        grav = h.g(h.r)
    r = h.r
    gpot = grav.copy()
    gpot[:-1] *= np.diff(r)
    gpot[-1] *= r[-1] - r[-2]
    gpot = - gpot.cumsum()
    return gpot
   
if __name__ == "__main__":
    try:
        if skm.Eos_Module.initialized != 1:
            skm.starkiller_initialization_module.starkiller_initialize('probin.pyeos')
        eos_type_module = skm.Eos_Type_Module()
        eos_state = eos_type_module.eos_t()
        eos_module = skm.Eos_Module()
    except:
        print 'eos init failed'
            
    m,l = create_model(0,1e11,20000,smooth=True,nsmooth=80,eos_type=eos_type_module,eos_state=eos_state,eos_module=eos_module)
    write_tab(m,l)

