## Basic Emcee

For fitting a straight line to data.

        import emcee
        import tqdm
        def line_model(x,m,c): return m*x+c

        def initial_guess(x,y,yerr):
            A = np.vander(x, 2)
            C = np.diag(yerr * yerr)
            ATA = np.dot(A.T, A / (yerr**2)[:, None])
            cov = np.linalg.inv(ATA)
            w = np.linalg.solve(ATA, np.dot(A.T, y / yerr**2))
            return w[1]

        def log_prior(m,c):
            if 18 < c and c < 25 and 0.5 < m and m < 1.5: return 0
            else: return -np.inf
            
        def log_likelihood(m, c, x, y, xerr, yerr):
            chi2 = (y - line_model(x,m,c))**2/(xerr**2 + yerr**2)
            chi2 = np.sum(chi2)
            logl = np.log10(1000/chi2)
            return logl

        def log_probability(theta, x, y, xerr, yerr):
            m,c = theta
            logP = log_prior(m,c)
            if not np.isfinite(logP):
                return -np.inf
            return logP + log_likelihood(m, c, x, y, xerr, yerr)

        def run_emcee(x,y,xerr,yerr, nchains= 64, niters=30000, nburnin=10000):
        #     positions = initial_guess(x,y,yerr)*0.1*np.random.rand(nchains,1)
            posm = 0.5+np.random.rand(nchains)
            posc = 18+7*np.random.rand(nchains)
            positions = np.array([[posm[ii],posc[ii]] for ii in range(nchains)])
        #     positions = np.random.rand(nchains,1)
            nwalkers, ndim = np.shape(positions)

            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, log_probability, args=(x, y, xerr, yerr)
            )

            state = sampler.run_mcmc(positions, nburnin, progress=True)
            sampler.reset()
            result = sampler.run_mcmc(state, niters, progress=True)
            
            mres = np.ndarray.flatten(sampler.get_chain()[:, :,0])
            cres = np.ndarray.flatten(sampler.get_chain()[:, :,1])
            
            return mres,cres 

#### Example
        xerr = 2*np.random.rand(100)
        x = np.linspace(1,10,100) + xerr

        slopes = 1+ 0.2*np.random.rand(100)
        intercepts = 20+2*np.random.rand(100)
        yerr = 2*np.random.rand(100)
        y = slopes*x + intercepts + yerr

        m,c= run_emcee(x,y,xerr,yerr,nchains= 100, niters=3000, nburnin=3000)
        m16,m50,m84 = np.percentile(m,[16,50,84])
        print('m',m16,m50,m84)
        c16,c50,c84 = np.percentile(c,[16,50,84])
        print('c',c16,c50,c84)

        plt.plot(x,y,'.k')
        plt.errorbar(x,y,xerr,yerr,c='orange',zorder=1,lw=1,alpha=0.5, ls='')

        plt.plot(x,m50*x+c50,color='dodgerblue',alpha=0.5,label='50ptile deviation')
        plt.fill_between(x=x, y1=m16*x+c16,y2=m84*x+c84, 
                             color='lightblue', alpha=0.2, zorder=1)


## Mass independant emcee

        import emcee
        import itertools
        def line_model(x,m,c): return m*x+c

        # Not using this for now
        def initial_guess(x,y,yerr):
            A = np.vander(x, 2)
            C = np.diag(yerr * yerr)
            ATA = np.dot(A.T, A / (yerr**2)[:, None])
            cov = np.linalg.inv(ATA)
            w = np.linalg.solve(ATA, np.dot(A.T, y / yerr**2))
            return w[1]

        def log_prior(m,c):
            if 18 < c and c < 25 and 0.5 < m and m < 1.5: return 0
            else: return -np.inf
            
        def log_likelihood(m, c, xbins,ybins,nobjs):
            N, M = len(xbins), len(ybins)
            tsum = 0
            for i,j in itertools.product(np.arange(N),np.arange(M)):
                x,y,nobj = xbins[i],ybins[j],nobjs[i,j]
                tsum = tsum + (nobj**2/(y - line_model(x,m,c))**2)
            logl = np.log10(tsum)
            return logl

        def log_probability(theta, xbins,ybins,nobjs):
            m,c = theta
            logP = log_prior(m,c)
            if not np.isfinite(logP):
                return -np.inf
            return logP + log_likelihood(m, c, xbins,ybins,nobjs)

        def run_emcee(xbins,ybins,nobjs, nchains= 128, niters=3000, nburnin=1000):
        #     positions = initial_guess(x,y,yerr)*0.1*np.random.rand(nchains,1)
            posm = 0.5+np.random.rand(nchains)
            posc = 18+7*np.random.rand(nchains)
            positions = np.array([[posm[ii],posc[ii]] for ii in range(nchains)])
        #     positions = np.random.rand(nchains,1)
            nwalkers, ndim = np.shape(positions)

            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, log_probability, args=(xbins,ybins,nobjs)
            )

            state = sampler.run_mcmc(positions, nburnin, progress=True)
            sampler.reset()
            result = sampler.run_mcmc(state, niters, progress=True)
            
            mres = np.ndarray.flatten(sampler.get_chain()[:, :,0])
            cres = np.ndarray.flatten(sampler.get_chain()[:, :,1])
            
            return mres,cres

#### How to run?

        # Heatmap first
        alpha = -0.7
        stackedpdf = np.zeros((69, 179))

        for igal in tqdm.notebook.tqdm(goodkeys):
            z = pdict[igal][pkeys['z_best']]

            lumdist = pdict[igal][pkeys['lum_dist']] *3.086e24
            flux150 = raddict[igal][radkeys['pflux']]
            fluxerr = raddict[igal][radkeys['prms']]
            # fluxes are in jy

            log_l150 = (1.e-23*flux150*4*np.pi*lumdist*lumdist*1.e-7)/( (1 + z)**(1+alpha) )
            err_l150 = (1.e-23*fluxerr*4*np.pi*lumdist*lumdist*1.e-7)/( (1 + z)**(1+alpha) )

            mu, sigma = log_l150, err_l150
            rad100 = np.random.normal(mu, sigma, 100)
            rad100[np.where(rad100 < 1.e17)[0]] = 1.e17

            mu = pdict[igal][pkeys['sfr_50']]
            splus = pdict[igal][pkeys['sfr_84']] - mu
            sminux = mu - pdict[igal][pkeys['sfr_16']]
            sigma = np.max([sminux,splus])

            sfr100 = np.random.normal(mu, sigma, 100)
            
            items, edgesx, edgesy = np.histogram2d(np.log10(sfr100), np.log10(rad100), bins=(np.linspace(-3,3,70),np.linspace(17,26,180)))
            stackedpdf = stackedpdf + items

        vals, xbins, ybins = stackedpdf, edgesx, edgesy
        midx = xbins + (xbins[1] - xbins[0])
        midy = ybins + (ybins[1] - ybins[0])

        m,c= run_emcee(midx[:-1],midy[:-1],vals,nchains= 16, niters=10000, nburnin=2000)

        m16,m50,m84 = np.percentile(m,[16,50,84])
        print('m 16: {0:.4f}, 50: {1:.4f}, 84: {2:.4f}'.format(m16,m50,m84))
        c16,c50,c84 = np.percentile(c,[16,50,84])
        print('c 16: {0:.4f}, 50: {1:.4f}, 84: {2:.4f}'.format(c16,c50,c84))

        ndata=np.sqrt(len(xbins)*len(ybins))
        msigplus, msigmin = m84-m50, m50-m16
        print('m+ = {0:.4f}, m- = {1:.4f}'.format(msigplus/ndata, msigmin/ndata))

        csigplus, csigmin = c84-c50, c50-c16
        print('c+ = {0:.4f}, c- = {1:.4f}'.format(csigplus/ndata, csigmin/ndata))

        plt.rcParams['figure.figsize'] = (12,8)
        plt.rcParams['font.size'] = 16
        cmap = plt.cm.get_cmap('gist_heat_r')
        new_cmap = truncate_colormap(cmap, 0, 0.9)
        # plt.rcParams['image.cmap'] = 'gist_heat_r'
        plt.pcolormesh(midx[:-1], midy[:-1],  stackedpdf.T, cmap=cmap)
        plt.plot(sfr, logL_gurkan, lw=1, ls='--')
        plt.ylabel(r'$\log_{10}(L_{150MHz}/W~Hz^{-1})$')
        plt.xlabel(r'$\log_{10}(SFR/M_\odot~yr^{-1})$')
        plt.title('Fig 4')
        plt.xlim(-3,3)
        plt.ylim(19,26)
        plt.colorbar()
        plt.show()


## Mass dependant emcee

        import emcee
        import itertools
        import tqdm
        def line_model(x,y,m,n,c): return m*x+n*y+c

        def log_prior(m,n,c):
            if 18 < c and c < 25 and 0.5 < m and m < 1.5 and 0.25 < n and n < 0.65: return 0
            else: return -np.inf
            
        def log_likelihood(m, n, c, xbins,ybins,zbins,nobjs):
            # xbins = sfr bins, y bins = mass bins, zbins = l150 bins
            N, M, P = len(xbins), len(ybins), len(zbins)
            tsum = 0
            for i,j,k in itertools.product(np.arange(N),np.arange(M),np.arange(P)):
                x,y,z,nobj = xbins[i],ybins[j],zbins[k],nobjs[i,j,k]
                tsum = tsum + (nobj**2/(z - line_model(x,y,m,n,c))**2)
            logl = np.log10(tsum)
            return logl

        def log_probability(theta, xbins,ybins,zbins,nobjs):
            m,n,c = theta
            logP = log_prior(m,n,c)
            if not np.isfinite(logP):
                return -np.inf
            return logP + log_likelihood(m, n, c, xbins,ybins,zbins,nobjs)

        def run_emcee(xbins,ybins,zbins,nobjs, nchains= 128, niters=3000, nburnin=1000):
        #     positions = initial_guess(x,y,yerr)*0.1*np.random.rand(nchains,1)
            posm = 0.5+np.random.rand(nchains)
            posn = 0.25+0.4*np.random.rand(nchains)
            posc = 18+7*np.random.rand(nchains)
            positions = np.array([[posm[ii],posn[ii],posc[ii]] for ii in range(nchains)])
        #     positions = np.random.rand(nchains,1)
            nwalkers, ndim = np.shape(positions)

            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, log_probability, args=(xbins,ybins,zbins,nobjs)
            )

            state = sampler.run_mcmc(positions, nburnin, progress=True)
            sampler.reset()
            result = sampler.run_mcmc(state, niters, progress=True)
            
            mres = np.ndarray.flatten(sampler.get_chain()[:, :,0])
            nres = np.ndarray.flatten(sampler.get_chain()[:, :,1])
            cres = np.ndarray.flatten(sampler.get_chain()[:, :,2])
            
            return mres,nres,cres



#### How to run?
        # Get heatmap
        def get_heatmap_mass(tkeys):
            alpha = -0.7
            stackedpdf = np.zeros((60,50,180))

            for igal in tqdm.notebook.tqdm(tkeys):
                z = pdict[igal][pkeys['z_best']]

                lumdist = pdict[igal][pkeys['lum_dist']] *3.086e24
                flux150 = raddict[igal][radkeys['pflux']]
                fluxerr = raddict[igal][radkeys['prms']]
                # fluxes are in jy

                log_l150 = (1.e-23*flux150*4*np.pi*lumdist*lumdist*1.e-7)/( (1 + z)**(1+alpha) )
                err_l150 = (1.e-23*fluxerr*4*np.pi*lumdist*lumdist*1.e-7)/( (1 + z)**(1+alpha) )

                mu, sigma = log_l150, err_l150
                rad100 = np.random.normal(mu, sigma, 100)

                mu = pdict[igal][pkeys['sfr_50']]
                splus = pdict[igal][pkeys['sfr_84']] - mu
                sminux = mu - pdict[igal][pkeys['sfr_16']]
                sigma = np.max([sminux,splus])

                sfr100 = np.random.normal(mu, sigma, 100)
                
                
                mu = pdict[igal][pkeys['mcurr_50']]
                splus = pdict[igal][pkeys['mcurr_84']] - mu
                sminux = mu - pdict[igal][pkeys['mcurr_16']]
                sigma = np.max([sminux,splus])

                mass100 = np.random.normal(mu, sigma, 100)

                coords = np.array([ [np.log10(sfr100[ii]),np.log10(mass100[ii]),np.log10(rad100[ii])] 
                                   for  ii in range(100)])
                H, edges = np.histogramdd(coords, bins=(60,50,180), range=((-3,3),(7.5,11.8),(17,26)) )
                stackedpdf = stackedpdf + H
                
            return edges[0],edges[1],edges[2], stackedpdf


        edgesx_mass, edgesy_mass, edgesz_mass, stackedpdf_mass = get_heatmap_mass(goodkeys)
        midx = edgesx_mass + (edgesx_mass[1] - edgesx_mass[0])
        midy = edgesy_mass + (edgesy_mass[1] - edgesy_mass[0])
        midz = edgesz_mass + (edgesz_mass[1] - edgesz_mass[0])

        mm, nn, cc = run_emcee(edgesx_mass[:-1],edgesy_mass[:-1],edgesz_mass[:-1],stackedpdf_mass,
                       nchains= 16, niters=300, nburnin=100)

        m16,m50,m84 = np.percentile(mm,[16,50,84])
        print('m 16: {0:.4f}, 50: {1:.4f}, 84: {2:.4f}'.format(m16,m50,m84))
        n16,n50,n84 = np.percentile(nn,[16,50,84])
        print('n 16: {0:.4f}, 50: {1:.4f}, 84: {2:.4f}'.format(n16,n50,n84))
        c16,c50,c84 = np.percentile(cc,[16,50,84])
        print('c 16: {0:.4f}, 50: {1:.4f}, 84: {2:.4f}'.format(c16,c50,c84))

        ndata=np.sqrt(len(xbins)*len(ybins))
        msigplus, msigmin = m84-m50, m50-m16
        print('m+ = {0:.4f}, m- = {1:.4f}'.format(msigplus/ndata, msigmin/ndata))
        nsigplus, nsigmin = n84-n50, n50-n16
        print('n+ = {0:.4f}, n- = {1:.4f}'.format(nsigplus/ndata, nsigmin/ndata))
        csigplus, csigmin = c84-c50, c50-c16
        print('c+ = {0:.4f}, c- = {1:.4f}'.format(csigplus/ndata, csigmin/ndata))