
addpath('../mcx/');
addpath('../mcx/mcxlab');


cfg.nphoton=1e6;
cfg.vol=permute(uint8(ones(100,100)), [3,1,2]); % from 2d to 3d
cfg.issrcfrom0=1;
cfg.srctype='pencil';
cfg.srcpos=[0,50,0];
cfg.srcdir=[0 0 1];
cfg.gpuid=1;
cfg.autopilot=1;

myprop=[0.02 10 0.9 1.37; 0.02 10 0.9 1.37];

cfg.prop=[0 0 1 1; myprop];
cfg.tstart=0;
cfg.tend=1e-8;
cfg.tstep=1e-8;
flux=mcxlab(cfg);

cw=squeeze(sum(flux.data,4));
figure; imagesc(log10(abs(cw)));