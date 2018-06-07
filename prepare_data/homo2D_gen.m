function homo2D_gen(pho_cnt, fname_save, rand_seed, volume, ...
    xpos, ypos, refract)

clear cfg
cfg.nphoton=pho_cnt;
cfg.vol= volume;
%cfg.vol=uint8(ones(1,60,60));
cfg.srcpos=[1 ypos xpos];  % change the last two pos [z, y, x]
cfg.srcdir=[0 0 1];
cfg.gpuid=1;
cfg.autopilot=1;
%
% configure optical properties here 
%
% cfg.prop=[0 0 1 1;0.005 1 0 1.37]; % scattering mus, anisotropy, mua, refractive index
cfg.prop=[0 0 1 1;0.005 1 0 refract];
cfg.tstart=0;
cfg.tend=5e-8;
cfg.tstep=5e-8;
cfg.seed = rand_seed; % each random seed will have different pattern 

% calculate the flux distribution with the given config
[flux,~]=mcxlab(cfg);

%image2D = flux.data;
%size(image2D)

%cw=squeeze(sum(flux.data,4));
%imagesc(log10(abs(cw)), [-3 7]);

currentImage = squeeze(sum(flux.data,4));
feval('save', fname_save, 'currentImage');

end
