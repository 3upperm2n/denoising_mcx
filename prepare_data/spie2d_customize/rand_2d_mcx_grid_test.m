function [cw, myimg, cfg]=rand_2d_mcx_grid_test(nphoton, myimg, randseed, mcxseed)
%
% Format:
%   [cw, myimg, cfg]=rand_2d_mcx(nphoton, maxprop, imsize, randseed, srcoffset)
% 
% Author: Qianqian Fang (q.fang at neu.edu)
%

cfg.nphoton=nphoton;
imsize = size(myimg);

if(nargin<3)
    randseed=123456789;
end

if(nargin>=2)
    rand('state',randseed);
    randn('state',randseed);
end

if(nargin<=3)
    mcxseed=randi(99999999);
end

% iput_img should be uint8 
maxprop=max(myimg(:));

cfg.vol=permute(uint8(myimg), [3,1,2]); % from 2d to 3d
cfg.issrcfrom0=1;
cfg.srctype='pencil';

%cfg.srcpos=[0 rand()*imsize(1) rand()*imsize(2)];
cfg.srcpos=[0 0.5*imsize(1) imsize(2) * 0];

cfg.srcdir=[0 imsize(1)*0.5-cfg.srcpos(2)  imsize(2)*0.5-cfg.srcpos(3)];
cfg.srcdir=cfg.srcdir/norm(cfg.srcdir);
cfg.gpuid=1;
cfg.seed=mcxseed;
% cfg.gpuid='11'; % use two GPUs together
cfg.autopilot=1;
musp=abs(randn(maxprop,1)+1);
g=rand(maxprop,1);
g=zeros(maxprop,1);
mus=musp./(1-g);
% myprop=[abs(randn(maxprop,1)*0.05+0.01), mus, g, rand(maxprop,1)+1];
% myprop(1,:)=[abs(randn*0.3+1)*0.01 abs(randn*0.1+1) 0.01 1.33];
% myprop(:,1)=myprop(:,1) * 3; % adjust absorption
% % myprop(:,2)=myprop(:,2) * 0.1; % adjust scattering
% % myprop(:,3)=myprop(:,3) * 10; % adjust anisotropy
% myprop(:,4)=myprop(:,4) * 0.1; % adjust refractive coef 

%
% overwrite myprop
%


%myprop=[0.0011 0.9161 0.0100 1.3300; 0.0107 0.1757 0 1.8652];
%myprop(2,4) = myprop(2,4) * 3;

%myprop=[0.005 1 0 1.37; 0.025 1 0 1.37];    % [mua,mus,g,n] :

% myprop=[0.005 1 0 1.37; 0.02 10 0.9 1.37];    % homo
myprop=[0.005 1 0 1.37; 0.1 10 0.9 1.37];    % absorber       
%myprop=[0.005 1 0 1.37; 0.02 10 0.9 6.85];    % refractive   



cfg.prop=[0 0 1 1; myprop];
cfg.tstart=0;
cfg.tend=1e-8;
cfg.tstep=1e-8;
% cfg.detpos=[0 50 50 200];
% cfg.issaveexit=1;
% calculate the flux distribution with the given config
flux=mcxlab(cfg);

cw=squeeze(sum(flux.data,4));

if(nargout==0)
    cla
    subplot(121);
    imagesc(myimg);
    set(gca,'ydir','normal');
    axis equal
    subplot(122);
    imagesc(log10(abs(cw)))
    axis equal;
end