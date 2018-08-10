function [cwdata, imdata]=gentraining_grid(nphoton, imsize, maxobj, maxrepeat, randseed)

if(nargin>=5)
%     rand('state',randseed);
%     randn('state',randseed);
%     rng(randseed);
    rng(randseed,'twister');
end

if(maxobj==0) % homogeneous case 
    cwdata=zeros(imsize(1),imsize(2),maxrepeat);
    imdata=zeros(imsize(1),imsize(2),maxrepeat);
    myseed=randi(99999999,1);
    mcxseed=randi(99999999,maxrepeat);
    for j=1:maxrepeat
        [cw, myimg]=rand_2d_mcx_grid(nphoton, 1, imsize, myseed, mcxseed(j));
        cwdata(:,:,j)=cw;
        imdata(:,:,j)=myimg;
    end
else
    cwdata=zeros(imsize(1),imsize(2),maxrepeat,maxobj);
    imdata=zeros(imsize(1),imsize(2),maxrepeat,maxobj);
    myseed=randi(99999999,maxobj,1);
    mcxseed=randi(99999999,maxobj*maxrepeat,1);
    for i=1:maxobj
        for j=1:maxrepeat
            [cw, myimg]=rand_2d_mcx_grid(nphoton, i+1, imsize, myseed(i), mcxseed((i-1)*maxrepeat+j));
            cwdata(:,:,j,i)=cw;
            imdata(:,:,j,i)=myimg;
        end
    end
end