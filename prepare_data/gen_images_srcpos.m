function gen_images_srcpos(tid, rand_seed, pho_cnt, volume, x_loc, y_loc,x,y,z, ...
    dir_phn_test)

    clear cfg
    cfg.nphoton=pho_cnt;
    
    cfg.vol= volume;
    cfg.srcpos=[x_loc y_loc 1];
    cfg.srcdir=[0 0 1]; % perpendicular to y
    cfg.gpuid=1;
    cfg.autopilot=1;
    
    %
    % configure optical properties here 
    %
    cfg.prop=[0 0 1 1;0.005 1 0 1.37];
    cfg.tstart=0;
    cfg.tend=5e-8;
    cfg.tstep=5e-8;
    cfg.seed = rand_seed; % each random seed will have different pattern 

    % calculate the flux distribution with the given config
    %[flux,detpos]=mcxlab(cfg);
    [flux,~]=mcxlab(cfg);

    image3D=flux.data;


    %------
    % x-axis
    %------
    dir_x = sprintf('%s/x', dir_phn_test);
    if ~exist(dir_x, 'dir')  mkdir(dir_x); end
        
    for imageID=1:x
        fname = sprintf('%s/osa_phn%1.0e_test%d_img%d.mat', dir_x, pho_cnt, tid, imageID);
        fprintf('Generating %s\n (x-axis)',fname);
        currentImage = squeeze(image3D(imageID,:,:));
        feval('save', fname, 'currentImage');
    end

    
    %------
    % y-axis
    %------
    dir_y = sprintf('%s/y', dir_phn_test);
    if ~exist(dir_y, 'dir')  mkdir(dir_y); end

    for imageID=1:y
        fname = sprintf('%s/osa_phn%1.0e_test%d_img%d.mat', dir_y, pho_cnt, tid, imageID);
        fprintf('Generating %s\n (y-axis)',fname);
        currentImage = squeeze(image3D(:,imageID,:));
        feval('save', fname, 'currentImage');
    end


    %------
    % z-axis
    %------
    dir_z = sprintf('%s/z', dir_phn_test);
    if ~exist(dir_z, 'dir')  mkdir(dir_z); end

    for imageID=1:z
        fname = sprintf('%s/osa_phn%1.0e_test%d_img%d.mat', dir_z, pho_cnt, tid, imageID);
        fprintf('Generating %s\n (z-axis)',fname);
        currentImage = squeeze(image3D(:,:,imageID));
        feval('save', fname, 'currentImage');
    end

end