clear all; close all; clc;

%
rng(123456789,'twister');

large_num = 9992039;

%
% generate obj locations in image 100x100
%
obj_r = [30, 50, 70];
obj_c = [30, 50, 70];

rows = length(obj_r);
cols = length(obj_c);

obj_locs = zeros(rows * cols,2);
rowID = uint32(1);
for ii = 1 : rows
    for jj = 1 : cols
        obj_locs(rowID,:)= [obj_r(ii) obj_c(jj)];
        rowID = rowID + uint32(1);
    end
end


%
% for each loc, generate 1 square with a rand shape and rand prop
% we iterate 100 times for all the locs

N = 100;
totalPos = rows * cols;

% generate N randseeds
rand_seed = randi([1 2^31-1], 1, N * totalPos);
if (length(unique(rand_seed)) < length(rand_seed)) ~= 0
    error('There are repeated random seeds!')
end



% each iteration
sid = uint32(1);
for ii = 1: N
    fprintf("\n");
    % go through each location 
    
    for jj = 1: totalPos
        % image bg
        img = ones(100,100);
        
        % obj loc
        c_rl = obj_locs(jj, :);
        c_r = c_rl(1);
        c_l = c_rl(2);
        %fprintf("%d, %d\n", c_r, c_l);
        
        h = 5+randi(10); % 6 to 15
        img(c_r-h:c_r+h, c_l-h:c_l+h)=2; % obj shape/size
        img = uint8(img);
        fprintf("c_r: %d \t c_l:%d \t h: %d\n", c_r, c_l, h);
        
        % run mcx simulation
        % mcxseed should be a large nummber (here we fix it for demo)
        rseed = rand_seed(sid);
        
        [cwdata, ~, cfg]  = rand_2d_mcx_grid_test(1e5, img, rseed, large_num + rseed);
        [cwdata1, ~, ~] = rand_2d_mcx_grid_test(1e6, img,   rseed, large_num + rseed);
        [cwdata2, ~, ~] = rand_2d_mcx_grid_test(1e7, img,   rseed, large_num + rseed);
        
        figure;
        subplot(221); imagesc(log10(abs(cwdata)));  colorbar;title('1e5');
        subplot(222); imagesc(log10(abs(cwdata1))); colorbar;title('1e6');
        subplot(223); imagesc(log10(abs(cwdata2))); colorbar;title('1e7');
        subplot(224); imagesc(img); colorbar;title('obj');
        
        cfg.prop
        
        sid = sid + uint32(1);
        
        %break;
        
        if jj == 3
            break;
        end
    end
    
    break;
    
%     if ii == 3
%        break; 
%     end
end