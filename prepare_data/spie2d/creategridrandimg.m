function myimg=creategridrandimg(numchar, imsize)
%
% Format:
%   randimg=createrandimg(numchar, imsize)
% 
% Author: Qianqian Fang (q.fang at neu.edu)
%

hf=figure; 
axis;
pos=get(hf,'position');
pos(3:4)=max(pos(3:4),imsize+20);
set(hf,'position',pos);
set(gca, 'Units','pixels','position',[1, 1, imsize(1), imsize(2)]);
myimg=zeros(imsize(1),imsize(2));

%charset=['.','O','o','-','c','C','i','!'];
charset=[('A'+0):('Z'+0),('a'+0):('z'+0)];


for i=1:1
    cla;
    %randchar=charset(randi(length(charset)));
    
    rows=randi(numchar,1);
    cols=randi(numchar,1);
    rows=max([rows,cols]);
    cols=rows;
    for ix=1:cols
        for iy=1:rows
            randchar=charset(randi(length(charset)));
            ht=text((ix-1)/cols,(iy-0.5)/rows, char(randchar),'units','normalized');
            set(ht,'fontunits','normalized');
            set(ht,'fontsize',min([1/rows,1/cols]));
            %set(ht,'rotation',rand()*2*pi);
            %set(ht,'FontName','Times')
        end
    end
    axis off;
    im=getframe();
    im=im.cdata(:,:,1);
    im=(im==0);
    if(size(im,2)==imsize(2)+1)
        im(:,1)=[];
    end
    myimg=myimg+im(1:imsize(2),1:imsize(1))';
end
delete(hf);