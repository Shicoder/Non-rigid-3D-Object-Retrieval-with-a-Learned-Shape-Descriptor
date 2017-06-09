function [vertex,faces,normal] = render(filename)
file ='1.obj';
[v,t,n]=read_obj(file);
load 0001.mat;
vt = desc(:,30);
imagesc(vt')
%p.ymin=0;
[vt1,ps] = mapminmax(vt',0,1);
fprintf('%d',vt1);
fprintf('%d',size(vt1));
saveResult(file,v,t,vt1);
fprintf('%d',size(v,1));
