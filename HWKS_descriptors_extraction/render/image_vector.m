function [vertex,faces,normal] = image_vector(filename)
load 0001.mat;
vt = desc(:,30);
imagesc(vt')
