function [vertex,faces,normal] = rendermatrix(filename)
load featureallq11.mat;
vt = featureallq11(118,1:201)
imagesc(vt')
