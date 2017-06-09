%读取数据到smat
load sfeaturewksh19.txt;
save('sfeaturewksh19','sfeaturewksh19')
label = sfeaturewksh19(:,101)
features = sfeaturewksh19(:,1:100)
Dist = pdist2(features, features);

%% Evaluate descriptors before learning
shreceval = eval_on_shrec_modif(Dist, label,0)