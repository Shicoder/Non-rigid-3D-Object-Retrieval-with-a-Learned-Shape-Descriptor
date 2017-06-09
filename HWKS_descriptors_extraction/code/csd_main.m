%% Init directories and files to work with
startup;

%% Choose dataset and resolution 
init_dataset;

%% Compute and save required quantities for the algorithm
precompute;

%% Get pooled descriptors 
%HksPooledMat = get_pooled_desc(DESC_TYPES.hks, '', 'L2'); 
WksPooledMat = get_pooled_desc(DESC_TYPES.wks, 'L2', 'L2');  
%HksPooledMat= get_pooled_desc4(DESC_TYPES.hks,'',  'L2');
%HksPooledMat = get_pooled_desc(DESC_TYPES.hks,'',  'L2');
%HksPooledMat = HksPooledMat(1:400,:)
%HksHistDesc = get_hist_desc(DESC_TYPES.wks, 'L2', 'L2');
%% Choose desired descriptor or combination of descriptors
%Smat = SihksPooledMat; 
%Smat = WksPooledMat;                                                        
Smat = [WksPooledMat];
%Smat = Smat(1:200,:);
%[m,n]=size(Smat);
%Smat=cricle_jiaocha(100,50);
%Smat = normalize(Smat, 'L2', 2);
%读取数据到smat
Dist = pdist2(Smat, Smat);

%% Evaluate descriptors before learning
shreceval = eval_on_shrec_modif(Dist, C)
