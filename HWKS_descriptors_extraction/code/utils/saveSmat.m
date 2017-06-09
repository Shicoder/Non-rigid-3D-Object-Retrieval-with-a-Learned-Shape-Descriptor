%save('data_hist_allminmax_64bin_wks.mat','C','HksHistDesc');
%m = load('data_hist_all.mat');
%data = m(:,1:128,1);
%imagesc(data)
data=table([Smat C]);
writetable(data,'sall200mean.csv');