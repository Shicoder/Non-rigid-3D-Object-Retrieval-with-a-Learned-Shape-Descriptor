function Smat_tmp=cricle_jiaocha(x,y)
global Smat;
Smat_tmp=zeros(400,5000);
for j=1:x
    for i=1:y
        temp=Smat(:,i)+Smat(:,j);
        Smat_tmp(:,(j-1)*y+i)=temp;
    end
    fprintf('%d \n',j);
end