function [ pooledDesc ] = get_hist_desc( descType, normPoint, normPooled, cellOpt )


global DIRS;
tic;
SHAPES = get_files(DIRS.EVECS);
DESCS = get_files(fullfile(DIRS.DESC, descType));
N = length(DESCS);

if nargin < 2
    normPoint = 'L2';
    normPooled = 'L2';
end
if nargin < 4
    cellOpt = 0;
end


X_desc=ones(100,64,N);
minvalues = ones(100,1);
maxvalues = zeros(100,1);
for i=1:N
    fprintf('Pooling descriptor %d/%d\n', i, N);
    
    % Load shape's area elements
    Area = load(SHAPES{i}, 'S');
    [m,n] = size(Area.S);
    if m~=1 && n~=1
        Area = full(diag(Area.S));
    else
        Area = Area.S;
    end    
    %Area=sort(Area)
    % Make area elements sum up to 1
    %fprintf('Size %d\n',size(Area,1));
    %MeanArea = ones(size(Area,1),1);
    %pArea =  MeanArea/size(Area,1);
    %pArea = Area / sum(Area);
    
    % Load shape's point descriptors
    Desc = load(DESCS{i}, 'desc');
    Desc = Desc.desc; % n x k 
    
    % Normalize point descriptors
    %Desc = normalize(Desc, normPoint, 2); % definitely for WKS
    [Dm Dn] = size(Desc);
    for j =1:Dn
        Desc_in = Desc(:,j);
        Desc_in = Desc_in.*Area;
        if minvalues(j,1)>min(Desc_in(:))
            minvalues(j,1) = min(Desc_in(:));
        end
        if maxvalues(j,1) < max(Desc_in(:))
            maxvalues(j,1) = max(Desc_in(:));
        end
        % Pooling weighted by the area elements
        %descPooled = pArea'*Desc; % 1xn x nxk = 1xk    
        
        % Normalize the pooled descriptor
        %a = normalize(a, 'L2', 2);
        %a = (a - min(a(:)))./(max(a(:))-min(a(:)));
        %a=mapminmax(a,0,1);
        %output = (output-min(output(:)))./(max(output(:))-min(output(:)));
       
    end
    %output=output(1:100,:)
    %imshow(output)
end
for i=1:N
    fprintf('Pooling descriptor %d/%d\n', i, N);
    
    % Load shape's area elements
    Area = load(SHAPES{i}, 'S');
    [m,n] = size(Area.S);
    if m~=1 && n~=1
        Area = full(diag(Area.S));
    else
        Area = Area.S;
    end    
    %Area=sort(Area)
    % Make area elements sum up to 1
    %fprintf('Size %d\n',size(Area,1));
    %MeanArea = ones(size(Area,1),1);
    %pArea =  MeanArea/size(Area,1);
    %pArea = Area / sum(Area);
    
    % Load shape's point descriptors
    Desc = load(DESCS{i}, 'desc');
    Desc = Desc.desc; % n x k 
    
    % Normalize point descriptors
    %Desc = normalize(Desc, normPoint, 2); % definitely for WKS
    [Dm Dn] = size(Desc);
    output = [];
    for j =1:Dn
        Desc_in = Desc(:,j);
        Desc_in = Desc_in.*Area;
        [a b]=hist_area(Desc_in,64,minvalues(j,1),maxvalues(j,1));
        % Pooling weighted by the area elements
        %descPooled = pArea'*Desc; % 1xn x nxk = 1xk    
        
        % Normalize the pooled descriptor
        %a = normalize(a, 'L2', 2);
        %a = (a - min(a(:)))./(max(a(:))-min(a(:)));
        %a=mapminmax(a,0,1);
        output = [output;a];
        %output = (output-min(output(:)))./(max(output(:))-min(output(:)));
       
    end
    %output=output(1:100,:)
    %imagesc(output)
    X_desc(:,:,i)=output;  
end

if ~cellOpt
    pooledDesc = X_desc;
end
toc;



