function [ pooledDesc1,pooledDesc3] = get_pooled_desc2( descType, normPoint, normPooled, cellOpt )


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


X_desc1 = cell(60,1);
%X_desc2 = cell(60,1);
X_desc3 = cell(51,1);
parfor i=1:N
    fprintf('Pooling descriptor %d/%d\n', i, N);
    
    % Load shape's area elements
    Area = load(SHAPES{i}, 'S');
    [m,n] = size(Area.S);
    if m~=1 && n~=1
        Area = full(diag(Area.S));
    else
        Area = Area.S;
    end    
    % Make area elements sum up to 1
    %fprintf('Size %d\n',size(Area,1));
    MeanArea = ones(size(Area,1),1);
    pArea =  MeanArea/size(Area,1);
    pArea2 = Area / sum(Area);
    
    % Load shape's point descriptors
    Desc = load(DESCS{i}, 'desc');
    Desc = Desc.desc; % n x k 
    
    % Normalize point descriptors
    Desc = normalize(Desc, normPoint, 2); % definitely for WKS
    
    % Pooling weighted by the area elements
    descPooled_mean = pArea'*Desc; % 1xn x nxk = 1xk    
    descPooled_weight = pArea2'*Desc;
    descPooled1 = descPooled_weight(1:30);
    %descPooled2 = descPooled(25:85);
    descPooled3 =descPooled_mean(31:end);
    % Normalize the pooled descriptor
    %descPooled1 = normalize(descPooled1, normPooled, 2);
    %descPooled2 = normalize(descPooled2, normPooled, 2);
    %descPooled3 = normalize(descPooled3, normPooled, 2);
    X_desc1{i} = descPooled1; 
    %X_desc2{i} = descPooled2; 
    X_desc3{i} = descPooled3;
end

if ~cellOpt
    pooledDesc1 = cell2mat(X_desc1);
    %pooledDesc2 = cell2mat(X_desc2);
    pooledDesc3 = cell2mat(X_desc3);
end
toc;



