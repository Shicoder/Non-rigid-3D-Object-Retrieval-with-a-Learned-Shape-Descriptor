%% Initialize thread pool
if isempty(gcp), parpool; end


%% Read mesh files
read_meshes; % VERT, TRIV to shape


%% Compute Laplace-Beltrami evals and evecs
compute_LBprop; % Area elements, evals, evecs


%% Compute point descriptors
% Choose the descriptor type
compute_descriptors2('wks'); 
compute_descriptors2('sihks'); 
compute_descriptors2('hks');

