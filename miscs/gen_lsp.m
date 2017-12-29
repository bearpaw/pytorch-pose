% Dataset link
% LSP: http://sam.johnson.io/research/lsp.html
% LSP extend: http://sam.johnson.io/research/lspet.html
function gen_lsp
addpath('jsonlab/')
% in cpp: real scale = param_.target_dist()/meta.scale_self = (41/35)/scale_input
targetDist = 41/35; % in caffe cpp file 41/35
oriTrTe = load('/home/wyang/Data/dataset/LSP/joints.mat');
extTrain = load('/home/wyang/Data/dataset/lspet_dataset/joints.mat');

% in LEEDS:
% 1  Right ankle
% 2  Right knee
% 3  Right hip
% 4  Left hip
% 5  Left knee
% 6  Left ankle
% 7  Right wrist
% 8  Right elbow
% 9  Right shoulder
% 10 Left shoulder
% 11 Left elbow
% 12 Left wrist
% 13 Neck
% 14 Head top
% 15,16 DUMMY
% We want to comply to MPII: (1 - r ankle, 2 - r knee, 3 - r hip, 4 - l hip, 5 - l knee, 6 - l ankle, ..
%                             7 - pelvis, 8 - thorax, 9 - upper neck, 10 - head top,
%                             11 - r wrist, 12 - r elbow, 13 - r shoulder, 14 - l shoulder, 15 - l elbow, 16 - l wrist)
ordering = [1 2 3, 4 5 6, 15 16, 13 14, 7 8 9, 10 11 12]; % should follow MPI 16 parts..?
oriTrTe.joints(:,[15 16],:) = 0;
oriTrTe.joints = oriTrTe.joints(:,ordering,:);
oriTrTe.joints(3,:,:) = 1 - oriTrTe.joints(3,:,:);
oriTrTe.joints = permute(oriTrTe.joints, [2 1 3]);

% pelvis
oriTrTe.joints(7, 1:2, :) = mean(oriTrTe.joints(3:4,1:2,:));
v1 = oriTrTe.joints(3,3,:) > 0;
v2 = oriTrTe.joints(4,3,:) > 0;
v = find(v1 .* v2 == 1);
oriTrTe.joints(7, 3, v) = 1;

% thorax
oriTrTe.joints(8, 1:2, :) = mean(oriTrTe.joints(13:14,1:2,:));
v1 = oriTrTe.joints(13,3,:) > 0;
v2 = oriTrTe.joints(14,3,:) > 0;
v = find(v1 .* v2 == 1);
oriTrTe.joints(8, 3, v) = 1;

extTrain.joints([15 16],:,:) = 0;
extTrain.joints = extTrain.joints(ordering,:,:);


% pelvis
extTrain.joints(7, 1:2, :) = mean(extTrain.joints(3:4,1:2,:));
extTrain.joints(7, 3, :) = 1;

% thorax
extTrain.joints(8, 1:2, :) = mean(extTrain.joints(13:14,1:2,:));
extTrain.joints(8, 3, :) = 1;

count = 1;

path = {'lspet_dataset/images/im%05d.jpg', 'lsp_dataset/images/im%04d.jpg'};
local_path = {'/home/wyang/Data/dataset/lspet_dataset/images/im%05d.jpg', '/home/wyang/Data/dataset/LSP/images/im%04d.jpg'};
num_image = [10000, 1000]; %[10000, 2000];

for dataset = 1:2
  for im = 1:num_image(dataset)
    % trivial stuff for LEEDS
    joint_all(count).dataset = 'LEEDS';
    joint_all(count).isValidation = 0;
    joint_all(count).img_paths = sprintf(path{dataset}, im);
    joint_all(count).numOtherPeople = 0;
    joint_all(count).annolist_index = count;
    joint_all(count).people_index = 1;
    % joints and w, h
    if(dataset == 1)
      joint_this = extTrain.joints(:,:,im);
    else
      joint_this = oriTrTe.joints(:,:,im);
    end
    path_this = sprintf(local_path{dataset}, im);
    [h,w,~] = size(imread(path_this));
    
    joint_all(count).img_width = w;
    joint_all(count).img_height = h;
    joint_all(count).joint_self = joint_this;
    % infer objpos
    invisible = (joint_all(count).joint_self(:,3) == 0);
    if(dataset == 1) %lspet is not tightly cropped
      joint_all(count).objpos(1) = (min(joint_all(count).joint_self(~invisible, 1)) + max(joint_all(count).joint_self(~invisible, 1))) / 2;
      joint_all(count).objpos(2) = (min(joint_all(count).joint_self(~invisible, 2)) + max(joint_all(count).joint_self(~invisible, 2))) / 2;
    else
      joint_all(count).objpos(1) = w/2;
      joint_all(count).objpos(2) = h/2;
    end
    
    count = count + 1;
    fprintf('processing %s\n', path_this);
  end
end

% ---- test data
dataset = 2;
for im = 1001:2000
  % trivial stuff for LEEDS
  joint_all(count).dataset = 'LEEDS';
  joint_all(count).isValidation = 1;
  joint_all(count).img_paths = sprintf(path{dataset}, im);
  joint_all(count).numOtherPeople = 0;
  joint_all(count).annolist_index = count;
  joint_all(count).people_index = 1;
  % joints and w, h
  if(dataset == 1)
    joint_this = extTrain.joints(:,:,im);
  else
    joint_this = oriTrTe.joints(:,:,im);
  end
  path_this = sprintf(local_path{dataset}, im);
  [h,w,~] = size(imread(path_this));
  
  joint_all(count).img_width = w;
  joint_all(count).img_height = h;
  joint_all(count).joint_self = joint_this;
  % infer objpos
  invisible = (joint_all(count).joint_self(:,3) == 0);
  if(dataset == 1) %lspet is not tightly cropped
    joint_all(count).objpos(1) = (min(joint_all(count).joint_self(~invisible, 1)) + max(joint_all(count).joint_self(~invisible, 1))) / 2;
    joint_all(count).objpos(2) = (min(joint_all(count).joint_self(~invisible, 2)) + max(joint_all(count).joint_self(~invisible, 2))) / 2;
  else
    joint_all(count).objpos(1) = w/2;
    joint_all(count).objpos(2) = h/2;
  end
  
  count = count + 1;
  fprintf('processing %s\n', path_this);
end



joint_all = insertMPILikeScale(joint_all, targetDist);


opt.FileName = '../data/lsp/LEEDS_annotations.json';
opt.FloatFormat = '%.3f';
opt.Compact = 1;
savejson('', joint_all, opt);


function joint_all = insertMPILikeScale(joint_all, targetDist)
% calculate scales for each image first
joints = cat(3, joint_all.joint_self);
joints([7 8],:,:) = [];
pa = [2 3 7, 5 4 7, 8 0, 10 11 7, 13 12 7];
x = permute(joints(:,1,:), [3 1 2]);
y = permute(joints(:,2,:), [3 1 2]);
vis = permute(joints(:,3,:), [3 1 2]);
validLimb = 1:14-1;

x_diff = x(:, [1:7,9:14]) - x(:, pa([1:7,9:14]));
y_diff = y(:, [1:7,9:14]) - y(:, pa([1:7,9:14]));
limb_vis = vis(:, [1:7,9:14]) .* vis(:, pa([1:7,9:14]));
l = sqrt(x_diff.^2 + y_diff.^2);

for p = 1:14-1 % for each limb. reference: 7th limb, which is 7 to pa(7) (neck to head)
  valid_compare = limb_vis(:,7) .* limb_vis(:,p);
  ratio = l(valid_compare==1, p) ./ l(valid_compare==1, 7);
  r(p) = median(ratio(~isnan(ratio), 1));
end

numFiles = size(x_diff, 1);
all_scales = zeros(numFiles, 1);

boxSize = 368;
psize = 64;
nSqueezed = 0;

for file = 1:numFiles %numFiles
  l_update = l(file, validLimb) ./ r(validLimb);
  l_update = l_update(limb_vis(file,:)==1);
  distToObserve = quantile(l_update, 0.75);
  scale_in_lmdb = distToObserve/35; % can't get too small. 35 is a magic number to balance to MPI
  scale_in_cpp = targetDist/scale_in_lmdb; % can't get too large to be cropped
  
  visibleParts = joints(:, 3, file);
  visibleParts = joints(visibleParts==1, 1:2, file);
  x_range = max(visibleParts(:,1)) - min(visibleParts(:,1));
  y_range = max(visibleParts(:,2)) - min(visibleParts(:,2));
  scale_x_ub = (boxSize - psize)/x_range;
  scale_y_ub = (boxSize - psize)/y_range;
  
  scale_shrink = min(min(scale_x_ub, scale_y_ub), scale_in_cpp);
  
  if scale_shrink ~= scale_in_cpp
    nSqueezed = nSqueezed + 1;
    fprintf('img %d: scale = %f %f %f shrink %d\n', file, scale_in_cpp, scale_shrink, min(scale_x_ub, scale_y_ub), nSqueezed);
  else
    fprintf('img %d: scale = %f %f %f\n', file, scale_in_cpp, scale_shrink, min(scale_x_ub, scale_y_ub));
  end
  
  joint_all(file).scale_provided = targetDist/scale_shrink; % back to lmdb unit
end

fprintf('total %d squeezed!\n', nSqueezed);
