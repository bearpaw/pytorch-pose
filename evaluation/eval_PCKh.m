% wei
addpath('./utils');

% set `debug = true` if you want to visualize the skeletons
% You also need to download the MPII dataset and specify the path of 
% annopath = `mpii_human_pose_v1_u12_1.mat`
debug = false; 
annopath = 'path_to/mpii_human_pose_v1_u12_1.mat';

load('data/detections.mat');
tompson_i = RELEASE_img_index;

threshold = 0.5;
SC_BIAS = 0.6; % THIS IS DEFINED IN util_get_head_size.m

pa = [2, 3, 7, 7, 4, 5, 8, 9, 10, 0, 12, 13, 8, 8, 14, 15];

load('data/detections_our_format.mat', 'dataset_joints', 'jnt_missing', 'pos_pred_src', 'pos_gt_src', 'headboxes_src');

% predictions
predfile = '/home/wyang/code/pose/pytorch-pose/checkpoint/mpii/hg_s2_b1_mean/preds_valid.mat';
preds = load(predfile,'preds');
pos_pred_src = permute(preds.preds, [2, 3, 1]);   

% DEBUG
if debug
  mat = load(annopath);
  
  for i = 1:length(tompson_i)
    imname = mat.RELEASE.annolist(tompson_i(i)).image.name;
    fprintf('%s\n', imname);
    im = imread(['/home/wyang/Data/dataset/mpii/images/' imname]);
    pred = pos_pred_src(:, :, i);
    showskeletons_joints(im, pred, pa);
    pause; clf;
  end
end

head = find(ismember(dataset_joints, 'head'));
lsho = find(ismember(dataset_joints, 'lsho'));
lelb = find(ismember(dataset_joints, 'lelb'));
lwri = find(ismember(dataset_joints, 'lwri'));
lhip = find(ismember(dataset_joints, 'lhip'));
lkne = find(ismember(dataset_joints, 'lkne'));
lank = find(ismember(dataset_joints, 'lank'));

rsho = find(ismember(dataset_joints, 'rsho'));
relb = find(ismember(dataset_joints, 'relb'));
rwri = find(ismember(dataset_joints, 'rwri'));
rhip = find(ismember(dataset_joints, 'rhip'));
rkne = find(ismember(dataset_joints, 'rkne'));
rank = find(ismember(dataset_joints, 'rank'));

% Calculate PCKh again for a few joints just to make sure our evaluation
% matches Leonid's...
jnt_visible = 1 - jnt_missing;
uv_err = pos_pred_src - pos_gt_src;
uv_err = sqrt(sum(uv_err .* uv_err, 2));
headsizes = headboxes_src(2,:,:) - headboxes_src(1,:,:);
headsizes = sqrt(sum(headsizes .* headsizes, 2));
headsizes = headsizes * SC_BIAS;
scaled_uv_err = squeeze(uv_err ./ repmat(headsizes, size(uv_err, 1), 1, 1));

% Zero the contribution of joints that are missing
scaled_uv_err = scaled_uv_err .* jnt_visible;
jnt_count = squeeze(sum(jnt_visible, 2));
less_than_threshold = (scaled_uv_err < threshold) .* jnt_visible;
PCKh = 100 * squeeze(sum(less_than_threshold, 2)) ./ jnt_count;

% save PCK all
range = (0:0.01:0.5);
pckAll = zeros(length(range),16);
for r = 1:length( range)
  threshold = range(r);
  less_than_threshold = (scaled_uv_err < threshold) .* jnt_visible;
  pckAll(r, :) = 100 * squeeze(sum(less_than_threshold, 2)) ./ jnt_count;

end

[~, name, ~] = fileparts(predfile);

% Uncomment if you want to save the result
% save(sprintf('pckAll-%s.mat', name), 'scaled_uv_err', 'pos_pred_src');

clc;
fprintf('      Head , Shoulder , Elbow , Wrist , Hip , Knee  , Ankle , Mean , \n');
fprintf('name , %.2f , %.2f , %.2f , %.2f , %.2f , %.2f , %.2f , %.2f% , \n',...
  PCKh(head), (PCKh(lsho)+PCKh(rsho))/2, (PCKh(lelb)+PCKh(relb))/2,...
  (PCKh(lwri)+PCKh(rwri))/2, (PCKh(lhip)+PCKh(rhip))/2, ...
  (PCKh(lkne)+PCKh(rkne))/2, (PCKh(lank)+PCKh(rank))/2, mean(PCKh([1:6, 9:16])));
fprintf('\n');

