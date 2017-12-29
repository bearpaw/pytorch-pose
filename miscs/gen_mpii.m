% Generate MPII train/validation split (Tompson et al. CVPR 2015)
% Code ported from 
% https://github.com/shihenw/convolutional-pose-machines-release/blob/master/training/genJSON.m
%
% in MPI: (0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee,
%          5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top,
%          10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder, 
%          14 - l elbow, 15 - l wrist)"


addpath('jsonlab/')

% Download MPII  http://human-pose.mpi-inf.mpg.de/#download
MPIIROOT = '/home/wyang/Data/dataset/mpii';  

% Download Tompson split from
% http://www.cims.nyu.edu/~tompson/data/mpii_valid_pred.zip
TOMPSONROOT = '/home/wyang/Data/dataset/mpii/Tompson_valid';

mat = load(fullfile(MPIIROOT, '/mpii_human_pose_v1_u12_1/mpii_human_pose_v1_u12_1.mat'));
RELEASE = mat.RELEASE;
trainIdx = find(RELEASE.img_train);

tompson = load(fullfile(TOMPSONROOT, '/mpii_predictions/data/detections'));
tompson_i_p = [tompson.RELEASE_img_index; tompson.RELEASE_person_index];

count = 1;
validationCount = 0;
trainCount = 0;

makeFigure = 0;       % Set as 1 for visualizing annotations

for i = trainIdx
  numPeople = length(RELEASE.annolist(i).annorect);
  fprintf('image: %d (numPeople: %d) last: %d\n', i, numPeople, trainIdx(end));
  
  for p = 1:numPeople
    loc = find(sum(~bsxfun(@minus, tompson_i_p, [i;p]))==2, 1);
    loc2 = find(tompson.RELEASE_img_index == i);
    if(~isempty(loc))
      validationCount = validationCount + 1;
      isValidation = 1;
    elseif (isempty(loc2))
      trainCount = trainCount + 1;
      isValidation = 0;
    else
      continue;
    end
    joint_all(count).dataset = 'MPI';
    joint_all(count).isValidation = isValidation;
    
    try % sometimes no annotation at all....
      anno = RELEASE.annolist(i).annorect(p).annopoints.point;
    catch
      continue;
    end
    
    % set image path
    joint_all(count).img_paths = RELEASE.annolist(i).image.name;
    [h,w,~] = size(imread(fullfile(MPIIROOT, '/images/', joint_all(count).img_paths)));
    joint_all(count).img_width = w;
    joint_all(count).img_height = h;
    joint_all(count).objpos = [RELEASE.annolist(i).annorect(p).objpos.x, RELEASE.annolist(i).annorect(p).objpos.y];
    % set part label: joint_all is (np-3-nTrain)
    
    
    % for this very center person
    for part = 1:length(anno)
      joint_all(count).joint_self(anno(part).id+1, 1) = anno(part).x;
      joint_all(count).joint_self(anno(part).id+1, 2) = anno(part).y;
      try % sometimes no is_visible...
        if(anno(part).is_visible == 0 || anno(part).is_visible == '0')
          joint_all(count).joint_self(anno(part).id+1, 3) = 0;
        else
          joint_all(count).joint_self(anno(part).id+1, 3) = 1;
        end
      catch
        joint_all(count).joint_self(anno(part).id+1, 3) = 1;
      end
    end
    
    % pad it into 16x3
    dim_1 = size(joint_all(count).joint_self, 1);
    dim_3 = size(joint_all(count).joint_self, 3);
    pad_dim = 16 - dim_1;
    joint_all(count).joint_self = [joint_all(count).joint_self; zeros(pad_dim, 3, dim_3)];
    
    % set scale
    joint_all(count).scale_provided = RELEASE.annolist(i).annorect(p).scale;
    
    % for other person on the same image
    count_other = 1;
    joint_others = cell(0,0);
    for op = 1:numPeople
      if(op == p), continue; end
      try % sometimes no annotation at all....
        anno = RELEASE.annolist(i).annorect(op).annopoints.point;
      catch
        continue;
      end
      joint_others{count_other} = zeros(16,3);
      for part = 1:length(anno)
        joint_all(count).joint_others{count_other}(anno(part).id+1, 1) = anno(part).x;
        joint_all(count).joint_others{count_other}(anno(part).id+1, 2) = anno(part).y;
        try % sometimes no is_visible...
          if(anno(part).is_visible == 0 || anno(part).is_visible == '0')
            joint_all(count).joint_others{count_other}(anno(part).id+1, 3) = 0;
          else
            joint_all(count).joint_others{count_other}(anno(part).id+1, 3) = 1;
          end
        catch
          joint_all(count).joint_others{count_other}(anno(part).id+1, 3) = 1;
        end
        % pad it into 16x3
        dim_1 = size(joint_all(count).joint_others{count_other}, 1);
        dim_3 = size(joint_all(count).joint_others{count_other}, 3);
        pad_dim = 16 - dim_1;
        joint_all(count).joint_others{count_other} = [joint_all(count).joint_others{count_other}; zeros(pad_dim, 3, dim_3)];
      end
      
      joint_all(count).scale_provided_other(count_other) = RELEASE.annolist(i).annorect(op).scale;
      joint_all(count).objpos_other{count_other} = [RELEASE.annolist(i).annorect(op).objpos.x RELEASE.annolist(i).annorect(op).objpos.y];
      
      count_other = count_other + 1;
    end
    
    if(makeFigure) % visualizing to debug
      imshow(imread(fullfile(MPIIROOT, '/images/', joint_all(count).img_paths)));
      hold on;
      visiblePart = joint_all(count).joint_self(:,3) == 1;
      invisiblePart = joint_all(count).joint_self(:,3) == 0;
      plot(joint_all(count).joint_self(visiblePart, 1), joint_all(count).joint_self(visiblePart,2), 'gx', 'MarkerSize', 10);
      plot(joint_all(count).joint_self(invisiblePart,1), joint_all(count).joint_self(invisiblePart,2), 'rx', 'MarkerSize', 10);
      plot(joint_all(count).objpos(1), joint_all(count).objpos(2), 'cs');
      if(~isempty(joint_all(count).joint_others))
        for op = 1:size(joint_all(count).joint_others, 3)
          visiblePart = joint_all(count).joint_others{op}(:,3) == 1;
          invisiblePart = joint_all(count).joint_others{op}(:,3) == 0;
          plot(joint_all(count).joint_others{op}(visiblePart,1), joint_all(count).joint_others{op}(visiblePart,2), 'mx', 'MarkerSize', 10);
          plot(joint_all(count).joint_others{op}(invisiblePart,1), joint_all(count).joint_others{op}(invisiblePart,2), 'cx', 'MarkerSize', 10);
        end
      end
      pause;
      close all;
    end
    joint_all(count).annolist_index = i;
    joint_all(count).people_index = p;
    joint_all(count).numOtherPeople = length(joint_all(count).joint_others);
    count = count + 1;
  end
end

opt.FileName = '../data/mpii/mpii_annotations.json';
opt.FloatFormat = '%.3f';
opt.Compact = 1;
savejson('', joint_all, opt);