%% Generate JSON file for MSCOCO keypoint data
clear all; close all;
addpath('jsonlab/')
addpath('../data/mscoco/cocoapi/MatlabAPI/');
trainval  = [1, 0];
personCnt = 0;
DEBUG     = false;
year      = 2014; % 2014 or 2017 

for isv = trainval
  isValidation = isv;
  %% initialize COCO api (please specify dataType/annType below)
  annTypes = {'person_keypoints' };
  if isValidation
    dataType = sprintf('val%d', year); annType=annTypes{1}; % specify dataType/annType
  else
    dataType = sprintf('train%d', year); annType=annTypes{1}; % specify dataType/annType
  end
  
  
  annFile = sprintf('../data/mscoco/annotations/%s_%s.json',annType,dataType);
  coco = CocoApi(annFile);
  
  %% display COCO categories and supercategories
  if( ~strcmp(annType,'captions') )
    cats = coco.loadCats(coco.getCatIds());
    sk = cats.skeleton;  % get skeleton
    skc = {'m', 'm', 'g', 'g', 'y', 'r', 'b', 'y', ... % 1-8
      'r', 'b', 'r', 'b', 'c', 'c', 'c', 'c', 'c', 'y', 'y'};
    nms={cats.name}; fprintf('COCO categories: ');
    fprintf('%s, ',nms{:}); fprintf('\n');
    nms=unique({cats.supercategory}); fprintf('COCO supercategories: ');
    fprintf('%s, ',nms{:}); fprintf('\n');
  end
  
  %% get all images containing given categories, select one at random
  catIds = coco.getCatIds('catNms',{'person'});
  imgIds = coco.getImgIds('catIds',catIds);
  imgId = imgIds(randi(length(imgIds)));
  
  imageCnt = 0;
  keypointsCnt = zeros(17, 1);
  fullBodyCnt = 0;
  meanArea = 0;
  
  for i = 1:length(imgIds)
    fprintf('%d | %d\n', i, length(imgIds));
    imgId = imgIds(i);
    img = coco.loadImgs(imgId);
    
    %% load and display annotations
    annIds = coco.getAnnIds('imgIds',imgId,'catIds',catIds,'iscrowd',[]);
    anns = coco.loadAnns(annIds);
    n=length(anns);
    hasKeypoints = false;
    for j=1:n
      a=anns(j); if(a.iscrowd), continue; end; hold on;
      if a.num_keypoints > 0
        hasKeypoints = true;
        
        kp=a.keypoints;
        x=kp(1:3:end)+1; y=kp(2:3:end)+1; v=kp(3:3:end);
        vi = find(v > 0);
        keypointsCnt(vi) = keypointsCnt(vi) + 1;
        meanArea = meanArea + a.area;
        
        scale = cocoScale(x, y, v);
%         if scale == -1 % connot compute scale
        if scale <= 0 % connot compute scale
          continue;
        end
        
        assert(scale ~= 0);
        personCnt = personCnt + 1;
        
        % write to json
        joint_all(personCnt).dataset = 'coco';
        joint_all(personCnt).isValidation = isValidation;
        
        joint_all(personCnt).img_paths = img.file_name;
        joint_all(personCnt).objpos = [mean(x(v>0)), mean(y(v>0))];
        joint_all(personCnt).joint_self = [x; y; v]';
        joint_all(personCnt).scale_provided = scale;
        
        if DEBUG 
          I = imread(sprintf('../data/mscoco/images/%s/%s',dataType,joint_all(personCnt).img_paths));
          imshow(I); hold on;
          x1 = x;
          y1 = y;
          objpos = joint_all(personCnt).objpos;
          viscircles(objpos,5);
          hold on;
          visiblePart = joint_all(personCnt).joint_self(:,3) >= 1;
          invisiblePart = joint_all(personCnt).joint_self(:,3) == 0;
          plot(joint_all(personCnt).joint_self(visiblePart, 1), joint_all(personCnt).joint_self(visiblePart,2), 'y.', 'MarkerSize', 20);
          plot(joint_all(personCnt).joint_self(invisiblePart,1), joint_all(personCnt).joint_self(invisiblePart,2), 'r.', 'MarkerSize', 20);
          plot(joint_all(personCnt).objpos(1), joint_all(personCnt).objpos(2), 'cs');
          pause;close;
        end
      end
      
      if a.num_keypoints == 17
        fullBodyCnt = fullBodyCnt + 1;
      end
    end
    
    if hasKeypoints
      imageCnt = imageCnt + 1;
    end
  end
end
fprintf('save %d person\n', personCnt);

opt.FileName = sprintf('../data/mscoco/coco_annotations_%d.json', year);
opt.FloatFormat = '%.3f';
opt.Compact = 1;
savejson('', joint_all, opt);


%
% clc;
%
% fprintf('validation: images: %d | persons: %d\n', imageCnt, personCnt);
%
% fprintf('%s\n', strjoin(cats.keypoints,', '))
% for i = 1:length(cats.keypoints)
%   fprintf('%d, ', keypointsCnt(i));
% end
%
% fprintf('\nFull body cnt: %d\n', fullBodyCnt);
% fprintf('mean area: %.4f\n', meanArea/personCnt);