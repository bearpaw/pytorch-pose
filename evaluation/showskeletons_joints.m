function h = showskeletons_joints(im, points, pa, msize, torsobox)
if nargin < 4
  msize = 4;
end
if nargin < 5
  torsobox = [];
end
p_no = numel(pa);

switch p_no
  case 26
    partcolor = {'g','g','y','r','r','r','r','y','y','y','m','m','m','m','y','b','b','b','b','y','y','y','c','c','c','c'};
  case 14
    partcolor = {'g','g','y','r','r','y','m','m','y','b','b','y','c','c'};    
  case 10
    partcolor = {'g','g','y','y','y','r','m','m','m','b','b','b','y','c','c'};
  case 18
    partcolor = {'g','g','y','r','r','r','r','y','y','y','y','b','b','b','b','y','y','y'};
  case 16
    partcolor = {'g','g','g','r','r','r','y','y','y','b','b','b','c','c','m','m'};
  otherwise
    error('showboxes: not supported');
end
h = imshow(im); hold on;
if ~isempty(points)
  x = points(:,1);
  y = points(:,2);
  for n = 1:size(x,1)
    for child = 1:p_no
      if child == 0 || pa(child) == 0 
        continue;
      end
      x1 = x(pa(child));
      y1 = y(pa(child));
      x2 = x(child);
      y2 = y(child);
      
      plot(x1, y1, 'o', 'color', partcolor{child}, ...
        'MarkerSize',msize, 'MarkerFaceColor', partcolor{child});
      plot(x2, y2, 'o', 'color', partcolor{child}, ...
        'MarkerSize',msize, 'MarkerFaceColor', partcolor{child});
      line([x1 x2],[y1 y2],'color',partcolor{child},'linewidth',round(msize/2));
    end
  end
end
if ~isempty(torsobox)
  plotbox(torsobox,'w--');
end
drawnow; hold off;
