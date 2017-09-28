function scale = cocoScale(x, y, v)
% Mean distance on MPII dataset
% rtorso,  ltorso, rlleg, ruleg, lulleg, llleg,
% rlarm, ruarm, luarm, llarm, head
meandist = [59.3535, 60.4532, 52.1800, 53.7957, 54.4153, 58.0402, ...
 27.0043, 32.8498, 33.1757, 27.0978, 33.3005];

sk = {[13, 7], [6, 12], [17, 15], [15, 13], [12, 14], [14, 16], ...
 [11, 9], [9, 7], [6, 8], [8, 10]};

scale = -1;
for i=1:length(sk), 
  s=sk{i}; 
  if(all(v(s)>0)), 
    scale = norm([x(s(1))-x(s(2)), y(s(1))-y(s(2))])/meandist(i);
    break;
  end; 
end
