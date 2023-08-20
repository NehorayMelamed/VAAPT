function [] = close_if_exist(fig_no)
gh = get(0, 'Children');
for ii=1:length(gh)
    if gh(ii).Number==fig_no;close(gh(ii));end
end
end

