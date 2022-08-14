function [i_opt,s_opt] = find_best_split(data,vals)
    
    x = data(1,:);
    y = data(2,:);

    %get splits
    [x_splits,y_splits] = get_splits(unique(x),unique(y));
   
    %store best error and split
    bestxerr = realmax;
    bestxsplit = nan;

    for i = x_splits
        xerr = get_error(i,x,vals);
        if xerr < bestxerr
            bestxerr = xerr;
            bestxsplit = i;
        end
    end
    
    %store best error and split
    bestyerr = realmax;  
    bestysplit = nan;
    for j = y_splits
        yerr = get_error(j,y,vals);
        if yerr < bestyerr
            bestyerr = yerr;  
            bestysplit = j;
        end
    end

    %choose best direction and split
    if bestyerr > bestxerr
        i_opt = 1; %split vertically
        s_opt = bestxsplit;
    else
        i_opt = 2; %split horizontally
        s_opt = bestysplit;
    end
end