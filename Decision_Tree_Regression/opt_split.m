function [direction,split_value] = opt_split(X,v)
    
    %split data into directions
    x = X(1,:);
    y = X(2,:);

    %get all the splits
    [x_splits,y_splits] = get_splits(unique(x),unique(y));

    %store best error and split
    bestx = nan;
    bestxerr = realmax;
    
    for i = x_splits
        lv = v(X(1,:) < i,:);
        rv = v(X(1,:) >= i,:);
        err = sum(sum((lv - mean(lv,1)).^2)) + sum(sum((rv - mean(rv,1)).^2));
        if err < bestxerr
            bestx = i;
            bestxerr = err;
        end
    end
    
    %store best error and split
    besty = nan;
    bestyerr = realmax;
    
    for i = y_splits
        lv = v(X(2,:) < i,:);
        rv = v(X(2,:) >= i,:);
        err = sum(sum((lv - mean(lv,1)).^2)) + sum(sum((rv - mean(rv,1)).^2));
        if err < bestyerr
            besty = i;
            bestyerr = err;
        end
    end
    
    %choose best split
    if bestxerr < bestyerr
        direction = 1;
        split_value = bestx;
    else
        direction = 2;
        split_value = besty;
    end

end