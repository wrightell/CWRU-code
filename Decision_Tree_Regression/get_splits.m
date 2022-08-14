function [x_splits,y_splits] = get_splits(x_data,y_data)
    
    %get lengths of possible splits
    n = length(x_data);
    m = length(y_data);
    x_splits = zeros(1,n-1);
    y_splits = zeros(1,m-1);
    
    %set splits to average between points
    for i = 2:n
        x_splits(i-1) = (x_data(i) + x_data(i-1))/2;
    end

    for j = 2:m
        y_splits(j-1) = (y_data(j) + y_data(j-1))/2;
    end

end