function dist = medoid_dist(x,y)
    x_yes = (x == 1);
    y_yes = (y == 1);
    x_no = (x == -1);
    y_no = (y == -1);
    total_casted = sum( (x ~= 0) & (y ~= 0) );
    dist = sum((x_yes .* y_no) + (x_no .* y_yes))/total_casted;
end