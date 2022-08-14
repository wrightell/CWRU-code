function plot_tree(R,leaves)
    
    %for each leaf plot a rectangle in its bouds 
    for i = leaves
        x = R(i).x;
        y = R(i).y;
        fill([x(1),x(2),x(2),x(1)],[y(1),y(1),y(2),y(2)],R(i).color,'LineStyle','none');
        hold on
    end

end