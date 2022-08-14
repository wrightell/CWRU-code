function [R,leaves] = build_tree(R,leaves)

    %t = 1;
    %besg = 0;
    %for j = 1:length(leaves)
    %    if length(R(leaves(j)).data(1,:)) > besg
    %        besg = length(R(leaves(j)).data(1,:));
    %        t = j;
    %    end
    %end
    %i = leaves(t);
    %leaves(t) = [];

    %get oldest leaf
    i = leaves(1);
    leaves(1) = [];

    %if pure, return
    if length(R(i).data(1,:)) == 1
        leaves = [leaves,i];
        return
    end

    %get optimate split and direction
    [R(i).dir,R(i).split] = opt_split(R(i).data,R(i).vals);
    
    %subset indiceis
    Il = R(i).data(R(i).dir,:) < R(i).split;
    Ir = R(i).data(R(i).dir,:) >= R(i).split;
    
   %subset data
    left = R(i).data(:,Il);
    right = R(i).data(:,Ir);
    
    %subset pixel values
    lvals = R(i).vals(Il,:);
    rvals = R(i).vals(Ir,:);
    
    l = length(R);

    %initialize nodes
    R(i).left = l + 1;
    R(i).right = l + 2;
    
    x = R(i).x;
    y = R(i).y;

        R(l+1).split = [];
        R(l+1).dir = [];
        R(l+1).data = left;
        R(l+1).left = [];
        R(l+1).right = [];
        if R(i).dir == 1
            R(l+1).x = [x(1);R(i).split];
            R(l+1).y = y;
        else
            R(l+1).x = x;
            R(l+1).y = [y(1);R(i).split];
        end

        R(l+1).color = mean(lvals,1);
        R(l+1).vals = lvals;

        
        R(l+2).split = [];
        R(l+2).dir = [];
        R(l+2).data = right;
        R(l+2).left = [];
        R(l+2).right = [];

        if R(i).dir == 1
            R(l+2).x = [R(i).split;x(2)];
            R(l+2).y = y;
        else
            R(l+2).x = x;
            R(l+2).y = [R(i).split;y(2)];
        end

        R(l+2).color = mean(rvals,1);
        R(l+2).vals = rvals;

        leaves = [leaves,l+1,l+2];
end