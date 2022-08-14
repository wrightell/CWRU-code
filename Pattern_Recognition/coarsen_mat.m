%function that discretizes matrix into k boxes
function C = coarsen_mat(mat,k)
    
    %find interval
    step = 1/k;

    %for each interval, update their values
    for i = 1:k
        mat((mat < step*i) & (mat >= step*(i-1))) = i;
    end
    C = mat;
end