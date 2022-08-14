%function to create glcm matrix taken from slides
function G = GLCM(mat,mu,nu,k,size)
    
    %offset will be nans
    C_shift= NaN(size,size);

    %take only the numbers that will exist 1 to end - offset
    C_shift(1:size-mu,1:size-nu) = mat(1+mu:size,1+nu:size);
    
    %initialize frequnecy to 0
    G = zeros(k,k);

    for i = 1:k
        
        %indices for pixels in ith layer
        Ii = (mat == i);

        for j = 1:k
            
            %indices for pixels in jth layer
            Ij = (C_shift == j);

            %set G(i,j) to the total times this is true in an image
            G(i,j) = sum(sum(Ii .* Ij)); 

        end
    end

    %normalize
    G = G * (1/sum(G,'all'));
end