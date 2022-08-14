function k = linear_k(X)
    
    [~,n] = size(X);
    k = zeros(n);

    for i = 1:n
        for j = i:n
            k(i,j) = X(:,i)'*X(:,j);
            k(j,i) = k(i,j);
        end
    end


end