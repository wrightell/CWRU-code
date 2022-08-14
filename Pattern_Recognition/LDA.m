function directions = LDA(X,I)
    
    %format the dataset and get the centers of the clusters
    [data,labels,cluster_means,k] = process_data(X,I);
    
    %center the mean matrix and calclulate the between cluster scatter
    %matrix
    global_mean = mean(cluster_means,2);
    centered_means = cluster_means - global_mean;
    Sb = centered_means*centered_means';
    
    %calculate the within cluster scatter matrix and make sure it is
    %regularized to ensure it is spd
    Sw = data*data';
    eigen = eigs(Sw,1);
    error = eigen*(10^-10);
    Sw = Sw + error*eye(length(X(:,1)));
    
    %find the cholesky facotrization
    K = chol(Sw);

    %the eigen vectors of this expression maximize its value
    [eigen_vectors,~] = eigs(inv(K') * Sb * inv(K),k-1);

    %get the diretions corresponding to these vectors
    directions = inv(K)*eigen_vectors;

end




